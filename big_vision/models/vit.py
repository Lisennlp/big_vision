# Copyright 2023 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A refactored and simplified ViT.

However, the names of modules are made to match the old ones for easy loading.
"""

import math  # XD
from typing import Any, Callable, Optional, Tuple, Union, overload, Sequence

from absl import logging
from big_vision import utils
from big_vision.models import common
import flax
import flax.linen as nn
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import numpy as np
import scipy.ndimage

from flax.linen.linear import PrecisionLike, default_kernel_init, DenseGeneral
from flax.linen import initializers  # XD
from einops import rearrange, repeat  # XD
from flax.linen.dtypes import promote_dtype
from flax import struct
from flax.linen.normalization import LayerNorm


PRNGKey = jax.Array
Shape = Tuple[int, ...]
Dtype = Any
Array = Any


def posemb_sincos_2d(h, w, width, temperature=10_000., dtype=jnp.float32):
  """Follows the MoCo v3 logic."""
  y, x = jnp.mgrid[:h, :w]

  assert width % 4 == 0, "Width must be mult of 4 for sincos posemb"
  omega = jnp.arange(width // 4) / (width // 4 - 1)
  omega = 1. / (temperature**omega)
  y = jnp.einsum("m,d->md", y.flatten(), omega)
  x = jnp.einsum("m,d->md", x.flatten(), omega)
  pe = jnp.concatenate([jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)], axis=1)
  return jnp.asarray(pe, dtype)[None, :, :]


def get_posemb(self, typ, seqshape, width, name, dtype=jnp.float32):
  if typ == "learn":
    return self.param(name, nn.initializers.normal(stddev=1/np.sqrt(width)),
                      (1, np.prod(seqshape), width), dtype)
  elif typ == "sincos2d":
    return posemb_sincos_2d(*seqshape, width, dtype=dtype)
  else:
    raise ValueError(f"Unknown posemb type: {typ}")



def unbind(ary, n, axis=0):
  return [jnp.squeeze(a, axis=axis) for a in jnp.split(ary, n, axis=axis)]

class DynamicWeightProjection(nn.Module):
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  n_splits: int = None
  num_heads: int = 0
  num_groups: int = 0
  input_dim: int = None
  dynamic_w_init: float = None
  dynamic_d_init: float = None
  dynamic_squeeze_ratio: int = None  # mqy
  decompose_dynamic_w: bool = True
  # dw_activation_cls: activations_lib.BaseActivation = None
  # dw1_norm_cls: normalizations.BaseNormalization = None  # not effective without learned bias # mqy
  dynamic_w_hidden_dim: int = None  # mqy
  # dynamic_d_hidden_dim: int = None
  merge_dynamic_w_hidden: bool = False
  # dw_hidden_activation_cls: activations_lib.BaseActivation = None  # mqy
  deterministic: bool = False
  dynamic_dropout_rate: Optional[float] = None

  def setup(self) -> None:
    self.num_heads_per_group = self.num_heads // self.num_groups
    kwargs = dict(
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      use_bias=False,
      precision=self.precision,
    )
    if self.dynamic_w_init is not None:
      dynamic_hidden_dim = self.num_heads_per_group // self.dynamic_squeeze_ratio \
        if self.dynamic_squeeze_ratio is not None else 2
      print(f'input_dim: {self.input_dim} dynamic_w_hidden_dim: {self.dynamic_w_hidden_dim}')
      self.dw1 = DenseGeneral(features=(self.num_groups, self.n_splits, self.dynamic_w_hidden_dim),
        kernel_init=initializers.normal(math.sqrt(2.0 / (self.input_dim + self.dynamic_w_hidden_dim))), **kwargs)
      self.dw_hidden_activation = nn.gelu

      G, K, M = self.num_groups, self.dynamic_w_hidden_dim, self.num_heads_per_group
      I = dynamic_hidden_dim * 2
      # if not self.decompose_dynamic_w: I = M
      shape = [G, self.n_splits, K, I, M]
      # self.qkw = DenseGeneral(axis=(-3, -2, -1), features=shape[3:],
      #   kernel_init=initializers.normal(self.dynamic_w_init), **kwargs)
      self.qkw = self.param('qkw', initializers.normal(self.dynamic_w_init), shape, self.param_dtype)
  
    if self.dynamic_d_init is not None:
      self.dd = DenseGeneral(features=(self.num_groups, self.num_heads_per_group * self.n_splits),
        kernel_init=initializers.normal(self.dynamic_d_init), **kwargs)

    self.dw_activation = nn.tanh
    self.dw1_norm = nn.RMSNorm(use_scale=False, **{k: v for k, v in kwargs.items() if k not in ['use_bias', 'precision']})

    if self.dynamic_dropout_rate is not None:
      self.dropout = nn.Dropout(self.dynamic_dropout_rate)

  def __call__(self, query_vec):
    print(f'dynamic_dropout_rate: {self.dynamic_dropout_rate}')
    if self.n_splits == 2:
      dw_hidden = self.dw_hidden_activation(self.dw1(query_vec))   # BTG2,64
      if self.dynamic_dropout_rate is not None:
        dw_hidden = self.dropout(dw_hidden, deterministic=self.deterministic)  # XD may add
      # w1, w2 = jnp.split(self.qkw(dw_hidden), 2, axis=-2)
      w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), 2, axis=-2)
      w1 = self.dw1_norm(w1)
      # w2 = self.dw_activation(w2)
      pre_w1, post_w1 = unbind(w1, 2, axis=3) # BTG2IM->[BTGIM]*2
      pre_w2, post_w2 = unbind(w2, 2, axis=3)

      dd = self.dd(query_vec) # jnp.einsum('BTD,DGM->BTGM', query_vec, theta.dd)
      dd = self.dw_activation(dd)
      if self.dynamic_dropout_rate is not None:
        dd = self.dropout(dd, deterministic=self.deterministic)  # XD may add
      pre_dd, post_dd = jnp.split(dd, 2, axis=-1)
      return (pre_w1, pre_w2, pre_dd), (post_w1, post_w2, post_dd)
    else:
      # dw_hidden = jnp.einsum('BTD,DGCK->BTGCK', query_vec, theta.dw1)  # C=4 [pre,post]*[query,key]
      # w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, theta.qkw), 2, axis=-2)
      dw_hidden = self.dw_hidden_activation(self.dw1(query_vec))
      if self.dynamic_dropout_rate is not None:
        dw_hidden = self.dropout(dw_hidden, deterministic=self.deterministic)  # XD may add
      # w1, w2 = jnp.split(self.qkw(dw_hidden), 2, axis=-2)
      w1, w2 = jnp.split(jnp.einsum('BTGCK,GCKIM->BTGCIM', dw_hidden, self.qkw), 2, axis=-2)
      w1 = self.dw1_norm(w1)
      # w2 = self.dw_activation(w2)
      pre_qw1, pre_kw1, post_qw1, post_kw1 = unbind(w1, 4, axis=3) # BTG4IM->[BTGIM]*4
      pre_qw2, pre_kw2, post_qw2, post_kw2 = unbind(w2, 4, axis=3)

      dd = self.dd(query_vec) # jnp.einsum('BTD,DGM->BTGM', query_vec, theta.dd)
      dd = self.dw_activation(dd)
      if self.dynamic_dropout_rate is not None:
        dd = self.dropout(dd, deterministic=self.deterministic)  # XD may add
      pre_qdd, pre_kdd, post_qdd, post_kdd = jnp.split(dd, 4, axis=-1)
      return (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd), \
        (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)

class CrossHeadProjection(nn.Module):
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None

  num_heads: int = 0
  num_groups: int = 0
  relative_scale: float = 0.1
  use_static_w: bool = True
  loop_over_dynamic_hd: bool = True
  tgt_dependent: bool = True
  src_dependent: bool = True

  def setup(self) -> None:
    self.num_heads_per_group = self.num_heads // self.num_groups
    kwargs = dict(
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      use_bias=False,
      precision=self.precision,
    )
    # self.w = DenseGeneral(axis=(1, 2), features=(self.num_heads_per_group,),  # BGMTS,GMN->BGNTS
    #   kernel_init=initializers.normal(math.sqrt(1. / self.num_heads_per_group) * self.relative_scale), **kwargs)
    shape = (self.num_groups, self.num_heads_per_group, self.num_heads_per_group)
    self.w = self.param('w', initializers.normal(math.sqrt(1. / self.num_heads_per_group) * self.relative_scale), shape, self.param_dtype)

  def __call__(self, inputs, qw1 = None, qw2 = None, kw1 = None, kw2 = None, qdd = None, kdd = None):
    shape = inputs.shape
    assert inputs.shape[1] == self.num_heads
    inputs = rearrange(inputs, 'B (G M) T S -> B G M T S', G=self.num_groups)
    inputs_label = 'BGMTS'

    ret = inputs
    # ret += self.w(inputs)  # BGMTS,GMN->BGNTS
    ret += jnp.einsum('BGMTS,GMN->BGNTS', inputs, self.w)

    if qw1 is not None:
      hidden_sym = 'I'; hidden_label = inputs_label.replace('M', 'I')
      for sym, (w1, w2) in zip(['T', 'S'], [(qw1, qw2), (kw1, kw2)]):
        dw_label = f'B{sym}G{hidden_sym}M' if w1.shape[-1] == self.num_heads_per_group \
          else f'B{sym}GM{hidden_sym}'  # w1.shape[-2] == self.num_heads_per_group
        dynamic_hidden_dim = w1.shape[dw_label.index(hidden_sym)]
        eqn1 = f'{inputs_label},{dw_label}->{hidden_label}' # 'BGMTS,BTGMI->BGITS'
        eqn2 = f'{hidden_label},{dw_label}->{inputs_label}' # 'BGITS,BTGMI->BGMTS'
        if sym == 'T' and self.tgt_dependent or sym == 'S' and self.src_dependent:
          if self.loop_over_dynamic_hd and dynamic_hidden_dim <= 2:
            for i in range(dynamic_hidden_dim):
              if dw_label[-1] == hidden_sym:
                hidden = jnp.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i])
                out = jnp.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i])
              else:
                assert dw_label[-2] == hidden_sym, dw_label
                hidden = jnp.einsum(eqn1.replace(hidden_sym, ''), inputs, w1[..., i, :])
                out = jnp.einsum(eqn2.replace(hidden_sym, ''), hidden, w2[..., i, :])
              ret = ret + out
          else:
            hidden = jnp.einsum(eqn1, inputs, w1)
            if self.decompose_dynamic_w:
              out = jnp.einsum(eqn2, hidden, w2)
              ret = ret + out
            else:
              ret = ret + hidden

    if qdd is not None:
      for sym, dd in zip(['T', 'S'], [qdd, kdd]):
        dd_label = f'B{sym}GM'
        if sym == 'T' and self.tgt_dependent or sym == 'S' and self.src_dependent or \
              not self.tgt_dependent and not self.src_dependent:
          dout = jnp.einsum(f'{inputs_label},{dd_label}->{inputs_label}', inputs, dd)
          ret = ret + dout
    return jnp.reshape(ret, shape)  # BGMTS->BNTS

class MultiHeadDotProductAttention(nn.Module):
  num_heads: int
  num_groups: int = 1
  dtype: Optional[Dtype] = None
  param_dtype: Dtype = jnp.float32
  qkv_features: Optional[int] = None
  broadcast_dropout: bool = True
  dropout_rate: float = 0.0
  deterministic: Optional[bool] = None
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_kernel_init
  bias_init: Callable[
    [PRNGKey, Shape, Dtype], Array
  ] = nn.zeros_init()
  use_bias: bool = True
  decode: bool = False
  # normalize_qk: bool = False
  dynamic_compose: bool = True  # XD
  is_cross_attention: bool = False  # XD
  dynamic_dropout_rate: float = None

  def setup(self):
    self.head_dim = self.qkv_features // self.num_heads
    kwargs = dict(
      dtype=self.dtype,
      param_dtype=self.param_dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      use_bias=self.use_bias,
      precision=self.precision,
    )

    self.query_dense = DenseGeneral(features=(self.num_heads, self.head_dim), **kwargs)
    self.key_dense = DenseGeneral(features=(self.num_heads, self.head_dim), **kwargs)
    self.value_dense = DenseGeneral(features=(self.num_heads, self.head_dim), **kwargs)
    self.o_dense = nn.Dense(features=self.qkv_features, **kwargs)

    if self.dynamic_compose:
      input_dim = self.num_heads * self.head_dim
      I = 2
      num_heads_per_group = self.num_heads // self.num_groups
      dynamic_w_hidden_dim = num_heads_per_group * I * 2
      if self.is_cross_attention:
        for name in ['q_dyn_w_proj', 'k_dyn_w_proj']:
          setattr(self, name, DynamicWeightProjection(
            num_heads=self.num_heads, num_groups=self.num_groups,
            input_dim=self.num_heads * self.head_dim, n_splits=2,
            dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
            dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
            dynamic_squeeze_ratio=num_heads_per_group // I,
            dynamic_w_hidden_dim=dynamic_w_hidden_dim,
            dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
            deterministic=self.deterministic,
            dynamic_dropout_rate=self.dynamic_dropout_rate,
          ))
      else:
        self.dyn_w_proj = DynamicWeightProjection(
          num_heads=self.num_heads, num_groups=self.num_groups,
          input_dim=self.num_heads * self.head_dim, n_splits=4,
          dynamic_w_init=math.sqrt(1 / dynamic_w_hidden_dim) * 2 / (num_heads_per_group + I) * 0.01,
          dynamic_d_init=math.sqrt(2 / (input_dim + num_heads_per_group)) * 0.005,
          dynamic_squeeze_ratio=num_heads_per_group // I,
          dynamic_w_hidden_dim=dynamic_w_hidden_dim,
          dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
          deterministic=self.deterministic,
          dynamic_dropout_rate=self.dynamic_dropout_rate,
        )
      for name in ['pre_proj', 'post_proj']:
        setattr(self, name, CrossHeadProjection(
          num_heads=self.num_heads, num_groups=self.num_groups,
          dtype=self.dtype, param_dtype=self.param_dtype, precision=self.precision,
        ))

  def dot_product_attention(
    self,
    query: Array,
    key: Array,
    value: Array,
    inputs_q: Optional[Array] = None,  # XD
    inputs_k: Optional[Array] = None,  # XD
    bias: Optional[Array] = None,
    mask: Optional[Array] = None,
    broadcast_dropout: bool = True,
    dropout_rng: Optional[PRNGKey] = None,
    dropout_rate: float = 0.0,
    deterministic: bool = False,
    dtype: Optional[Dtype] = None,
    precision: PrecisionLike = None,
    module: Optional[nn.Module] = None,
  ):
   # 所有参数转为dtype类型
    query, key = promote_dtype(query, key, dtype=dtype)
    dtype = query.dtype
    assert query.ndim == key.ndim, 'q, k must have same rank.'
    assert query.shape[:-3] == key.shape[:-3], 'q, k batch dims must match.'
    assert query.shape[-2] == key.shape[-2], 'q, k num_heads must match.'
    assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'
    # calculate attention matrix
    depth = query.shape[-1]
    query = query / jnp.sqrt(depth).astype(dtype)
    # attn weight shape is (batch..., num_heads, q_length, kv_length)
    attn_weights = jnp.einsum(
      '...qhd,...khd->...hqk', query, key, precision=precision
    )
    if self.dynamic_compose:  # XD
      if self.is_cross_attention:
        (pre_qw1, pre_qw2, pre_qdd), (post_qw1, post_qw2, post_qdd) = self.q_dyn_w_proj(inputs_q)
        (pre_kw1, pre_kw2, pre_kdd), (post_kw1, post_kw2, post_kdd) = self.k_dyn_w_proj(inputs_k)
      else:
        (pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd), \
        (post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd) = self.dyn_w_proj(inputs_q)
      attn_weights = self.pre_proj(attn_weights, pre_qw1, pre_qw2, pre_kw1, pre_kw2, pre_qdd, pre_kdd)

    # apply attention bias: masking, dropout, proximity bias, etc.
    if bias is not None:
      attn_weights = attn_weights + bias
    # apply attention mask
    if mask is not None:
      big_neg = jnp.finfo(dtype).min
      attn_weights = jnp.where(mask, attn_weights, big_neg)
    # normalize the attention weights
    attn_weights = jax.nn.softmax(attn_weights).astype(dtype)
    if self.dynamic_compose:  # XD
      attn_weights = self.post_proj(attn_weights, post_qw1, post_qw2, post_kw1, post_kw2, post_qdd, post_kdd)
    if module:
      module.sow('intermediates', 'attention_weights', attn_weights)
    # apply attention dropout
    if not deterministic and dropout_rate > 0.0:
      keep_prob = 1.0 - dropout_rate
      if broadcast_dropout:
        # dropout is broadcast across the batch + head dimensions
        dropout_shape = tuple([1] * (key.ndim - 2)) + attn_weights.shape[-2:]
        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)  # type: ignore
      else:
        keep = jax.random.bernoulli(dropout_rng, keep_prob, attn_weights.shape)  # type: ignore
      multiplier = keep.astype(dtype) / jnp.asarray(keep_prob, dtype=dtype)
      attn_weights = attn_weights * multiplier

    outputs = jnp.einsum(
      '...hqk,...khd->...qhd', attn_weights, value, precision=precision
    )
    return outputs

  @overload
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    ...

  @overload
  def __call__(
    self,
    inputs_q: Array,
    *,
    inputs_kv: Array = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    ...

  @nn.compact
  def __call__(
    self,
    inputs_q: Array,
    inputs_k: Optional[Array] = None,
    inputs_v: Optional[Array] = None,
    *,
    inputs_kv: Optional[Array] = None,
    mask: Optional[Array] = None,
    deterministic: Optional[bool] = None,
    dropout_rng: Optional[PRNGKey] = None,
    sow_weights: bool = False,
  ):
    if inputs_kv is not None:
      if inputs_k is not None or inputs_v is not None:
        raise ValueError(
          'If either `inputs_k` or `inputs_v` is not None, '
          '`inputs_kv` must be None. If `inputs_kv` is not None, both `inputs_k` '
          'and `inputs_v` must be None. We recommend using `inputs_k` and '
          '`inputs_v` args, since `inputs_kv` will be deprecated soon. See '
          'https://github.com/google/flax/discussions/3389 for more '
          'information.'
        )
      inputs_k = inputs_v = inputs_kv
      warnings.warn(
        'The inputs_kv arg will be deprecated soon. '
        'Use inputs_k and inputs_v instead. See '
        'https://github.com/google/flax/discussions/3389 '
        'for more information.',
        DeprecationWarning,
      )
    else:
      if inputs_k is None:
        if inputs_v is not None:
          raise ValueError(
            '`inputs_k` cannot be None if `inputs_v` is not None. '
            'To have both `inputs_k` and `inputs_v` be the same value, pass in the '
            'value to `inputs_k` and leave `inputs_v` as None.'
          )
        inputs_k = inputs_q
      if inputs_v is None:
        inputs_v = inputs_k
      elif inputs_v.shape[-1] == inputs_v.shape[-2]:
        warnings.warn(
          f'You are passing an array of shape {inputs_v.shape} '
          'to the `inputs_v` arg, when you may have intended '
          'to pass it to the `mask` arg. As of Flax version '
          '0.7.4, the function signature of '
          "MultiHeadDotProductAttention's `__call__` method "
          'has changed to `__call__(inputs_q, inputs_k=None, '
          'inputs_v=None, *, inputs_kv=None, mask=None, '
          'deterministic=None)`. Use the kwarg `mask` instead. '
          'See https://github.com/google/flax/discussions/3389 '
          'and read the docstring for more information.',
          DeprecationWarning,
        )
    bsz, length, model_dim = inputs_q.shape
    # features = self.out_features or inputs_q.shape[-1]
    qkv_features = self.qkv_features or inputs_q.shape[-1]
    assert qkv_features % self.num_heads == 0, (
      f'Memory dimension ({qkv_features}) must be divisible by number of'
      f' heads ({self.num_heads}).'
    )
    # dimensions are then [batch..., length, n_heads, n_features_per_head]
    query = self.query_dense(inputs_q)
    key = self.key_dense(inputs_k)
    value = self.value_dense(inputs_v)
    # query = query.reshape(bsz, length, self.num_heads, self.head_dim)
    # key = key.reshape(bsz, length, self.num_heads, self.head_dim)
    # value = value.reshape(bsz, length, self.num_heads, self.head_dim)

    # if self.normalize_qk:
    if self.dynamic_compose:  # XD
      # Normalizing query and key projections stabilizes training with higher
      # LR. See ViT-22B paper http://arxiv.org/abs/2302.05442 for analysis.
      query = LayerNorm(name='query_ln', use_bias=False)(query)  # type: ignore[call-arg]
      key = LayerNorm(name='key_ln', use_bias=False)(key)  # type: ignore[call-arg]

    # During fast autoregressive decoding, we feed one position at a time,
    # and cache the keys and values step by step.
    if self.decode:
      # detect if we're initializing by absence of existing cache data.
      is_initialized = self.has_variable('cache', 'cached_key')
      cached_key = self.variable(
        'cache', 'cached_key', jnp.zeros, key.shape, key.dtype
      )
      cached_value = self.variable(
        'cache', 'cached_value', jnp.zeros, value.shape, value.dtype
      )
      cache_index = self.variable(
        'cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32)
      )
      if is_initialized:
        (
          *batch_dims,
          max_length,
          num_heads,
          depth_per_head,
        ) = cached_key.value.shape
        # shape check of cached keys against query input
        expected_shape = tuple(batch_dims) + (1, num_heads, depth_per_head)
        if expected_shape != query.shape:
          raise ValueError(
            'Autoregressive cache shape error, '
            'expected query shape %s instead got %s.'
            % (expected_shape, query.shape)
          )
        # update key, value caches with our new 1d spatial slices
        cur_index = cache_index.value
        zero = jnp.array(0, dtype=lax.dtype(cur_index.dtype))
        indices: tuple[Union[int, jax.Array], ...] = (zero,) * len(batch_dims) + (
          cur_index,
          zero,
          zero,
        )
        key = lax.dynamic_update_slice(cached_key.value, key, indices)
        value = lax.dynamic_update_slice(cached_value.value, value, indices)
        cached_key.value = key
        cached_value.value = value
        cache_index.value = cache_index.value + 1
        # causal mask for cached decoder self-attention:
        # our single query position should only attend to those key
        # positions that have already been generated and cached,
        # not the remaining zero elements.
        mask = nn.combine_masks(
          mask,
          jnp.broadcast_to(
            jnp.arange(max_length) <= cur_index,
            tuple(batch_dims) + (1, 1, max_length),
          ),
        )

    if (
      self.dropout_rate > 0.0
    ):  # Require `deterministic` only if using dropout.
      m_deterministic = nn.merge_param(
        'deterministic', self.deterministic, deterministic
      )
      if not m_deterministic and dropout_rng is None:
        dropout_rng = self.make_rng('dropout')
    else:
      m_deterministic = True
    # bsz * lenght * n * head_dim
    x = self.dot_product_attention(
        query,
        key,
        value,
        inputs_q=inputs_q,  # XD
        inputs_k=inputs_k,  # XD
        mask=mask,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        broadcast_dropout=self.broadcast_dropout,
        deterministic=m_deterministic,
        dtype=self.dtype,
        precision=self.precision,
        module=self if sow_weights else None,
      )  # pytype: disable=wrong-keyword-args
    x = x.reshape(bsz, length, -1)
    out = self.o_dense(x)
    return out


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  dropout: float = 0.0
  dtype_mm: str = "float32"

  @nn.compact
  def __call__(self, x, deterministic=True):
    """Applies Transformer MlpBlock module."""
    inits = dict(
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6),
    )

    n, l, d = x.shape  # pylint: disable=unused-variable
    x = nn.Dense(self.mlp_dim or 4 * d, dtype=self.dtype_mm, **inits)(x)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout)(x, deterministic)
    x = nn.Dense(d, dtype=self.dtype_mm, **inits)(x)
    return x


class Encoder1DBlock(nn.Module):
  """Single transformer encoder block (MHSA + MLP)."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  dtype_mm: str = "float32"
  dc_config: dict = None

  @nn.compact
  def __call__(self, x, deterministic=True):
    if self.dc_config is None:
        self.dc_config = {}
    out = {}
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    y = nn.LayerNorm()(x)

    y = out['sa'] = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype_mm,
        qkv_features=y.shape[-1],
        kernel_init=nn.initializers.xavier_uniform(),
        deterministic=deterministic,
        **self.dc_config
    )(y)

    # y = out["sa"] = nn.MultiHeadDotProductAttention(
    #     num_heads=self.num_heads,
    #     kernel_init=nn.initializers.xavier_uniform(),
    #     deterministic=deterministic,
    #     dtype=self.dtype_mm,
    # )(y, y)

    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+sa"] = x + y

    y = nn.LayerNorm()(x)
    y = out["mlp"] = MlpBlock(
        mlp_dim=self.mlp_dim, dropout=self.dropout,
        dtype_mm=self.dtype_mm,
    )(y, deterministic)
    y = nn.with_logical_constraint(y, ("act_batch", "act_len", "act_emb"))
    y = nn.Dropout(rate=self.dropout)(y, deterministic)
    x = out["+mlp"] = x + y
    x = nn.with_logical_constraint(x, ("act_batch", "act_len", "act_emb"))
    return x, out


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""
  depth: int
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  dropout: float = 0.0
  scan: bool = False
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"
  dc_config: dict = None

  @nn.compact
  def __call__(self, x, deterministic=True):
    out = {}

    if self.scan:
      block = nn.remat(
          Encoder1DBlock,
          prevent_cse=False,
          static_argnums=(-1,),
          policy=getattr(jax.checkpoint_policies, self.remat_policy, None),
          )
      x, _ = nn.scan(block,
                     variable_axes={"params": 0},
                     split_rngs={"params": True, "dropout": True},
                     in_axes=nn.broadcast,
                     length=self.depth)(
                         name="encoderblock",
                         dtype_mm=self.dtype_mm,
                         mlp_dim=self.mlp_dim,
                         num_heads=self.num_heads,
                         dropout=self.dropout)(x, deterministic)
    else:
      # Input Encoder
      for lyr in range(self.depth):
        block_cur = Encoder1DBlock(
            name=f"encoderblock_{lyr}",
            dtype_mm=self.dtype_mm,
            mlp_dim=self.mlp_dim, num_heads=self.num_heads,
            dc_config=self.dc_config,
            dropout=self.dropout)
        x, out[f"block{lyr:02d}"] = block_cur(x, deterministic)
      out["pre_ln"] = x  # Alias for last block, but without the number in it.

    return nn.LayerNorm(name="encoder_norm")(x), out


class MAPHead(nn.Module):
  """Multihead Attention Pooling."""
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12

  @nn.compact
  def __call__(self, x):
    # TODO
    n, l, d = x.shape  # pylint: disable=unused-variable
    probe = self.param("probe", nn.initializers.xavier_uniform(),
                       (1, 1, d), x.dtype)
    probe = jnp.tile(probe, [n, 1, 1])

    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        kernel_init=nn.initializers.xavier_uniform())(probe, x)

    # TODO: dropout on head?
    y = nn.LayerNorm()(x)
    x = x + MlpBlock(mlp_dim=self.mlp_dim)(y)
    return x[:, 0]


class _Model(nn.Module):
  """ViT model."""

  num_classes: Optional[int] = None
  patch_size: Sequence[int] = (16, 16)
  width: int = 768
  depth: int = 12
  mlp_dim: Optional[int] = None  # Defaults to 4x input dim
  num_heads: int = 12
  posemb: str = "learn"  # Can also be "sincos2d"
  rep_size: Union[int, bool] = False
  dropout: float = 0.0
  pool_type: str = "gap"  # Can also be "map" or "tok"
  head_zeroinit: bool = True
  scan: bool = False
  # or "dots_with_no_batch_dims_saveable" for more speed (memory costly)
  remat_policy: str = "nothing_saveable"
  dtype_mm: str = "float32"
  dc_config: dict = None

  @nn.compact
  def __call__(self, image, *, train=False):
    logging.info(f'width: {self.width} depth: {self.depth} num_heads: {self.num_heads} dc_config: {self.dc_config}')
    out = {}
    image = jnp.asarray(image, self.dtype_mm)

    # Patch extraction,
    # (16, 16, 3, 768)， conv shape: (*patch_size, 3, width)
    # image:  batch * 长 * 宽 * 3
    # 得到的向量x: batch * (长 * 宽 / patch_size) * 3 * width
    x = out["stem"] = nn.Conv(
        self.width, self.patch_size, strides=self.patch_size,
        padding="VALID", name="embedding", dtype=self.dtype_mm)(image)
    # n: batch,  h: 长/patch   w: 宽/patch  c: width
    n, h, w, c = x.shape
    # logging.info(f'n: {n} h: {h} w: {w} c: {c}')
    x = jnp.reshape(x, [n, h * w, c])

    # Add posemb before adding extra token.
    x = out["with_posemb"] = x + get_posemb(
        self, self.posemb, (h, w), c, "pos_embedding", x.dtype)

    if self.pool_type == "tok":
      cls = self.param("cls", nn.initializers.zeros, (1, 1, c), x.dtype)
      x = jnp.concatenate([jnp.tile(cls, [n, 1, 1]), x], axis=1)

    n, l, c = x.shape  # pylint: disable=unused-variable
    x = nn.Dropout(rate=self.dropout)(x, not train)

    x, out["encoder"] = Encoder(
        depth=self.depth,
        mlp_dim=self.mlp_dim,
        num_heads=self.num_heads,
        dropout=self.dropout,
        scan=self.scan,
        remat_policy=self.remat_policy,
        dtype_mm=self.dtype_mm,
        dc_config=self.dc_config,
        name="Transformer")(
            x, deterministic=not train)
    encoded = out["encoded"] = x

    if self.pool_type == "map":
      x = out["head_input"] = MAPHead(
          num_heads=self.num_heads, mlp_dim=self.mlp_dim)(x)
    elif self.pool_type == "gap":
      x = out["head_input"] = jnp.mean(x, axis=1)
    elif self.pool_type == "0":
      x = out["head_input"] = x[:, 0]
    elif self.pool_type == "tok":
      x = out["head_input"] = x[:, 0]
      encoded = encoded[:, 1:]
    else:
      raise ValueError(f"Unknown pool type: '{self.pool_type}'")

    x_2d = jnp.reshape(encoded, [n, h, w, -1])

    if self.rep_size:
      rep_size = self.width if self.rep_size is True else self.rep_size
      hid = nn.Dense(rep_size, name="pre_logits")
      # NOTE: In the past we did not include tanh in pre_logits.
      # For few-shot, it should not matter much, as it whitens anyways.
      x_2d = nn.tanh(hid(x_2d))
      x = nn.tanh(hid(x))

    out["pre_logits_2d"] = x_2d
    out["pre_logits"] = x

    if self.num_classes:
      kw = {"kernel_init": nn.initializers.zeros} if self.head_zeroinit else {}
      head = nn.Dense(self.num_classes, name="head", **kw)
      x_2d = out["logits_2d"] = head(x_2d)
      x = out["logits"] = head(x)

    return x, out


def Model(num_classes=None, *, variant=None, **kw):  # pylint: disable=invalid-name
  """Factory function, because linen really don't like what I'm doing!"""
  return _Model(num_classes, **{**decode_variant(variant), **kw})


def decode_variant(variant):
  """Converts a string like "B" or "B/32" into a params dict."""
  if variant is None:
    return {}

  v, patch = variant, {}
  if "/" in variant:
    v, patch = variant.split("/")
    patch = {"patch_size": (int(patch), int(patch))}

  return {
      # pylint:disable=line-too-long
      # Reference: Table 2 of https://arxiv.org/abs/2106.04560.
      "width": {"mu": 32, "Ti": 192, "S": 384, "M": 512, "B": 768, "L": 1024, "So400m": 1152, "H": 1280, "g": 1408, "g-opt": 1536, "G": 1664, "G-opt": 1536, "e": 1792}[v],
      "depth": {"mu": 1, "Ti": 12, "S": 12, "M": 12, "B": 12, "L": 24, "So400m": 27, "H": 32, "g": 40, "g-opt": 40, "G": 48, "G-opt": 48, "e": 56}[v],
      "mlp_dim": {"mu": 128, "Ti": 768, "S": 1536, "M": 2048, "B": 3072, "L": 4096, "So400m": 4304, "H": 5120, "g": 6144, "g-opt": 6144, "G": 8192, "G-opt": 8192, "e": 15360}[v],
      "num_heads": {"mu": 2, "Ti": 3, "S": 6, "M": 8, "B": 12, "L": 16, "So400m": 16, "H": 16, "g": 16, "g-opt": 16, "G": 16, "G-opt": 16, "e": 16}[v],
      # pylint:enable=line-too-long
      **patch
  }


def resample_posemb(old, new):
  """This function implements "high-res finetuning" for transformer models."""
  # Rescale the grid of position embeddings. Param shape is (1,N,1024)
  if old.shape == new.shape:
    return old

  logging.info("ViT: resize %s to %s", old.shape, new.shape)
  gs_old = int(np.sqrt(old.shape[1]))
  gs_new = int(np.sqrt(new.shape[1]))
  logging.info("ViT: grid-size from %s to %s", gs_old, gs_new)
  grid = old.reshape(gs_old, gs_old, -1)

  zoom = (gs_new/gs_old, gs_new/gs_old, 1)
  grid = scipy.ndimage.zoom(grid, zoom, order=1)
  grid = grid.reshape(1, gs_new*gs_new, -1)
  return grid


def fix_old_checkpoints(params):
  """Fix small bwd incompat that can't be resolved with names in model def."""

  params = flax.core.unfreeze(
      flax.training.checkpoints.convert_pre_linen(params))

  # Original ViT paper variant had posemb in a module:
  if "posembed_input" in params["Transformer"]:
    logging.info("ViT: Loading and fixing VERY old posemb")
    posemb = params["Transformer"].pop("posembed_input")
    params["pos_embedding"] = posemb["pos_embedding"]

  # Widely used version before 2022 had posemb in Encoder:
  if "pos_embedding" in params["Transformer"]:
    logging.info("ViT: Loading and fixing old posemb")
    params["pos_embedding"] = params["Transformer"].pop("pos_embedding")

  # Old vit.py used to first concat [cls] token, then add posemb.
  # This means a B/32@224px would have 7x7+1 posembs. This is useless and clumsy
  # so we changed to add posemb then concat [cls]. We can recover the old
  # checkpoint by manually summing [cls] token and its posemb entry.
  if "pos_embedding" in params:
    pe = params["pos_embedding"]
    if int(np.sqrt(pe.shape[1])) ** 2 + 1 == int(pe.shape[1]):
      logging.info("ViT: Loading and fixing combined cls+posemb")
      pe_cls, params["pos_embedding"] = pe[:, :1], pe[:, 1:]
      if "cls" in params:
        params["cls"] += pe_cls

  # MAP-head variants during ViT-G development had it inlined:
  if "probe" in params:
    params["MAPHead_0"] = {
        k: params.pop(k) for k in
        ["probe", "MlpBlock_0", "MultiHeadDotProductAttention_0", "LayerNorm_0"]
    }

  return params


def pyloop_to_scan(params_pyloop):
  """Converts a python for-loop ViT checkpoint to a lax.scan based one."""
  # On a high level, they are the same except that the for loop has separate
  # array pytrees for each encoderblock, while the scan one has just one
  # encoderblock pytree, with all block's params concatenated.

  params_scan = jax.tree_map(lambda x: x, params_pyloop)  # Structural copy
  t = params_scan["Transformer"]

  # Find highest index of encoderblocks in the checkpoint (they start at 0):
  encoderblocks = {k for k in t if k.startswith("encoderblock_")}
  depth = 1 + max({int(k.split("_")[-1]) for k in encoderblocks})

  def stack(*values):
    return np.stack(values)

  # Stack all encoderblocks into a single one:
  t["encoderblock"] = jax.tree_map(
      stack, *[t[f"encoderblock_{lyr}"] for lyr in range(depth)])

  for lyr in range(depth):
    del t[f"encoderblock_{lyr}"]

  return params_scan


def scan_to_pyloop(params_scan):
  """Converts a lax.scan ViT checkpoint to a python for-loop based one."""
  # See comment in pyloop_to_scan.

  params_scan = jax.tree_map(lambda x: x, params_scan)  # Structural copy
  t = params_scan["Transformer"]

  # Find out how many encoderblocks there are
  depth = len(t["encoderblock"]["LayerNorm_0"]["bias"])

  # Create that many encoderblocks, each with their slice of their sub-pytree.
  for lyr in range(depth):
    block = jax.tree_map(lambda x, lyr=lyr: x[lyr], t["encoderblock"])
    t[f"encoderblock_{lyr}"] = block

  del t["encoderblock"]
  return params_scan


def load(init_params, init_file, model_cfg, dont_load=()):  # pylint: disable=invalid-name because we had to CamelCase above.
  """Load init from checkpoint, both old model and this one. +Hi-res posemb."""
  init_file = VANITY_NAMES.get(init_file, init_file)
  restored_params = utils.load_params(init_file)

  restored_params = fix_old_checkpoints(restored_params)

  # Detect attempts to load non-scan checkpoint into scan model.
  if (model_cfg.get("scan") and
      "encoderblock" not in restored_params["Transformer"]):
    restored_params = pyloop_to_scan(restored_params)
  if (not model_cfg.get("scan")
      and "encoderblock" in restored_params["Transformer"]):
    restored_params = scan_to_pyloop(restored_params)

  # possibly use the random init for some of the params (such as, the head).
  restored_params = common.merge_params(restored_params, init_params, dont_load)

  # resample posemb if needed.
  # TODO: Take this from model_cfg to avoid need for init_params.
  if init_params and "pos_embedding" in init_params:
    restored_params["pos_embedding"] = resample_posemb(
        old=restored_params["pos_embedding"],
        new=init_params["pos_embedding"])

  return restored_params


# Shortcut names for some canonical paper checkpoints:
VANITY_NAMES = {
    # pylint: disable=line-too-long
    # Recommended models from https://arxiv.org/abs/2106.10270
    # Many more models at https://github.com/google-research/vision_transformer
    "howto-i21k-Ti/16": "gs://vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-S/32": "gs://vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_none-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-S/16": "gs://vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz",
    "howto-i21k-B/32": "gs://vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/16": "gs://vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-B/8": "gs://vit_models/augreg/B_8-i21k-300ep-lr_0.001-aug_medium2-wd_0.1-do_0.0-sd_0.0.npz",
    "howto-i21k-L/16": "gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0.npz",

    # Better plain vit-s16 baselines from https://arxiv.org/abs/2205.01580
    "i1k-s16-90ep": "gs://big_vision/vit_s16_i1k_90ep.npz",
    "i1k-s16-150ep": "gs://big_vision/vit_s16_i1k_150ep.npz",
    "i1k-s16-300ep": "gs://big_vision/vit_s16_i1k_300ep.npz",

    # DeiT-3 checkpoints from https://github.com/facebookresearch/deit/blob/main/README_revenge.md
    # First layer converted to take inputs in [-1,1]
    "deit3_S_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_1k.npz",
    "deit3_S_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_224_21k.npz",
    "deit3_S_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_1k.npz",
    "deit3_S_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_small_384_21k.npz",
    "deit3_B_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_1k.npz",
    "deit3_B_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_224_21k.npz",
    "deit3_B_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_1k.npz",
    "deit3_B_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_base_384_21k.npz",
    "deit3_L_224_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_1k.npz",
    "deit3_L_224_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_224_21k.npz",
    "deit3_L_384_1k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_1k.npz",
    "deit3_L_384_21k": "gs://big_vision/zoo/deit3/bv_deit_3_large_384_21k.npz",

    # SigLIP image encoder checkpoints from https://arxiv.org/abs/2303.15343
    "SigLIP B/16 224": "gs://big_vision/siglip/webli_en_b16_224_63724782.npz:img",
    "SigLIP B/16 256": "gs://big_vision/siglip/webli_en_b16_256_60500360.npz:img",
    "SigLIP B/16 384": "gs://big_vision/siglip/webli_en_b16_384_68578854.npz:img",
    "SigLIP B/16 512": "gs://big_vision/siglip/webli_en_b16_512_68580893.npz:img",
    "SigLIP L/16 256": "gs://big_vision/siglip/webli_en_l16_256_60552751.npz:img",
    "SigLIP L/16 384": "gs://big_vision/siglip/webli_en_l16_384_63634585.npz:img",
    "SigLIP So400m/14 224": "gs://big_vision/siglip/webli_en_so400m_224_57633886.npz:img",
    "SigLIP So400m/14 384": "gs://big_vision/siglip/webli_en_so400m_384_58765454.npz:img",
    "SigLIP B/16-i18n 256": "gs://big_vision/siglip/webli_i18n_b16_256_66117334.npz:img",
    # pylint: enable=line-too-long
}
