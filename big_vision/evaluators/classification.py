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

"""Evaluator for the classfication task."""
# pylint: disable=consider-using-from-import

import functools

import big_vision.datasets.core as ds_core
import big_vision.input_pipeline as input_pipeline
import big_vision.pp.builder as pp_builder
import big_vision.utils as u
import jax
import jax.numpy as jnp


# Temporary global flag to facilitate backwards compatability. Will be removed
# by the end of year 2023.
API = 'jit'


# To avoid re-compiling the function for every new instance of the same
# evaluator on a different dataset!
@functools.cache
def get_eval_fn(predict_fn, loss_name, topk=1):
  """Produces eval function, also applies pmap."""
  @jax.jit
  def _eval_fn(train_state, batch, labels, mask):
    logits, *_ = predict_fn(train_state, batch)

    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)

    loss = getattr(u, loss_name)(
        logits=logits, labels=labels, reduction=False)
    loss = jnp.sum(loss * mask)
    print(f'logits: {logits.shape}')
    #lsp: batch * topk
    topk_values, topk_idx = jax.lax.top_k(logits, k=topk)
    top1_correct = jnp.zeros_like(mask, dtype=jnp.bool_)
    corrects = []
    for i in range(topk):
        top1_idx = topk_idx[:,i]
        correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
        correct = correct.astype(jnp.bool_)
        top1_correct |= correct
        top1_sum_correct = jnp.sum(top1_correct * mask)
        corrects.append(top1_sum_correct)

    nseen = jnp.sum(mask)
    print(f'nseen: {nseen}')
    return corrects, loss, nseen
  return _eval_fn


class Evaluator:
  """Classification evaluator."""

  def __init__(self, predict_fn, data, pp_fn, batch_size, loss_name,
               cache_final=True, cache_raw=False, prefetch=1, topk=1,
               label_key='labels', *, devices):
    self.topk = topk
    data = ds_core.get(**data)
    pp_fn = pp_builder.get_preprocess_fn(pp_fn)
    self.ds, self.steps = input_pipeline.make_for_inference(
        data.get_tfdata(ordered=True), pp_fn, batch_size,
        num_ex_per_process=data.num_examples_per_process(),
        cache_final=cache_final, cache_raw=cache_raw)
    self.data_iter = input_pipeline.start_global(self.ds, devices, prefetch)
    self.eval_fn = get_eval_fn(predict_fn, loss_name, topk)
    self.label_key = label_key

  def run(self, train_state):
    """Computes all metrics."""
    ncorrects, loss, nseen = [0] * self.topk, 0, 0
    for _, batch in zip(range(self.steps), self.data_iter):
      labels, mask = batch.pop(self.label_key), batch.pop('_mask')
      batch_ncorrects, batch_losses, batch_nseen = jax.device_get(
          self.eval_fn(train_state, batch, labels, mask))
      # print(f'labels: {jax.device_get(labels)}')
      # print(f'mask: {jax.device_get(mask)}')
      loss += batch_losses
      nseen += batch_nseen

      for i in range(self.topk):
        ncorrects[i] += batch_ncorrects[i]

    for i in range(self.topk):
      yield (f'prec@{i + 1}', ncorrects[i] / nseen)
    yield ('loss', loss / nseen)