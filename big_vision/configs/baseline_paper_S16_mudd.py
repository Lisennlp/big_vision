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

# pylint: disable=line-too-long
r"""Pre-training ViT-S/16 on ILSVRC-2012 following https://arxiv.org/abs/2205.01580.

This should take 6-7h to finish 90ep on a TPU-v3-8 and reach 76.5%,
see the tech report for more details.

Command to run:

big_vision.train \
    --config big_vision/configs/vit_s16_i1k.py \
    --workdir gs://[your_bucket]/big_vision/`date '+%m-%d_%H%M'`

To run for 300ep, add `--config.total_epochs 300` to the command.
"""

import ml_collections as mlc


def get_config():
  """Config for training."""
  config = mlc.ConfigDict()

  config.seed = 9876
  config.total_epochs = 90
  config.num_classes = 1000
  config.loss = 'softmax_xent'

  config.input = {}
  config.input.data = dict(
      name='imagenet2012',
      split='train[:99%]',
  )
  config.input.batch_size = 1024 * 4
  config.input.cache_raw = True  # Needs up to 120GB of RAM!
  config.input.shuffle_buffer_size = 250_000

  pp_common = (
      '|value_range(-1, 1)'
      '|onehot(1000, key="{lbl}", key_result="labels")'
      '|keep("image", "labels")'
  )
  config.input.pp = (
      'decode_jpeg_and_inception_crop(224)|flip_lr|randaug(2,10)' +
      pp_common.format(lbl='label')
  )
  pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.log_training_steps = 5
  config.ckpt_steps = 250

  # Model section
  config.model_name = 'vit'
  config.model = dict(
      variant='S/16',
      rep_size=True,
      pool_type='gap', # tok: cls, gap: avg
      posemb='sincos2d',
      scan=False,
      dropout=0.1
  )

  # Optimizer section
  config.grad_clip_norm = 1.0
  config.optax_name = 'scale_by_adam'
  config.optax = dict(mu_dtype='bfloat16')
  # lsp
  config.dropout = 0.1

  config.lr = 0.003
  config.wd = 0.0009 # real wd: config.wd / config.lr = 0.003/0.0009 = 0.3
  config.schedule = dict(warmup_steps=10_000, decay_type='cosine')

  config.mixup = dict(p=0.2, fold_in=None)

  config.dc_config = dict(
    dynamic_compose=False,
    dynamic_dropout_rate=0.0,
    dynamic_dense_type = 'qkvm',
    dynamic_dense_fix_last_layer = True,
    dynamic_dense_hidden_expand = 1,
    dynamic_dense_hidden_round = True, # True -> false
    dynamic_dense_act_cls = 'gelu',
    static = False,
    dense_proj1_init_scale = 1.0,
    dynamic_qkvm_tanh = False, # hc tanh
    dynamic_qkv_tanh = False,
    dynamic_m_tanh = False,
    dense_coef = ['LLL', 0.01], # A: qkvml共享β，C：qkvm使用不同β，L：每层使用不同β，CL：qkvm，每层均使用不同β
    mudd_dropout = 0.0,
    last_layer_static = False,
    dense2_bias_init_value = 1.0,  # prepost norm的时候初始化为0
    mudd_prenorm = False,
    mudd_postnorm = False,
    inner_scale=False,
    mudd_postnorm_residual_qkv = False,
  )

  config.resume = ''
  config.only_eval = False
  config.topk = 10
  config.save_checkpoint = True

  # Eval section
  def get_eval(split, dataset='imagenet2012'):
    return dict(
        type='classification',
        data=dict(name=dataset, split=split),
        pp_fn=pp_eval.format(lbl='label'),
        loss_name=config.loss,
        log_steps=625,  # Very fast O(seconds) so it's fine to run it often.
        topk=config.topk,  # lsp
    )
  config.evals = {}
  config.evals.train = get_eval('train[:2%]')
  config.evals.minival = get_eval('train[99%:]')
  config.evals.val = get_eval('validation')
  config.evals.v2 = get_eval('test', dataset='imagenet_v2')
  config.evals.real = get_eval('validation', dataset='imagenet2012_real')
  config.evals.real.pp_fn = pp_eval.format(lbl='real_label')

  return config
