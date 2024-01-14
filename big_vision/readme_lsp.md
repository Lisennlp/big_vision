如果处理imagenet_v2报错
/home/lishengping/.local/lib/python3.9/site-packages/tensorflow_datasets/datasets/imagenet_v2/imagenet_v2_dataset_builder.py
_generate_examples(self
class_id长这样：100/, 11/
if class_id.endswith('/'): 
    class_id = class_id[:-1]
features = 
    ...
    int(class_id) -> int(class_id)
    ...


## 测试tpu是否可用
TPU_NAME=llm-jax-v4-16-10
ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/;/home/lishengping/miniconda3/bin/python -c 'import jax; print(jax.devices())'"
## train
# 安装环境
TPU_NAME=llm-jax-v4-16-10
ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects;sudo rm -r big_vision; git clone https://github.com/Lisennlp/big_vision.git"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects/big_vision/big_vision/; /home/lishengping/miniconda3/bin/pip  install -r requirements_lsp.txt"
# train
TPU_NAME=llm-jax-v4-16-10
ZONE=us-central2-b

gcloud compute tpus tpu-vm ssh $NAME --zone=$ZONE --worker=all --command "TFDS_DATA_DIR=gs://$GS_BUCKET_NAME/tensorflow_datasets bash big_vision/run_tpu.sh big_vision.train --config big_vision/configs/bit_i1k.py  --workdir gs://$GS_BUCKET_NAME/big_vision/workdir/`date '+%m-%d_%H%M'`"



FLAGS="--config.num_train_steps=100000 --config.warmup_steps=1000 --config.checkpoint_every_steps=1000 --config.per_device_batch_size=16 --config.save_checkpoints=True --config.checkpoint_every_steps = 5000"
WOKRDIR=gs://jax_llm_data/dcformer_compare_experiments/logs/wmt_256/transformer_base/
CODEDIR=/home/lishengping/projects/flax/examples/wmt

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects/;sudo rm -r flax; git clone https://github.com/Lisennlp/flax.git"

gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/;export TFDS_DATA_DIR=gs://jax_llm_data/dcformer_compare_experiments/;/home/lishengping/miniconda3/bin/python $CODEDIR/main.py --workdir=$WOKRDIR --config=$CODEDIR/configs/default.py $FLAGS| tee train.log"