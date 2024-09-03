## Note

If you process imagenet_v2, the imagenet_v2_dataset_builder file of tensorflow_datasets package may report an error:

```
/home/xxx/.local/lib/python3.9/site-packages/tensorflow_datasets/datasets/imagenet_v2/imagenet_v2_dataset_builder.py
```
FunctionL: _generate_examples(self......) 

Because class_id like：100/, 11/
so, int(class_id) will occur error.，You can modify it in tensorflow_datasets package as follows:

```bash
if class_id.endswith('/'): 
    class_id = class_id[:-1]
features = ...
```

## Test tpu
```
TPU_NAME=llm-jax-v4-16-10
ZONE=us-central2-b
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/;/home/xxx/miniconda3/bin/python -c 'import jax; print(jax.devices())'"
```
## train
### remote sh
```
TPU_NAME=llm-jax-v3-8-10
ZONE=us-central1-a
CODEDIR=/home/xxx/projects/big_vision
CONFIG=vit_M16_i1k.py
WORKDIR=gs://jax_llm_data/dcformer_compare_experiments/logs/vit/m16_eval_models
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects;sudo rm -r big_vision; git clone https://github.com/Lisennlp/big_vision.git"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects/big_vision/big_vision/; /home/xxx/miniconda3/bin/pip  install -r requirements_lsp.txt"

# config.only_eval=False when train mode, ohterwise, set config.only_eval=True
FLAGS="--config.total_epochs=300  --config.ckpt_steps=5000 --config.resume=$WORKDIR/checkpoint.bv-000090000-tmp/ --config.only_eval=True"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "killall main.py; sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/; export TFDS_DATA_DIR=gs://jax_llm_data/dcformer_compare_experiments/vit; cd projects/big_vision/; /home/xxx/miniconda3/bin/python  -m big_vision.train --config $CODEDIR/big_vision/configs/$CONFIG  $FLAGS --workdir $WORKDIR"
```

# auto eval
```
TPU_NAME=llm-jax-v3-8-10
ZONE=us-central1-a
CODEDIR=/home/xxx/projects/big_vision
CONFIG=vit_M16_i1k.py
WORKDIR=gs://jax_llm_data/dcformer_compare_experiments/logs/vit/m16
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects;sudo rm -r big_vision; git clone https://github.com/Lisennlp/big_vision.git"
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command="cd projects/big_vision/big_vision/; /home/xxx/miniconda3/bin/pip  install -r requirements_lsp.txt"
START=80000
END=80000
for ((step=$START; step<=$END; step++)); do
  echo $step
  if [ $step -eq $END ] || [ $((step % 5000)) -eq 0 ]; then
    padded_step=$(printf "%09d" $step)
    echo $padded_step
    FLAGS="--config.total_epochs=300  --config.ckpt_steps=5000 --config.resume=$WORKDIR/checkpoint.bv-$padded_step-tmp/ --config.only_eval=True"
    gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$ZONE --worker=all --command "killall main.py; sudo lsof -w /dev/accel0 |cut -c 9-14|awk 'NR>1 {print $1}'| xargs sudo kill -9; sudo rm -f /tmp/libtpu_lockfile;sudo chmod +777 -R /tmp/tpu_logs/; export TFDS_DATA_DIR=gs://jax_llm_data/dcformer_compare_experiments/vit; cd projects/big_vision/; /home/xxx/miniconda3/bin/python  -m big_vision.train --config $CODEDIR/big_vision/configs/$CONFIG  $FLAGS --workdir $WORKDIR"
  fi
done
```


