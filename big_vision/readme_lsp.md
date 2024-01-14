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