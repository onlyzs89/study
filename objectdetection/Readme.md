## 1.Capture Image

`python capture_img.py`

Prerequisite:
- opencv-python
- web camera

## 2.Labeling

`python labeling.py`

Prerequisite:
- opencv-python

## 3.Create Datasets

`python create_tf_record`

## 4.Training

```
python object_detection/train.py 
  --pipeline_config_path=../../data/ssd_mobilenet_v1.config 
  --train_dir=../../data/train 
  --logtostderr
```

## 5.Evaling

```
python object_detection/eval.py
  --pipeline_config_path=../../data/ssd_mobilenet_v1.config
  --checkpoint_dir=../../data/train
  --eval_dir=../../data/train
  --logtostderr
```

## 6.Tensorboard

`tensorboard --logdir=train`

## 7.Freeze Model

```
python object_detection/export_inference_graph.py
  --input_type image_tensor
  --pipeline_config_path ../../data/ssd_mobilenet_v1.config
  --trained_checkpoint_prefix ../../data/train/model.ckpt
  --output_directory ../../data/save
```

## 8.Use Model

`python detect.py`
