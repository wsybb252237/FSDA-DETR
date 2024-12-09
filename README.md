# FSDA-DETR: Few-shot Domain Adaptive Object Detection Transformer in Remote Sensing Imagery

This repository contains the implementation accompanying our paper FSDA-DETR: Few-shot Domain Adaptive Object Detection Transformer in Remote Sensing Imagery.


![](/figs/CSR_module.png)

## Acknowledgment
This implementation is bulit upon [DINO](https://github.com/IDEA-Research/DINO/).

## Installation
Please refer to the instructions [here](requirements.txt). We leave our system information for reference.

* OS: Ubuntu 20.04
* Python: 3.9.19
* CUDA: 12.2
* PyTorch: 2.1.0
* torchaudio：2.1.0
* torchvision: 0.16.0

## Dataset Preparation
Please construct the datasets following these steps:

- Download the datasets from their sources.

- Convert the annotation files into COCO-format annotations.

- Modify the dataset path setting within the script [coco_FSDA.py](./datasets/coco_FSDA.py)

- All the scenes can be found within the script [__init__.py](./datasets/__init__.py).


  - Training with single GPU
```
sh scripts/DINO_train.sh
```
- Training with Multi-GPU
```
sh scripts/DINO_train_dist.sh
```

We provide an evaluation script to evaluate the pre-trained model. --dataset_file is used to specify the test dataset, and --resume is used to specify the path for loading the model.
- Evaluation Model.
```
sh scripts/DINO_eval.sh
```


We provide inference script to visualize detection results. See [inference.py](inference.py) for details
- Inference Model.
```
python inference.py
```

## Reference
https://github.com/IDEA-Research/DINO
