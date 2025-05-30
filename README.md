# FSDA-DETR: Few-shot Domain Adaptive Object Detection Transformer in Remote Sensing Imagery

This repository contains the implementation accompanying our paper FSDA-DETR: Few-shot Domain Adaptive Object Detection Transformer in Remote Sensing Imagery.

## 📄 Citation

If you find this project useful in your research, please consider citing our paper:

BibTeX:
```bibtex
@ARTICLE{11016953,
  author={Yang, Binbin and Han, Jianhong and Hou, Xinghai and Zhou, Dehao and Liu, Wenkai and Bi, Fukun},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={FSDA-DETR: Few-shot Domain Adaptive Object Detection Transformer in Remote Sensing Imagery}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Remote sensing;Feature extraction;Object detection;Detectors;Training;Adaptation models;Benchmark testing;Visualization;Training data;Prototypes;Few-shot;Domain Adaptive;Object Detection;remote sensing imagery},
  doi={10.1109/TGRS.2025.3574245}
}
```

## 🖼️ Method Overview
![](/figs/CSR_module.png)

## 🙏 Acknowledgment

This implementation is built upon [DINO](https://github.com/IDEA-Research/DINO/).

---

## ⚙️ Installation

Please refer to the instructions in [requirements.txt](requirements.txt). We leave our system configuration below for reference:

- 🖥️ OS: Ubuntu 20.04  
- 🐍 Python: 3.9.19  
- 🔧 CUDA: 12.2  
- 🔥 PyTorch: 2.1.0  
- 🎧 torchaudio: 2.1.0  
- 🖼️ torchvision: 0.16.0  

---

## 📁 Dataset Preparation

Please construct the datasets following the steps below:

1. 📥 **Download** the datasets from their official sources.
2. 🧩 **Convert** annotation files into **COCO format**.
3. 🛠️ **Modify** dataset path in [`coco_FSDA.py`](./datasets/coco_FSDA.py).
4. 📂 Dataset scenes are listed in [`__init__.py`](./datasets/__init__.py).

---

## 🏋️ Training

### 🔹 Train with Single GPU
```bash
sh scripts/xView2DOTA/DINO_train.sh

