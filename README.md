<div align="center">

<img src="assets/title.png" width="250" style="margin-bottom: -10px;">

# A Conformation-Centric Generative Foundation Model for Linear Polymer Modeling and Design

<div align="center">
    <p>
        <strong>Fanmeng Wang<sup>1,2</sup>, Ruochao Wang<sup>3</sup>, Shan Mei<sup>3</sup>, Wentao Guo<sup>2</sup>, Hongshuai Wang<sup>2</sup>, <br>Qi Ou<sup>3,&#42;</sup>, Zhifeng Gao<sup>2,&#42;</sup>, Hongteng Xu<sup>1,&#42;</sup></strong>
    </p>
    <p>
        <sup>1</sup>Gaoling School of Artificial Intelligence, Renmin University of China<br>
        <sup>2</sup>DP Technology<br>
        <sup>3</sup>SINOPEC Research Institute of Petroleum Processing Co., Ltd.<br>
        <sup>&#42;</sup>Corresponding authors
    </p>
</div>

[![arXiv](https://img.shields.io/badge/arXiv-2510.16023-b31b1b?style=flat&logo=arxiv)](https://arxiv.org/abs/2510.16023)
[![Checkpoint](https://img.shields.io/badge/Download-Checkpoint-brightgreen?style=flat&logo=zenodo)](https://zenodo.org/records/17577742)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue?logo=gnu)](LICENSE)

</div>

---


## 📜 Table of Contents
- [📖 Overview](#-overview)
- [⚒️ Dependencies](#️-dependencies)
- [📦 Datasets](#-datasets)
- [🚀 Quick Inference](#-quick-inference)
- [💪 Train from Scratch](#-train-from-scratch)
- [👍 Acknowledgments](#-acknowledgments)

---

## 📖 Overview
**PolyConFM** is the first foundation model for linear polymer modeling and design via conformation-centric generative pretraining. In particular, PolyConFM achieves state-of-the-art performance on three fundamental tasks: 
  
1.  **Polymer Conformation Generation**: Predict the stable 3D structures of polymers.
2.  **Polymer Property Prediction**: Forecast key physical and chemical properties.
3.  **Polymer Design**: Generate novel polymers satisfying specific conditions.

By seamlessly bridging polymer structure, properties, and design, PolyConFM serves as a powerful tool for advancing polymer science.  

<p align="center" style="margin-top: 20px;">
  <img src="assets/overview.png" alt="PolyConFM Overview" width="90%">
</p>

---

## ⚒️ Dependencies
### Option 1: Using Docker (Recommended)
We now provide a pre-configured Docker image containing all necessary dependencies
* Use the following command to pull this Docker image:
  ```bash
  docker pull dp-ve-registry-cn-beijing.cr.volces.com/dplc/polyconfm:0.0.1
  ```
* ✅ Then you can directly run our project within this container! It offers a ready-to-use environment with zero manual configuration required.

### Option 2: Manual Installation
If you prefer manual setup, please ensure you use Python 3.10.6
* [Uni-Core](https://github.com/dptech-corp/Uni-Core), please check its [Installation Documentation](https://github.com/dptech-corp/Uni-Core#installation).
* Other dependencies are listed in `requirements.txt`, please execute the following command:
  ```bash
  pip install -r requirements.txt
  pip install git+https://github.com/igor-krawczuk/mini-moses
  ```

---

## 📦 Datasets
All datasets used in this work are provided on [Zenodo](https://zenodo.org/records/17568899). Please download and organize them into the `./datasets` directory as follows:
```
PolyConFM
├──datasets
│   ├── pretrain_dataset
│   ├── property_dataset
│   ├── design_dataset
│
```

---

## 🚀 Quick Inference
All model weights are available on [Zenodo](https://zenodo.org/records/17577742). Please download and organize them into the `./ckpts` directory as follows:
```
PolyConFM
├──ckpts
│   ├── pretrain_ckpt
│   ├── property_ckpt
│   ├── design_ckpt
│
```
Then you can easily run inference for three fundamental tasks using the following scripts:

### 1. Polymer Conformation Generation
```bash
bash ./scripts/conf_script/conf_gen.sh
python ./scripts/conf_script/conf_eval.py
```

### 2. Polymer Property Prediction
```bash
bash ./scripts/property_script/property_inference.sh
python ./scripts/property_script/property_eval.py
```

### 3. Polymer Design
```bash
bash ./scripts/design_script/design_inference.sh
python ./scripts/design_script/design_eval.py
```

---

## 💪 Train from Scratch
You can also train PolyConFM from scratch using the following scripts:

### 1. Conformation-Centric Pretraining
This is the core training process that enables all downstream tasks.
```bash
bash ./scripts/pretrain.sh
```
*Note: The conformation generation capability is unlocked directly after pretraining and does not require a separate finetuning step.*

### 2. Finetuning for the Downstream Polymer Property Prediction Task
```bash
bash ./scripts/property_script/property_train.sh
```

### 3. Finetuning for the Downstream Polymer Design Task
```bash
bash ./scripts/design_script/design_train.sh
```

---

## 👍 Acknowledgments
 We extend our sincere gratitude to [Uni-Core](https://github.com/dptech-corp/Uni-Core), [Uni-Mol](https://github.com/dptech-corp/Uni-Mol), [MolCLR](https://github.com/yuyangw/MolCLR), [MAR](https://github.com/LTH14/mar), [TorsionalDiff](https://github.com/gcorso/torsional-diffusion), [FrameDiff](https://github.com/jasonkyuyim/se3_diffusion), [GraphDiT](https://github.com/liugangcode/Graph-DiT) for their great work and codebase, which served as the foundation for developing PolyConFM.