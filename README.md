# 🧠 Polyp Segmentation Using U-Net, ResUNet, and Custom Inception-U-Net

## Overview

This repository implements a deep learning-based model for **polyp segmentation** in medical images, specifically using various variants of U-Net architecture. The models included in this repository are:

- **U-Net**: A standard architecture for semantic segmentation tasks.
- **ResUNet**: A variant of U-Net with residual connections, improving performance in deeper networks.
- **ResUNet++**: A further enhancement of ResUNet that incorporates dense skip pathways.
- **Custom Inception U-Net**: A novel custom architecture combining Inception modules and U-Net, with a unique **Wavelet Pooling** technique for improved feature extraction and segmentation results.

Moreover, an **Explanation of layerwise output** is done to visualize the process of obtaining the output. This part is adopted from the paper,

[**LeXNet++: Layer-wise eXplainable ResUNet++ framework for segmentation of colorectal polyp cancer images**](https://link.springer.com/article/10.1007/s00521-024-10441-6)

The dataset used for the project is CVC-ClinicDB. The dataset can be downloaded [from here](https://www.kaggle.com/datasets/balraj98/cvcclinicdb). But the code can be used for any segmentation task if the images and masks are arranged in the specified format.


## Table of Contents

- [Prerequisites](#prerequisites)
- [📁 Project Structure](#-project-structure)
- [🚀 Models](#-models)
- [📷 Sample Result](#-sample-result)
- [📦 Installation](#-installation)
- [🏃‍♂️ How to Run](#-how-to-run)
- [📊 Results & Evaluation](#-results--evaluation)
- [📚 Citations](#-citations)
- [🤝 Contributing](#-contributing)


## Prerequisites

- Python 3.x
- TensorFlow
- NumPy
- OpenCV
- Keras
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required dependencies using the following:

```bash
pip install -r requirements.txt
```

## 📁 Project Structure

The project should be structured as following to run properly,

```bash
data/                
├── Original/         # Contains input images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── Ground Truth/     # Corresponding masks (ground truth)
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── models/                 
├── train.py               
├── main.py                 
├── utils
├── dataset.py                 
├── requirements.txt        
└── README.md               
```


## 🚀 Models

| Model                                | Description                                                                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **UNet**                             | Standard baseline for biomedical segmentation                                                                  |
| **ResUNet**                          | UNet with residual connections                                                                                 |
| **ResUNet++**                        | Enhanced ResUNet with attention & dilated convolutions                                                         |
| **Inception-UNet + Wavelet Pooling** | Our custom architecture with inception modules and wavelet pooling for improved multi-scale feature extraction |


## 📷 Sample Result

<img width="604" height="479" alt="image" src="https://github.com/user-attachments/assets/39516714-a430-4a6b-b931-bd7c703f4177" />


From the left a) Original Image  b) Ground Truth  c) Unet Prediction  d) ResUnet++ Prediction   e) Custom Model Prediction


## 📦 Installation

1. Clone The Repo
```bash
git clone https://github.com/jabirahsan/polyp-segmentation-wavelet-unet.git
cd polyp-segmentation
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Data Preparation

Place your dataset inside the data/ directory.
```bash
data/
├── Original/         # Contains input images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── Ground Truth/     # Corresponding masks (ground truth)
│   ├── image1.png
│   ├── image2.png
│   └── ...

```
Make sure the image and mask filenames align.

## 🏃‍♂️ How to Run

1. Train each model (UNet, ResUNet++, and CustomNet):

```bash
python train.py --model unet
python train.py --model resunetplus
python train.py --model customnet

```

In case you don't want to train the models. [***Download the Model Weights from here***](https://drive.google.com/drive/folders/1kMuCQCwtqTzJFuFzgRX6_zSYTMNLYn2F?usp=sharing). After downloading place the files as following,
```bash
Final/
├── unet.keras        # Contains input images
├── resunetplus.keras
│── customnet.keras

```
2. Run Inference

Once all models are trained, run evaluation & explanation:

```bash
python main.py
```
This will:

- Load all trained models

- Perform segmentation on the test set

- Generate evaluation metrics and visualizations

- Generate Layerwise Heatmap of Models

## 📊 Results & Evaluation

The model evaluation on validation and test set is,

<img width="695" height="94" alt="image" src="https://github.com/user-attachments/assets/0debab99-0ab1-43fb-89a4-5964a1f04799" />



# 📚 Citations

If you use this repository, please cite the following papers (if applicable):

-UNet: Ronneberger et al., 2015

-ResUNet: Zhang et al., 2018

-ResUNet++: Jha et al., 2019

-Wavelet Pooling: Williams et al., 2018 (if used as basis)


## 🤝 Contributing
Feel free to open issues or pull requests if you'd like to contribute.










    








