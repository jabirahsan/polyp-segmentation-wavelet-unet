# Polyp Segmentation Using U-Net, ResUNet, and Custom Inception-U-Net

## Overview

This repository implements a deep learning-based model for **polyp segmentation** in medical images, specifically using various variants of U-Net architecture. The models included in this repository are:

- **U-Net**: A standard architecture for semantic segmentation tasks.
- **ResUNet**: A variant of U-Net with residual connections, improving performance in deeper networks.
- **ResUNet++**: A further enhancement of ResUNet that incorporates dense skip pathways.
- **Custom Inception U-Net**: A novel custom architecture combining Inception modules and U-Net, with a unique **Wavelet Pooling** technique for improved feature extraction and segmentation results.

Moreover, an **Explanation of layerwise output** is done to visualize the process of obtaining the output. This part is adopted from the paper,

[**LeXNet++: Layer-wise eXplainable ResUNet++ framework for segmentation of colorectal polyp cancer images**](https://link.springer.com/article/10.1007/s00521-024-10441-6)


## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Models](#models)
- [Training Instructions](#training-instructions)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

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
├── original/         # Contains input images
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── ground_truth/     # Corresponding masks (ground truth)
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── models/                 # Contains model definitions: unet, resunet, etc.
├── train.py                # Script to train the models
├── main.py                 # Script to run inference/visualization
├── utils/                  # Utility functions (data loaders, metrics, etc.)
├── requirements.txt        # Python dependencies
└── README.md               # You're here!
```


## 🚀 Models

| Model                                | Description                                                                                                    |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| **UNet**                             | Standard baseline for biomedical segmentation                                                                  |
| **ResUNet**                          | UNet with residual connections                                                                                 |
| **ResUNet++**                        | Enhanced ResUNet with attention & dilated convolutions                                                         |
| **Inception-UNet + Wavelet Pooling** | Our custom architecture with inception modules and wavelet pooling for improved multi-scale feature extraction |







