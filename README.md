# Prototype-Based Interpretable Breast Cancer Prediction Models: Analysis and Challenges

Prototype-based breast cancer prediction on mammography images with 3 prototype-based models - ProtoPNet, BRAIxProtoPNet++ and PIP-Net.
Each model contains the prototype evaluation framework for coherence code for evaluating the quality of the prototypes.

## Prerequisites
- Python 3.10.6
- Pytorch 1.13.0
- Cuda 11.7
- matplotlib 3.5.3
- opencv-python 4.7.0
- pandas 1.4.4
- openpyxl 3.0.10
- scikit-learn 1.1.1
- seaborn 0.11.2

## Dataset

### Access the dataset

We used 3 public datasets in our work - CBIS, VinDr and CMMD.

- CBIS can be downloaded from [here](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629).
- VinDr can be downloaded from [here](https://vindr.ai/datasets/mammo).
- CMMD can be downloaded from [here](https://www.cancerimagingarchive.net/collection/cmmd/).

### Preprocess the dataset

We preprocessed the dataset as follows:

1. Convert the dicom images to png with this [script](/src/data_processing/dicom_to_png.py). <br/>
2. Convert the original png images to preprocessed png images (to remove irrelevant information and remove extra black background) according to our [image cleaning script](/src/data_processing/image_cleaning.py). Example of the results of our image preprocessing algorithm can be found [here](/image-preprocessing). We used these preprocessed images as input to our model.

## Model Training

### ProtoPNet

Train ProtoPNet model on breast cancer 

### BRAIxProtoPNet++

### PIP-Net

### Black-box models

### Prototype Evaluation Framework for Coherence (PEF-Coh)
