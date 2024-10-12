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

1. Convert the dicom images to png with this [script](data-processing/cmmd/dicom_to_png.py). <br/>
2. Convert the original png images to preprocessed png images (to remove irrelevant information and remove extra black background) according to our [image cleaning script](data-processing/cmmd/image_cleaning.py). We used these preprocessed images as input to our model.

[This folder](data-processing) contains data processing script used for all datasets - CBIS, VinDr and CMMD. 

### Preparation of Input csv files

Sample of the input csv file containing details about each instance in the dataset can be found [here](sample-input-csv-file).
Input csv files for CMMD can be created using our script [here](data-processing/cmmd/utilities.py). 

## Model Training and Global and Local Explanation

### ProtoPNet

Train ProtoPNet model on breast cancer:
> python main.py

Optionally, arguments can be passed to main.py <br/>
-gpuid, -disable_cuda, -start_epoch to resume training from a certain epoch, -best_score is the last best score before resuming training, needed for model checkpoint

Local explanations. Visualize top-k prototypes with similarity*weight score per image. 
> python local_analysis.py

Global explanations. Visualize top-k activated patches from the training set per prototype.
> python global_analysis_new.py

### BRAIxProtoPNet++

### PIP-Net

### Black-box models

## Prototype Evaluation Framework for Coherence (PEF-Coh)
