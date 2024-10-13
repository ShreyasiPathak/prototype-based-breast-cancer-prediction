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
We provide the script for creating the input csv files for [CMMD](data-processing/cmmd/utilities.py), [CBIS](data-processing/cbis/input_csv_file_creation_cbis.py) and [VinDr](data-processing/vindr/utilities.py).  

## Model Training and Global and Local Explanation

### ProtoPNet

Train ProtoPNet model on breast cancer:
> python main.py

Optionally, arguments can be passed to main.py <br/>
-gpuid (gpu id), <br/>
-disable_cuda (disable cuda training) <br/>
-start_epoch to resume training from a certain epoch <br/>
-best_score is the last best score before resuming training, needed for model checkpoint <br/>
-mode: train or localization. Localization is used for calculation of localization measure.

In protopnet/settings.py, you can add the settings for training the model.

Local explanations. Visualize top-k prototypes with similarity*weight score per image. 
> python protopnet/local_analysis.py

Global explanations. Visualize top-k activated patches from the training set per prototype.
> python protopnet/global_analysis_new.py

### BRAIxProtoPNet++
Train BRAIxProtoPNet++ model on breast cancer:
> python main.py

Optionally, arguments can be passed to main.py <br/>
-gpuid (gpu id), <br/>
-disable_cuda (disable cuda training) <br/>
-start_epoch to resume training from a certain epoch <br/>
-mode: train or localization. Localization is used for calculation of localization measure.

In braixprotopnet/settings.py, you can add the settings for training the model.

Local explanations. Visualize top-k prototypes with similarity*weight score per image. 
> python braixprotopnet/local_analysis.py

Global explanations. Visualize top-k activated patches from the training set per prototype.
> python braixprotopnet/global_analysis_new.py

### PIP-Net

Train PIP-Net on breast cancer prediction
> python3 main.py --net convnext_tiny_13 --optimizer Adam --randseedother 8 --randseeddata 8 --epochs_pretrain 10 --batch_size 6 --freeze_epochs 0 --epochs 60 --num_workers 8 --dataset cmmd --lr_net 0.00001 --lr 0.05 --lr_block 0.00001 --batch_size_pretrain 30 --log_dir ./runs/modelid12_cmmd_ct13_10_60_.05_0.0001_bs6_bsp30_clwt3 --weighted_loss --image_size 1536 768 --datasplit customsplit --SIL_csvfilepath /deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/cmmd_singleinstance_groundtruth.csv --preprocessed_imagepath /deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/processed_png_8bit --numclasses 2 --modelid 12 --classtype diagnosis --usevalidation --align_pf_weight 5 --tanhloss_weight 2 --classloss_weight 2

Global explanation and local explanation both are in main.py. Global explanation is generated automatically after training. For local explanation, uncomment the lines marked for local explanation.

### Black-box models

To train the black-box models (EfficientNet, ConvNext and GMIC) in the paper, please refer to this repository: https://github.com/ShreyasiPathak/multiinstance-learning-mammography

## Prototype Evaluation Framework for Coherence (PEF-Coh)
Prototype evaluation framework script location for [ProtoPNet](protopnet/proto_eval_framework.py), [BRAIxProtoPNet++](braixprotopnet/proto_eval_framework.py) and [PIP-Net](pipnet/src/util/proto_eval_framework.py).

For calculating the measures in PEF-Coh for ProtoPNet and BRAIxProtoPNet++, follow:

1. Generate protopnet_cbis_topk.csv, which is generated during global explanation generation
> python global_analysis_new.py

2. Generate class distribution of each abnormality type if this information is available in the dataset (available for cbis). This is needed for the class-specific measure.

First, create the ROI csv file with all information about the ROIs ([sample of the ROI can be found here](sample-input-csv-file/cbis/MG_training_files_cbis-ddsm_roi_groundtruth.csv)). Inside the script below, check that file_creation = 'ROI' in main function. If not, then set it and run the folllowing:
> python data-processing/cbis/input_csv_file_creation_cbis.py

The file generated from the above script is needed for running the script below. Run the following line to generate cbisddsm_abnormalitygroup_malignant_benign_count.csv.
> python data-processing/cbis/abnormalitytype_diagnosis.py

3. Calculate relevance, specialization, uniqueness, coverage and class-specific measures.
> python protopnet/proto_eval_framework.py --dataset cbis-ddsm --patch_size 130 130 --state_dict_dir_net /home/pathaks/PhD/prototype-model-evaluation/protopnet/saved_models/cbis-ddsm/convnext_tiny_13/019/net_trained_best_8_8 --patch_proto_csv /home/pathaks/PhD/prototype-model-evaluation/protopnet/saved_models/cbis-ddsm/convnext_tiny_13/019/net_trained_best_8_8_nearest_train_protopnet/protopnet_cbis_topk.csv --image_size 1536 768

4. Calculate localization measure
> python main.py -mode localization

For calculating the measures in PEF-Coh for PIP-Net, follow:
