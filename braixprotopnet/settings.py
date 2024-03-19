#parameters for the code
randseedother = 24
randseeddata = 8
experiment_run = '006-1'
base_architecture =  'convnext_tiny_13' #'resnet34' #'vgg19' 'convnext_tiny_13', 'efficientnet_b0'
img_size = [1536, 768]
prototype_shape = (400, 128, 1, 1)
num_classes = 2
prototype_activation_function = 'exp'
add_on_layers_type = 'regular'
num_workers = 8
batch_size = 12 #24 #80
dataset = 'cbis-ddsm' #'cbis-ddsm', 'cmmd', 'vindr'
datasplit = 'officialtestset' #, 'customsplit', 'officialtestset'
usevalidation = True
flipimage = False
weighted_loss = True
classtype= 'diagnosis'
viewsinclusion = 'all'
dataaug = 'gmic'
if classtype=='diagnosis':
    groundtruthdic = {'benign': 0, 'malignant': 1}
elif classtype=='diagnosis_with_normal':
    groundtruthdic = {'normal': 0, 'benign': 1, 'malignant': 2}
elif classtype == 'birads':
    groundtruthdic = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}

#input path and output path
SIL_csvfilepath = "/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/cbis-ddsm_singleinstance_groundtruth.csv" #"/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/cbis-ddsm_singleinstance_groundtruth.csv" #"/groups/dso/spathak/vindr/MG_training_files_vindr_singleinstance_groundtruth.csv" #"/deepstore/datasets/dmb/medical/breastcancer/mammography/vindr/MG_training_files_vindr_singleinstance_groundtruth.csv" #'/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_groundtruth.csv' #'/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/cmmd_singleinstance_groundtruth.csv' #'/projects/dso_mammovit/project_kushal/data/cbis-ddsm_singleinstance_groundtruth.csv' # # # # #
preprocessed_imagepath = "/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/multiinstance_data_8bit" #"/deepstore/datasets/dmb/medical/breastcancer/mammography/vindr/processed_png_8bit" #"/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/multiinstance_data_8bit" #"/groups/dso/spathak/vindr/processed_png_8bit" #'/projects/dso_mammovit/project_kushal/data/multiinstance_data_8bit' #'/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/processed_png_8bit' #'/projects/dso_mammovit/project_kushal/data/multiinstance_data_8bit'  # # # # #
'''data_path = '/projects/dso_mammovit/project_kushal/data/pipnet/' #'./datasets/cub200_cropped/'
train_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_push_dir = data_path + 'train/'
'''
log_dir = './saved_models/' + dataset + '/' + base_architecture + '/' + experiment_run + '/'

#training scheme
#backbone and globalnet
stage1_optimizer_lrs = {'features': 1e-5,
                        #'add_on_layers_globalnet': 3e-3,
                        'last_layer_globalnet': 1e-5}

#protopnet layer
stage2_optimizer_lrs = {'add_on_layers': 1e-5,
                        'prototype_vectors': 1e-5,
                        'last_layer': 1e-5}

#joint training
joint_optimizer_lrs = {'features': 1e-5,
                       'add_on_layers': 1e-4, #1e-5,
                       'prototype_vectors': 1e-4, #1e-5,
                       'last_layer': 1e-2, #1e-5,
                       #'add_on_layers_globalnet': 3e-3,
                       'last_layer_globalnet': 1e-2 #1e-5
                       }
joint_lr_step_size = 10
lrdecay_gamma = 0.5

#last layer training
last_layer_optimizer_lr = 1e-2 #1e-5 #0.05

lrscheduler = 'fixedlr' #fixedr, cosineannealing

num_stage1_epochs = 3
num_stage2_epochs = 3
num_train_epochs = 60 #1000

#loss function parameters
coefs = {
    'crs_ent_protopnet': 1,
    'clst': 0.1,
    'sep': 0.1,
    #'l1': 1e-4,
    'crs_ent_globalnet': 1,
    'kd_loss': 0.5,
    'kd_loss_margin': 0.2,
    'hinge_loss_sep': 10,
    'temp_param': 128
}
