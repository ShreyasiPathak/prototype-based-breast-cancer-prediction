import re
import cv2
import torch
import argparse
import os,shutil
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from proto_join import join_prototypes, join_prototypes_by_activations

import model
import find_nearest_new
from log import create_logger
from helpers import makedir
import train_and_test as tnt
from preprocess import preprocess_input_function
from data_loader import input_file_creation, dataloader_visualize

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                    prototype_activation_function, add_on_layers_type, coefs, randseedother, \
                    randseeddata, joint_optimizer_lrs, stage1_optimizer_lrs, stage2_optimizer_lrs

def load_model(net, path):
    checkpoint = torch.load(path, map_location='cuda:0')
    net.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['epoch_joint'])
    print(checkpoint['epoch_stage1'])
    print(checkpoint['epoch_stage2'])
    return net

def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)

def set_device(args):
    gpu_list = args.gpuid[0].split(',')
    device_ids = []
    if args.gpuid!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpuid))
        elif len(device_ids)==0:
            device = torch.device('cuda')
            print("CUDA device set without id specification", flush=True)
            device_ids.append(torch.cuda.current_device())
        else:
            print("This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU.", flush=True)
            device_str = ''
            for d in device_ids:
                device_str+=str(d)
                device_str+=","
            device = torch.device('cuda:'+str(device_ids[0]))
    else:
        device = torch.device('cpu')
     
    # Log which device was actually used
    print("Device used: ", device, "with id", device_ids, flush=True)
    return device, device_ids

#read arguments passed
# Usage: python3 global_analysis.py -modeldir='./saved_models/' -model=''
parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-start_epoch', type=int)
parser.add_argument('-disable_cuda', action='store_true', help='Flag that disables GPU usage if set')
args = parser.parse_args()

device, device_ids = set_device(args)

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
load_model_dir = args.modeldir[0]
load_model_name = args.model[0]
load_model_path = os.path.join(load_model_dir, load_model_name)
#epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(args.start_epoch) #int(epoch_number_str)
k = 10 #number of top k patches to visualize per prototype

# load the model
print('load model from ',load_model_path)

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)

ppnet = torch.nn.DataParallel(ppnet)
ppnet = load_model(ppnet, load_model_path)
ppnet = ppnet.to(device)#cuda()
#ppnet_multi = torch.nn.DataParallel(ppnet)
#ppnet = torch.nn.DataParallel(ppnet)

log, logclose = create_logger(log_filename=os.path.join(load_model_dir, 'train_visualize.log_'+str(randseedother)+'_'+str(randseeddata)))

#print model details
print("num prototypes", ppnet.module.num_prototypes,flush=True)
img_size = ppnet.module.img_size
print("img size: ", img_size, flush=True)
proto_percentile = 0.999 #0.95
print("proto percentile: ", proto_percentile, flush=True)

# load the data
# must use unaugmented (original) dataset
#trained_model_path = './saved_models/densenet121/001/net_trained_push_sparsity' #'./saved_models/resnet34_hoffman/004/10_18push0.7958.pth'

#img_size = 224

df_train, _, df_test, batches_train, _, batches_test = input_file_creation(log)
train_loader, test_loader, test_loader_normalize = dataloader_visualize(df_train, df_test)

#define optimizer
stage1_optimizer_specs = \
[{'params': ppnet.module.features.parameters(), 'lr': stage1_optimizer_lrs['features'], 'weight_decay': 1e-5},
 #{'params': ppnet.add_on_layers_globalnet.parameters(), 'lr': stage1_optimizer_lrs['add_on_layers_globalnet'], 'weight_decay': 1e-5},
 {'params': ppnet.module.last_layer_globalnet.parameters(), 'lr': stage1_optimizer_lrs['last_layer_globalnet'], 'weight_decay': 1e-5},
]
stage1_optimizer = torch.optim.Adam(stage1_optimizer_specs)

stage2_optimizer_specs = \
[{'params': ppnet.module.add_on_layers.parameters(), 'lr': stage2_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-5},
 {'params': ppnet.module.prototype_vectors, 'lr': stage2_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.module.last_layer.parameters(), 'lr': stage2_optimizer_lrs['last_layer']}
]
stage2_optimizer = torch.optim.Adam(stage2_optimizer_specs)

joint_optimizer_specs = \
[{'params': ppnet.module.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-5}, # bias are now also being regularized
 #{'params': ppnet.add_on_layers_globalnet.parameters(), 'lr': joint_optimizer_lrs['add_on_layers_globalnet'], 'weight_decay': 1e-5},
 {'params': ppnet.module.last_layer_globalnet.parameters(), 'lr': joint_optimizer_lrs['last_layer_globalnet'], 'weight_decay': 1e-5},
 {'params': ppnet.module.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-5},
 {'params': ppnet.module.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': ppnet.module.last_layer.parameters(), 'lr': joint_optimizer_lrs['last_layer']}
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)

#check test accuracy
accu = tnt.test(model=ppnet, dataloader=test_loader_normalize, class_specific=True, coefs=coefs, device=device, epoch=start_epoch_number)
log("accu protopnet: {0}".format(accu))

# join_info = join_prototypes_by_activations(ppnet_multi, proto_percentile, train_push_loader, joint_optimizer, warm_optimizer, last_layer_optimizer, no_p=200)
#join_info = join_prototypes(ppnet_multi, proto_percentile, joint_optimizer, warm_optimizer, last_layer_optimizer,)

#prototype folders for train and test set
root_dir_for_saving_train_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_train_protopnet')
root_dir_for_saving_test_images = os.path.join(load_model_dir,
                                                load_model_name.split('.pth')[0] + '_nearest_test_protopnet')

if os.path.exists(root_dir_for_saving_train_images) and os.path.isdir(root_dir_for_saving_train_images):
    shutil.rmtree(root_dir_for_saving_train_images)
if os.path.exists(root_dir_for_saving_test_images) and os.path.isdir(root_dir_for_saving_test_images):
    shutil.rmtree(root_dir_for_saving_test_images)
makedir(root_dir_for_saving_train_images)
makedir(root_dir_for_saving_test_images)

# save prototypes in original images
load_img_dir = os.path.join(load_model_dir, 'img'+'_'+str(randseedother)+'_'+str(randseeddata))
'''prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+str(start_epoch_number), 'bb'+str(start_epoch_number)+'.npy'))

for j in range(ppnet.module.num_prototypes):
    makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
    makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_train_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          epoch=start_epoch_number,
                                          index=j,
                                          bbox_height_start=prototype_info[j][1],
                                          bbox_height_end=prototype_info[j][2],
                                          bbox_width_start=prototype_info[j][3],
                                          bbox_width_end=prototype_info[j][4],
                                          color=(0, 255, 255))
    save_prototype_original_img_with_bbox(fname=os.path.join(root_dir_for_saving_test_images, str(j),
                                                             'prototype_in_original_pimg.png'),
                                          epoch=start_epoch_number,
                                          index=j,
                                          bbox_height_start=prototype_info[j][1],
                                          bbox_height_end=prototype_info[j][2],
                                          bbox_width_start=prototype_info[j][3],
                                          bbox_width_end=prototype_info[j][4],
                                          color=(0, 255, 255))
'''

#prototypes from train set
find_nearest_new.find_k_nearest_patches_to_prototypes(
        dataloader=train_loader, # pytorch dataloader (must be unnormalized in [0,1])
        df=df_train,
        prototype_network_parallel=ppnet, # pytorch network with prototype_vectors
        k=k+1,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=print,
        device=device)

#prototypes from test set
'''find_nearest_new.find_k_nearest_patches_to_prototypes(
        dataloader=test_loader, # pytorch dataloader (must be unnormalized in [0,1])
        df=df_test,
        prototype_network_parallel=ppnet, # pytorch network with prototype_vectors
        k=k,
        preprocess_input_function=preprocess_input_function, # normalize if needed
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=print,
        device=device)
'''