##### MODEL AND DATA LOADING
import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import re

import os
import copy

from helpers import makedir, find_high_activation_crop
import model
import push
import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function, undo_preprocess_input_function

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                    prototype_activation_function, add_on_layers_type

import argparse

def load_model(net, path):
    checkpoint = torch.load(path)
    #print(checkpoint['model_state_dict'], flush=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['epoch_joint'])
    print(checkpoint['epoch_stage1'])
    print(checkpoint['epoch_stage2'])
    return net

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

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-modeldir', nargs=1, type=str)
parser.add_argument('-model', nargs=1, type=str)
parser.add_argument('-imgdir', nargs=1, type=str)
parser.add_argument('-img', nargs=1, type=str)
parser.add_argument('-imgclass', nargs=1, type=int, default=-1)
parser.add_argument('-epoch_number', type=int, default=1)
parser.add_argument('-disable_cuda', action='store_true', help='Flag that disables GPU usage if set')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]

# specify the test image to be analyzed
test_image_dir = args.imgdir[0] #'./local_analysis/Painted_Bunting_Class15_0081/'
test_image_name = args.img[0] #'Painted_Bunting_0081_15230.jpg'
test_image_label = args.imgclass[0] #15

test_image_path = os.path.join(test_image_dir, test_image_name)

# load the model
check_test_accu = False

load_model_dir = args.modeldir[0] #'./saved_models/vgg19/003/'
load_model_name = args.model[0] #'10_18push0.7822.pth'

#if load_model_dir[-1] == '/':
#    model_base_architecture = load_model_dir.split('/')[-3]
#    experiment_run = load_model_dir.split('/')[-2]
#else:
#    model_base_architecture = load_model_dir.split('/')[-2]
#    experiment_run = load_model_dir.split('/')[-1]

model_base_architecture = load_model_dir.split('/')[6]
experiment_run = '/'.join(load_model_dir.split('/')[7:])

save_analysis_path = os.path.join(load_model_dir, 'local_analysis', test_image_dir.split('/')[-1], test_image_name)
makedir(save_analysis_path)
print(save_analysis_path)

log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
#epoch_number_str = re.search(r'\d+', load_model_name).group(0)
#start_epoch_number = int(epoch_number_str)
epoch_number_str = str(args.epoch_number)
start_epoch_number = int(args.epoch_number)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)
log('epoch number:' + epoch_number_str)
log('analysis path ' + save_analysis_path)

device, device_ids = set_device(args)

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)

#ppnet = torch.load(load_model_path)
#ppnet = ppnet.cuda()
#ppnet_multi = torch.nn.DataParallel(ppnet)

ppnet = torch.nn.DataParallel(ppnet)
ppnet = load_model(ppnet, load_model_path)
ppnet = ppnet.to(device)#cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)

print(ppnet, flush=True)
print(ppnet.module.last_layer_globalnet.weight.shape, flush=True)

#img_size = ppnet_multi.module.img_size
#prototype_shape = ppnet.prototype_shape
max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

class_specific = True

normalize = transforms.Normalize(mean=mean,
                                 std=std)

# load the test data and check test accuracy
'''from settings import test_dir
if check_test_accu:
    test_batch_size = 100

    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    log('test set size: {0}'.format(len(test_loader.dataset)))

    accu = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=print)
'''
##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')

prototype_info = np.load(os.path.join(load_img_dir, 'epoch-'+epoch_number_str, 'bb'+epoch_number_str+'.npy'))
prototype_img_identity = prototype_info[:, -1]

log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
log('Their class identities are: ' + str(prototype_img_identity))

# confirm prototype connects most strongly to its own class
prototype_max_connection = torch.argmax(ppnet.module.last_layer.weight, dim=0)
prototype_max_connection = prototype_max_connection.cpu().numpy()
if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.module.num_prototypes:
    log('All prototypes connect most strongly to their respective classes.')
else:
    log('WARNING: Not all prototypes connect most strongly to their respective classes.')

##### HELPER FUNCTIONS FOR PLOTTING
def save_preprocessed_img(fname, preprocessed_imgs, index=0):
    img_copy = copy.deepcopy(preprocessed_imgs[index:index+1])
    undo_preprocessed_img = undo_preprocess_input_function(img_copy)
    print('image index {0} in batch'.format(index))
    undo_preprocessed_img = undo_preprocessed_img[0]
    undo_preprocessed_img = undo_preprocessed_img.detach().cpu().numpy()
    undo_preprocessed_img = np.transpose(undo_preprocessed_img, [1,2,0])
    
    plt.imsave(fname, undo_preprocessed_img)
    return undo_preprocessed_img

def save_prototype(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)
    
def save_prototype_self_activation(fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    'prototype-img-original_with_self_act'+str(index)+'.png'))
    #plt.axis('off')
    plt.imsave(fname, p_img)

def save_prototype_original_img_with_bbox(fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), 'prototype-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=5)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    #plt.imshow(p_img_rgb)
    #plt.axis('off')
    plt.imsave(fname, p_img_rgb)

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=5)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    #plt.imshow(img_rgb_float)
    #plt.axis('off')
    plt.imsave(fname, img_rgb_float)
    return img_rgb_float

def map_region_to_fixed_patch_size(bounding_box):
    h_min_224 = bounding_box[0]
    h_max_224 = bounding_box[1]
    w_min_224 = bounding_box[2]
    w_max_224 = bounding_box[3]
    h_min_224, h_max_224, w_min_224, w_max_224 = float(h_min_224), float(h_max_224), float(w_min_224), float(w_max_224)
    diffh = h_max_224 - h_min_224
    diffw = w_max_224 - w_min_224
    if diffh > 130: #patch size too big, we take the center. otherwise the bigger the patch, the higher the purity. #32
        correction = diffh-130
        h_min_224 = h_min_224 + correction//2.
        h_max_224 = h_max_224 - correction//2.
        if h_max_224 - h_min_224 == 131:
            h_max_224 -= 1
    elif diffh < 130:
        if h_min_224 - (130-diffh) <0:
            h_max_224 = h_max_224 + (130-diffh)
        else:
            h_min_224 -= (130-diffh)
        
    if diffw > 130:
        correction = diffw-130
        w_min_224 = w_min_224 + correction//2.
        w_max_224 = w_max_224 - correction//2.
        if w_max_224 - w_min_224 == 131:
            w_max_224 -= 1
    elif diffw < 130:
        if w_min_224 - (130-diffw) <0:
            w_max_224 = w_max_224 + (130-diffw)
        else:
            w_min_224 -= (130-diffw)
    # print(imgname, "corrected sizes: ", h_min_224, h_max_224, w_min_224, w_max_224)
    h_min_224, h_max_224, w_min_224, w_max_224 = int(h_min_224), int(h_max_224), int(w_min_224), int(w_max_224)
    return h_min_224, h_max_224, w_min_224, w_max_224

# load the test image and forward it through the network
preprocess = transforms.Compose([
   transforms.Resize((img_size[0],img_size[1])),
   transforms.Grayscale(3),
   transforms.ToTensor(),
   normalize
])

img_pil = Image.open(test_image_path)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))

images_test = img_variable.cuda()
labels_test = torch.tensor([test_image_label])

logits, min_distances, logits_global = ppnet_multi(images_test)
log("logits protopnet branch:"+ str(logits))
log("logits globalnet branch:"+ str(logits_global))
log("min distances:"+str(min_distances))
conv_output, distances = ppnet.module.push_forward(images_test)
prototype_activations = ppnet.module.distance_2_similarity(min_distances)
prototype_activation_patterns = ppnet.module.distance_2_similarity(distances)
if ppnet.module.prototype_activation_function == 'linear':
    prototype_activations = prototype_activations + max_dist
    prototype_activation_patterns = prototype_activation_patterns + max_dist

tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables[-1]))
    notpredicted_class = torch.argmin(logits, dim=1)[i].item()

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls))
original_img = save_preprocessed_img(os.path.join(save_analysis_path, 'original_img.png'),
                                     images_test, idx)

##### All non-zero SIM*WEIGHT (NEAREST) PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls)))

log('All non-zero prototypes of this image for class'+str(predicted_cls)+':')
grid_img_tensor = []
simwt_total=0
simwt = np.zeros(prototype_shape[0])
for i in range(0, prototype_shape[0]):
    simwt[i] = prototype_activations[idx][i].item() * ppnet.module.last_layer.weight[predicted_cls, i].item()
    print(i, prototype_activations[idx][i].item(), ppnet.module.last_layer.weight[predicted_cls, i].item(), simwt[i], flush=True)

'''simwt_sorted, simwt_sorted_indices = torch.sort(torch.from_numpy(simwt), descending=True)
for i in simwt_sorted_indices:
    if simwt[i] > 0.01:
        log('prototype index: {0}'.format(i))
        log('prototype class identity: {0}'.format(prototype_img_identity[i].item()))
        if prototype_max_connection[i] != prototype_img_identity[i].item():
            log('prototype connection identity: {0}'.format(prototype_max_connection[i].item()))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][i]))
        log('last layer connection with predicted class: {0}'.format(ppnet.module.last_layer.weight[predicted_cls][i].item()))
        log('similarity x weight: {0}'.format(simwt[i]))
        
        activation_pattern = prototype_activation_patterns[idx][i].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size[1], img_size[0]),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern, percentile = 99.9)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        img_rgb_float = imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                overlayed_img)
        
        overlayed_ten = transforms.ToTensor()(overlayed_img)
        grid_img_tensor.append(overlayed_ten)
        ori_img_ten = transforms.ToTensor()(img_rgb_float)
        grid_img_tensor.append(ori_img_ten)
        
        log('--------------------------------------------------------------')
    
print("simwt_total (should equal to the logit for the predicted class):", sum(simwt), flush=True)
grid = torchvision.utils.make_grid(grid_img_tensor, nrow=10, padding=1)
torchvision.utils.save_image(grid,os.path.join(save_analysis_path, 'highest_simwt_prototypes'+'class_'+str(predicted_cls),"grid_activated_proto_all.png"))
'''
##### Top 10 prototypes SIM*WEIGHT (NEAREST) PROTOTYPES OF THIS IMAGE
simwt_sorted, simwt_sorted_indices = torch.sort(torch.from_numpy(simwt), descending=True)
for i in simwt_sorted_indices[:10]:
    i = i.item()
    if simwt[i] > 0.01:
        log('top {0} activated prototype for this image:'.format(i))
        save_prototype(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls),
                                'top-%d_activated_prototype.png' % i), start_epoch_number, i)
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls),
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=i,
                                          bbox_height_start=prototype_info[i][1],
                                          bbox_height_end=prototype_info[i][2],
                                          bbox_width_start=prototype_info[i][3],
                                          bbox_width_end=prototype_info[i][4],
                                          color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls),
                                                'top-%d_activated_prototype_self_act.png' % i), start_epoch_number, i)
        log('prototype index: {0}'.format(i))
        log('prototype class identity: {0}'.format(prototype_img_identity[i].item()))
        if prototype_max_connection[i] != prototype_img_identity[i].item():
            log('prototype connection identity: {0}'.format(prototype_max_connection[i].item()))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][i]))
        log('last layer connection with predicted class: {0}'.format(ppnet.module.last_layer.weight[predicted_cls][i].item()))
        log('similarity x weight: {0}'.format(simwt[i]))
        
        activation_pattern = prototype_activation_patterns[idx][i].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size[1], img_size[0]),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern, percentile = 99.9)
        #high_act_patch_indices = map_region_to_fixed_patch_size(high_act_patch_indices)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls),
                                'most_highly_activated_patch_by_top-%d_prototype_protopnetviz.png' % i),
                high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        img_rgb_float = imsave_with_bbox(fname=os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls),
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls),
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                overlayed_img)
        
        grid_img_tensor = []
        overlayed_ten = transforms.ToTensor()(overlayed_img)
        grid_img_tensor.append(overlayed_ten)
        ori_img_ten = transforms.ToTensor()(img_rgb_float)
        grid_img_tensor.append(ori_img_ten)
        grid = torchvision.utils.make_grid(grid_img_tensor, nrow=10, padding=1)
        torchvision.utils.save_image(grid, os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(predicted_cls), "proto_"+str(i)+".png"))

        log('--------------------------------------------------------------')

##### All non-zero SIM*WEIGHT (NEAREST) PROTOTYPES OF THE NOT PREDICTED CLASS FOR THIS IMAGE
makedir(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class)))

log('All non-zero prototypes of this image for class_'+str(notpredicted_class)+':')
grid_img_tensor_notpred_class = []
simwt_notpred_class_total=0
simwt_notpred_class = np.zeros(prototype_shape[0])
for i in range(0, prototype_shape[0]):
    simwt_notpred_class[i] = prototype_activations[idx][i].item() * ppnet.module.last_layer.weight[notpredicted_class, i].item()
    print(i, prototype_activations[idx][i].item(), ppnet.module.last_layer.weight[notpredicted_class, i].item(), simwt[i], flush=True)

'''simwt_sorted_notpred_class, simwt_sorted_indices_notpred_class = torch.sort(torch.from_numpy(simwt_notpred_class), descending=True)
for i in simwt_sorted_indices_notpred_class:
    if simwt_notpred_class[i] > 0.01:
        log('prototype index: {0}'.format(i))
        log('prototype class identity: {0}'.format(prototype_img_identity[i].item()))
        if prototype_max_connection[i] != prototype_img_identity[i].item():
            log('prototype connection identity: {0}'.format(prototype_max_connection[i].item()))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][i]))
        log('last layer connection with predicted class: {0}'.format(ppnet.module.last_layer.weight[notpredicted_class][i].item()))
        log('similarity x weight: {0}'.format(simwt_notpred_class[i]))
        
        activation_pattern = prototype_activation_patterns[idx][i].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size[1], img_size[0]),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern, percentile = 99.9)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        img_rgb_float = imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                overlayed_img)
        
        overlayed_ten = transforms.ToTensor()(overlayed_img)
        grid_img_tensor_notpred_class.append(overlayed_ten)
        ori_img_ten = transforms.ToTensor()(img_rgb_float)
        grid_img_tensor_notpred_class.append(ori_img_ten)
        
        log('--------------------------------------------------------------')
    
print("simwt_total (should equal to the logit for the not predicted class):", sum(simwt_notpred_class), flush=True)
grid = torchvision.utils.make_grid(grid_img_tensor_notpred_class, nrow=10, padding=1)
torchvision.utils.save_image(grid,os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class),"grid_activated_proto_all.png"))
'''

simwt_sorted_notpred_class, simwt_sorted_indices_notpred_class = torch.sort(torch.from_numpy(simwt_notpred_class), descending=True)
for i in simwt_sorted_indices_notpred_class[:10]:
    i = i.item()
    if simwt_notpred_class[i] > 0.01:
        log('top {0} activated prototype for this image:'.format(i))
        save_prototype(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class),
                                'top-%d_activated_prototype.png' % i), start_epoch_number, i)
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class),
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=i,
                                          bbox_height_start=prototype_info[i][1],
                                          bbox_height_end=prototype_info[i][2],
                                          bbox_width_start=prototype_info[i][3],
                                          bbox_width_end=prototype_info[i][4],
                                          color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class),
                                                'top-%d_activated_prototype_self_act.png' % i), start_epoch_number, i)
        log('prototype index: {0}'.format(i))
        log('prototype class identity: {0}'.format(prototype_img_identity[i].item()))
        if prototype_max_connection[i] != prototype_img_identity[i].item():
            log('prototype connection identity: {0}'.format(prototype_max_connection[i].item()))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][i]))
        log('last layer connection with predicted class: {0}'.format(ppnet.module.last_layer.weight[notpredicted_class][i].item()))
        log('similarity x weight: {0}'.format(simwt_notpred_class[i]))
        
        activation_pattern = prototype_activation_patterns[idx][i].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size[1], img_size[0]),
                                                interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern, percentile = 99.9)
        #high_act_patch_indices = map_region_to_fixed_patch_size(high_act_patch_indices)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                    high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class),
                                'most_highly_activated_patch_by_top-%d_prototype_protopnetviz.png' % i),
                high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        img_rgb_float = imsave_with_bbox(fname=os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class),
                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class),
                                'prototype_activation_map_by_top-%d_prototype.png' % i),
                overlayed_img)
        
        grid_img_tensor_notpred_class = []
        overlayed_ten = transforms.ToTensor()(overlayed_img)
        grid_img_tensor_notpred_class.append(overlayed_ten)
        ori_img_ten = transforms.ToTensor()(img_rgb_float)
        grid_img_tensor_notpred_class.append(ori_img_ten)
        
        grid = torchvision.utils.make_grid(grid_img_tensor_notpred_class, nrow=2, padding=1)
        torchvision.utils.save_image(grid, os.path.join(save_analysis_path, 'highest_simwt_prototypes_'+'class_'+str(notpredicted_class), "proto_"+str(i)+".png"))

        log('--------------------------------------------------------------')

'''
##### MOST ACTIVATED (NEAREST) 10 PROTOTYPES OF THIS IMAGE
makedir(os.path.join(save_analysis_path, 'most_activated_prototypes'))

log('Most activated 10 prototypes of this image:')
array_act, sorted_indices_act = torch.sort(prototype_activations[idx]) #similarity scores
for i in range(1,11):
    log('top {0} activated prototype for this image:'.format(i))
    save_prototype(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                'top-%d_activated_prototype.png' % i),
                   start_epoch_number, sorted_indices_act[-i].item())
    save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                             'top-%d_activated_prototype_in_original_pimg.png' % i),
                                          epoch=start_epoch_number,
                                          index=sorted_indices_act[-i].item(),
                                          bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                          bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                          bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                          bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                          color=(0, 255, 255))
    save_prototype_self_activation(os.path.join(save_analysis_path, 'most_activated_prototypes',
                                                'top-%d_activated_prototype_self_act.png' % i),
                                   start_epoch_number, sorted_indices_act[-i].item())
    log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
    log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
    if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
        log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
    log('activation value (similarity score): {0}'.format(array_act[-i]))
    log('last layer connection with predicted class: {0}'.format(ppnet.module.last_layer.weight[predicted_cls][sorted_indices_act[-i].item()]))
    
    activation_pattern = prototype_activation_patterns[idx][sorted_indices_act[-i].item()].detach().cpu().numpy()
    upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size[1], img_size[0]),
                                              interpolation=cv2.INTER_CUBIC)
    
    # show the most highly activated patch of the image by this prototype
    high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern, percentile = 99.9)
    high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                  high_act_patch_indices[2]:high_act_patch_indices[3], :]
    log('most highly activated patch of the chosen image by this prototype:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_by_top-%d_prototype.png' % i),
               high_act_patch)
    log('most highly activated patch by this prototype shown in the original image:')
    imsave_with_bbox(fname=os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                     img_rgb=original_img,
                     bbox_height_start=high_act_patch_indices[0],
                     bbox_height_end=high_act_patch_indices[1],
                     bbox_width_start=high_act_patch_indices[2],
                     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
    
    # show the image overlayed with prototype activation map
    rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
    rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[...,::-1]
    overlayed_img = 0.5 * original_img + 0.3 * heatmap
    log('prototype activation map of the chosen image:')
    #plt.axis('off')
    plt.imsave(os.path.join(save_analysis_path, 'most_activated_prototypes',
                            'prototype_activation_map_by_top-%d_prototype.png' % i),
               overlayed_img)
    log('--------------------------------------------------------------')

##### PROTOTYPES FROM TOP-k CLASSES
k = 2
log('Prototypes from top-%d classes:' % k)
topk_logits, topk_classes = torch.topk(logits[idx], k=k)
for i,c in enumerate(topk_classes.detach().cpu().numpy()):
    makedir(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1)))

    log('top %d predicted class: %d' % (i+1, c))
    log('logit of the class: %f' % topk_logits[i])
    class_prototype_indices = np.nonzero(ppnet.module.prototype_class_identity.detach().cpu().numpy()[:, c])[0]
    class_prototype_activations = prototype_activations[idx][class_prototype_indices]
    _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

    prototype_cnt = 1
    for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
        prototype_index = class_prototype_indices[j]
        save_prototype(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                    'top-%d_activated_prototype.png' % prototype_cnt),
                       start_epoch_number, prototype_index)
        save_prototype_original_img_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                                 'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                              epoch=start_epoch_number,
                                              index=prototype_index,
                                              bbox_height_start=prototype_info[prototype_index][1],
                                              bbox_height_end=prototype_info[prototype_index][2],
                                              bbox_width_start=prototype_info[prototype_index][3],
                                              bbox_width_end=prototype_info[prototype_index][4],
                                              color=(0, 255, 255))
        save_prototype_self_activation(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                                    'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                       start_epoch_number, prototype_index)
        log('prototype index: {0}'.format(prototype_index))
        log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))
        if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
        log('activation value (similarity score): {0}'.format(prototype_activations[idx][prototype_index]))
        log('last layer connection: {0}'.format(ppnet.module.last_layer.weight[c][prototype_index]))
        
        activation_pattern = prototype_activation_patterns[idx][prototype_index].detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size[1], img_size[0]),
                                                  interpolation=cv2.INTER_CUBIC)
        
        # show the most highly activated patch of the image by this prototype
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern, percentile = 99.9)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                      high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                   high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                            'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                         img_rgb=original_img,
                         bbox_height_start=high_act_patch_indices[0],
                         bbox_height_end=high_act_patch_indices[1],
                         bbox_width_start=high_act_patch_indices[2],
                         bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))
        
        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img + 0.3 * heatmap
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, 'top-%d_class_prototypes' % (i+1),
                                'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                   overlayed_img)
        log('--------------------------------------------------------------')
        prototype_cnt += 1
    log('***************************************************************')
'''
if predicted_cls == correct_cls:
    log('Prediction is correct.')
else:
    log('Prediction is wrong.')

logclose()

