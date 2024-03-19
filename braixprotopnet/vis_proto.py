import os
import csv
import torch
import argparse
import torchvision
from PIL import Image
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw as D

import model
from settings import prototype_shape, base_architecture, img_size, num_classes, \
                prototype_activation_function, add_on_layers_type

def load_model(net, path):
    checkpoint = torch.load(path, map_location='cuda:0')
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def vis_proto_csv(ppnet, csvfile, logdir):
    classification_weight = ppnet.module.last_layer.weight
    print(classification_weight.shape)
    patches_per_prototype = dict()
    all_tensors=[]
    k=10
    with open (csvfile, newline='') as f:
        filereader = csv.reader(f, delimiter=',')
        next(filereader) #skip header
        for (prototype, imgname, h_min_224, h_max_224, w_min_224, w_max_224) in filereader:
            p = prototype
            print("prototype:", p)
            if p not in patches_per_prototype.keys():
                patches_per_prototype[p] = []
            imgpath = image_path_root + '/' + imgname
            img = Image.open(imgpath).convert('RGB')
           
            img_orig_width, img_orig_height = img.size
            # print("orig widht: ", img_orig_width, img_orig_height)
            h_min_224, h_max_224, w_min_224, w_max_224 = float(h_min_224), float(h_max_224), float(w_min_224), float(w_max_224)
            # print("first sizes: ", h_min_224, h_max_224, w_min_224, w_max_224)
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
            # orig_img_location_h_min = int((img_orig_height/224.) * h_min_224)
            # orig_img_location_h_max = int((img_orig_height/224.) * h_max_224)
            # orig_img_location_w_min = int((img_orig_width/224.) * w_min_224)
            # orig_img_location_w_max = int((img_orig_width/224.) * w_max_224)

            # print(orig_img_location_h_min, orig_img_location_h_max, orig_img_location_w_min, orig_img_location_w_max, flush=True)
            img = transforms.Resize(size=(1536, 768))(img)
            img_tensor = transforms.ToTensor()(img).unsqueeze_(0) #shape (1, 3, h, w)
           
            # print(img_tensor.shape)
            imagepatch = img_tensor[0,:,h_min_224:h_max_224, w_min_224:w_max_224]
            
            if len(patches_per_prototype[p])<k:
                patches_per_prototype[p].append(imagepatch)     

    for p in patches_per_prototype.keys():
        if 0 <= int(p) < (prototype_shape[0]/2):
            class_name = 'B'
        else:
            class_name = 'M'
        
        text_part1 = class_name+str(p)+'\n'+str(round(classification_weight[0, int(p)].item(),1))+'\n'+str(round(classification_weight[1, int(p)].item(),1))
        txtimage = Image.new("RGB", (imagepatch.shape[2],imagepatch.shape[1]), (0, 0, 0))
        draw = D.Draw(txtimage)
        #font = ImageFont.truetype(r'C:\Users\System-Pc\Desktop\arial.ttf', 20)  
        #fnt = ImageFont.load_default(size=15)
        draw.text((imagepatch.shape[2]//2, imagepatch.shape[1]//2), text_part1, anchor='mm', fill="white")
        txttensor = transforms.ToTensor()(txtimage)
        patches_per_prototype[p].append(txttensor)

        grid = torchvision.utils.make_grid(patches_per_prototype[p], nrow=k+1, padding=1)
        torchvision.utils.save_image(grid,os.path.join(logdir,"grid_%s.png"%(str(p))))
        all_tensors+=patches_per_prototype[p]
    
    grid = torchvision.utils.make_grid(all_tensors, nrow=k+1, padding=1)
    torchvision.utils.save_image(grid,os.path.join(logdir,"grid_topk_all.png"))

parser = argparse.ArgumentParser()
parser.add_argument('-modelpath', type=str, default='')
args = parser.parse_args()

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)

ppnet = torch.nn.DataParallel(ppnet)
ppnet = load_model(ppnet, args.modelpath)
device = torch.device('cuda')
ppnet = ppnet.to(device)

'''csvfile_path = '/home/pathaks/braixxprotopnet/saved_models/cmmd/convnext_tiny_13/005-1/net_trained_best_8_8_nearest_train_protopnet/protopnet_cbis_topk.csv'
visualization_path = '/home/pathaks/braixxprotopnet/saved_models/cmmd/convnext_tiny_13/005-1/net_trained_best_8_8_nearest_train_protopnet'
image_path_root = "/deepstore/datasets/dmb/medical/breastcancer/mammography/cmmd/processed_png_8bit" 
#adjusted_bounds_csvfile_path = '/home/pathaks/braixxprotopnet/saved_models/cmmd/convnext_tiny_13/005-1/net_trained_best_8_8_nearest_train_protopnet/protopnet_cbis_topk.csv'
'''
csvfile_path = '/home/pathaks/braixxprotopnet/saved_models/cbis-ddsm/convnext_tiny_13/006-1/net_trained_best_8_8_nearest_train_protopnet/protopnet_cbis_topk.csv'
visualization_path = '/home/pathaks/braixxprotopnet/saved_models/cbis-ddsm/convnext_tiny_13/006-1/net_trained_best_8_8_nearest_train_protopnet'
image_path_root = "/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/multiinstance_data_8bit" 

vis_proto_csv(ppnet, csvfile_path, visualization_path)