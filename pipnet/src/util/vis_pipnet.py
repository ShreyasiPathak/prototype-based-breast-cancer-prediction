from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F
import torch.utils.data
import os, shutil
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import statistics
import torchvision
from util.func import get_patch_size
import random
import csv

from util.data_augmentation import MyHorizontalFlip

@torch.no_grad()                    
def visualize_topk(net, projectloader, df, num_classes, device, foldername, args: argparse.Namespace, k=10):
    print("Visualizing prototypes for topk...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

        near_imgs_dirs = dict()
        seen_max = dict()
        saved = dict()
        saved_ys = dict()
        tensors_per_prototype = dict()
        
        for p in range(net.module._num_prototypes):
            near_imgs_dir = os.path.join(dir, str(p))
            near_imgs_dirs[p]=near_imgs_dir
            seen_max[p]=0.
            saved[p]=0
            saved_ys[p]=[]
            tensors_per_prototype[p]=[]
        
        patchsize, skip_h, skip_w = get_patch_size(args)
        print("patchsize: ", patchsize, "skip: ", skip_h, skip_w, flush=True)

        #imgs = projectloader.dataset.imgs
        
        # Make sure the model is in evaluation mode
        net.eval()
        classification_weights = net.module._classification.weight

        # Show progress on progress bar
        img_iter = tqdm(enumerate(projectloader),
                        total=len(projectloader),
                        mininterval=50.,
                        desc='Collecting topk',
                        ncols=0)

        # Iterate through the data
        images_seen = 0
        topks = dict()

        csvfile = open(os.path.join(dir,'pipnet_cbis_topk.csv'), "w")
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(["prototype", "img name", "h_min_224", "h_max_224", "w_min_224", "w_max_224"])

        # Iterate through the training set
        for i, (_, xs, ys, _) in img_iter:
            images_seen+=1
            xs, ys = xs.to(device), ys.to(device)

            with torch.no_grad():
                # Use the model to classify this batch of input data
                pfs, pooled, _ = net(xs, inference=True)
                pooled = pooled.squeeze(0) # channels (prototypes), H, W
                pfs = pfs.squeeze(0) 
                
                for p in range(pooled.shape[0]):
                    c_weight = torch.max(classification_weights[:,p]) 
                    #if c_weight > 1e-3:#ignore prototypes that are not relevant to any class
                    if p not in topks.keys():
                        topks[p] = []
                        
                    if len(topks[p]) < k:
                        topks[p].append((i, pooled[p].item()))
                    else:
                        topks[p] = sorted(topks[p], key=lambda tup: tup[1], reverse=True)
                        if topks[p][-1][1] < pooled[p].item():
                            topks[p][-1] = (i, pooled[p].item())
                        if topks[p][-1][1] == pooled[p].item():
                            # equal scores. randomly chose one (since dataset is not shuffled so latter images with same scores can now also get in topk).
                            replace_choice = random.choice([0, 1])
                            if replace_choice > 0:
                                topks[p][-1] = (i, pooled[p].item())
                
                #print(ys.item(), pooled[87].item(), classification_weights[0, 87].item(), flush=True)

        alli = []
        prototypes_not_used = []
        for p in topks.keys():
            found = False
            for idx, score in topks[p]:
                alli.append(idx)
                if score > 0.1:  #in case prototypes have fewer than k well-related patches
                    found = True
            if not found:
                prototypes_not_used.append(p)

        print(len(prototypes_not_used), "prototypes do not have any similarity score > 0.1. Will be ignored in visualisation.")
        abstained = 0
        # Show progress on progress bar
        img_iter = tqdm(enumerate(projectloader),
                        total=len(projectloader),
                        mininterval=50.,
                        desc='Visualizing topk',
                        ncols=0)
        for i, (idx_img, xs, ys, view) in img_iter: #shuffle is false so should lead to same order as in imgs
            if i in alli:
                xs, ys = xs.to(device), ys.to(device)
                for p in topks.keys():
                    if p not in prototypes_not_used:
                        for idx, score in topks[p]:
                            if idx == i:
                                # Use the model to classify this batch of input data
                                with torch.no_grad():
                                    softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (1, num_prototypes, W, H), feature map resulting from softmax over each patch.
                                    outmax = torch.amax(out,dim=1)[0] #shape ([1]) because batch size of projectloader is 1
                                    if outmax.item() == 0.:
                                        # print("PIP-Net abstained from a decision for this img", imgs[i][0], outmax[k].item(), flush=True)
                                        abstained+=1
                                
                                # Take the max per prototype.                             
                                max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
                                max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
                                max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1) #shape (num_prototypes)
                                
                                c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
                                if (c_weight > 1e-10) or ('pretrain' in foldername):
                                    
                                    h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                                    w_idx = max_idx_per_prototype_w[p]
                                    
                                    img_name = df.iloc[idx_img.item()]['ShortPath']
                                    print("img name:", img_name)
                                    img_path = args.preprocessed_imagepath + '/' + img_name
                                    img_to_open = img_path
                                    print("path:", img_to_open)
                                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                                        img_to_open = img_to_open[0]
                                    
                                    if args.flipimage:
                                        ori_img = Image.open(img_to_open).convert("RGB")
                                        breast_side = view[0]
                                        hflip_img = MyHorizontalFlip()
                                        ori_img = hflip_img(ori_img, breast_side)
                                    else:
                                        ori_img = Image.open(img_to_open).convert("RGB")

                                    image = transforms.Resize(size=(args.image_size[0], args.image_size[1]))(ori_img)
                                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip_h, skip_w, h_idx, w_idx)
                                    try:
                                        h_coor_min = h_coor_min.item()
                                    except:
                                        pass
                                    try:
                                        h_coor_max = h_coor_max.item()
                                    except:
                                        pass
                                    try:
                                        w_coor_min = w_coor_min.item()
                                    except:
                                        pass
                                    try:
                                        w_coor_max = w_coor_max.item()
                                    except:
                                        pass
                                    writer.writerow([p, img_name, h_coor_min, h_coor_max, w_coor_min, w_coor_max])
                                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                                    saved[p]+=1
                                    tensors_per_prototype[p].append(img_tensor_patch)
        csvfile.close()

        print("Abstained: ", abstained, flush=True)
        all_tensors = []
        for p in range(net.module._num_prototypes):
            if saved[p]>0:
                total_score = 0
                for _, score in topks[p]:
                    total_score+= score

                if args.numclasses == 5:
                    text_part1 = "P "+str(p)+';'+'w '+'1 '
                    text_part2 = str(round(classification_weights[0,p].item(),1))+' 2 '+str(round(classification_weights[1,p].item(),1))
                    text_part3 = ' 3 '+str(round(classification_weights[2,p].item(),1))+' 4 '
                    text_part4 = str(round(classification_weights[3,p].item(),1)) + ' 5 '+str(round(classification_weights[4,p].item(),1))
                    text_part5 = ';sim '+ str(round(total_score/len(topks[p]),1))
                    tail = 5
                    #print("annotation:", text_part1+text_part2+text_part3)
                    
                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part1, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)
                    
                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part2, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)
                    
                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part3, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)
                    
                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part4, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)

                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part5, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)
                
                elif args.numclasses == 2:
                    text_part1 = "P "+str(p)+';'+'w '+'B '
                    text_part2 = str(round(classification_weights[0,p].item(),1))+' M '+str(round(classification_weights[1,p].item(),1))
                    text_part3 = ';sim '+ str(round(total_score/len(topks[p]),1))
                    tail = 3
                    #print("annotation:", text_part1+text_part2+text_part3)
                    
                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part1, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)
                    
                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part2, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)
                    
                    txtimage = Image.new("RGB", (img_tensor_patch.shape[1],img_tensor_patch.shape[2]), (0, 0, 0))
                    draw = D.Draw(txtimage)
                    draw.text((img_tensor_patch.shape[0]//2, img_tensor_patch.shape[1]//2), text_part3, anchor='mm', fill="white")
                    txttensor = transforms.ToTensor()(txtimage)
                    tensors_per_prototype[p].append(txttensor)
                
                try:
                    #grid = torchvision.utils.make_grid(tensors_per_prototype[p], nrow=k+tail, padding=1)
                    #torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_%s.png"%(str(p))))
                    if saved[p]>=k:
                        all_tensors+=tensors_per_prototype[p]
                except:
                    pass
        grid = torchvision.utils.make_grid(all_tensors, nrow=k+tail, padding=1)
        torchvision.utils.save_image(grid,os.path.join(dir,"grid_topk_all.png"))
        return topks
    else:
        return None
        

def visualize(net, projectloader, df, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

        near_imgs_dirs = dict()
        seen_max = dict()
        saved = dict()
        saved_ys = dict()
        tensors_per_prototype = dict()
        abstainedimgs = set()
        notabstainedimgs = set()
        
        for p in range(net.module._num_prototypes):
            near_imgs_dir = os.path.join(dir, str(p))
            near_imgs_dirs[p]=near_imgs_dir
            seen_max[p]=0.
            saved[p]=0
            saved_ys[p]=[]
            tensors_per_prototype[p]=[]
        
        patchsize, skip_h, skip_w = get_patch_size(args)

        #imgs = projectloader.dataset.imgs
        
        # skip some images for visualisation to speed up the process
        if df.shape[0]/num_classes <10:
            skip_img=10
        elif df.shape[0]/num_classes < 50:
            skip_img=5
        else:
            skip_img = 2
        print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

        # Make sure the model is in evaluation mode
        net.eval()
        classification_weights = net.module._classification.weight
        # Show progress on progress bar
        img_iter = tqdm(enumerate(projectloader),
                        total=len(projectloader),
                        mininterval=100.,
                        desc='Visualizing',
                        ncols=0)

        # Iterate through the data
        images_seen_before = 0
        for i, (idx_img, xs, ys, view) in img_iter: #shuffle is false so should lead to same order as in imgs
            if i % skip_img == 0:
                images_seen_before+=xs.shape[0]
                continue
            
            xs, ys = xs.to(device), ys.to(device)
            # Use the model to classify this batch of input data
            with torch.no_grad():
                softmaxes, _, out = net(xs, inference=True) 

            max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
            # In PyTorch, images are represented as [channels, height, width]
            max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
            max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
            for p in range(0, net.module._num_prototypes):
                c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
                if c_weight>0:
                    h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                    w_idx = max_idx_per_prototype_w[p]
                    idx_to_select = max_idx_per_prototype[p,h_idx, w_idx].item()
                    found_max = max_per_prototype[p,h_idx, w_idx].item()
                    
                    imgname = args.preprocessed_imagepath + '/' + df.loc[images_seen_before+idx_to_select, 'ShortPath']
                    #imgname = imgs[images_seen_before+idx_to_select]
                    if out.max() < 1e-8:
                        # print("abstained: ", p, imgname)
                        abstainedimgs.add(imgname)
                    else:
                        notabstainedimgs.add(imgname)
                    
                    if found_max > seen_max[p]:
                        seen_max[p]=found_max
                
                    if found_max > 0.5:
                        img_to_open = args.preprocessed_imagepath + '/' + df.loc[images_seen_before+idx_to_select, 'ShortPath']
                        #img_to_open = imgs[images_seen_before+idx_to_select]
                        if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                            imglabel = img_to_open[1]
                            img_to_open = img_to_open[0]
                        else:
                            imglabel = df.loc[images_seen_before+idx_to_select, 'Groundtruth']
                        
                        if args.flipimage:
                            ori_img = Image.open(img_to_open).convert("RGB")
                            breast_side = view[0]
                            hflip_img = MyHorizontalFlip()
                            ori_img = hflip_img(ori_img, breast_side)
                        else:
                            ori_img = Image.open(img_to_open).convert("RGB")

                        image = transforms.Resize(size=(args.image_size[0], args.image_size[1]))(ori_img)
                        img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                        h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip_h, skip_w, h_idx, w_idx)
                        img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                        saved[p]+=1
                        tensors_per_prototype[p].append((img_tensor_patch, found_max))
                        
                        save_path = os.path.join(dir, "prototype_%s")%str(p)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        draw = D.Draw(image)
                        draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                        image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))
                        
            
            images_seen_before+=len(ys)

        print("num images abstained: ", len(abstainedimgs), flush=True)
        print("num images not abstained: ", len(notabstainedimgs), flush=True)
        for p in range(net.module._num_prototypes):
            if saved[p]>0:
                try:
                    sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                    sorted_ps = [i[0] for i in sorted_by_second]
                    grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                    torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
                except RuntimeError:
                    pass

# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip_h, skip_w, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0,(h_idx-1)*skip_h+4)
        if h_idx < softmaxes_shape[-1]-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0,(w_idx-1)*skip_w+4)
        if w_idx < softmaxes_shape[-1]-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx*skip_h
        h_coor_max = min(img_size[0], h_idx*skip_h+patchsize)
        w_coor_min = w_idx*skip_w
        w_coor_max = min(img_size[1], w_idx*skip_w+patchsize)                                    
    
    if h_idx == softmaxes_shape[1]-1:
        h_coor_max = img_size[0]
    if w_idx == softmaxes_shape[2] -1:
        w_coor_max = img_size[1]
    if h_coor_max == img_size[0]:
        h_coor_min = img_size[0]-patchsize
    if w_coor_max == img_size[1]:
        w_coor_min = img_size[1]-patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max
    