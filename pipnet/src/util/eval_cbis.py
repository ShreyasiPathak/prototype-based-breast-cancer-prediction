import os
import cv2
import math
import torch
import argparse
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.patches as patches
import torchvision.transforms as transforms

from util.func import get_patch_size
from util.vis_pipnet import get_img_coordinates

def create_mask_file(filepath_roiloc, filepath_newimagesize):
    """
    Function that combines the location coordinates of the cleaned image with the 
    location coordinates of the ROI. This is valid for vindr as vindr dataset has 
    provided a roi location coordinate, unlike, cbis, which has provided a mask image.
    """
    df_roiloc = pd.read_csv(filepath_roiloc).rename(columns={'image_id':'ImageName'}).sort_values(by='ImageName') #20485
    df_newimagesize = pd.read_csv(filepath_newimagesize,sep=';').sort_values(by='ImageName') #19999
    df_merged = df_newimagesize.merge(df_roiloc, on ='ImageName', how = 'left') #20485
    df_nonan = df_merged[['xmin', 'ymin', 'xmax', 'ymax']].dropna()
    df_merged = df_merged.loc[df_nonan.index] #2254
    df_merged.to_csv('/groups/dso/spathak/vindr/'+'vindr_singleinstance_imgpreprocessing_size_withroiloc.csv', sep=';', na_rep='NULL', index=False)

def mask_paths(config_params):
    """
    Function that returns the location of the mask and 
    file path of bounding box location after image cleaning for all datasets 
    """
    if config_params.dataset == 'cbis-ddsm':
        mask_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/roi-images/'
        image_size_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv'
    
    elif config_params.dataset == 'vindr':
        mask_path = '/groups/dso/spathak/vindr/physionet.org/files/vindr-mammo/1.0.0/finding_annotations.csv'
        image_size_path = '/groups/dso/spathak/vindr/vindr_singleinstance_imgpreprocessing_size.csv'

    return mask_path, image_size_path

def dice_similarity_score(trueLoc, predLoc):
    """
    Function that calculates IOU given the true ROI location and predicted ROI location
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(trueLoc[0], predLoc[0])
    yA = max(trueLoc[1], predLoc[1])
    xB = min(trueLoc[2], predLoc[2])
    yB = min(trueLoc[3], predLoc[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (trueLoc[2] - trueLoc[0] + 1) * (trueLoc[3] - trueLoc[1] + 1)
    boxBArea = (predLoc[2] - predLoc[0] + 1) * (predLoc[3] - predLoc[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    dsc = 2*interArea / float(boxAArea + boxBArea)
    # return the intersection over union value
    return dsc

def intersection_over_union(trueLoc, predLoc):
    """
    Function that calculates IOU given the true ROI location and predicted ROI location
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(trueLoc[0], predLoc[0])
    yA = max(trueLoc[1], predLoc[1])
    xB = min(trueLoc[2], predLoc[2])
    yB = min(trueLoc[3], predLoc[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (trueLoc[2] - trueLoc[0] + 1) * (trueLoc[3] - trueLoc[1] + 1)
    boxBArea = (predLoc[2] - predLoc[0] + 1) * (predLoc[3] - predLoc[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def bounding_box_true_pred(original_image, true_mask_loc_all, pred_mask_loc_all, view_id, views_names, figure, ax):
    """
    Function that calculates the bounding box around the original ROI mask image
    :true_mask: ROI mask image
    :pred_mask_loc: min_x, min_y, max_x, max_y, where x = width, y = height
    :original_image: pytorch, C, H, W
    """
    original_image = original_image.numpy()[0,:,:]
    ax[view_id].imshow(original_image, aspect='equal', cmap='gray')
    for true_mask_loc in true_mask_loc_all:
        print("True mask loc:", true_mask_loc)
        rect1 = patches.Rectangle(tuple(true_mask_loc[:2]), true_mask_loc[2]-true_mask_loc[0], true_mask_loc[3]-true_mask_loc[1], linewidth=1, edgecolor='g', facecolor='none')
        ax[view_id].add_patch(rect1)
    for pred_mask_loc in pred_mask_loc_all:
        print("Pred mask loc:", pred_mask_loc)
        rect2 = patches.Rectangle(tuple(pred_mask_loc[:2]), pred_mask_loc[2]-pred_mask_loc[0], pred_mask_loc[3]-pred_mask_loc[1], linewidth=1, edgecolor='r', facecolor='none')
        ax[view_id].add_patch(rect2)
    return figure, ax

def match_to_mask_images_cbis(config_params, original_image, exam_name, model_patch_locations, seg_eval_metric):
    """
    Function to calculate how much does the patch extracted by the model match to the true ROI 
    """
    #Map this exam (image or case) to a name that is a substring of the ROI folder name
    mask_path, image_size_path = mask_paths(config_params)
    image_folder_name = "_".join(exam_name.split('/')[-1].split('_')[:-1])
    print("image_folder_name:", image_folder_name, flush=True)

    #which ROI folders belong to this exam
    roi_folders = os.listdir(mask_path)
    roi_folder_name = [roi_folder for roi_folder in roi_folders if image_folder_name in roi_folder]
    print("roi folder name:", roi_folder_name, flush=True)
    
    #Read the bounding box coordinates generated after passing the original image through the cleaning algorithm
    df_img_size = pd.read_csv(image_size_path, sep=';')
    df_img_size = df_img_size[df_img_size['ImageName'].str.split('_').str[:5].str.join('_')==image_folder_name]

    #select the location of the highest attention patch
    #maxattn_patch_location = model_patch_locations[0, np.argmax(model_patch_attentions), :]

    #calculate segmentation evaluation metric over all ROIs belonging to that exam
    iou_all_over_each_roi = []
    iou_max_over_each_roi = []
    iou_all_over_each_roi_top10 = []
    iou_max_over_each_roi_top10 = []
    iou_all_over_each_roi_allproto = []
    iou_max_over_each_roi_allproto = []
    true_mask_loc_all = [] 
    for roi_folder in roi_folder_name:
        #correcting mask size: original to preprocessed (cleaning algo) to resize (2944x1920)
        true_mask_image = cv2.imread(mask_path+'/'+roi_folder+'/'+'1-2.png') 
        true_mask_image = true_mask_image[df_img_size['pro_min_y'].item():df_img_size['pro_max_y'].item(), df_img_size['pro_min_x'].item():df_img_size['pro_max_x'].item()]
        true_mask_image = cv2.resize(true_mask_image, dsize=(config_params.image_size[1], config_params.image_size[0]))
        x,y,w,h = cv2.boundingRect(true_mask_image[:,:,0])
        true_mask_loc = [x,y,x+w,y+h]
        true_mask_loc_all.append(true_mask_loc)
        
        iou_over_all_patches = []
        iou_over_all_patches_top10 = []
        iou_over_all_patches_allproto = []
        #pred_mask_loc_all = []
        #IoU calculation over the top-1 prototype
        for pred_mask_loc in model_patch_locations[:1]:
            #pred_mask_loc = model_patch_locations[0, idx, :]
            if seg_eval_metric=='IOU':
                iou_over_all_patches.append(intersection_over_union(true_mask_loc, pred_mask_loc))
            elif seg_eval_metric=='DSC':
                iou_over_all_patches.append(dice_similarity_score(true_mask_loc, pred_mask_loc))
            #print("iou_over_all_patches in the function:", iou_over_all_patches, flush=True)
            #pred_mask_loc_all.append(pred_mask_loc)
        
        #max iou over all patches for each roi; [R] where R is number of patches
        #print("model_patch_locations:", model_patch_locations[:1], flush=True)
        #print("iou over all patches:", iou_over_all_patches, flush=True)
        iou_max_over_each_roi.append(max(iou_over_all_patches)) # size = no.of ROIs

        #append iou over all patches for each roi
        iou_all_over_each_roi.append(iou_over_all_patches) #size = #ROIs x #patches

        #IoU calculation over the top-10 prototypes
        for pred_mask_loc in model_patch_locations[:10]:
            #pred_mask_loc = model_patch_locations[0, idx, :]
            if seg_eval_metric=='IOU':
                iou_over_all_patches_top10.append(intersection_over_union(true_mask_loc, pred_mask_loc))
            elif seg_eval_metric=='DSC':
                iou_over_all_patches_top10.append(dice_similarity_score(true_mask_loc, pred_mask_loc))
            #pred_mask_loc_all.append(pred_mask_loc)
        
        #max iou over all patches for each roi; [R] where R is number of patches
        iou_max_over_each_roi_top10.append(max(iou_over_all_patches_top10)) # size = no.of ROIs

        #append iou over all patches for each roi
        iou_all_over_each_roi_top10.append(iou_over_all_patches_top10) #size = #ROIs x #patches
        
        #IoU calculation over all prototypes
        for pred_mask_loc in model_patch_locations:
            #pred_mask_loc = model_patch_locations[0, idx, :]
            if seg_eval_metric=='IOU':
                iou_over_all_patches_allproto.append(intersection_over_union(true_mask_loc, pred_mask_loc))
            elif seg_eval_metric=='DSC':
                iou_over_all_patches_allproto.append(dice_similarity_score(true_mask_loc, pred_mask_loc))
            #pred_mask_loc_all.append(pred_mask_loc)
        
        #max iou over all patches for each roi; [R] where R is number of patches
        iou_max_over_each_roi_allproto.append(max(iou_over_all_patches_allproto)) # size = no.of ROIs

        #append iou over all patches for each roi
        iou_all_over_each_roi_allproto.append(iou_over_all_patches_allproto) #size = #ROIs x #patches      
    
    #fig, ax = bounding_box_true_pred(original_image, true_mask_loc_all, pred_mask_loc_all, view_id, views_names, fig, ax)

    #iou_any_roi = max(iou_max_over_each_roi)
    return iou_max_over_each_roi, iou_all_over_each_roi, iou_max_over_each_roi_top10, iou_all_over_each_roi_top10, iou_max_over_each_roi_allproto, iou_all_over_each_roi_allproto

def get_images(net, vis_test_loader, df_test, device, args: argparse.Namespace, seg_eval_metric):
    # Make sure the model is in evaluation mode
    net.eval()

    df_iou = {}
    df_iou_top10 = {}
    df_iou_allproto = {}
    iou_sum_any_roi = 0
    iou_sum_all_roi = 0
    iou_sum_any_roi_top10 = 0
    iou_sum_all_roi_top10 = 0
    iou_sum_any_roi_allproto = 0
    iou_sum_all_roi_allproto = 0
    local_exp_pos = []
    local_exp_neg = []

    patchsize, skip_h, skip_w = get_patch_size(args)

    '''num_workers = args.num_workers

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(args.image_size[0], args.image_size[1])),
                            transforms.ToTensor(),
                            normalize])

    vis_test_set = torchvision.datasets.ImageFolder(vis_test_dir, transform=transform_no_augment)
    vis_test_loader = torch.utils.data.DataLoader(vis_test_set, batch_size = 1,
                                                shuffle=False, pin_memory=not args.disable_cuda and torch.cuda.is_available(),
                                                num_workers=num_workers)
    imgs = vis_test_set.imgs
    '''

    for k, (_, xs, ys, _) in enumerate(vis_test_loader): #shuffle is false so should lead to same order as in imgs
        xs, ys = xs.to(device), ys.to(device)
        #img = imgs[k][0]
        img_name = df_test.loc[k, 'ShortPath']
        y_label = args.groundtruthdic[df_test.loc[k, 'Groundtruth']]
        img_path = args.preprocessed_imagepath + '/' + img_name
        model_patch_locations = []
        #iou_any_roi_max = []
        #iou_all_roi_mean = []
        #iou_any_roi_max_top10 = []
        #iou_all_roi_mean_top10 = []
        
        print("global non zero proto: ", net.module._classification.weight.any(dim=0).sum().item(), flush=True)
        
        with torch.no_grad():
            softmaxes, pooled, out = net(xs, inference=True) #softmaxes has shape (bs, num_prototypes, W, H), pooled has shape (bs, num_prototypes), out has shape (bs, num_classes)
            sorted_out, sorted_out_indices = torch.sort(out.squeeze(0), descending=True)
            print("sorted_out_indices:", sorted_out_indices, flush=True)
            #for pred_class_idx in sorted_out_indices[:3]:
            #pred_class = classes[pred_class_idx]
            #sorted_pooled, sorted_pooled_indices = torch.sort(pooled.squeeze(0), descending=True)
            simweights = np.zeros(pooled.shape[1])
            #print("pooled:", pooled, flush=True)
            for p in range(0, pooled.shape[1]): #pooled shape - 1x768
                simweights[p] = pooled[0][p].item() * net.module._classification.weight[y_label, p].item()
                #print(p, pooled[0][p].item(), net.module._classification.weight[y_label, p].item(), simweights[p], flush=True)
            #print("global non zero proto: ", net.module._classification.weight.any(dim=0).sum().item(), flush=True)
            local_exp_pos.append(np.sum(simweights>0))
            local_exp_neg.append(np.sum(simweights<0))

            simwt_sorted, simwt_sorted_indices = torch.sort(torch.from_numpy(simweights), descending=True)
            #print("simwt sorted:", simwt_sorted, flush=True)
            for prototype_idx in simwt_sorted_indices:
                #simweight = pooled[0,prototype_idx].item() * net.module._classification.weight[pred_class_idx, prototype_idx].item()
                #simweight = pooled[0,prototype_idx].item() * net.module._classification.weight[y_label, prototype_idx].item()
                #simweights.append(simweight)
                
                #if abs(simweight) > 0:
                if abs(simweights[prototype_idx])>0:
                    max_h, max_idx_h = torch.max(softmaxes[0, prototype_idx, :, :], dim=0)
                    max_w, max_idx_w = torch.max(max_h, dim=0)
                    max_idx_h = max_idx_h[max_idx_w].item()
                    max_idx_w = max_idx_w.item()
                    original_image = Image.open(img_path)
                    #image = transforms.Resize(size=(args.image_size[0], args.image_size[1]))(original_image)
                    #img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip_h, skip_w, max_idx_h, max_idx_w)
                    model_patch_locations.append([w_coor_min, h_coor_min, w_coor_max, h_coor_max])
            
            if model_patch_locations: 
                iou_max_over_each_roi, _, iou_max_over_each_roi_top10, _, iou_max_over_each_roi_allproto, _ = match_to_mask_images_cbis(args, original_image, img_path, model_patch_locations, seg_eval_metric)
            else:
                iou_max_over_each_roi = [0.0]
                iou_max_over_each_roi_top10 = [0.0]
                iou_max_over_each_roi_allproto = [0.0]
            
            
            print("iou_view_each_roi top-1:", iou_max_over_each_roi, flush=True)
            print("iou_view_each_roi top-10:", iou_max_over_each_roi_top10, flush=True)
            
            if iou_max_over_each_roi!=[]:
                iou_any_roi_max = max(iou_max_over_each_roi) # max IOU for any of the ROIs
                iou_all_roi_mean = np.mean(np.array(iou_max_over_each_roi))
            
                df_iou[img_path] = [img_path, iou_any_roi_max, iou_all_roi_mean]
                iou_sum_any_roi = iou_sum_any_roi + iou_any_roi_max
                iou_sum_all_roi = iou_sum_all_roi + iou_all_roi_mean
                print("iou exam any roi iou top-1:", iou_any_roi_max, flush=True)

            if iou_max_over_each_roi_top10!=[]:
                iou_any_roi_max_top10 = max(iou_max_over_each_roi_top10) # max IOU for any of the ROIs
                iou_all_roi_mean_top10 = np.mean(np.array(iou_max_over_each_roi_top10))
            
                df_iou_top10[img_path] = [img_path, iou_any_roi_max_top10, iou_all_roi_mean_top10]
                iou_sum_any_roi_top10 = iou_sum_any_roi_top10 + iou_any_roi_max_top10
                iou_sum_all_roi_top10 = iou_sum_all_roi_top10 + iou_all_roi_mean_top10
                print("iou exam any roi iou top-10:", iou_any_roi_max_top10, flush=True)

            if iou_max_over_each_roi_allproto!=[]:
                iou_any_roi_max_allproto = max(iou_max_over_each_roi_allproto) # max IOU for any of the ROIs
                iou_all_roi_mean_allproto = np.mean(np.array(iou_max_over_each_roi_allproto))
            
                df_iou_allproto[img_path] = [img_path, iou_any_roi_max_allproto, iou_all_roi_mean_allproto]
                iou_sum_any_roi_allproto = iou_sum_any_roi_allproto + iou_any_roi_max_allproto
                iou_sum_all_roi_allproto = iou_sum_all_roi_allproto + iou_all_roi_mean_allproto
                print("iou exam any roi iou all proto:", iou_any_roi_max_allproto, flush=True)
            
    
    #average IoU on the test set over top-1 activated prototypes  
    df_img_iou = pd.DataFrame.from_dict(df_iou, orient='index', columns=['ImageName', 'iou_any_roi_max', 'iou_all_roi_mean'])
    df_img_iou.to_csv(os.path.join(args.log_dir, seg_eval_metric+"_score_test_set_top1_proto"+"_"+str(args.randseedother)+"_"+str(args.randseeddata)+".csv"), sep=';',na_rep='NULL',index=False)
    print("df image iou shape:", df_img_iou.shape[0], flush=True)
    iou_avg_any_roi = iou_sum_any_roi/df_img_iou.shape[0]
    iou_avg_all_roi = iou_sum_all_roi/df_img_iou.shape[0]
    print("iou avg any roi top-1:", iou_avg_any_roi, flush=True)
    print("iou avg all roi top-1:", iou_avg_all_roi, flush=True) 

    #average IoU on the test set over top-10 activated prototypes 
    df_img_iou_top10 = pd.DataFrame.from_dict(df_iou_top10, orient='index', columns=['ImageName', 'iou_any_roi_max', 'iou_all_roi_mean'])
    df_img_iou_top10.to_csv(os.path.join(args.log_dir, seg_eval_metric+"_score_test_set_top10_proto"+"_"+str(args.randseedother)+"_"+str(args.randseeddata)+".csv"), sep=';',na_rep='NULL',index=False)
    iou_avg_any_roi_top10 = iou_sum_any_roi_top10/df_img_iou_top10.shape[0]
    iou_avg_all_roi_top10 = iou_sum_all_roi_top10/df_img_iou_top10.shape[0]
    print("iou avg any roi top-10:", iou_avg_any_roi_top10, flush=True)
    print("iou avg all roi top-10:", iou_avg_all_roi_top10, flush=True)
    
    #average IoU on the test set over all activated prototypes 
    df_img_iou_allproto = pd.DataFrame.from_dict(df_iou_allproto, orient='index', columns=['ImageName', 'iou_any_roi_max', 'iou_all_roi_mean'])
    df_img_iou_allproto.to_csv(os.path.join(args.log_dir, seg_eval_metric+"_score_test_set_all_proto"+"_"+str(args.randseedother)+"_"+str(args.randseeddata)+".csv"), sep=';',na_rep='NULL',index=False)
    iou_avg_any_roi_allproto = iou_sum_any_roi_allproto/df_img_iou_allproto.shape[0]
    iou_avg_all_roi_allproto = iou_sum_all_roi_allproto/df_img_iou_allproto.shape[0]
    print("iou avg any roi all proto:", iou_avg_any_roi_allproto, flush=True)
    print("iou avg all roi all proto:", iou_avg_all_roi_allproto, flush=True)

    print("local exp positive prototypes avg size:", np.mean(np.array(local_exp_pos)), flush=True)
    print("local exp positive prototypes std dev:", np.std(np.array(local_exp_pos)), flush=True)
    print("local exp negative prototypes avg size:", np.mean(np.array(local_exp_neg)), flush=True)
    print("local exp negative prototypes std dev:", np.std(np.array(local_exp_neg)), flush=True)
    f = open('local_exp_pos_cbis'+"_"+str(args.randseedother)+"_"+str(args.randseeddata), 'wb')
    np.save(f, np.array(local_exp_pos))
    f.close()
    f1 = open('local_exp_neg_cbis'+"_"+str(args.randseedother)+"_"+str(args.randseeddata), 'wb')
    np.save(f1, np.array(local_exp_neg))
    f1.close()
'''
def match_to_mask_images_vindr(config_params, original_image, exam_name, model_patch_attentions, model_patch_locations, seg_eval_metric, view_id, views_names, fig, ax):
    #filepath_roiloc, filepath_newimagesize = mask_paths(config_params)
    #create_mask_file(filepath_roiloc, filepath_newimagesize) #20485
    #input('halt')

    #which ROI folders belong to this exam
    print("image_name:", exam_name)
    df_roi = pd.read_csv('/groups/dso/spathak/vindr/vindr_singleinstance_imgpreprocessing_size_withroiloc.csv', sep=';')
    roi_folder_df = df_roi[df_roi['ImageName']==exam_name]#.split('.')[0].split('_')[1]]
    print("roi folder name:", roi_folder_df)
    
    iou_all_over_each_roi = []
    iou_max_over_each_roi = []
    iou_highestattnwt_max_over_each_roi = []
    true_mask_loc_all = [] 
    
    if not roi_folder_df.empty:
        for idx in roi_folder_df.index:
            try:
                roi_row = roi_folder_df.loc[idx]
                #print(roi_row)
                true_mask_image = np.zeros((roi_row['ori_height'], roi_row['ori_width']), dtype=np.uint8)
                mask_white = np.ones((math.ceil(roi_row['ymax']) - math.ceil(roi_row['ymin']), math.ceil(roi_row['xmax']) - math.ceil(roi_row['xmin'])))
                true_mask_image[math.ceil(roi_row['ymin']):math.ceil(roi_row['ymax']), math.ceil(roi_row['xmin']):math.ceil(roi_row['xmax'])] = mask_white
                #plt.imsave('./mask.png', true_mask_image, cmap='gray')
                true_mask_image = true_mask_image * 255
                true_mask_image = true_mask_image[roi_row['pro_min_y'].item():roi_row['pro_max_y'].item(), roi_row['pro_min_x'].item():roi_row['pro_max_x'].item()]
                true_mask_image = data_augmentation_utils.myhorizontalflip(true_mask_image, roi_row['laterality'])
                #plt.imsave('./mask_hf.png', true_mask_image, cmap='gray')
                true_mask_image = cv2.resize(true_mask_image, dsize=(config_params['resize'][1], config_params['resize'][0]))
                #plt.imsave('./mask_rs.png', true_mask_image, cmap='gray')
                #print(true_mask_image.shape)
                #print(true_mask_image.dtype)
                x,y,w,h = cv2.boundingRect(true_mask_image) 
                true_mask_loc = [x,y,x+w,y+h]
                true_mask_loc_all.append(true_mask_loc)
            
                iou_over_all_patches = []
                pred_mask_loc_all = []
                for idx in range(model_patch_locations.shape[1]):
                    patch_location = model_patch_locations[0, idx, :]
                    #extract min_x, min_y, max_x, max_y position from the upper left patch location (extracted by the model)
                    pred_mask_loc = extract_patch_position_wrt_image(original_image, config_params['crop_shape'], patch_location)
                    if seg_eval_metric=='IOU':
                        iou_over_all_patches.append(intersection_over_union(true_mask_loc, pred_mask_loc))
                    elif seg_eval_metric=='DSC':
                        iou_over_all_patches.append(dice_similarity_score(true_mask_loc, pred_mask_loc))
                    pred_mask_loc_all.append(pred_mask_loc)
            
                #max iou over all patches for each roi; [R] where R is number of patches
                iou_max_over_each_roi.append(max(iou_over_all_patches))
                
                #highest attention weighted patch
                iou_highestattnwt_max_over_each_roi.append(iou_over_all_patches[np.argmax(model_patch_attentions)])

                #append iou over all patches for each roi
                iou_all_over_each_roi.append(iou_over_all_patches)
            except:
                pass
    
        #fig, ax = bounding_box_true_pred(original_image, true_mask_loc_all, pred_mask_loc_all, view_id, views_names, fig, ax)
    
    return iou_max_over_each_roi, iou_all_over_each_roi, iou_highestattnwt_max_over_each_roi, fig, ax
'''