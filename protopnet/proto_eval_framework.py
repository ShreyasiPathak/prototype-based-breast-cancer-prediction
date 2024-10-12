import argparse
import torch.nn.functional as F
import torch.utils.data
import os
import numpy as np
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import pandas as pd
import cv2
import csv
import matplotlib.pyplot as plt 

# book keeping namings and code
import model 
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                    prototype_activation_function, add_on_layers_type


def load_model(net, path):
    checkpoint = torch.load(path, map_location=torch.device('cuda:0'))
    #print(checkpoint['model_state_dict'], flush=True)
    net.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['epoch_joint'])
    #print(checkpoint['epoch_stage1'])
    #print(checkpoint['epoch_stage2'])
    return net

def mask_paths(config_params):
    """
    Function that returns the location of the mask and 
    file path of bounding box location after image cleaning for all datasets 
    """
    if config_params.dataset == 'cbis-ddsm':
        mask_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/roi-images/'
        image_size_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/cbis-ddsm_singleinstance_imgpreprocessing_size_mod.csv'
        roi_file_path = '/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/MG_training_files_cbis-ddsm_roi_groundtruth.csv'
    
    elif config_params.dataset == 'vindr':
        mask_path = '/groups/dso/spathak/vindr/physionet.org/files/vindr-mammo/1.0.0/finding_annotations.csv'
        image_size_path = '/groups/dso/spathak/vindr/vindr_singleinstance_imgpreprocessing_size.csv'
        roi_file_path = ''

    return mask_path, image_size_path, roi_file_path

def read_roi_file(args):
    csvfile = open(os.path.join('/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/','cbis_roi_details_bounding_box.csv'), "w")
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerow(["ImageFolderName", "Patient_Id", "BreastDensity", "BreastSide", "ImageView", "AbnormalityID",	"AbnormalityType", "MassShape", "MassMargins", "Assessment", "Subtlety", "CalcType", "CalcDistribution", "FullPath", "Views", "Groundtruth", "FolderName", "roi_w_min", "roi_h_min", "roi_w_max", "roi_h_max"])

    #Map this exam (image or case) to a name that is a substring of the ROI folder name
    mask_path, image_size_path, roi_file_path = mask_paths(args)
    
    df_img_size = pd.read_csv(image_size_path, sep=';')
    df_roi = pd.read_csv(roi_file_path, sep=';')
    roi_dic ={}
    for ind_img, row in df_img_size.iterrows():
        print("ind_img:", ind_img, flush=True)
        #read image name
        image_folder_name = "_".join(row['ImageName'].split('_')[:-1])
        print("image folder name:", image_folder_name, flush=True)

        #which ROI folders belong to this image
        #print("roi foldername:", df_roi['FolderName'].str.split('_').str[:-1].str.join('_'), flush=True)
        df_roi_folder_names = df_roi[df_roi['FolderName'].str.split('_').str[:-1].str.join('_')==image_folder_name]
        print("roi folder name:", df_roi_folder_names, flush=True)
        
        #Read the bounding box coordinates generated after passing the original image through the cleaning algorithm
        df_img_size1 = df_img_size[df_img_size['ImageName'].str.split('_').str[:5].str.join('_')==image_folder_name]
        #print("df_img_size:", df_img_size1, flush=True)
        
        #store all roi details for this image
        #all_roi_details_per_image = []
        for ind_roi, roi_row in df_roi_folder_names.iterrows():
            #correcting mask size: original to preprocessed (cleaning algo) to resize (2944x1920)
            roi_folder = roi_row['FolderName']
            true_mask_image = cv2.imread(mask_path+'/'+roi_folder+'/'+'1-2.png') 
            true_mask_image = true_mask_image[df_img_size1['pro_min_y'].item():df_img_size1['pro_max_y'].item(), df_img_size1['pro_min_x'].item():df_img_size1['pro_max_x'].item()]
            true_mask_image = cv2.resize(true_mask_image, dsize=(args.image_size[1], args.image_size[0]))
            x,y,w,h = cv2.boundingRect(true_mask_image[:,:,0])
            true_mask_loc = [x,y,x+w,y+h]
            roi_details = [image_folder_name] + list(roi_row.values) + true_mask_loc
            writer.writerow(roi_details)
            #print("roi details:", roi_details, flush=True)
            #all_roi_details_per_image.append(roi_details)
        #roi_dic[image_folder_name] = all_roi_details_per_image
        #print("roi dic image folder name:", roi_dic[image_folder_name], flush=True)
    csvfile.close()

def map_region_to_fixed_patch_size(bounding_box, patch_size):
    h_min_224 = bounding_box[0]
    h_max_224 = bounding_box[1]
    w_min_224 = bounding_box[2]
    w_max_224 = bounding_box[3]
    h_min_224, h_max_224, w_min_224, w_max_224 = float(h_min_224), float(h_max_224), float(w_min_224), float(w_max_224)
    diffh = h_max_224 - h_min_224
    diffw = w_max_224 - w_min_224
    if diffh > patch_size[0]: #patch size too big, we take the center. otherwise the bigger the patch, the higher the purity. #32
        correction = diffh-patch_size[0]
        h_min_224 = h_min_224 + correction//2.
        h_max_224 = h_max_224 - correction//2.
        if h_max_224 - h_min_224 == patch_size[0]+1:
            h_max_224 -= 1
    elif diffh < patch_size[0]:
        if h_min_224 - (patch_size[0]-diffh) <0:
            h_max_224 = h_max_224 + (patch_size[0]-diffh)
        else:
            h_min_224 -= (patch_size[0]-diffh)
        
    if diffw > patch_size[1]:
        correction = diffw-patch_size[1]
        w_min_224 = w_min_224 + correction//2.
        w_max_224 = w_max_224 - correction//2.
        if w_max_224 - w_min_224 == patch_size[1]+1:
            w_max_224 -= 1
    elif diffw < patch_size[1]:
        if w_min_224 - (patch_size[1]-diffw) <0:
            w_max_224 = w_max_224 + (patch_size[1]-diffw)
        else:
            w_min_224 -= (patch_size[1]-diffw)
    # print(imgname, "corrected sizes: ", h_min_224, h_max_224, w_min_224, w_max_224)
    h_min_224, h_max_224, w_min_224, w_max_224 = int(h_min_224), int(h_max_224), int(w_min_224), int(w_max_224)
    return h_min_224, h_max_224, w_min_224, w_max_224

def class_specific_score(args):
    prototype_purity_file = '/home/pathaks/PhD/prototype-model-evaluation/prototype_eval_framwork/protopnet_cbis_8_8_prototypeevalframe_res.csv' #created from the function eval_prototypes_purity() in this code
    dataset_category_file = '/home/pathaks/PhD/prototype-model-evaluation/prototype_eval_framwork/cbisddsm_abnormalitygroup_malignant_benign_count.csv' #created using the code in data-processing/cbis/abnormalitytype_diagnosis.py
    
    purity_file = pd.read_csv(prototype_purity_file, sep = ';')
    dataset_category_file = pd.read_csv(dataset_category_file, sep = ';')
    
    # construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    
    ppnet = torch.nn.DataParallel(ppnet)
    ppnet = load_model(ppnet, args.state_dict_dir_net)
    ppnet = ppnet.to('cuda')#cuda()
    match_count = 0
    for i in range(0, purity_file.shape[0]):
        protoid = purity_file.loc[i, 'ProtoId']
        protoname = purity_file.loc[i, 'AbnormalityType'] + '-' + purity_file.loc[i, 'MassShape/CalcMorph'] + '-' + purity_file.loc[i, 'MassMargin/CalcDist']
        print("protoname:", protoname, flush=True)
        wt_benign = ppnet.module.last_layer.weight[0, protoid].item()
        wt_malign = ppnet.module.last_layer.weight[1, protoid].item()
        dataset_category_row = dataset_category_file[dataset_category_file['Category']==protoname]
        print("dataset row:", dataset_category_row, flush=True)
        print("wt benign:", wt_benign, flush=True)
        print("wt malign:", wt_malign, flush=True)
        if not dataset_category_row.empty:
            print("dataset benign:", dataset_category_row['Benign'].item(), flush=True)
            print("dataset malignant:", dataset_category_row['Malignant'].item(), flush=True)
            if (wt_benign > wt_malign) and (dataset_category_row['Benign'].item() > dataset_category_row['Malignant'].item()):
                match_count+=1
            elif (wt_benign < wt_malign) and (dataset_category_row['Benign'].item() < dataset_category_row['Malignant'].item()):
                match_count+=1
            print("dataset row:", dataset_category_row, flush=True)
            print("model weight row:", i, protoid, protoname, wt_benign, wt_malign, flush=True)
            print("match count:", match_count, flush=True)
    
    print("match count average:", match_count/purity_file.shape[0], flush=True)

def class_specific_score_dataset_cldist_plot(args):
    prototype_purity_file = '/home/pathaks/PhD/prototype-model-evaluation/prototype_eval_framwork/protopnet_cbis_8_8_prototypeevalframe_res.csv' #created from the function eval_prototypes_purity() in this code 
    dataset_category_file = '/home/pathaks/PhD/prototype-model-evaluation/prototype_eval_framwork/cbisddsm_abnormalitygroup_malignant_benign_count.csv' #created using the code in data-processing/cbis/abnormalitytype_diagnosis.py

    purity_file = pd.read_csv(prototype_purity_file, sep = ';')
    dataset_category_file = pd.read_csv(dataset_category_file, sep = ';')

    # construct the model
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    
    ppnet = torch.nn.DataParallel(ppnet)
    ppnet = load_model(ppnet, args.state_dict_dir_net)
    ppnet = ppnet.to('cuda')#cuda()
    match_count = 0
    category_dic = {}
    for i in range(0, purity_file.shape[0]):
        protoid = purity_file.loc[i, 'ProtoId']
        protoname = purity_file.loc[i, 'AbnormalityType'] + '-' + purity_file.loc[i, 'MassShape/CalcMorph'] + '-' + purity_file.loc[i, 'MassMargin/CalcDist']
        print("protoname:", protoname, flush=True)
        wt_benign = ppnet.module.last_layer.weight[0, protoid].item()
        wt_malign = ppnet.module.last_layer.weight[1, protoid].item()
        dataset_category_row = dataset_category_file[dataset_category_file['Category']==protoname]
        print("dataset row:", dataset_category_row, flush=True)
        print("wt benign:", wt_benign, flush=True)
        print("wt malign:", wt_malign, flush=True)
        if not dataset_category_row.empty:
            print("dataset benign:", dataset_category_row['Benign'].item(), flush=True)
            print("dataset malignant:", dataset_category_row['Malignant'].item(), flush=True)
            
            if protoname not in category_dic.keys():
                category_dic[protoname] = [dataset_category_row['Benign'].item()/(dataset_category_row['Benign'].item()+dataset_category_row['Malignant'].item()), dataset_category_row['Malignant'].item()/(dataset_category_row['Benign'].item()+dataset_category_row['Malignant'].item()), 0, 0]
                #category_dic[protoname] = [dataset_category_row['Benign'].item(), dataset_category_row['Malignant'].item(), 0, 0]
                #category_dic[protoname] = [dataset_category_row['Malignant'].item()/dataset_category_row['Benign'].item(), 0, 0]

            if (wt_benign > wt_malign):
                category_dic[protoname][2]+=1
                #wt_div = wt_malign/wt_benign
                #if wt_div<0:
                #    wt_div_assign = wt_div
                #elif wt_div>0:
                #    wt_div_assign = -wt_div
            
            elif (wt_benign < wt_malign):
                category_dic[protoname][3]+=1
                #wt_div = wt_malign/wt_benign
                #if wt_div<0:
                #    wt_div_assign = -wt_div
                #elif wt_div>0:
                #    wt_div_assign = wt_div
            
            #category_dic[protoname].append(wt_div_assign)
            
            print("dataset row:", dataset_category_row, flush=True)
            print("model weight row:", i, protoid, protoname, wt_benign, wt_malign, flush=True)
            #print("match count:", match_count, flush=True)
    
    print("category dic:", category_dic)
    
    barWidth = 0.18
    fig, ax = plt.subplots(figsize =(20, 15)) 

    #ax.set_yscale('log')
    # set height of bar 
    y_malignant = []
    y_benign = []
    true_benign = []
    true_malignant = []
    for key in category_dic.keys():
        true_benign.append(category_dic[key][0])
        true_malignant.append(category_dic[key][1])
        y_benign.append(category_dic[key][2]/(category_dic[key][2]+category_dic[key][3]))
        y_malignant.append(category_dic[key][3]/(category_dic[key][2]+category_dic[key][3]))

    # Set position of bar on X axis 
    br1 = np.arange(len(y_malignant)) 
    br2 = np.array([x + barWidth for x in br1]) 
    br3 = np.array([x + barWidth for x in br2]) 
    br4 = np.array([x + barWidth for x in br3])

    ax.bar(br1, true_benign, color ='cornflowerblue', width = barWidth, label ='benign instances (CBIS)')
    ax.bar(br2, true_malignant, color ='crimson', width = barWidth, label ='malignant instances (CBIS)')#, bottom=true_benign) #edgecolor ='grey',
    ax.bar(br3, y_benign, color ='lightskyblue', width = barWidth, label ='#wt-b>wt-m prototypes') 
    ax.bar(br4, y_malignant, color ='lightsalmon', width = barWidth, label ='#wt-b<wt-m prototypes') 

    # Adding Xticks 
    plt.xlabel('Categories', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Count', fontweight ='bold', fontsize = 15) 
    print(len(category_dic.keys()))
    print(len(br1))
    print(br1+barWidth)
    print(list(category_dic.keys()))
    #xlabels = ['ARCHDIS-SPICULATED', 'IRREGULAR-ILLDEFINED', 'IRREGULAR-SPICULATED', 'IRREGULAR-ARCHDIS-SPICULATED', 'LOBULATED-CIRCUMSCRIBED', 'LOBULATED-ILLDEFINED', 'LOBULATED-MICROLOBULATED', 'LOBULATED-OBSCURED', 'OVAL-CIRCUMSCRIBED', 'OVAL-ILLDEFINED', 'OVAL-MICROLOBULATED', 'OVAL-OBSCURED', 'OVAL-SPICULATED', 'ROUND-CIRCUMSCRIBED', 'ROUND-ILLDEFINED', 'ROUND-OBSCURED', 'ROUND-SPICULATED']
    #xlabels = ['AMORPHOUS-CLUSTERED', 'AMORPHOUS-SEGMENTAL', 'FINELINEARBRANCH-CLUSTERED', 'FINELINEARBRANCH-LINEAR', 'PLEOMORPHIC-CLUSTERED', 'PLEOMORPHIC-LINEAR', 'PLEOMORPHIC-REGIONAL', 'PLEOMORPHIC-SEGMENTAL', 'PUNCTATE-CLUSTERED', 'PUNCTATE-SEGMENTAL', 'PUNCTATE-PLEOMORPHIC-CLUSTERED']
    xlabels = list(category_dic.keys())
    ax.set_xticks(br1+(barWidth), xlabels, rotation = 35, ha='right', rotation_mode='anchor')
    
    #plt.ylim(0,2)
    plt.legend()
    #plt.show() 

    fig.savefig("./category_truedata_benign_malign_wt"+'.pdf', format='pdf', bbox_inches='tight')

    #print("match count average:", match_count/purity_file.shape[0], flush=True)

# Evaluates purity of CUB prototypes from csv file. General method that can be used for other part-prototype methods as well
# Assumes that coordinates in csv file apply to a 224x224 image!
def eval_prototypes_purity(args):
    df_roi = pd.read_csv(os.path.join('/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/','cbis_roi_details_bounding_box.csv'), sep=';')
    print("df_roi:", df_roi, flush=True)
    proto_parts_presences_roi = dict()
    proto_parts_presences_abnormalitytype = dict()
    proto_parts_presences_massshape = dict()
    proto_parts_presences_massmargin = dict()
    proto_parts_presences_calcmorphology = dict()
    proto_parts_presences_calcdistribution = dict()
    count = dict()

    with open(args.patch_proto_csv, newline='') as f:
        filereader = csv.reader(f, delimiter=',')
        next(filereader) #skip header
        for (prototype, imgname, h_min_224, h_max_224, w_min_224, w_max_224) in filereader:
            if prototype not in count.keys():
                count[prototype] = 1
            else:
                count[prototype]+= 1 
            
            if count[prototype]>10:
                print("count prototype:", count[prototype], flush=True)
                continue
            
            h_min_224, h_max_224, w_min_224, w_max_224 = float(h_min_224), float(h_max_224), float(w_min_224), float(w_max_224)
            bounding_box = [h_min_224, h_max_224, w_min_224, w_max_224]
            h_min_224, h_max_224, w_min_224, w_max_224 = map_region_to_fixed_patch_size(bounding_box, args.patch_size)

            roi_ids_img = df_roi[df_roi['ImageFolderName']=="_".join(imgname.split('/')[-1].split('_')[:-1])]
            print("prototype details:", prototype, imgname, h_min_224, h_max_224, w_min_224, w_max_224, flush=True)
            #print("img details:", roi_ids_img, flush=True)
            if roi_ids_img.shape[0] > 1: 
                print("img details:", roi_ids_img, flush=True)
            
            if prototype not in proto_parts_presences_abnormalitytype.keys():
                proto_parts_presences_abnormalitytype[prototype]=dict()
            if prototype not in proto_parts_presences_massshape.keys():
                proto_parts_presences_massshape[prototype]=dict()
            if prototype not in proto_parts_presences_massmargin.keys():
                proto_parts_presences_massmargin[prototype]=dict()
            if prototype not in proto_parts_presences_calcmorphology.keys():
                proto_parts_presences_calcmorphology[prototype]=dict()
            if prototype not in proto_parts_presences_calcdistribution.keys():
                proto_parts_presences_calcdistribution[prototype]=dict()
            
            p = prototype
                       
            #part_dict_img = roi_dic[img_id]
            for idx_roi, roi in roi_ids_img.iterrows():
                y_center = int((roi['roi_h_min'] + roi['roi_h_max'])/2)   
                x_center = int((roi['roi_w_min'] + roi['roi_w_max'])/2)        
                part_in_patch = 0 
                if y_center >= h_min_224 and y_center <=h_max_224:
                    if x_center >= w_min_224 and x_center <= w_max_224:
                        part_in_patch = 1
                        break
            
            if type(roi['AbnormalityType']) != float:
                if 'ROI' not in proto_parts_presences_roi[p].keys():
                    proto_parts_presences_roi[p]['ROI']=[]
                proto_parts_presences_roi[p]['ROI'].append(part_in_patch)

            if type(roi['AbnormalityType']) != float:
                if roi['AbnormalityType'] not in proto_parts_presences_abnormalitytype[p].keys():
                    proto_parts_presences_abnormalitytype[p][roi['AbnormalityType']]=[]
                proto_parts_presences_abnormalitytype[p][roi['AbnormalityType']].append(part_in_patch)

            if type(roi['MassShape']) != float:
                if roi['MassShape'] not in proto_parts_presences_massshape[p].keys():
                    proto_parts_presences_massshape[p][roi['MassShape']]=[]
                proto_parts_presences_massshape[p][roi['MassShape']].append(part_in_patch)

            if type(roi['MassMargins']) != float:
                if roi['MassMargins'] not in proto_parts_presences_massmargin[p].keys():
                    proto_parts_presences_massmargin[p][roi['MassMargins']]=[]
                proto_parts_presences_massmargin[p][roi['MassMargins']].append(part_in_patch)

            if type(roi['CalcType']) != float:
                if roi['CalcType'] not in proto_parts_presences_calcmorphology[p].keys():
                    proto_parts_presences_calcmorphology[p][roi['CalcType']]=[]
                proto_parts_presences_calcmorphology[p][roi['CalcType']].append(part_in_patch)

            if type(roi['CalcDistribution']) != float:
                if roi['CalcDistribution'] not in proto_parts_presences_calcdistribution[p].keys():
                    proto_parts_presences_calcdistribution[p][roi['CalcDistribution']]=[]
                proto_parts_presences_calcdistribution[p][roi['CalcDistribution']].append(part_in_patch)
            

    print("Number of prototypes in parts_presences abnormality type: ", len(proto_parts_presences_roi.keys()), flush=True)
    print("Proto parts presence abnormality type:", proto_parts_presences_roi, flush=True)

    print("Number of prototypes in parts_presences abnormality type: ", len(proto_parts_presences_abnormalitytype.keys()), flush=True)
    print("Proto parts presence abnormality type:", proto_parts_presences_abnormalitytype, flush=True)
    
    print("Number of prototypes in parts_presences abnormality type: ", len(proto_parts_presences_massshape.keys()), flush=True)
    print("Proto parts presence abnormality type:", proto_parts_presences_massshape, flush=True)

    print("Number of prototypes in parts_presences abnormality type: ", len(proto_parts_presences_massmargin.keys()), flush=True)
    print("Proto parts presence abnormality type:", proto_parts_presences_massmargin, flush=True)

    print("Number of prototypes in parts_presences abnormality type: ", len(proto_parts_presences_calcmorphology.keys()), flush=True)
    print("Proto parts presence abnormality type:", proto_parts_presences_calcmorphology, flush=True)

    print("Number of prototypes in parts_presences abnormality type: ", len(proto_parts_presences_calcdistribution.keys()), flush=True)
    print("Proto parts presence abnormality type:", proto_parts_presences_calcdistribution, flush=True)
    
    
    max_presence_purity_roi = dict()
    part_most_present_roi = dict()
    for proto in proto_parts_presences_roi.keys():
        max_presence_purity_roi[proto]= 0.
        part_most_present_roi[proto] = ('0',0)
        total_entry_roi = 0.
        part_score_roi = {}
        for part in proto_parts_presences_roi[proto].keys():
            #presence_purity = np.mean(proto_parts_presences_abnormalitytype[proto][part])
            part_score_roi[part] = np.array(proto_parts_presences_roi[proto][part]).sum()
            total_entry_roi+= np.array(proto_parts_presences_roi[proto][part]).shape[0]
        for part in proto_parts_presences_roi[proto].keys():
            part_score_roi[part] = part_score_roi[part]/total_entry_roi
            if part_score_roi[part] > max_presence_purity_roi[proto]:
                max_presence_purity_roi[proto] = part_score_roi[part]
                part_most_present_roi[proto] = (part, part_score_roi[part])

        #print("part score:", proto, part_score, flush= True)
    print("purity all proto abnormality type:", part_most_present_roi, flush= True)
    
    max_presence_purity = dict()
    part_most_present = dict()
    for proto in proto_parts_presences_abnormalitytype.keys():
        max_presence_purity[proto]= 0.
        part_most_present[proto] = ('0',0)
        total_entry = 0.
        part_score = {}
        for part in proto_parts_presences_abnormalitytype[proto].keys():
            #presence_purity = np.mean(proto_parts_presences_abnormalitytype[proto][part])
            part_score[part] = np.array(proto_parts_presences_abnormalitytype[proto][part]).sum()
            total_entry+= np.array(proto_parts_presences_abnormalitytype[proto][part]).shape[0]
        for part in proto_parts_presences_abnormalitytype[proto].keys():
            part_score[part] = part_score[part]/total_entry
            if part_score[part] > max_presence_purity[proto]:
                max_presence_purity[proto] = part_score[part]
                part_most_present[proto] = (part, part_score[part])

        #print("part score:", proto, part_score, flush= True)
    print("purity all proto abnormality type:", part_most_present, flush= True)

    max_presence_purity_massshape = dict()
    part_most_present_massshape = dict()
    for proto in proto_parts_presences_massshape.keys():
        max_presence_purity_massshape[proto]= 0.
        part_most_present_massshape[proto] = ('0',0)
        total_entry_massshape = 0.
        part_score_massshape = {}
        for part in proto_parts_presences_massshape[proto].keys():
            part_score_massshape[part] = np.array(proto_parts_presences_massshape[proto][part]).sum()
            total_entry_massshape+= np.array(proto_parts_presences_massshape[proto][part]).shape[0]

        for part in proto_parts_presences_massshape[proto].keys():
            part_score_massshape[part] = part_score_massshape[part]/total_entry_massshape
            if part_score_massshape[part] > max_presence_purity_massshape[proto]:
                max_presence_purity_massshape[proto] = part_score_massshape[part]
                part_most_present_massshape[proto] = (part, part_score_massshape[part])
        #print("part score:", proto, part_score_massshape, flush= True)
    print("purity all proto mass shape:", part_most_present_massshape, flush= True)

    max_presence_purity_massmargin= dict()
    part_most_present_massmargin = dict()
    for proto in proto_parts_presences_massmargin.keys():
        max_presence_purity_massmargin[proto]= 0.
        part_most_present_massmargin[proto] = ('0',0)
        total_entry_massmargin = 0.
        part_score_massmargin = {}
        for part in proto_parts_presences_massmargin[proto].keys():
            part_score_massmargin[part] = np.array(proto_parts_presences_massmargin[proto][part]).sum()
            total_entry_massmargin+= np.array(proto_parts_presences_massmargin[proto][part]).shape[0]

        for part in proto_parts_presences_massmargin[proto].keys():
            part_score_massmargin[part] = part_score_massmargin[part]/total_entry_massmargin
            if part_score_massmargin[part] > max_presence_purity_massmargin[proto]:
                max_presence_purity_massmargin[proto] = part_score_massmargin[part]
                part_most_present_massmargin[proto] = (part, part_score_massmargin[part])
        #print("part score:", proto, part_score_massmargin, flush= True)
    print("purity all proto mass margin:", part_most_present_massmargin, flush= True)

    max_presence_purity_calcmorphology = dict()
    part_most_present_calcmorphology = dict()
    for proto in proto_parts_presences_calcmorphology.keys():
        max_presence_purity_calcmorphology[proto]= 0.
        part_most_present_calcmorphology[proto] = ('0',0)
        total_entry_calcmorphology = 0.
        part_score_calcmorphology = {}
        for part in proto_parts_presences_calcmorphology[proto].keys():
            part_score_calcmorphology[part] = np.array(proto_parts_presences_calcmorphology[proto][part]).sum()
            total_entry_calcmorphology+= np.array(proto_parts_presences_calcmorphology[proto][part]).shape[0]

        for part in proto_parts_presences_calcmorphology[proto].keys():
            part_score_calcmorphology[part] = part_score_calcmorphology[part]/total_entry_calcmorphology
            if part_score_calcmorphology[part] > max_presence_purity_calcmorphology[proto]:
                max_presence_purity_calcmorphology[proto] = part_score_calcmorphology[part]
                part_most_present_calcmorphology[proto] = (part, part_score_calcmorphology[part])
        #print("part score:", proto, part_score_calcmorphology, flush= True)
    print("purity all proto calc morphology:", part_most_present_calcmorphology, flush= True)

    max_presence_purity_calcdistribution = dict()
    part_most_present_calcdistribution = dict()
    for proto in proto_parts_presences_calcdistribution.keys():
        max_presence_purity_calcdistribution[proto]= 0.
        part_most_present_calcdistribution[proto] = ('0',0)
        total_entry_calcdistribution = 0.
        part_score_calcdistribution = {}
        for part in proto_parts_presences_calcdistribution[proto].keys():
            part_score_calcdistribution[part] = np.array(proto_parts_presences_calcdistribution[proto][part]).sum()
            total_entry_calcdistribution+= np.array(proto_parts_presences_calcdistribution[proto][part]).shape[0]

        for part in proto_parts_presences_calcdistribution[proto].keys():
            part_score_calcdistribution[part] = part_score_calcdistribution[part]/total_entry_calcdistribution
            if part_score_calcdistribution[part] > max_presence_purity_calcdistribution[proto]:
                max_presence_purity_calcdistribution[proto] = part_score_calcdistribution[part]
                part_most_present_calcdistribution[proto] = (part, part_score_calcdistribution[part])
        #print("part score:", proto, part_score_calcdistribution, flush= True)
    #(name of the prototype, purity score)
    print("purity all proto calc distribution:", part_most_present_calcdistribution, flush= True)
    
    part_proto_present_all_granularity = dict()
    massshape = 0
    massmargin = 0
    calcmorph = 0
    calcdist = 0
    abnormtype = 0
    totalmass = 0
    totalcalc = 0
    description = []
    
    roitype = 0
    for proto in proto_parts_presences_roi.keys():
        roitype+= part_most_present_roi[proto][1]
    print("roi value:", roitype/len(proto_parts_presences_roi))

    '''for proto in proto_parts_presences_abnormalitytype.keys():
        if part_most_present[proto][0] == 'mass':
            part_proto_present_all_granularity[proto] = [part_most_present[proto][0], part_most_present[proto][1], part_most_present_massshape[proto][0], part_most_present_massshape[proto][1], part_most_present_massmargin[proto][0], part_most_present_massmargin[proto][1]]
            massshape+=part_most_present_massshape[proto][1]
            massmargin+=part_most_present_massmargin[proto][1]
            description.append(part_most_present[proto][0] + '-' + part_most_present_massshape[proto][0] + '-' + part_most_present_massmargin[proto][0])
            totalmass+=1

        elif part_most_present[proto][0] == 'calcification':
            part_proto_present_all_granularity[proto] = [part_most_present[proto][0], part_most_present[proto][1], part_most_present_calcmorphology[proto][0], part_most_present_calcmorphology[proto][1], part_most_present_calcdistribution[proto][0], part_most_present_calcdistribution[proto][1]]
            calcmorph+=part_most_present_calcmorphology[proto][1]
            calcdist+=part_most_present_calcdistribution[proto][1]
            description.append(part_most_present[proto][0] + '-' + part_most_present_calcmorphology[proto][0] + '-' + part_most_present_calcdistribution[proto][0]) 
            totalcalc+=1

        abnormtype+=part_most_present[proto][1] 
    '''
    '''df_part_proto_present_all_granularity= pd.DataFrame.from_dict(part_proto_present_all_granularity, orient='index', columns=['AbnormalityType', 'PurityScore', 'MassShape/CalcMorph', 'PurityScore', 'MassMargin/CalcDist', 'PurityScore'])
    df_part_proto_present_all_granularity.to_csv('./protopnet_cbis_8_8_prototypeevalframe_res.csv', sep=';',na_rep='NULL',index=True)
    print("part proto all granularity:", part_proto_present_all_granularity, flush=True)
    print("relevance:", len(part_proto_present_all_granularity.keys()))
    print("all values for purity:", abnormtype/(totalmass+totalcalc), massshape/totalmass, massmargin/totalmass, calcmorph/totalcalc, calcdist/totalcalc, flush=True)
    print("uniqueness:", np.unique(np.array(description)).shape)
    '''
    #print("Number of part-related prototypes (purity>0.5): ", prototypes_part_related, flush=True)

    #print("Mean purity of prototypes (corresponding to purest part): ", np.mean(list(max_presence_purity.values())), "std: ", np.std(list(max_presence_purity.values())), flush=True)
    #print("Prototypes with highest-purity part (no contraints): ", max_presence_purity_part, flush=True)
    #print("Prototype with part that has most often overlap with prototype: ", part_most_present, flush=True)

    #log.log_values('log_epoch_overview', "p_cub_"+str(args.epoch), "mean purity (averaged over all prototypes, corresponding to purest part)", "std purity", "mean purity (averaged over all prototypes, corresponding to part with most often overlap)", "std purity", "# prototypes in csv", "#part-related prototypes (purity > 0.5)","","")
    
    #log.log_values('log_epoch_overview', "p_cub_"+str(args.epoch), np.mean(list(max_presence_purity.values())), np.std(list(max_presence_purity.values())), np.mean(list(most_often_present_purity.values())), np.std(list(most_often_present_purity.values())), len(list(proto_parts_presences.keys())), prototypes_part_related, "", "")
    

if __name__ == '__main__':
    #read arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpuid', type=str, default='0') # python3 main.py -gpuid=0,1,2,3
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--patch_size', nargs='+', type=int, default=32)
    parser.add_argument('--state_dict_dir_net', type=str, default='')
    parser.add_argument('--patch_proto_csv', type=str, default='')
    parser.add_argument('--image_size', nargs='+', type=int, default=224)

    args = parser.parse_args()

    #create the cbis_roi_details_bounding_box.csv
    if not os.path.exists('/deepstore/datasets/dmb/medical/breastcancer/mammography/cbis-ddsm/cbis_roi_details_bounding_box.csv'):
        read_roi_file(args)

    #calculate metrics relevance, specialization, uniqueness, coverage
    eval_prototypes_purity(args)

    #calculate metrics class-specific score
    class_specific_score(args)
    
    #plot the classification layer weight vs the class distribution in the dataset for the abnormality category
    class_specific_score_dataset_cldist_plot(args)