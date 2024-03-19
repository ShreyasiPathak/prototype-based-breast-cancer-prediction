import sys
import math
import torch
import random
import torchvision
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from data_augmentation import MyHorizontalFlip, data_transform_pipnet, data_transform_gmic, cbisddsm_transforms
from helpers import stratifiedgroupsplit
from settings import img_size, batch_size, dataset, datasplit, usevalidation, weighted_loss, classtype, \
    SIL_csvfilepath, preprocessed_imagepath, flipimage, viewsinclusion, groundtruthdic, num_workers, \
    randseeddata, randseedother, dataaug
from helpers import stratifiedgroupsplit_train_val

def stratified_class_count(df):
    class_count=df.groupby(by=['Groundtruth']).size()
    return class_count

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def input_file_creation(log):
    csv_file_path = SIL_csvfilepath
    df_modality = pd.read_csv(csv_file_path, sep=';')
    log("df modality shape:{0}".format(df_modality.shape))
    df_modality = df_modality[~df_modality['Views'].isnull()]
    log("df modality no null view:{0}".format(df_modality.shape))
    df_modality['FullPath'] = preprocessed_imagepath + '/' + df_modality['ShortPath']
    
    if classtype == 'diagnosis':
        df_modality['Groundtruth'] = df_modality['ImageLabel']
    elif classtype == 'birads':
        df_modality['Groundtruth'] = df_modality['BIRADS']
    
    total_instances = df_modality.shape[0]

    if datasplit == 'officialtestset':
        if dataset == 'cbis-ddsm':
            df_train = df_modality[df_modality['ImageName'].str.contains('Training')]
            if usevalidation:
                patient_col = 'Patient_Id'
                df_train, df_val = stratifiedgroupsplit_train_val(df_train, randseeddata, patient_col)
                #df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
            df_test = df_modality[df_modality['ImageName'].str.contains('Test')]
        
        elif dataset == 'vindr':
            df_train = df_modality[df_modality['Split'] == 'training']
            if usevalidation:
                patient_col = 'StudyInstanceUID'
                df_train, df_val = stratifiedgroupsplit_train_val(df_train, randseeddata, patient_col)
                #df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
            df_test = df_modality[df_modality['Split'] == 'test']  
    
    elif datasplit == 'customsplit':
        if dataset == 'cmmd':
            df_modality['Groundtruth'] = df_modality['Groundtruth'].str.lower()
            #following InterNRL
            df_train_index = df_modality[df_modality['Patient_Id']=='D2-0247'].index[-1]
            df_train = df_modality[:df_train_index]
            if usevalidation:
                patient_col = 'Patient_Id'
                df_train, df_val = stratifiedgroupsplit_train_val(df_train, randseeddata, patient_col)
                #df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
            df_test =  df_modality[(df_train_index+1):]
            '''if usevalidation:
                df_train, df_val, df_test = stratifiedgroupsplit(df_modality, randseeddata)
            else:
                df_train, df_test = stratifiedgroupsplit(df_modality, randseeddata)
            log("Check starting between perfect transfer of patients from case based to single instance based")
            train_check = df_train['Patient_Id'].unique().tolist()
            val_check = df_val['Patient_Id'].unique().tolist()
            test_check = df_test['Patient_Id'].unique().tolist()
            train_check.sort()
            val_check.sort()
            test_check.sort()
            log(str(len(train_check)))
            log(str(len(val_check)))
            log(str(len(test_check)))'''
    
    log("Total instances:{0}".format(total_instances))
    
    #df_train = df_train[100:300]
    #df_val = df_val[100:300]
    #df_test = df_test[100:300]

    #reset index     
    df_train = df_train.reset_index()
    train_instances = df_train.shape[0]
    log("Train: {0}".format(stratified_class_count(df_train)))
    log("training instances: {0}".format(train_instances))
    if usevalidation:
        df_val = df_val.reset_index()
        val_instances = df_val.shape[0]
        log("Val:{0}".format(stratified_class_count(df_val)))
        log("Validation instances:{0}".format(val_instances))
    df_test = df_test.reset_index()
    test_instances = df_test.shape[0]
    log("Test:{0}".format(stratified_class_count(df_test))) 
    log("Test instances:{0}".format(test_instances)) 

    numbatches_train = int(math.ceil(train_instances/batch_size))
    if usevalidation:
        numbatches_val = int(math.ceil(val_instances/batch_size))
    numbatches_test = int(math.ceil(test_instances/batch_size))
    
    if usevalidation:
        return df_train, df_val, df_test, numbatches_train, numbatches_val, numbatches_test
    else:
        return df_train, df_test, numbatches_train, numbatches_test

class BreastCancerDatasetTwoAugView_generator(Dataset):
    def __init__(self, df, transform1=None, transform2=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img = collect_images(data)

        if self.transform1 and self.transform2:
            img = self.transform1(img)
            if self.transform2:
                img1 = self.transform2(img)
                img2 = self.transform2(img)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
        return idx, img1, img2, torch.tensor(groundtruthdic[data['Groundtruth']]), data['Views']

class BreastCancerDatasetSingleView_generator(Dataset):
    def __init__(self, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img = collect_images(data)

        if self.transform:
            img = self.transform(img)
        img = img.unsqueeze(0)
        
        return idx, img, torch.tensor(groundtruthdic[data['Groundtruth']]), data['Views']

def collect_images(data):
    views_allowed = views_allowed_dataset()
    img = collect_images_8bits(data, views_allowed)
    return img

def collect_images_8bits(data, views_allowed):
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        img = Image.open(img_path)
        img= img.convert('RGB')
        breast_side = data['Views'][0]
        if flipimage:
            hflip_img = MyHorizontalFlip()
            img = hflip_img(img, breast_side)
        return img
    else:
        print('error in view')
        sys.exit()

def views_allowed_dataset():
    if dataset == 'zgt' and viewsinclusion == 'all':
        views_allowed=['LCC', 'LLM', 'LML', 'LMLO', 'LXCCL', 'RCC', 'RLM', 'RML', 'RMLO', 'RXCCL']
    else:
        views_allowed = ['LCC','LMLO','RCC','RMLO']
    return views_allowed

def MyCollate(batch):
    i=0
    index=[]
    target=[]
    if len(batch[0]) == 4:
        for item in batch:
            if i==0:
                data = batch[i][1]
                views_names = [item[3]]
            else:
                data=torch.cat((data,batch[i][1]),dim=0)
                views_names.append(item[3])
            index.append(item[0])
            target.append(item[2])
            i+=1
        index = torch.LongTensor(index)
        target = torch.LongTensor(target)
        return [index, data, target, views_names]
    
    elif len(batch[0]) == 5:
        for item in batch:
            if i==0:
                data1 = batch[i][1]
                data2 = batch[i][2]
                views_names = [item[4]]
            else:
                data1=torch.cat((data1,batch[i][1]),dim=0)
                data2=torch.cat((data2,batch[i][2]),dim=0)
                views_names.append(item[4])
            index.append(item[0])
            target.append(item[3])
            i+=1
        index = torch.LongTensor(index)
        target = torch.LongTensor(target)
        return [index, data1, data2, target, views_names]

def dataloader(df_train, df_val, df_test, g):
    if dataaug == 'pipnet':
        transform_train, transform_val, transform_push = data_transform_pipnet(img_size)
    elif dataaug == 'gmic':
        transform_train, transform_val, transform_push = data_transform_gmic(img_size)

    if weighted_loss:
        train_indices = list(range(df_train.shape[0]))
        train_targets = torch.LongTensor(df_train['Groundtruth'].map(groundtruthdic).values.tolist())
        if train_targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor([(train_targets[train_indices] == t).sum() for t in torch.unique(train_targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in train_targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        to_shuffle = False

    #train set data generator and dataloaders
    dataset_gen_train = BreastCancerDatasetSingleView_generator(df_train, transform_train)
    #transform1_train, transform2_train, _ = cbisddsm_transforms(img_size)
    #dataset_gen_train = BreastCancerDatasetTwoAugView_generator(df_train, transform1_train, transform2_train)
    #for full training
    dataloader_train = DataLoader(dataset_gen_train, batch_size=batch_size, shuffle=to_shuffle, num_workers=num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker, generator=g, sampler=sampler)    
    #for visualizing the most similar patches from the train set
    dataset_gen_train_project = BreastCancerDatasetSingleView_generator(df_train, transform_push)
    dataloader_train_projectloader = DataLoader(dataset_gen_train_project, batch_size = batch_size, shuffle=False, num_workers=num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker)
    
    #validation set dataset generator and dataloaders for early stopping on validation AUC 
    if usevalidation:
        dataset_gen_val = BreastCancerDatasetSingleView_generator(df_val, transform_val)
        dataloader_val = DataLoader(dataset_gen_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker, generator=g)
    
    #test set data generator and dataloaders
    #for evaluating the model on the test set after full training
    dataset_gen_test = BreastCancerDatasetSingleView_generator(df_test, transform_val)
    dataloader_test = DataLoader(dataset_gen_test, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker, generator=g)

    if usevalidation:
        return dataloader_train, dataloader_train_projectloader, dataloader_val, dataloader_test
    else:
        return dataloader_train, dataloader_train_projectloader, dataloader_test
    
def dataloader_visualize(df_train, df_test):
    if dataaug == 'pipnet':
        transform_train, transform_val, transform_push = data_transform_pipnet(img_size)
    elif dataaug == 'gmic':
        transform_train, transform_val, transform_push = data_transform_gmic(img_size)

    dataset_gen_train_project = BreastCancerDatasetSingleView_generator(df_train, transform_push)
    dataloader_train_projectloader = DataLoader(dataset_gen_train_project, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker)
    
    dataset_gen_test_project = BreastCancerDatasetSingleView_generator(df_test, transform_push)
    dataloader_test_projectloader = DataLoader(dataset_gen_test_project, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker)

    dataset_gen_test_normalize = BreastCancerDatasetSingleView_generator(df_test, transform_val)
    dataloader_test_normalize = DataLoader(dataset_gen_test_normalize, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker)

    return dataloader_train_projectloader, dataloader_test_projectloader, dataloader_test_normalize