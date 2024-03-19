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

from util.data_augmentation import MyHorizontalFlip, cbisddsm_transforms
from util.utils import stratified_class_count, stratifiedgroupsplit, stratifiedgroupsplit_train_val

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def input_file_creation(config_params):
    csv_file_path = config_params.SIL_csvfilepath
    df_modality = pd.read_csv(csv_file_path, sep=';')
    print("df modality shape:",df_modality.shape)
    df_modality = df_modality[~df_modality['Views'].isnull()]
    print("df modality no null view:",df_modality.shape)
    df_modality['FullPath'] = config_params.preprocessed_imagepath+'/'+df_modality['ShortPath']
    
    if config_params.classtype == 'diagnosis':
        df_modality['Groundtruth'] = df_modality['ImageLabel']
    elif config_params.classtype == 'birads':
        df_modality['Groundtruth'] = df_modality['BIRADS']
    
    total_instances = df_modality.shape[0]

    if config_params.datasplit == 'officialtestset':
        if config_params.dataset == 'cbis-ddsm':
            df_train = df_modality[df_modality['ImageName'].str.contains('Training')]
            if config_params.usevalidation:
                patient_col = 'Patient_Id'
                df_train, df_val = stratifiedgroupsplit_train_val(df_train, config_params.randseeddata, patient_col)
            df_test = df_modality[df_modality['ImageName'].str.contains('Test')]
        
        elif config_params.dataset == 'vindr':
            df_train = df_modality[df_modality['Split'] == 'training']
            if config_params.usevalidation:
                patient_col = 'StudyInstanceUID'
                df_train, df_val = stratifiedgroupsplit_train_val(df_train, config_params.randseeddata, patient_col)
                #df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
            df_test = df_modality[df_modality['Split'] == 'test']
    
    elif config_params.datasplit == 'customsplit':
        if config_params.dataset == 'cmmd':
            df_modality['Groundtruth'] = df_modality['Groundtruth'].str.lower()
            df_train_index = df_modality[df_modality['Patient_Id']=='D2-0247'].index[-1]
            df_train = df_modality[:df_train_index]
            if config_params.usevalidation:
                patient_col = 'Patient_Id'
                df_train, df_val = stratifiedgroupsplit_train_val(df_train, config_params.randseeddata, patient_col)
                #df_train, df_val = train_test_split(df_train, test_size=0.10, shuffle=True, stratify=df_train['Groundtruth'])
            df_test =  df_modality[(df_train_index+1):]
            '''df_modality['Groundtruth'] = df_modality['Groundtruth'].str.lower()
            df_train, df_val, df_test = stratifiedgroupsplit(df_modality, config_params.randseeddata)
            print("Check starting between perfect transfer of patients from case based to single instance based")
            train_check = df_train['Patient_Id'].unique().tolist()
            val_check = df_val['Patient_Id'].unique().tolist()
            test_check = df_test['Patient_Id'].unique().tolist()
            train_check.sort()
            val_check.sort()
            test_check.sort()
            print(len(train_check))
            print(len(val_check))
            print(len(test_check))'''

    print("Total instances:", total_instances)
    
    #df_train = df_train[100:300]
    #df_val = df_val[100:300]
    #df_test = df_test[100:300]

    #reset index     
    df_train = df_train.reset_index()
    train_instances = df_train.shape[0]
    print("Train:", stratified_class_count(df_train))
    print("training instances:", train_instances)
    if config_params.usevalidation:
        df_val = df_val.reset_index()
        val_instances = df_val.shape[0]
        print("Val:", stratified_class_count(df_val))
        print("Validation instances:", val_instances)
    df_test = df_test.reset_index()
    test_instances = df_test.shape[0]
    print("Test:", stratified_class_count(df_test)) 
    print("Test instances:", test_instances) 

    numbatches_train = int(math.ceil(train_instances/config_params.batch_size))
    if config_params.usevalidation:
        numbatches_val = int(math.ceil(val_instances/config_params.batch_size))
    numbatches_test = int(math.ceil(test_instances/config_params.batch_size))
    
    if config_params.usevalidation:
        return df_train, df_val, df_test, numbatches_train, numbatches_val, numbatches_test
    else:
        return df_train, df_test, numbatches_train, numbatches_test

class BreastCancerDatasetTwoAugView_generator(Dataset):
    def __init__(self, config_params, df, transform1=None, transform2=None):
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
        self.config_params = config_params

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img = collect_images(self.config_params, data)

        if self.transform1 and self.transform2:
            img = self.transform1(img)
            if self.transform2:
                img1 = self.transform2(img)
                img2 = self.transform2(img)
                img1 = img1.unsqueeze(0)
                img2 = img2.unsqueeze(0)
        return idx, img1, img2, torch.tensor(self.config_params.groundtruthdic[data['Groundtruth']]), data['Views']

class BreastCancerDatasetSingleView_generator(Dataset):
    def __init__(self, config_params, df, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.transform = transform
        self.config_params = config_params

    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        img = collect_images(self.config_params, data)

        if self.transform:
            img = self.transform(img)
        img = img.unsqueeze(0)
        
        return idx, img, torch.tensor(self.config_params.groundtruthdic[data['Groundtruth']]), data['Views']

def collect_images(config_params, data):
    views_allowed = views_allowed_dataset(config_params)
    img = collect_images_8bits(config_params, data, views_allowed)
    return img

def collect_images_8bits(config_params, data, views_allowed):
    #collect images for the model
    if data['Views'] in views_allowed:
        img_path = str(data['FullPath'])
        img = Image.open(img_path)
        img= img.convert('RGB')
        breast_side = data['Views'][0]
        if config_params.flipimage:
            hflip_img = MyHorizontalFlip()
            img = hflip_img(img, breast_side)
        return img
    else:
        print('error in view')
        sys.exit()

def views_allowed_dataset(config_params):
    if config_params.dataset == 'zgt' and config_params.viewsinclusion == 'all':
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

def dataloader(config_params, df_train, df_val, df_test, g):
    transform1_train, transform2_train, transform_val = cbisddsm_transforms(config_params.image_size)

    if config_params.weighted_loss:
        train_indices = list(range(df_train.shape[0]))
        train_targets = torch.LongTensor(df_train['Groundtruth'].map(config_params.groundtruthdic).values.tolist())
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

    
    '''trainvalset = torchvision.datasets.ImageFolder('/projects/dso_mammovit/project_kushal/data/pipnet/train')
    trainset = TwoAugSupervisedDataset(trainvalset, transform1=transform1_train, transform2=transform2_train)
    trainloader_pretraining = torch.utils.data.DataLoader(trainset,
                                            batch_size=config_params.batch_size_pretrain,
                                            shuffle=to_shuffle,
                                            sampler=sampler,
                                            num_workers=config_params.num_workers,
                                            worker_init_fn=np.random.seed(config_params.randseedother),
                                            drop_last = True
                                            )
    '''

    #train set data generator and dataloaders
    dataset_gen_train = BreastCancerDatasetTwoAugView_generator(config_params, df_train, transform1_train, transform2_train)
    #for pretraining
    dataloader_train_pretraining = DataLoader(dataset_gen_train, batch_size=config_params.batch_size_pretrain, shuffle=to_shuffle, num_workers=config_params.num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker, generator=g, sampler=sampler)
    #for full training
    dataloader_train = DataLoader(dataset_gen_train, batch_size=config_params.batch_size, shuffle=to_shuffle, num_workers=config_params.num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker, generator=g, sampler=sampler)    
    #for visualizing the most similar patches from the train set
    dataset_gen_train_project = BreastCancerDatasetSingleView_generator(config_params, df_train, transform_val)
    dataloader_train_projectloader = DataLoader(dataset_gen_train_project, batch_size = 1, shuffle=False, num_workers=config_params.num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker)
    
    #validation set dataset generator and dataloaders for early stopping on validation AUC 
    if config_params.usevalidation:
        dataset_gen_val = BreastCancerDatasetSingleView_generator(config_params, df_val, transform_val)
        dataloader_val = DataLoader(dataset_gen_val, batch_size=config_params.batch_size, shuffle=False, num_workers=config_params.num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker, generator=g)
    
    #test set data generator and dataloaders
    #for evaluating the model on the test set after full training
    dataset_gen_test = BreastCancerDatasetSingleView_generator(config_params, df_test, transform_val)
    dataloader_test = DataLoader(dataset_gen_test, batch_size=config_params.batch_size, shuffle=False, num_workers=config_params.num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker, generator=g)
    #for image-level prototype visualization on the test set after full training
    dataloader_test_projectloader = DataLoader(dataset_gen_test, batch_size=1, shuffle=False, num_workers=config_params.num_workers, collate_fn=MyCollate, worker_init_fn=seed_worker)

    if config_params.usevalidation:
        return dataloader_train_pretraining, dataloader_train, dataloader_train_projectloader, dataloader_val, dataloader_test, dataloader_test_projectloader
    else:
        return dataloader_train_pretraining, dataloader_train, dataloader_train_projectloader, dataloader_test, dataloader_test_projectloader


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2):
        self.dataset = dataset
        self.classes = dataset.classes
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
            # print("imgs: ", self.imgs[200:500], flush=True)
        self.transform1 = transform1
        self.transform2 = transform2
        

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = self.transform1(image)
        return index, self.transform2(image), self.transform2(image), target, 'LCC'

    def __len__(self):
        return len(self.dataset)