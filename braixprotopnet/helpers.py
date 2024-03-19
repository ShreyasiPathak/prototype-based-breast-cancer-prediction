import os
import torch
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

def list_of_distances(X, Y):
    return torch.sum((torch.unsqueeze(X, dim=2) - torch.unsqueeze(Y.t(), dim=0)) ** 2, dim=1)

def make_one_hot(target, target_one_hot):
    target = target.view(-1,1)
    target_one_hot.zero_()
    target_one_hot.scatter_(dim=1, index=target, value=1.)

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)

def print_and_write(str, file):
    print(str)
    file.write(str + '\n')

def find_high_activation_crop(activation_map, percentile=95):
    threshold = np.percentile(activation_map, percentile)
    #print("activation map:", activation_map)
    #print("max and min val activation map:", activation_map.max(), activation_map.min())
    #print("threshold:", threshold)
    mask = np.ones(activation_map.shape)
    #print("mask before", mask)
    #print("non zero before:", np.count_nonzero(mask))
    mask[activation_map < threshold] = 0
    #print("mask after:", mask)
    #print("non zero after:", np.count_nonzero(mask))
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    for i in range(mask.shape[0]):
        if np.amax(mask[i]) > 0.5:
            lower_y = i
            break
    for i in reversed(range(mask.shape[0])):
        if np.amax(mask[i]) > 0.5:
            upper_y = i
            break
    for j in range(mask.shape[1]):
        if np.amax(mask[:,j]) > 0.5:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:,j]) > 0.5:
            upper_x = j
            break
    return lower_y, upper_y+1, lower_x, upper_x+1
    
def stratifiedgroupsplit(df, rand_seed, usevalidation):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_test = []
    all_val = []
    train_testsplit = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state=rand_seed)
    if usevalidation:
        train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group['Patient_Id'].isin(all_train+all_val+all_test)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds1, test_inds = next(train_testsplit.split(group, groups=group['Patient_Id']))
        if usevalidation:
            train_inds, val_inds = next(train_valsplit.split(group.iloc[train_inds1], groups=group.iloc[train_inds1]['Patient_Id']))
    
        all_train += group.iloc[train_inds1].iloc[train_inds]['Patient_Id'].tolist()
        if usevalidation:
            all_val += group.iloc[train_inds1].iloc[val_inds]['Patient_Id'].tolist()
        all_test += group.iloc[test_inds]['Patient_Id'].tolist()
        
    train = df[df['Patient_Id'].isin(all_train)]
    if usevalidation:
        val = df[df['Patient_Id'].isin(all_val)]
    test = df[df['Patient_Id'].isin(all_test)]
    
    '''form_train = set(train['Patient_Id'].tolist())
    form_val = set(val['Patient_Id'].tolist())
    form_test = set(test['Patient_Id'].tolist())
    inter1 = form_train.intersection(form_test)
    inter2 = form_train.intersection(form_val)
    inter3 = form_val.intersection(form_test)
    print(df.groupby('Groundtruth').size())
    print(train.groupby('Groundtruth').size())
    print(val.groupby('Groundtruth').size())
    print(test.groupby('Groundtruth').size())
    print(inter1) # this should be empty
    print(inter2) # this should be empty
    print(inter3) # this should be empty
    print(train[train['Patient_Id'].isin(test['Patient_Id'].unique().tolist())])
    print(test[test['Patient_Id'].isin(train['Patient_Id'].unique().tolist())])'''
    if usevalidation:
        return train, val, test
    else:
        return train, test

def stratifiedgroupsplit_train_val(df, rand_seed, patient_col):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_val = []
    train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group[patient_col].isin(all_train+all_val)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds, val_inds = next(train_valsplit.split(group, groups=group[patient_col]))
    
        all_train += group.iloc[train_inds][patient_col].tolist()
        all_val += group.iloc[val_inds][patient_col].tolist()
        
    train = df[df[patient_col].isin(all_train)]
    val = df[df[patient_col].isin(all_val)]
    return train, val