import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def stratified_class_count(df):
    class_count=df.groupby(by=['Groundtruth']).size()
    return class_count

def save_model(model, optimizer_net, optimizer_classifier, train_loss, path_to_model, epoch_pretrain, epoch_finetune, model_eval_score=None):
    state = {
        'epoch_pretrain': epoch_pretrain,
        'epoch_finetune': epoch_finetune,
        'model_state_dict': model.state_dict(),
        'optimizer_net_state_dict': optimizer_net.state_dict(),
        'optimizer_classifier_state_dict': optimizer_classifier.state_dict(),
        'train_loss': train_loss,
        'val_score': model_eval_score
    }
    torch.save(state, path_to_model)

def load_model_for_testing(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("checkpoint epoch and loss:", checkpoint['epoch_finetune'], checkpoint['train_loss'])
    return model 

def stratifiedgroupsplit(df, rand_seed):
    groups = df.groupby('Groundtruth')
    all_train = []
    all_test = []
    all_val = []
    train_testsplit = GroupShuffleSplit(test_size=0.15, n_splits=2, random_state=rand_seed)
    train_valsplit = GroupShuffleSplit(test_size=0.10, n_splits=2, random_state=rand_seed)
    for group_id, group in groups:
        # if a group is already taken in test or train it must stay there
        group = group[~group['Patient_Id'].isin(all_train+all_val+all_test)]
        # if group is empty 
        if group.shape[0] == 0:
            continue
        train_inds1, test_inds = next(train_testsplit.split(group, groups=group['Patient_Id']))
        train_inds, val_inds = next(train_valsplit.split(group.iloc[train_inds1], groups=group.iloc[train_inds1]['Patient_Id']))
    
        all_train += group.iloc[train_inds1].iloc[train_inds]['Patient_Id'].tolist()
        all_val += group.iloc[train_inds1].iloc[val_inds]['Patient_Id'].tolist()
        all_test += group.iloc[test_inds]['Patient_Id'].tolist()
        
    train = df[df['Patient_Id'].isin(all_train)]
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
    return train, val, test

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