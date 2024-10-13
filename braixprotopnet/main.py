import re
import os
import torch
import shutil
import random
import datetime
import argparse
import numpy as np
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

import copy
import save
import push, push_braixx
import prune
import model
from helpers import makedir
import train_and_test as tnt
from log import create_logger, create_results
from preprocess import mean, std, preprocess_input_function
from eval_cbis import get_images

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run, \
                    log_dir, randseeddata, randseedother, coefs, dataset

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import batch_size, usevalidation, num_train_epochs, num_stage1_epochs, num_stage2_epochs                                              

# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size, stage1_optimizer_lrs, stage2_optimizer_lrs, last_layer_optimizer_lr, lrscheduler, lrdecay_gamma

from evaluation import results_store_excel, write_results_xlsx_confmat

from data_loader import input_file_creation, dataloader

def set_random_seed(randseedother, randseeddata):
    #random state initialization of the code - values - 8, 24, 30
    torch.manual_seed(randseedother) 
    torch.cuda.manual_seed(randseedother)
    torch.cuda.manual_seed_all(randseedother)
    np.random.seed(randseeddata)
    random.seed(randseeddata)
    g = torch.Generator()
    g.manual_seed(randseedother)
    torch.backends.cudnn.deterministic = True
    return g

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
                             
def load_model(net, stage1_optimizer, stage2_optimizer, joint_optimizer, path):
    if path!='':
        checkpoint = torch.load(path, map_location='cuda:0')
        net.load_state_dict(checkpoint['model_state_dict'])
        stage1_optimizer.load_state_dict(checkpoint['optimizer_stage1'])
        stage2_optimizer.load_state_dict(checkpoint['optimizer_stage2'])
        joint_optimizer.load_state_dict(checkpoint['optimizer_joint'])
        epoch_stage1 = checkpoint['epoch_stage1'] + 1
        epoch_stage2 = checkpoint['epoch_stage2'] + 1
        epoch_joint = checkpoint['epoch_joint'] + 1
        if epoch_stage1 <= num_stage1_epochs:
            start_epoch = epoch_stage1
        elif num_stage1_epochs < epoch_stage2 <= (num_stage1_epochs+num_stage2_epochs):
            start_epoch = epoch_stage2
        else:
            start_epoch = epoch_joint
    else:
        start_epoch = 1
        epoch_stage1 = 0
        epoch_stage2 = 0
        epoch_joint = 0
    return net, stage1_optimizer, stage2_optimizer, joint_optimizer, start_epoch, epoch_stage1, epoch_stage2, epoch_joint

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', type=str, default='0') # python3 main.py -gpuid=0,1,2,3
parser.add_argument('-disable_cuda', action='store_true', help='Flag that disables GPU usage if set')
parser.add_argument('-start_epoch', type=int, default=0)
parser.add_argument('-state_dict_dir_net', type=str, default='')
parser.add_argument('-mode', type=str, default='train')

args = parser.parse_args()

device, device_ids = set_device(args)

#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
#print(os.environ['CUDA_VISIBLE_DEVICES'])

begin_time = datetime.datetime.now()

#set random seed
g = set_random_seed(randseedother, randseeddata)

#feature extractor
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

#set path for saved models
model_dir = './saved_models/' + dataset + '/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

#create logging path for results
log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log_' + str(randseedother) + '_' + str(randseeddata)))
create_results(log_dir, 'results'+'_'+str(randseedother)+'_'+str(randseeddata))
img_dir = os.path.join(model_dir, 'img'+'_'+str(randseedother)+'_'+str(randseeddata))
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'

# Obtain the dataset and dataloaders
#trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
if usevalidation:
    df_train, df_val, df_test, batches_train, batches_val, batches_test = input_file_creation(log)
    train_loader, train_push_loader, val_loader, test_loader = dataloader(df_train, df_val, df_test, g)
else:
    df_train, df_test, batches_train,batches_test = input_file_creation(log)
    train_loader, train_push_loader, test_loader = dataloader(df_train, None, df_test, g)
    val_loader = None

# all datasets
# train set
#normalize = transforms.Normalize(mean=mean, std=std)
'''train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))'''
'''train_dataset = datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        TrivialAugmentWideNoColor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=(img_size[0], img_size[1]), scale=(0.95, 1.)),
        transforms.ToTensor(),
        normalize,
    ]))
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=train_batch_size, shuffle=True,
    num_workers=8, pin_memory=False)
# push set
train_push_dataset = datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        transforms.ToTensor(),
    ]))
train_push_loader = torch.utils.data.DataLoader(
    train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
    num_workers=8, pin_memory=False)
# test set
test_dataset = datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size[0], img_size[1])),
        transforms.ToTensor(),
        normalize,
    ]))
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=8, pin_memory=False)
'''

# we should look into distributed sampler more carefully at torch.utils.data.distributed.DistributedSampler(train_dataset)
#log('training set size: {0}'.format(len(train_loader.dataset)))
#log('push set size: {0}'.format(len(train_push_loader.dataset)))
#log('test set size: {0}'.format(len(test_loader.dataset)))
log('batch size: {0}'.format(batch_size))

# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)

#if prototype_activation_function == 'linear':
#    ppnet.set_last_layer_incorrect_connection(incorrect_strength=0)
ppnet = ppnet.to(device)
ppnet = torch.nn.DataParallel(ppnet, device_ids=device_ids)
ppnet = ppnet.to(device)
#print(ppnet)
class_specific = True

#count number of model parameters to be trained 
count_param=0
for name, param in ppnet.named_parameters():
    if param.requires_grad:
        log(name)
        count_param+=1
pytorch_total_params = sum(p.numel() for p in ppnet.parameters() if p.requires_grad)
log("Total model parameters: {}".format(pytorch_total_params))
log('Number of layers that require gradient: \t{0}'.format(count_param))

#create parameter groups
'''params_features = []
params_add_on_layers = []
params_last_layer = []
params_last_layer_globalnet = []
params_prototype_vector = []

# set up optimizer
if 'densenet' in base_architecture: 
    # freeze resnet50 except last convolutional layer
    for name,param in ppnet.module.features.named_parameters():
        if 'features' in name:
            params_features.append(param)
        elif 'add_on_layers' in name:
            params_add_on_layers.append(param)
        elif 'last_layer' in name:
            params_last_layer.append(param)
        elif 'last_layer_globalnet' in name:
            params_last_layer_globalnet.append(param)
        elif 'prototype_vectors' in name:
            params_prototype_vector.append(param)
        else: #such that model training fits on one gpu 
            param.requires_grad = False
            # params_backbone.append(param)
print("total params saved:", len(params_features)+len(params_add_on_layers)+len(params_last_layer)+len(params_last_layer_globalnet))
'''

# define optimizer
'''stage1_optimizer_specs = \
[{'params': params_features, 'lr': stage1_optimizer_lrs['features'], 'weight_decay': 1e-5},
 #{'params': ppnet.add_on_layers_globalnet.parameters(), 'lr': stage1_optimizer_lrs['add_on_layers_globalnet'], 'weight_decay': 1e-5},
 {'params': params_last_layer_globalnet, 'lr': stage1_optimizer_lrs['last_layer_globalnet'], 'weight_decay': 1e-5},
]
stage1_optimizer = torch.optim.Adam(stage1_optimizer_specs)

from settings import stage2_optimizer_lrs
stage2_optimizer_specs = \
[{'params': params_add_on_layers, 'lr': stage2_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-5},
 {'params': params_prototype_vector, 'lr': stage2_optimizer_lrs['prototype_vectors']},
 {'params': params_last_layer, 'lr': stage2_optimizer_lrs['last_layer']}
]
stage2_optimizer = torch.optim.Adam(stage2_optimizer_specs)

joint_optimizer_specs = \
[{'params': params_features, 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-5}, # bias are now also being regularized
 #{'params': ppnet.add_on_layers_globalnet.parameters(), 'lr': joint_optimizer_lrs['add_on_layers_globalnet'], 'weight_decay': 1e-5},
 {'params': params_last_layer_globalnet, 'lr': joint_optimizer_lrs['last_layer_globalnet'], 'weight_decay': 1e-5},
 {'params': params_add_on_layers, 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-5},
 {'params': params_prototype_vector, 'lr': joint_optimizer_lrs['prototype_vectors']},
 {'params': params_last_layer, 'lr': joint_optimizer_lrs['last_layer']}
]
joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
#joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
'''

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


'''joint_optimizer_specs = \
[{'params': ppnet.module.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-5}, # bias are now also being regularized
 {'params': ppnet.module.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-5},
 {'params': ppnet.module.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']}
]
joint_optimizer_backbone = torch.optim.Adam(joint_optimizer_specs)

joint_optimizer_specs_lastlayer = \
[{'params': ppnet.module.last_layer_globalnet.parameters(), 'lr': joint_optimizer_lrs['last_layer_globalnet'], 'weight_decay': 1e-5},
 {'params': ppnet.module.last_layer.parameters(), 'lr': joint_optimizer_lrs['last_layer']}
]
joint_optimizer_lastlayer = torch.optim.Adam(joint_optimizer_specs_lastlayer)
joint_optimizer = [joint_optimizer_backbone, joint_optimizer_lastlayer]
'''

if lrscheduler == 'cosineannealing':
    joint_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(joint_optimizer, T_max=len(train_loader)*(num_train_epochs - (num_stage2_epochs+num_stage1_epochs)), eta_min=joint_optimizer_lrs['features']/100.)
    #joint_lr_scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(joint_optimizer_backbone, T_max=len(train_loader)*(num_train_epochs - (num_stage2_epochs+num_stage1_epochs)), eta_min=joint_optimizer_lrs['features']/100.)
    #joint_lr_scheduler_lastlayer = torch.optim.lr_scheduler.CosineAnnealingLR(joint_optimizer_lastlayer, T_max=len(train_loader)*(num_train_epochs - (num_stage2_epochs+num_stage1_epochs)), eta_min=0.001)
    #joint_lr_scheduler = [joint_lr_scheduler_backbone, joint_lr_scheduler_lastlayer]
    #joint_lr_scheduler = joint_lr_scheduler_lastlayer
elif lrscheduler == 'lrdecay':
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=lrdecay_gamma)
else:
    joint_lr_scheduler = None

#last_layer_optimizer_specs = [{'params': ppnet.module.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
#last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
#scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(last_layer_optimizer, T_0=10, eta_min=0.001, T_mult=1, verbose=False)

'''from settings import last_layer_optimizer_lr
last_layer_optimizer_specs = [{'params': params_last_layer, 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
'''

ppnet, stage1_optimizer, stage2_optimizer, joint_optimizer, start_epoch, epoch_stage1, epoch_stage2, epoch_joint = load_model(ppnet, stage1_optimizer, stage2_optimizer, joint_optimizer, args.state_dict_dir_net)
if args.start_epoch!=0:
    start_epoch = args.start_epoch
lrs = []

print("start epoch:", start_epoch, flush=True)

# train the model
log('start training')

modelcheckpoint = save.ModelCheckpoint(log_dir+'/'+'net_trained_best'+'_'+str(randseedother)+'_'+str(randseeddata), 'auc')
#modelcheckpoint_wtmacro = save.ModelCheckpoint(log_dir+'/'+'net_trained_best_wtmacro'+'_'+str(randseedother)+'_'+str(randseeddata), 'auc_wtmacro')

if args.mode == 'train':
    for epoch in range(start_epoch, num_train_epochs+1):
        log('epoch: \t{0}'.format(epoch))

        # train backbone and globalnet
        if epoch <= num_stage1_epochs:
            tnt.train_backbone_globalnet_only(model=ppnet, log=log)
            info_train, _ = tnt.train(model=ppnet, dataloader=train_loader, optimizer=stage1_optimizer,
                        epoch=epoch, last_lr=stage1_optimizer_lrs['last_layer_globalnet'], class_specific=class_specific, coefs=coefs, log=log, device=device)
            if usevalidation:
                info_val, _ = tnt.val(model=ppnet, dataloader=val_loader, epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device)
            
            last_lr = stage1_optimizer_lrs['last_layer_globalnet']
            optimizer = stage1_optimizer
            epoch_stage1 = epoch
            epoch_stage2 = 0
            epoch_joint = 0
            lrs.append(last_lr)
        
        # train protopnet
        elif num_stage1_epochs < epoch <= (num_stage2_epochs+num_stage1_epochs):
            tnt.train_protopnet_only(model=ppnet, log=log)
            info_train, _ = tnt.train(model=ppnet, dataloader=train_loader, optimizer=stage2_optimizer,
                        epoch=epoch, last_lr=stage2_optimizer_lrs['last_layer'], class_specific=class_specific, coefs=coefs, log=log, device=device)
            if usevalidation:
                info_val, _ = tnt.val(model=ppnet, dataloader=val_loader, epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device)
            
            last_lr = stage2_optimizer_lrs['last_layer']
            optimizer = stage2_optimizer
            epoch_stage1 = num_stage1_epochs
            epoch_stage2 = epoch
            epoch_joint = 0
            lrs.append(last_lr)
        
        # training the whole network
        else:
            tnt.joint(model=ppnet, log=log)
            info_train, lrs = tnt.train(model=ppnet, dataloader=train_loader, optimizer=joint_optimizer, epoch=epoch, 
                        last_lr=joint_optimizer_lrs['features'], lrs=lrs, lr_scheduler=joint_lr_scheduler, class_specific=class_specific, coefs=coefs, log=log, device=device)
            if usevalidation:
                info_val, _ = tnt.val(model=ppnet, dataloader=val_loader, epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device)
            
            if lrscheduler == 'lrdecay':
                joint_lr_scheduler.step()
                lrs.append(joint_lr_scheduler.get_last_lr()[0])
            elif lrscheduler=='fixedlr':
                lrs.append(joint_optimizer_lrs['features'])

            last_lr = lrs[-1]
            optimizer = joint_optimizer
            epoch_joint = epoch
            epoch_stage1 = num_stage1_epochs
            epoch_stage2 = num_stage1_epochs + num_stage2_epochs
            print("After epoch:", epoch, ppnet.module.last_layer.weight)
        
        info_test, _ = tnt.test(model=ppnet, dataloader=test_loader,
                        epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device)
        #accu = info_test['correct_both']/info_test['total_image']
        #save.save_model(ppnet, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, accu, log_dir+'/'+'net_trained'+'_nopush')
        #save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu,
        #                            target_accu=0.50, log=log)
        
        # update the prototype vector and sparsity of last layer of protopnet
        if epoch > num_stage1_epochs:
            push_braixx.push_prototypes(
                train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                prototype_network_parallel=ppnet, # pytorch network with prototype_vectors
                class_specific=class_specific,
                preprocess_input_function=preprocess_input_function, # normalize if needed
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
                epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                prototype_img_filename_prefix=prototype_img_filename_prefix,
                prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                save_prototype_class_identity=True,
                log=log,
                device=device)
            info_test, _ = tnt.test(model=ppnet, dataloader=test_loader,
                            epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device)
            #accu = info_test['correct_both']/info_test['total_image']
            #save.save_model(ppnet, stage1_optimizer, stage2_optimizer, joint_optimizer,  epoch_stage1, epoch_stage2, epoch_joint, accu, log_dir+'/'+'net_trained'+'_push'+str(randseedother)+'_'+str(randseeddata))
            #save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
            #                            target_accu=0.50, log=log)
        
        print("Again after epoch:", epoch, ppnet.module.last_layer.weight)

        if usevalidation:
            modelcheckpoint(info_val['avg_loss'], ppnet, info_train['conf_mat_global'], info_val['conf_mat_global'], stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, info_val['auc_both'])
            #modelcheckpoint_wtmacro(info_val['avg_loss'], ppnet, info_train['conf_mat_global'], info_val['conf_mat_global'], stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, info_val['auc_both_wtmacro'])
            results_store_excel(True, True, False, None, info_train, info_val, epoch, last_lr, None, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx')
            write_results_xlsx_confmat(list(range(num_classes)), modelcheckpoint.conf_mat_train_best, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'confmat_train_val_test')
            write_results_xlsx_confmat(list(range(num_classes)), modelcheckpoint.conf_mat_test_best, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'confmat_train_val_test')
            save.save_model(ppnet, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, info_val['auc_both'], log_dir+'/'+'net_trained_last'+'_'+str(randseedother)+'_'+str(randseeddata))
        else:
            results_store_excel(True, False, False, None, info_train, None, epoch, last_lr, None, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx')
            write_results_xlsx_confmat(list(range(num_classes)), info_train['conf_mat_global'], log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'confmat_train_val_test')
            save.save_model(ppnet, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, info_train['auc_both'], log_dir+'/'+'net_trained_last'+'_'+str(randseedother)+'_'+str(randseeddata))

#training the last layer for sparsity
'''if prototype_activation_function != 'linear':
    tnt.last_only(model=ppnet, log=log)
    for i in range(1):
        log('iteration: \t{0}'.format(i))
        _ = tnt.train(model=ppnet, dataloader=train_loader, optimizer=last_layer_optimizer,
                        epoch=epoch, last_lr=last_layer_optimizer_lr, class_specific=class_specific, coefs=coefs, log=log)
        accu = tnt.test(model=ppnet, dataloader=test_loader,
                        epoch=epoch, class_specific=class_specific, coefs=coefs, log=log)
        save.save_model(ppnet, stage1_optimizer, stage2_optimizer, joint_optimizer,  epoch_stage1, epoch_stage2, epoch_joint, accu, log_dir+'/'+'net_trained'+'_push_sparsity')
        #save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
        #                            target_accu=0.50, log=log)
'''
'''
    plt.clf()
    plt.plot(lrs)
    plt.savefig(os.path.join(model_dir,'lr_braixx'+'_'+str(randseedother)+'_'+str(randseeddata)+'.png'))
'''
if args.mode == 'localization':
    if dataset == 'cbis-ddsm':
        get_images(ppnet, test_loader, df_test, device, args, 'IOU')
        get_images(ppnet, test_loader, df_test, device, args, 'DSC')

end_time = datetime.datetime.now()
log("Start time: {0}".format(str(begin_time)))
log("End time: {0}".format(str(end_time)))
log("Execution time: {0}".format(str(end_time - begin_time)))

#info_test, _ = tnt.test(model=ppnet, dataloader=test_loader, epoch=60, class_specific=class_specific, coefs=coefs, log=log, device=device)

logclose()