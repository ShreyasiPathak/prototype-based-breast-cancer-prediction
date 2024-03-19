import os
import sys
import torch
import random
import datetime
import numpy as np
import torch.nn as nn
from shutil import copy
import matplotlib.pyplot as plt

from util.log import Log
from pipnet.test import eval_pipnet
from pipnet.train import train_pipnet
from util import evaluation, data_loader
from util.func import init_weights_xavier
from pipnet.pipnet import PIPNet, get_network
from util.visualize_prediction import vis_pred
from util.vis_pipnet import visualize, visualize_topk
from util.args import get_args, save_args, get_optimizer_nn

from util.pytorchtools import EarlyStopping, ModelCheckpoint
from util.utils import save_model, load_model_for_testing
from util.eval_cbis import get_images

def set_random_seed(args):
    #random state initialization of the code - values - 8, 24, 30
    torch.manual_seed(args.randseedother) 
    torch.cuda.manual_seed(args.randseedother)
    torch.cuda.manual_seed_all(args.randseedother)
    np.random.seed(args.randseeddata)
    random.seed(args.randseeddata)
    g = torch.Generator()
    g.manual_seed(args.randseedother)
    #torch.backends.cudnn.deterministic = True
    return g

def set_device(args):
    gpu_list = args.gpu_ids.split(',')
    device_ids = []
    if args.gpu_ids!='':
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))
    
    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids)==1:
            device = torch.device('cuda:{}'.format(args.gpu_ids))
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

def load_model(args, net, optimizer_net, optimizer_classifier, num_prototypes):
    with torch.no_grad():
        if args.state_dict_dir_net != '':
            checkpoint = torch.load(args.state_dict_dir_net,map_location=device)
            net.load_state_dict(checkpoint['model_state_dict'],strict=True) 
            print("Pretrained network loaded", flush=True)
            start_epoch_pretrain = checkpoint['epoch_pretrain'] + 1
            start_epoch_finetune = checkpoint['epoch_finetune'] + 1
            try:
                optimizer_net.load_state_dict(checkpoint['optimizer_net_state_dict']) 
            except:
                pass
            if torch.mean(net.module._classification.weight).item() > 1.0 and torch.mean(net.module._classification.weight).item() < 3.0 and torch.count_nonzero(torch.relu(net.module._classification.weight-1e-5)).float().item() > 0.8*(num_prototypes*args.numclasses): #assume that the linear classification layer is not yet trained (e.g. when loading a pretrained model only)
                print("We assume that the classification layer is not yet trained. We re-initialize it...", start_epoch_pretrain, start_epoch_finetune, flush=True)
                torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
                torch.nn.init.constant_(net.module._multiplier, val=2.)
                net.module._multiplier.requires_grad = False
                print("Multiplier initialized with", torch.mean(net.module._multiplier).item(), flush=True)
                print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.)
            else:
                if 'optimizer_classifier_state_dict' in checkpoint.keys():
                    optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])
                    print("Classification layer already trained. Loaded pretrained classifier. Start training from epoch: ", start_epoch_pretrain, start_epoch_finetune, flush = True)
            
        else:
            start_epoch_pretrain = 1
            start_epoch_finetune = 1
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0,std=0.1) 
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.)
            torch.nn.init.constant_(net.module._multiplier, val=2.)
            net.module._multiplier.requires_grad = False

            print("Classification layer initialized with mean", torch.mean(net.module._classification.weight).item(), flush=True)
            print("Multiplier initialized with", torch.mean(net.module._multiplier).item(), flush=True)

    return net, start_epoch_pretrain, start_epoch_finetune

def self_training_phase(args, log, net, params_to_train, params_to_freeze, params_backbone, trainloader_pretraining, optimizer_net, optimizer_classifier, criterion, start_epoch):
    # Define classification loss function and scheduler
    
    if args.scheduler_net == 'cosineannealing':
        scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader_pretraining)*args.epochs_pretrain, eta_min=args.lr_block/100., last_epoch=-1)
    else:
        scheduler_net = None

    lrs_pretrain_net = []
    # PRETRAINING PROTOTYPES PHASE
    for epoch_pretrain in range(start_epoch, args.epochs_pretrain+1):
        for param in params_to_train:
            param.requires_grad = True
        for param in net.module._add_on.parameters():
            param.requires_grad = True
        for param in net.module._classification.parameters():
            param.requires_grad = False
        for param in params_to_freeze:
            param.requires_grad = True # can be set to False when you want to freeze more layers
        for param in params_backbone:
            param.requires_grad = False #can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet)
        
        print("\nPretrain Epoch", epoch_pretrain, "with batch size", trainloader_pretraining.batch_size, flush=True)
        
        # Pretrain prototypes
        train_info = train_pipnet(net, trainloader_pretraining, optimizer_net, optimizer_classifier, scheduler_net, None, criterion, epoch_pretrain, args.epochs_pretrain, device, args.numclasses, align_pf_weight_val=args.align_pf_weight, tanh_weight_val=args.tanhloss_weight, cl_weight_val=args.classloss_weight, pretrain=True, finetune=False)
        lrs_pretrain_net+=train_info['lrs_net']
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(os.path.join(args.log_dir,'lr_pretrain_net'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.png'))
        log.log_values('log_epoch_overview'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), epoch_pretrain, "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", "n.a.", train_info['loss'])

        with torch.no_grad():
            net.eval()
            save_model(net, optimizer_net, optimizer_classifier, train_info['loss'], os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_pretrained'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)), epoch_pretrain, 0)
            #torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_pretrained'))
            net.train()
    return net

def full_training_phase(args, log, net, trainloader, valloader, testloader, criterion, start_epoch):
    # SECOND TRAINING PHASE
    # re-initialize optimizers and schedulers for second training phase
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)            
    
    if args.scheduler_net == 'cosineannealing':
        scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, T_max=len(trainloader)*args.epochs, eta_min=args.lr_net/100.)
    else:
        scheduler_net = None
    
    # scheduler for the classification layer is with restarts, such that the model can re-active zeroed-out prototypes. Hence an intuitive choice. 
    if args.epochs<=30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False)
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False)
    #scheduler_classifier = None

    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True
    
    #Setup early stopping
    if args.usevalidation:
        if args.patienceepochs:
            modelcheckpoint = EarlyStopping(path_to_model=os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_best'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)), best_score=args.valscore_resumetrain, early_stopping_criteria=args.early_stopping_criteria, patience=args.patienceepochs, verbose=True)
        else:
            modelcheckpoint = ModelCheckpoint(path_to_model=os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_best'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)), criteria=args.early_stopping_criteria, patience=args.patienceepochs, verbose=True)

    frozen = True
    lrs_net = []
    lrs_classifier = []
    
    print("I am in full training:", start_epoch, args.epochs + 1)
    for epoch in range(start_epoch, args.epochs + 1):                      
        epochs_to_finetune = 3 #during finetuning, only train classification layer and freeze rest. usually done for a few epochs (at least 1, more depends on size of dataset)
        if epoch <= epochs_to_finetune and (args.epochs_pretrain > 0 or args.state_dict_dir_net != ''):
            for param in net.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True
        
        else: 
            finetune=False          
            if frozen:
                # unfreeze backbone
                if epoch>(args.freeze_epochs):
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True   
                    frozen = False
                # freeze first layers of backbone, train rest
                else:
                    for param in params_to_freeze:
                        param.requires_grad = True #Can be set to False if you want to train fewer layers of backbone
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False
        
        print("\n Epoch", epoch, "frozen:", frozen, flush=True)            
        if (epoch==args.epochs or epoch%30==0) and args.epochs>1:
            # SET SMALL WEIGHTS TO ZERO
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data, min=0.)) 
                #net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 0.001, min=0.)) 
                print("Classifier weights: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
                if args.bias:
                    print("Classifier bias: ", net.module._classification.bias, flush=True)
                torch.set_printoptions(profile="default")

        #train model
        train_info = train_pipnet(net, trainloader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, args.epochs, device, args.numclasses, align_pf_weight_val=args.align_pf_weight, tanh_weight_val=args.tanhloss_weight, cl_weight_val=args.classloss_weight, pretrain=False, finetune=finetune)
        lrs_net+=train_info['lrs_net']
        lrs_classifier+=train_info['lrs_class']
        
        #validation set
        if args.usevalidation:
            val_info = eval_pipnet(args, net, valloader, epoch, device, criterion, 'val', args.tanhloss_weight, args.classloss_weight, log)
            if scheduler_classifier is not None:
                classifier_lr = scheduler_classifier.get_last_lr()[0]
            else:
                classifier_lr = args.lr
            evaluation.results_store_excel(True, True, False, None, train_info, val_info, epoch, classifier_lr, val_info['num non-zero prototypes'], args.log_dir+'/results'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.xlsx')
        else:
            evaluation.results_store_excel(True, False, False, None, train_info, val_info, epoch, classifier_lr, val_info['num non-zero prototypes'], args.log_dir+'/results'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.xlsx')
            
        
        #test set
        eval_info = eval_pipnet(args, net, testloader, epoch, device, criterion, 'test', args.tanhloss_weight, args.classloss_weight, log)
        log.log_values('log_epoch_overview'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), epoch, eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], train_info['train_accuracy'], train_info['loss'])

        with torch.no_grad():
            net.eval()
            #torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last'))
            if args.usevalidation:
                save_model(net, optimizer_net, optimizer_classifier, train_info['loss'], os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)), args.epochs_pretrain, epoch, val_info['AUC_test'])
                if args.patienceepochs:
                    modelcheckpoint(val_info['AUC_test'], net, optimizer_net, optimizer_classifier, args.epochs_pretrain, epoch, train_info['conf_mat_train'], val_info['confusion_matrix'], train_info['loss'])
                    '''if modelcheckpoint.early_stop:
                        print("Early stopping",epoch+1, flush = True)
                        break
                    '''
                else:
                    modelcheckpoint(val_info['AUC_test'], net, optimizer_net, optimizer_classifier, args.epochs_pretrain, epoch, train_info['conf_mat_train'], val_info['confusion_matrix'], train_info['loss'])
            else:
                save_model(net, optimizer_net, optimizer_classifier, train_info['loss'], os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)), args.epochs_pretrain, epoch, None)

            '''if epoch%30 == 0:
                net.eval()
                torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_%s'%str(epoch)))            
            '''

            # save learning rate in figure
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(os.path.join(args.log_dir,'lr_net'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.png'))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(os.path.join(args.log_dir,'lr_class'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.png'))
    
    if args.epochs > 0:
        if args.usevalidation:
            evaluation.write_results_xlsx_confmat(list(range(args.numclasses)), modelcheckpoint.conf_mat_train_best, args.log_dir+'/results'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.xlsx', 'confmat_train_val_test')
            evaluation.write_results_xlsx_confmat(list(range(args.numclasses)), modelcheckpoint.conf_mat_test_best, args.log_dir+'/results'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.xlsx', 'confmat_train_val_test')
            net = load_model_for_testing(net, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_best'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)))
        else:
            evaluation.write_results_xlsx_confmat(list(range(args.numclasses)), train_info['conf_mat_train'], args.log_dir+'/results'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.xlsx', 'confmat_train_val_test')

    #net.eval()
    #torch.save({'model_state_dict': net.state_dict(), 'optimizer_net_state_dict': optimizer_net.state_dict(), 'optimizer_classifier_state_dict': optimizer_classifier.state_dict()}, os.path.join(os.path.join(args.log_dir, 'checkpoints'), 'net_trained_last'))
    return net

def run_pipnet(args=None):
    '''
    if not os.path.isdir(os.path.join(args.log_dir, "files")):
        os.mkdir(os.path.join(args.log_dir, "files"))
    copy('main.py', os.path.join(args.log_dir, "files")+'/main.py') #save python file
    copy(os.path.join('pipnet', 'train.py'), os.path.join(args.log_dir, "files")+'/train.py') #save python file
    copy(os.path.join('pipnet', 'pipnet.py'), os.path.join(args.log_dir, "files")+'/pipnet.py') #save python file
    '''
    #set random seed
    #g = set_random_seed(args)

    #get args
    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger
    log = Log(args.log_dir)
    print("Log dir: ", args.log_dir, flush=True)
    
    # Log the run arguments
    save_args(args, log.metadata_dir)
    
    #set device for the model
    device, device_ids = set_device(args)
    
    # Obtain the dataset and dataloaders
    #trainloader, trainloader_pretraining, trainloader_normal, trainloader_normal_augment, projectloader, testloader, test_projectloader, classes = get_dataloaders(args, device)
    if args.usevalidation:
        df_train, df_val, df_test, batches_train, batches_val, batches_test = data_loader.input_file_creation(args)
        dataloader_train_pretraining, dataloader_train, dataloader_train_projectloader, dataloader_val, dataloader_test, dataloader_test_projectloader = data_loader.dataloader(args, df_train, df_val, df_test, g)
    else:
        df_train, df_test, batches_train,batches_test = data_loader.input_file_creation(args)
        dataloader_train_pretraining, dataloader_train, dataloader_train_projectloader, dataloader_test, dataloader_test_projectloader = data_loader.dataloader(args, df_train, None, df_test, g)
        dataloader_val = None

    '''if args.validation_size == 0.:
        print("Classes: ", testloader.dataset.class_to_idx, flush=True)
    else:
        print("Classes: ", str(classes), flush=True)
    '''
    
    #model initialization
    # Create a convolutional network based on arguments and add 1x1 conv layer
    feature_net, add_on_layers, pool_layer, classification_layer, num_prototypes = get_network(args.numclasses, args)
    # Create a PIP-Net
    net = PIPNet(num_classes=args.numclasses,
                    num_prototypes=num_prototypes,
                    feature_net = feature_net,
                    args = args,
                    add_on_layers = add_on_layers,
                    pool_layer = pool_layer,
                    classification_layer = classification_layer
                )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids = device_ids)   
    
    #optimizer
    optimizer_net, optimizer_classifier, params_to_freeze, params_to_train, params_backbone = get_optimizer_nn(net, args)   

    # Initialize or load model
    net, start_epoch_pretrain, start_epoch_finetune = load_model(args, net, optimizer_net, optimizer_classifier, num_prototypes)

    # Forward one batch through the backbone to get the latent output size
    with torch.no_grad():
        _, xs1, _, _, _ = next(iter(dataloader_train))
        xs1 = xs1.to(device)
        proto_features, _, _ = net(xs1)
        wshape = proto_features.shape[-1]
        hshape = proto_features.shape[-2]
        args.wshape = wshape #needed for calculating image patch size
        args.hshape = hshape #needed for calculating image patch size
        print("Output shape: ", proto_features.shape, flush=True)
    
    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), 'results'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), 'epoch', 'test_top1_acc', 'test_f1', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
        print("Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance.", flush=True)
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch
        log.create_log('log_epoch_overview'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), 'results'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), 'epoch', 'test_top1_acc', 'test_top5_acc', 'almost_sim_nonzeros', 'local_size_all_classes','almost_nonzeros_pooled', 'num_nonzero_prototypes', 'mean_train_acc', 'mean_train_loss_during_epoch')
    
    #loss criterion
    criterion = nn.NLLLoss(reduction='mean').to(device)

    #first training phase
    #net = self_training_phase(args, log, net, params_to_train, params_to_freeze, params_backbone, dataloader_train_pretraining, optimizer_net, optimizer_classifier, criterion, start_epoch_pretrain)
           
    #visualizing top k prototypes after self-training phase
    '''with torch.no_grad():
        if 'convnext' in args.net and args.epochs_pretrain > 0:
            topks = visualize_topk(net, dataloader_train_projectloader, df_train, args.numclasses, device, 'visualised_pretrained_prototypes_topk'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), args)
    '''

    #second training phase    
    #net = full_training_phase(args, log, net, dataloader_train, dataloader_val, dataloader_test, criterion, start_epoch_finetune)
    #eval_info = eval_pipnet(args, net, dataloader_test, 'test', device, criterion, 'test', args.tanhloss_weight, args.classloss_weight, log)

    #visualizing top k prototypes after full training phase
    '''topks = visualize_topk(net, dataloader_train_projectloader, df_train, args.numclasses, device, 'visualised_prototypes_topk'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), args)
    
    # set weights of prototypes that are never really found in projection set to 0
    set_to_zero = []
    if topks:
        for prot in topks.keys():
            found = False
            for (i_id, score) in topks[prot]:
                if score > 0.1:
                    found = True
            if not found:
                torch.nn.init.zeros_(net.module._classification.weight[:,prot])
                set_to_zero.append(prot)
        print("Weights of prototypes", set_to_zero, "are set to zero because it is never detected with similarity>0.1 in the training set", flush=True)
        eval_info = eval_pipnet(args, net, dataloader_test, "notused0", device, criterion, 'test', args.tanhloss_weight, args.classloss_weight, log)
        log.log_values('log_epoch_overview'+'_'+str(args.randseedother)+'_'+str(args.randseeddata), "notused0", eval_info['top1_accuracy'], eval_info['top5_accuracy'], eval_info['almost_sim_nonzeros'], eval_info['local_size_all_classes'], eval_info['almost_nonzeros'], eval_info['num non-zero prototypes'], "n.a.", "n.a.")
    '''
    '''print("classifier weights: ", net.module._classification.weight, flush=True)
    print("Classifier weights nonzero: ", net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)], (net.module._classification.weight[net.module._classification.weight.nonzero(as_tuple=True)]).shape, flush=True)
    print("Classifier bias: ", net.module._classification.bias, flush=True)
    
    # Print weights and relevant prototypes per class
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c,:]
        for p in range(net.module._classification.weight.shape[1]):
            #if proto_weights[p]> 1e-3:
            relevant_ps.append((p, proto_weights[p].item()))
        if args.validation_size == 0.:
            #print("Class", c, "(", list(dataloader_test.dataset.class_to_idx.keys())[list(dataloader_test.dataset.class_to_idx.values()).index(c)],"):","has", len(relevant_ps),"relevant prototypes: ", relevant_ps, flush=True)
            print("Class", c, "(", list(args.groundtruthdic.keys())[list(args.groundtruthdic.values()).index(c)],"):","has", len(relevant_ps),"relevant prototypes: ", relevant_ps, flush=True)
    '''
    # visualize all prototypes and not just top k predictions 
    #visualize(net, dataloader_train_projectloader, df_train, args.numclasses, device, 'visualised_prototypes', args)
    
    #visualize the prediction for each instance in test set
    #testset_img0_path = dataloader_test_projectloader.dataset.samples[0][0]
    #test_path = os.path.split(os.path.split(testset_img0_path)[0])[0]
    #vis_pred(net, dataloader_test_projectloader, df_test, args.numclasses, device, args) 
    
    #IoU score for cbis-ddsm
    if args.dataset == 'cbis-ddsm':
        get_images(net, dataloader_test_projectloader, df_test, device, args, 'IOU')
        #get_images(net, dataloader_test_projectloader, df_test, device, args, 'DSC')
    
    print("Done!", flush=True)

if __name__ == '__main__':
    args = get_args()
    if args.classtype == 'diagnosis':
        args.groundtruthdic = {'benign': 0, 'malignant': 1}
    elif args.classtype=='diagnosis_with_normal':
        groundtruthdic = {'normal': 0, 'benign': 1, 'malignant': 2}
    elif args.classtype == 'birads':
        args.groundtruthdic = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    
    #torch.manual_seed(args.seed)
    #torch.cuda.manual_seed_all(args.seed)
    #random.seed(args.seed)
    #np.random.seed(args.seed)
    
    begin_time = datetime.datetime.now()
    
    g = set_random_seed(args)

    print_dir = os.path.join(args.log_dir,'out_iouallproto_trial'+'_'+str(args.randseedother)+'_'+str(args.randseeddata)+'.txt')
    tqdm_dir = os.path.join(args.log_dir,'tqdm'+'_iouallproto_trial'+str(args.randseedother)+'_'+str(args.randseeddata)+'.txt')
    if not os.path.isdir(args.log_dir):
        os.mkdir(args.log_dir)
    
    sys.stdout.close()
    sys.stderr.close()
    sys.stdout = open(print_dir, 'w')
    sys.stderr = open(tqdm_dir, 'w')
    
    run_pipnet(args)
    
    end_time = datetime.datetime.now()
    print("Start time:", str(begin_time), flush=True)
    print("End time:", str(end_time), flush = True)
    print("Execution time:", str(end_time - begin_time), flush=True)

    sys.stdout.close()
    sys.stderr.close()
