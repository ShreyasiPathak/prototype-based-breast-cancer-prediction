from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import math
import numpy as np

from util.evaluation import conf_mat_create

def train_pipnet(net, train_loader, optimizer_net, optimizer_classifier, scheduler_net, scheduler_classifier, criterion, epoch, nr_epochs, device, numclasses, align_pf_weight_val = 5., tanh_weight_val=2., cl_weight_val=2., pretrain=False, finetune=False, progress_prefix: str = 'Train Epoch'):

    # Make sure the model is in train mode
    net.train()
    
    if pretrain:
        # Disable training of classification layer
        net.module._classification.requires_grad = False
        progress_prefix = 'Pretrain Epoch'
    else:
        # Enable training of classification layer (disabled in case of pretraining)
        net.module._classification.requires_grad = True
    
    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_align_loss = 0.
    total_tanh_loss = 0.
    total_class_loss = 0.
    total_acc = 0.
    correct_train = 0
    total_images_train = 0
    total_images = 0
    conf_mat_train = np.zeros((numclasses, numclasses))

    iters = len(train_loader)
    # Show progress on progress bar. 
    train_iter = tqdm(enumerate(train_loader),
                    total=len(train_loader),
                    desc=progress_prefix+'%s'%epoch,
                    mininterval=2.,
                    ncols=0)
    
    count_param=0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param+=1           
    print("Number of parameters that require gradient: ", count_param, flush=True)

    if pretrain:
        align_pf_weight = (epoch/nr_epochs)*1.
        unif_weight = 0.5 #ignored
        t_weight = 5.
        cl_weight = 0.
    else:
        align_pf_weight = align_pf_weight_val #5. 
        t_weight = tanh_weight_val #2.
        unif_weight = 0.
        cl_weight = cl_weight_val #2.

    
    print("Align weight: ", align_pf_weight, ", U_tanh weight: ", t_weight, "Class weight:", cl_weight, flush=True)
    print("Pretrain?", pretrain, "Finetune?", finetune, flush=True)
    
    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network
    for i, (_, xs1, xs2, ys, _) in train_iter:       
        
        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)
       
        # Reset the gradients
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)
       
        # Perform a forward pass through the network
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        #print("out shape:", out.shape, flush=True) #pretrain batch size 50 gets doubled to 100.
        loss, align_loss_val, tanh_loss_val, class_loss_val, acc, correct_train, total_images_train, conf_mat_train = calculate_loss(proto_features, pooled, out, ys, align_pf_weight, t_weight, unif_weight, cl_weight, net.module._classification.normalization_multiplier, pretrain, finetune, criterion, train_iter, numclasses, correct_train, total_images_train, conf_mat_train, print=True, EPS=1e-8)
        
        # Compute the gradient
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()  
            if scheduler_classifier is not None: 
                scheduler_classifier.step(epoch - 1 + (i/iters))
                lrs_class.append(scheduler_classifier.get_last_lr()[0])
            else:
                lrs_class.append(0.01)
     
        if not finetune:
            if scheduler_net is not None:
                optimizer_net.step()
                scheduler_net.step() 
                lrs_net.append(scheduler_net.get_last_lr()[0])
            else:
                optimizer_net.step()
                lrs_net.append(0.00001)
        else:
            lrs_net.append(0.)
            
        with torch.no_grad():
            total_acc+=acc
            total_loss+=out.shape[0] * loss.item()
            total_images+= out.shape[0]
            
            total_align_loss+= out.shape[0] * align_loss_val
            total_tanh_loss+= out.shape[0] * tanh_loss_val
            total_class_loss+= out.shape[0] * class_loss_val

        if not pretrain:
            with torch.no_grad():
                net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data, min=0.))
                #net.module._classification.weight.copy_(torch.clamp(net.module._classification.weight.data - 1e-3, min=0.)) #set weights in classification layer < 1e-3 to zero
                net.module._classification.normalization_multiplier.copy_(torch.clamp(net.module._classification.normalization_multiplier.data, min=1.0)) 
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(torch.clamp(net.module._classification.bias.data, min=0.))  

    

    train_info['train_accuracy'] = total_acc/float(i+1)
    #train_info['loss'] = total_loss/float(i+1)
    train_info['loss'] = total_loss/total_images
    train_info['align_loss'] = total_align_loss/total_images
    train_info['tanh_loss'] = total_tanh_loss/total_images
    train_info['class_loss'] = total_class_loss/total_images
    
    train_info['lrs_net'] = lrs_net
    train_info['lrs_class'] = lrs_class

    train_info['correct_train'] = correct_train
    train_info['total_images_train'] = total_images_train
    train_info['conf_mat_train'] = conf_mat_train
    
    return train_info

def calculate_loss(proto_features, pooled, out, ys1, align_pf_weight, t_weight, unif_weight, cl_weight, net_normalization_multiplier, pretrain, finetune, criterion, train_iter, numclasses, correct_train, total_images_train, conf_mat_train, print=True, EPS=1e-10):
    ys = torch.cat([ys1,ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)

    embv2 = pf2.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0,2,1).flatten(end_dim=1)
    
    a_loss_pf = (align_loss(embv1, embv2.detach())+ align_loss(embv2, embv1.detach()))/2.
    tanh_loss = -(torch.log(torch.tanh(torch.sum(pooled1,dim=0))+EPS).mean() + torch.log(torch.tanh(torch.sum(pooled2,dim=0))+EPS).mean())/2.

    if not finetune:
        loss = (align_pf_weight*a_loss_pf) + (t_weight * tanh_loss)
        align_loss_val = (align_pf_weight*a_loss_pf).item()
        tanh_loss_val = (t_weight * tanh_loss).item()
    else:
        align_loss_val = 0.
        tanh_loss_val = 0.
    
    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs),dim=1),ys)
        
        if finetune:
            loss = cl_weight * class_loss
            class_loss_val = (cl_weight * class_loss).item()
        else:
            loss+= cl_weight * class_loss
            class_loss_val = (cl_weight * class_loss).item()
    else:
        class_loss_val = 0.
    # Our tanh-loss optimizes for uniformity and was sufficient for our experiments. However, if pretraining of the prototypes is not working well for your dataset, you may try to add another uniformity loss from https://www.tongzhouwang.info/hypersphere/ Just uncomment the following three lines
    # else:
    #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
    #     loss += unif_weight * uni_loss

    acc=0.
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))
        correct_train, total_images_train, conf_mat_train, _ = conf_mat_create(ys_pred_max, ys, correct_train, total_images_train, conf_mat_train, list(range(numclasses)))
    
    if print: 
        with torch.no_grad():
            if pretrain:
                train_iter.set_postfix_str(
                f'L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}',refresh=False)
            else:
                if finetune:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)
                else:
                    train_iter.set_postfix_str(
                    f'L:{loss.item():.3f},LC:{class_loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1:{torch.count_nonzero(torch.relu(pooled-0.1),dim=1).float().mean().item():.1f}, Ac:{acc:.3f}',refresh=False)            
    return loss, align_loss_val, tanh_loss_val, class_loss_val, acc, correct_train, total_images_train, conf_mat_train

# Extra uniform loss from https://www.tongzhouwang.info/hypersphere/. Currently not used. 
def uniform_loss(x, t=2):
    # print("sum elements: ", torch.sum(torch.pow(x,2), dim=1).shape, torch.sum(torch.pow(x,2), dim=1)) #--> should be ones
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss

# from https://gitlab.com/mipl/carl/-/blob/main/losses.py
def align_loss(inputs, targets, EPS=1e-12):
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False
    
    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss

