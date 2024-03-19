import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

import evaluation
from helpers import list_of_distances, make_one_hot

from settings import num_classes, randseedother, randseeddata, log_dir, num_train_epochs, lrscheduler

def _train_or_test(model, dataloader, optimizer=None, epoch=None, last_lr=None, lrs=None, lr_scheduler=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, device=None, mode=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0

    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    y_prob = []
    y_preds = []
    y_trues = []
    info = {}
    total_loss = 0.
    correct = 0
    total_images = 0
    auc = 0.0
    conf_mat = np.zeros((num_classes, num_classes))
    num_batches = len(dataloader)

    count_param=0
    if is_train:
        for name, param in model.named_parameters():
            if param.requires_grad:
                count_param+=1           
        log('Number of layers that require gradient: \t{0}'.format(count_param))

    for i, dataloader_input in enumerate(dataloader):
        '''if is_train:
            #(_, image1, image2, label, _)
            #input1 = image1.cuda()
            #input2 = image2.cuda()
            #target = label.cuda()
            input = dataloader_input[1].cuda()
            target = dataloader_input[2].cuda()
            label = dataloader_input[3]
            label = torch.cat([label, label])
            target = torch.cat([target, target]).cuda()
        else:'''
        input = dataloader_input[1].to(device) #cuda()
        target = dataloader_input[2].to(device) #cuda()
        label = dataloader_input[2]

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, min_distances = model(input)
            
            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (model.module.prototype_shape[1]
                            * model.module.prototype_shape[2]
                            * model.module.prototype_shape[3])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:,label]).to(device) #cuda()
                inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = \
                    torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.module.prototype_class_identity).to(device) #cuda()
                    l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.module.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(min_distances, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.module.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            y_preds += predicted.detach().tolist()
            y_trues += target.detach().tolist()
            y_prob += F.softmax(output, dim=1).detach().tolist()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

            correct, total_images, conf_mat, _ = evaluation.conf_mat_create(predicted, target, correct, total_images, conf_mat, list(range(num_classes)))

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                if lrscheduler == 'cosineannealing':
                    lr_scheduler.step()
                    lrs.append(lr_scheduler.get_last_lr()[0])
                    log("Lr after step:".format(lr_scheduler.get_last_lr()[0]))
            total_loss += target.shape[0] * loss.item()

        else:
            loss = (coefs['crs_ent'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * separation_cost
                    + coefs['l1'] * l1)    
            
            total_loss += target.shape[0] * loss.item()

        if i%20==0:
            log("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(epoch, num_train_epochs, i, num_batches, loss.item()))

        del input
        del target
        del output
        del predicted
        del min_distances

    end = time.time()

    #loss values calculation
    info['loss_crossent'] = coefs['crs_ent'] * total_cross_entropy / n_batches
    info['loss_cluster'] = coefs['clst'] * total_cluster_cost / n_batches
    info['loss_separation'] = coefs['sep'] * total_separation_cost / n_batches
    info['loss_avg_separation'] = total_avg_separation_cost / n_batches
    info['L1'] = coefs['l1'] * model.module.last_layer.weight.norm(p=1).item()
    info['avg_loss'] = total_loss/total_images
    info['total_image'] = total_images

    info['correct'] = correct
    info['conf_mat'] = conf_mat

    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(info['loss_crossent']))
    log('\tcluster: \t{0}'.format(info['loss_cluster']))
    if class_specific:
        log('\tseparation:\t{0}'.format(info['loss_separation']))
        log('\tavg separation:\t{0}'.format(info['loss_avg_separation']))
    log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    log('\tl1: \t\t{0}'.format(info['L1']))
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))
    
    if mode == 'test':
        if num_classes > 2:
            per_model_metrics = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds), np.array(y_prob))
        else:
            per_model_metrics = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds), np.array(y_prob)[:,1])
        
        per_model_metrics_loss = [info['avg_loss'], info['loss_crossent'], info['loss_cluster'], info['loss_separation'], info['loss_avg_separation'], info['L1']]
        per_model_metrics = [epoch] + per_model_metrics_loss + per_model_metrics
        print("Test set result model metric:", per_model_metrics, flush=True)

        class_specific_metric = evaluation.classspecific_performance_metrics(list(range(num_classes)), np.array(y_trues), np.array(y_preds))
        print("Test set result class specific:", class_specific_metric, flush=True)
        
        evaluation.write_results_xlsx_confmat(list(range(num_classes)), conf_mat, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'confmat_train_val_test')
        evaluation.write_results_xlsx(per_model_metrics, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'test_results')
        evaluation.write_results_classspecific(class_specific_metric, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'test_results')

    else:
        if num_classes > 2:
            auc = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob), multi_class='ovo')
        else:
            auc = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob)[:,1])

    info['auc'] = auc

    return info, lrs


def train(model, dataloader, optimizer, epoch, last_lr, lrs=None, lr_scheduler=None, class_specific=False, coefs=None, log=print, device=None):
    assert(optimizer is not None)
    
    log('\ttrain')
    model.train()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=optimizer,
                          epoch=epoch, last_lr=last_lr, lrs=lrs, lr_scheduler=lr_scheduler, class_specific=class_specific, coefs=coefs, log=log, device=device, mode='train')

def val(model, dataloader, epoch=None, class_specific=False, coefs=None, log=print, device=None):
    log('\tval')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device, mode='val')

def test(model, dataloader, epoch=None, class_specific=False, coefs=None, log=print, device=None):
    log('\ttest')
    model.eval()
    return _train_or_test(model=model, dataloader=dataloader, optimizer=None,
                          epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device, mode='test')

def last_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')

def warm_only(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = False
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\twarm')

def joint(model, log=print):
    for p in model.module.features.parameters():
        p.requires_grad = True
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
