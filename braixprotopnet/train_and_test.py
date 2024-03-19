import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics

import evaluation
from helpers import list_of_distances, make_one_hot

from settings import num_classes, randseedother, randseeddata, log_dir, num_stage1_epochs, num_stage2_epochs, num_train_epochs, lrscheduler

'''def _train_or_test(model, dataloader, dataloader_val, optimizer=None, epoch=None, last_lr=None, lrs=None, lr_scheduler=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, device=None):
    
    #model: the multi-gpu model
    #dataloader:
    #optimizer: if None, will be test evaluation
    
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
    y_preds =[]
    y_trues =[]
    info = {}
    total_train_loss = 0.
    correct = 0
    total_images = 0
    conf_mat = np.zeros((num_classes, num_classes))
    num_batches = len(dataloader)

    count_param=0
    if is_train:
        for name, param in model.named_parameters():
            if param.requires_grad:
                count_param+=1           
        log('Number of layers that require gradient: \t{0}'.format(count_param))
    
    for i, dataloader_input in enumerate(dataloader):
        #(image, label)
        #input = image.cuda()
        #target = label.cuda()
        input = dataloader_input[1].to(device) #cuda()
        target = dataloader_input[2].to(device) #cuda()
        label = dataloader_input[2]
        
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            #print("input:", input, flush=True)
            output_protopnet, min_distances, output_globalnet = model(input)

            # compute cross-entropy loss of protopnet and globalnet
            cross_entropy = torch.nn.functional.cross_entropy(output_protopnet, target)
            cross_entropy_globalnet = torch.nn.functional.cross_entropy(output_globalnet, target)

            # compute knowlegde distillation loss
            kd_loss = knowledge_distillation_loss(target, F.softmax(output_globalnet, dim=1), F.softmax(output_protopnet, dim=1), coefs)

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
                separation_cost = torch.clamp(coefs['hinge_loss_sep'] - separation_cost, min=0)

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
            _, predicted = torch.max(output_protopnet.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            y_preds += predicted.detach().tolist()
            y_trues += target.detach().tolist()
            y_prob += F.softmax(output_protopnet, dim=1).detach().tolist()

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
                    if epoch < num_stage1_epochs:
                        loss = coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                    
                    elif (num_stage2_epochs + num_stage1_epochs) > epoch >= num_stage1_epochs:
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            #+ coefs['l1'] * l1
                            )
                    
                    elif epoch >= (num_stage2_epochs + num_stage1_epochs):
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            #+ coefs['l1'] * l1
                            + coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                            + coefs['kd_loss'] * kd_loss)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    if epoch < num_stage1_epochs:
                        loss = coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                    
                    elif (num_stage2_epochs + num_stage1_epochs) > epoch >= num_stage1_epochs:
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            #+ coefs['l1'] * l1
                            )
                    
                    elif epoch >= (num_stage2_epochs + num_stage1_epochs):
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            #+ coefs['l1'] * l1
                            + coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                            + coefs['kd_loss'] * kd_loss)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
                lrs.append(lr_scheduler.get_last_lr()[0])
                log("Lr after step:".format(lr_scheduler.get_last_lr()[0]))
            total_train_loss += target.shape[0] * loss.item()
        
        else:
            if epoch < num_stage1_epochs:
                loss = coefs['crs_ent_globalnet'] * cross_entropy_globalnet
            
            elif (num_stage2_epochs + num_stage1_epochs) > epoch >= num_stage1_epochs:
                loss = (coefs['crs_ent_protopnet'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    #+ coefs['l1'] * l1
                    )
            
            elif epoch >= (num_stage2_epochs + num_stage1_epochs):
                loss = (coefs['crs_ent_protopnet'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    #+ coefs['l1'] * l1
                    + coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                    + coefs['kd_loss'] * kd_loss)

        if i%20==0:
            log("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(epoch, num_train_epochs, i, num_batches, loss.item()))

        del input
        del target
        del output_protopnet
        del output_globalnet
        del predicted
        del min_distances

    end = time.time()

    #loss values calculation
    info['loss_crossent'] = total_cross_entropy / n_batches
    info['loss_cluster'] = total_cluster_cost / n_batches
    info['loss_separation'] = total_separation_cost / n_batches
    info['loss_avg_separation'] = total_avg_separation_cost / n_batches
    info['L1'] = model.module.last_layer.weight.norm(p=1).item()

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
    

    accu, _ = test(model, dataloader=dataloader_val, epoch=epoch, class_specific=class_specific, coefs=coefs, log=log, device=device)

    if is_train:
        evaluation.results_store_excel(True, True, False, None, correct_train, total_images_train, loss_train, correct_test, total_images_val, loss_val, epoch, conf_mat_train, conf_mat_val, current_lr, auc_val, path_to_results_xlsx, path_to_results_text)
        evaluation.results_store_excel(True, False, False, None, correct, total_images, total_train_loss/total_images, None, None, None, None, epoch, conf_mat, None, last_lr, None, None, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx')
        evaluation.write_results_xlsx_confmat(list(range(num_classes)), conf_mat, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'confmat_train_val_test')
    else:
        per_model_metrics = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds), np.array(y_prob)[:,1])
        per_model_metrics = [epoch, info['loss_crossent'], info['loss_cluster'], info['loss_separation'], info['loss_avg_separation'], info['L1']] + per_model_metrics
        class_specific_metric = evaluation.classspecific_performance_metrics(list(range(num_classes)), np.array(y_trues), np.array(y_preds))
        print("Test set result model metric:", per_model_metrics, flush=True)
        print("Test set result class specific:", class_specific_metric, flush=True)
        evaluation.write_results_xlsx_confmat(list(range(num_classes)), conf_mat, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'confmat_train_val_test')
        evaluation.write_results_xlsx(per_model_metrics, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'test_results')
        evaluation.write_results_classspecific(class_specific_metric, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'test_results')

    return n_correct / n_examples, lrs
'''

def _train_or_test(model, dataloader, optimizer=None, epoch=None, last_lr=None, lrs=None, lr_scheduler=None, class_specific=True, use_l1_mask=True,
                   coefs=None, log=print, device=None, mode=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    if isinstance(optimizer, list):
        optimizer_net = optimizer[0] 
        optimizer_classifier = optimizer[1] 
        is_train = optimizer[0] is not None
    else:
        is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0

    n_correct_global = 0
    y_preds_global = []
    y_prob_global = []
    total_cross_entropy_global = 0
    total_kd_loss = 0

    n_correct_both = 0
    y_preds_both = []
    y_prob_both = []

    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    y_prob = []
    y_preds =[]
    y_trues =[]
    info = {}
    total_loss = 0.
    correct = 0
    total_images = 0
    auc = 0.0
    conf_mat = np.zeros((num_classes, num_classes))
    num_batches = len(dataloader)

    correct_global = 0
    total_images_global = 0 
    conf_mat_global = np.zeros((num_classes, num_classes))

    correct_both = 0
    total_images_both = 0
    conf_mat_both = np.zeros((num_classes, num_classes))

    count_param=0
    if is_train:
        for name, param in model.named_parameters():
            if param.requires_grad:
                count_param+=1           
        log('Number of layers that require gradient: \t{0}'.format(count_param))
    
    for i, dataloader_input in enumerate(dataloader):
        #(image, label)
        #input = image.cuda()
        #target = label.cuda()
        input = dataloader_input[1].to(device) #cuda()
        target = dataloader_input[2].to(device) #cuda()
        label = dataloader_input[2]
        
        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            #print("input:", input, flush=True)
            output_protopnet, min_distances, output_globalnet = model(input)

            # compute cross-entropy loss of protopnet and globalnet
            cross_entropy = torch.nn.functional.cross_entropy(output_protopnet, target)
            cross_entropy_globalnet = torch.nn.functional.cross_entropy(output_globalnet, target)

            # compute knowlegde distillation loss
            kd_loss = knowledge_distillation_loss(target, F.softmax(output_globalnet, dim=1), F.softmax(output_protopnet, dim=1), coefs)

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
                separation_cost = torch.clamp(coefs['hinge_loss_sep'] - separation_cost, min=0)

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

            n_examples += target.size(0)
            y_trues += target.detach().tolist()

            # evaluation protopnet statistics
            _, predicted = torch.max(output_protopnet.data, 1)
            n_correct += (predicted == target).sum().item()
            y_preds += predicted.detach().tolist()
            y_prob += F.softmax(output_protopnet, dim=1).detach().tolist()

            # evaluation globalnet statistics
            _, predicted_global = torch.max(output_globalnet.data, 1)
            n_correct_global += (predicted_global == target).sum().item()
            y_preds_global += predicted_global.detach().tolist()
            y_prob_global += F.softmax(output_globalnet, dim=1).detach().tolist()

            #average protopnet and globalnet
            avg_output = (output_protopnet + output_globalnet)/2
            _, predicted_both = torch.max(avg_output.data, 1)
            n_correct_both += (predicted_both == target).sum().item()
            y_preds_both += predicted_both.detach().tolist()
            y_prob_both += F.softmax(avg_output, dim=1).detach().tolist()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            total_cross_entropy_global +=cross_entropy_globalnet.item()
            total_kd_loss += kd_loss.item()

            correct, total_images, conf_mat, _ = evaluation.conf_mat_create(predicted, target, correct, total_images, conf_mat, list(range(num_classes)))
            correct_global, total_images_global, conf_mat_global, _ = evaluation.conf_mat_create(predicted_global, target, correct_global, total_images_global, conf_mat_global, list(range(num_classes)))
            correct_both, total_images_both, conf_mat_both, _ = evaluation.conf_mat_create(predicted_both, target, correct_both, total_images_both, conf_mat_both, list(range(num_classes))) 

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    if epoch <= num_stage1_epochs:
                        loss = coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                    
                    elif num_stage1_epochs < epoch <= (num_stage2_epochs+num_stage1_epochs):
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            #+ coefs['l1'] * l1
                            )
                    
                    elif epoch > (num_stage2_epochs + num_stage1_epochs):
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            + coefs['sep'] * separation_cost
                            #+ coefs['l1'] * l1
                            + coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                            + coefs['kd_loss'] * kd_loss)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    if epoch <= num_stage1_epochs:
                        loss = coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                    
                    elif num_stage1_epochs < epoch <= (num_stage2_epochs+num_stage1_epochs):
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            #+ coefs['l1'] * l1
                            )
                    
                    elif epoch > (num_stage2_epochs + num_stage1_epochs):
                        loss = (coefs['crs_ent_protopnet'] * cross_entropy
                            + coefs['clst'] * cluster_cost
                            #+ coefs['l1'] * l1
                            + coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                            + coefs['kd_loss'] * kd_loss)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            
            
            if isinstance(optimizer,list):
                optimizer_net.zero_grad()
                optimizer_classifier.zero_grad()
            else:
                optimizer.zero_grad()
            loss.backward()
            if isinstance(optimizer,list):
                optimizer_net.step()
                optimizer_classifier.step()
            else:
                optimizer.step()
            if lr_scheduler is not None:
                if lrscheduler == 'cosineannealing':
                    if isinstance(lr_scheduler,list):
                        lr_scheduler[0].step()
                        lr_scheduler[1].step()
                        lrs.append(lr_scheduler[0].get_last_lr()[0])
                        log("Lr after step:".format(str(lr_scheduler[0].get_last_lr()[0])))
                    else:
                        lr_scheduler.step()
                        lrs.append(lr_scheduler.get_last_lr()[0])
                        log("Lr after step:".format(str(lr_scheduler.get_last_lr()[0])))
            total_loss += target.shape[0] * loss.item()
        
        else:
            if epoch <= num_stage1_epochs:
                loss = coefs['crs_ent_globalnet'] * cross_entropy_globalnet
            
            elif num_stage1_epochs < epoch <= (num_stage2_epochs+num_stage1_epochs):
                loss = (coefs['crs_ent_protopnet'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * separation_cost
                    #+ coefs['l1'] * l1
                    )
            
            elif epoch > (num_stage2_epochs + num_stage1_epochs):
                loss = (coefs['crs_ent_protopnet'] * cross_entropy
                    + coefs['clst'] * cluster_cost
                    + coefs['sep'] * separation_cost
                    #+ coefs['l1'] * l1
                    + coefs['crs_ent_globalnet'] * cross_entropy_globalnet
                    + coefs['kd_loss'] * kd_loss)
            
            total_loss += target.shape[0] * loss.item()

        if i%20==0:
            log("Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}".format(epoch, num_train_epochs, i, num_batches, loss.item()))

        del input
        del target
        del output_protopnet
        del output_globalnet
        del predicted
        del min_distances

    end = time.time()

    assert total_images == total_images_global
    assert total_images == total_images_both

    #loss values calculation
    info['loss_crossent_proto'] = coefs['crs_ent_protopnet'] * total_cross_entropy / n_batches
    info['loss_cluster'] = coefs['clst'] * total_cluster_cost / n_batches
    info['loss_separation'] = coefs['sep'] * total_separation_cost / n_batches
    info['loss_avg_separation'] = total_avg_separation_cost / n_batches
    info['loss_crossent_global'] = coefs['crs_ent_globalnet'] * total_cross_entropy_global / n_batches
    info['kd_loss'] = coefs['kd_loss'] * total_kd_loss / n_batches
    info['L1'] = model.module.last_layer.weight.norm(p=1).item()
    info['avg_loss'] = total_loss/total_images
    info['total_image'] = total_images
    
    info['correct_proto'] = correct
    info['conf_mat_proto'] = conf_mat
    info['correct_global'] = correct_global
    info['conf_mat_global'] = conf_mat_global
    info['correct_both'] = correct_both
    info['conf_mat_both'] = conf_mat_both
    
    log('\ttime: \t{0}'.format(end -  start))
    log('\tcross ent: \t{0}'.format(info['loss_crossent_proto']))
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
            per_model_metrics_proto = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds), np.array(y_prob))
            per_model_metrics_global = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds_global), np.array(y_prob_global))
            per_model_metrics_both = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds_both), np.array(y_prob_both))
        else:
            per_model_metrics_proto = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds), np.array(y_prob)[:,1])
            per_model_metrics_global = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds_global), np.array(y_prob_global)[:,1])
            per_model_metrics_both = evaluation.aggregate_performance_metrics(num_classes, np.array(y_trues), np.array(y_preds_both), np.array(y_prob_both)[:,1])
        
        per_model_metrics_loss = [info['avg_loss'], info['loss_crossent_proto'], info['loss_cluster'], info['loss_separation'], info['loss_crossent_global'], info['kd_loss'], info['loss_avg_separation'], info['L1']]
        per_model_metrics = [epoch] + per_model_metrics_loss + per_model_metrics_proto + per_model_metrics_global + per_model_metrics_both
        print("Test set result model metric:", per_model_metrics, flush=True)

        class_specific_metric_proto = evaluation.classspecific_performance_metrics(list(range(num_classes)), np.array(y_trues), np.array(y_preds))
        class_specific_metric_global = evaluation.classspecific_performance_metrics(list(range(num_classes)), np.array(y_trues), np.array(y_preds_global))
        class_specific_metric_both = evaluation.classspecific_performance_metrics(list(range(num_classes)), np.array(y_trues), np.array(y_preds_both))
        print("Test set result class specific:", class_specific_metric_both, flush=True)
        
        evaluation.write_results_xlsx_confmat(list(range(num_classes)), conf_mat, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'confmat_train_val_test')
        evaluation.write_results_xlsx(per_model_metrics, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'test_results')
        evaluation.write_results_classspecific(class_specific_metric_proto, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'classpecific_test_results')
        evaluation.write_results_classspecific(class_specific_metric_global, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'classpecific_test_results')
        evaluation.write_results_classspecific(class_specific_metric_both, log_dir+'/results'+'_'+str(randseedother)+'_'+str(randseeddata)+'.xlsx', 'classpecific_test_results')

    else:
        if num_classes > 2:
            auc = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob), average='macro', multi_class='ovo')
            auc_global = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob_global), average='macro', multi_class='ovo')
            auc_both = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob_both), average='macro', multi_class='ovo')

            if mode!='train':
                auc_wtmacro = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob), average='weighted', multi_class='ovo')
                auc_global_wtmacro = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob_global), average='weighted', multi_class='ovo')
                auc_both_wtmacro = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob_both), average='weighted', multi_class='ovo')
            else:
                auc_wtmacro = 0.0
                auc_global_wtmacro = 0.0
                auc_both_wtmacro = 0.0

        else:
            auc = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob)[:,1])
            auc_global = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob_global)[:,1])
            auc_both = metrics.roc_auc_score(np.array(y_trues), np.array(y_prob_both)[:,1])

            auc_wtmacro = 0.0
            auc_global_wtmacro = 0.0
            auc_both_wtmacro = 0.0
        
        info['auc_proto'] = auc
        info['auc_global'] = auc_global
        info['auc_both'] = auc_both

        info['auc_proto_wtmacro'] = auc_wtmacro
        info['auc_global_wtmacro'] = auc_global_wtmacro
        info['auc_both_wtmacro'] = auc_both_wtmacro

    return info, lrs

def knowledge_distillation_loss(y_true, y_prob_global, y_prob_protopnet, coefs):
    y_g = y_prob_global[torch.arange(y_true.shape[0]), y_true]
    y_l = y_prob_protopnet[torch.arange(y_true.shape[0]), y_true]

    kd_loss = torch.clamp(y_g-y_l+coefs['kd_loss_margin'], min=0)
    kd_loss = kd_loss.mean()
    return kd_loss

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
    #backbone
    for p in model.module.features.parameters():
        p.requires_grad = False
    #globalnet
    #for p in model.module.add_on_layers_globalnet.parameters():
    #    p.requires_grad = False
    for p in model.module.last_layer_globalnet.parameters():
        p.requires_grad = False
    #protopnet
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tlast layer')

#Stage 1
def train_backbone_globalnet_only(model, log=print):
    #backbone
    for p in model.module.features.parameters():
        p.requires_grad = True
    #globalnet
    #for p in model.module.add_on_layers_globalnet.parameters():
    #    p.requires_grad = True
    for p in model.module.last_layer_globalnet.parameters():
        p.requires_grad = True
    #protopnet
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = False
    
    log('\ttrain backbone and globalnet')

#Stage 2
def train_protopnet_only(model, log=print):
    #backbone
    for p in model.module.features.parameters():
        p.requires_grad = False
    #globalnet
    #for p in model.module.add_on_layers_globalnet.parameters():
    #    p.requires_grad = False
    for p in model.module.last_layer_globalnet.parameters():
        p.requires_grad = False
    #protopnet
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\ttrain protopnet')

#Stage 3
def joint(model, log=print):
    #backbone
    for p in model.module.features.parameters():
        p.requires_grad = True
    #globalnet
    #for p in model.module.add_on_layers_globalnet.parameters():
    #    p.requires_grad = True
    for p in model.module.last_layer_globalnet.parameters():
        p.requires_grad = True
    #protopnet
    for p in model.module.add_on_layers.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True
    
    log('\tjoint')
