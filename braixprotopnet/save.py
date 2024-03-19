import os
import torch
import numpy as np

class ModelCheckpoint:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path_to_model, criteria, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            path_to_model: The folder in your computer where the model gets saved
        """
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.conf_mat_train_best=np.zeros((2,2))
        self.conf_mat_test_best=np.zeros((2,2))
        self.path_to_model=path_to_model
        self.criteria = criteria

    def __call__(self, val_loss, model, conf_mat_train1, conf_mat_test1, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, val_auc):

        if self.criteria == 'loss':        
            score = -val_loss
        elif self.criteria == 'auc' or self.criteria == 'auc_wtmacro':
            score = val_auc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, conf_mat_train1, conf_mat_test1, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, val_auc)
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model, conf_mat_train1, conf_mat_test1, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, val_auc)

    def save_checkpoint(self, val_loss, model, conf_mat_train1, conf_mat_test1, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, val_auc):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        state = {
            'epoch_stage1': epoch_stage1,
            'epoch_stage2': epoch_stage2,
            'epoch_joint': epoch_joint,
            'model_state_dict': model.state_dict(),
            'optimizer_stage1': stage1_optimizer.state_dict(),
            'optimizer_stage2': stage2_optimizer.state_dict(),
            'optimizer_joint': joint_optimizer.state_dict(),
            #'optimizer_joint_lastlayer': joint_optimizer[1].state_dict(),
            'auc': val_auc
        }
        torch.save(state,self.path_to_model)
        #torch.save(model.state_dict(), 'checkpoint.pt')
        self.conf_mat_train_best=conf_mat_train1
        self.conf_mat_test_best=conf_mat_test1
        self.val_loss_min = val_loss

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def save_model(model, stage1_optimizer, stage2_optimizer, joint_optimizer, epoch_stage1, epoch_stage2, epoch_joint, val_auc, path_to_model):
    state = {
        'epoch_stage1': epoch_stage1,
        'epoch_stage2': epoch_stage2,
        'epoch_joint': epoch_joint,
        'model_state_dict': model.state_dict(),
        'optimizer_stage1': stage1_optimizer.state_dict(),
        'optimizer_stage2': stage2_optimizer.state_dict(),
        'optimizer_joint': joint_optimizer.state_dict(),
        #'optimizer_joint_lastlayer': joint_optimizer[1].state_dict(),
        'auc': val_auc
    }
    torch.save(state, path_to_model)