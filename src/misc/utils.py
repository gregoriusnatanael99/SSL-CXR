import numpy as np
import torch
import os
import pandas as pd
import time
from sklearn.utils.class_weight import compute_class_weight
import json
import warnings

np.random.seed(42)
torch.manual_seed(42)

def create_exp_path(cfg_data):
    exp_name = ""
    exp_name += time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    exp_name +="_"
    exp_name += cfg_data['training']['train_mode']
        
    exp_path = "exp/{}/{}/{}".format(cfg_data['model']['backbone'], cfg_data['model']['tl_algo'], exp_name)
    os.makedirs(exp_path,exist_ok=True)
    print(exp_path+" directory created!")

    return exp_path

def create_part_results_exp_path(exp_path,hp_dict):
    exp_path = exp_path + "/"
    for i in hp_dict.keys():
        exp_path += hp_dict[i]+"_"
    os.makedirs(exp_path,exist_ok=True)
    return exp_path

def log_train_data(exp_path,df,cfg_dict,tl_algo,hp_dict=None,unfrozen_blocks=None):
    if hp_dict:
        warnings.warn("Usage of hp_dict will be deprecated in the future", FutureWarning)
    if unfrozen_blocks:
        warnings.warn("Usage of unfrozen blocks will be deprecated in the future", FutureWarning)
    print("Saving train-val loss data . . .")
    f_name = "train_val_loss_{}.csv".format(tl_algo)
    df.to_csv(os.path.join(exp_path, f_name),index=False)

    # with open(exp_path+"/config.txt",'w') as file:
    #     file.write('Epochs: {}\nOptimizer: {}\nLearning rate: {}\nWeight decay: {}\nUnfrozen blocks: {}\n'.format(hp_dict['epochs'], hp_dict['optimizer'],hp_dict['lr'], hp_dict['weight_decay'],unfrozen_blocks))
    #     file.write(f'\nModel: {tl_algo}')

    with open(exp_path+"/config.json",'w') as file:
        json.dump(cfg_dict,file)

def save_model_state_dict(exp_path,model):
    print("Saving model . . .")
    torch.save(model,os.path.join(exp_path,"best_model.pth"))

def calculate_class_weights(image_dataset):
    targets = [i[1] for i in image_dataset.samples]
    return list(compute_class_weight(class_weight='balanced',classes=np.unique(targets),y=targets))
    
def export_conf_mat(conf_mat,class_names,exp_path):
    res_df = pd.DataFrame(conf_mat.astype('int'))
    for i in range(len(res_df.columns)):
        res_df = res_df.rename(columns={res_df.columns[i]:"pred_"+class_names[i]}, 
                                        index={res_df.columns[i]:"gt_"+class_names[i]})
    return res_df

def log_test_data(exp_path,res_dict,res_df):
    print("Saving test results data . . .")
    res_df.to_csv(exp_path+"/evaluation_results.csv")
    with open(exp_path+"/evaluation_summary.txt",'w') as file:
        # file.write('Loss: {}\nAccuracy: {}\nMacro AUC: {}\nMacro Precision: {}\nMacro Recall: {}\nMacro F1: {}\nMicro AUC: {}\nMicro Precision: {}\nMicro Recall: {}\nMicro F1: {}'.format(res_dict['loss'], res_dict['acc'],res_dict['macro_auc'], res_dict['macro_prec'], res_dict['macro_rec'], res_dict['macro_f1'],res_dict['micro_auc'], res_dict['micro_prec'], res_dict['micro_rec'], res_dict['micro_f1']))
        file.write('Loss: {}\nAccuracy: {}\nMacro Precision: {}\nMacro Recall: {}\nMacro F1: {}\nMacro AUC: {}\n'.format(res_dict['loss'], res_dict['acc'], res_dict['macro_prec'], res_dict['macro_rec'], res_dict['macro_f1'], res_dict['macro_auc']))

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss