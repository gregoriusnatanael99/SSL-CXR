from cmath import exp
from operator import index
import numpy as np
import torch
import os
import pandas as pd
import time
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(1234)
torch.manual_seed(1234)

def create_exp_path(cfg_data):
    exp_name = ""
    exp_name += time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    exp_name +="_"
    exp_name += cfg_data.TRAIN_MODE
        
    exp_path = "exp/{}/{}".format(cfg_data['TL_ALGO'], exp_name)
    os.makedirs(exp_path,exist_ok=True)
    print(exp_path+" directory created!")

    return exp_path

def create_part_results_exp_path(exp_path,hp_dict):
    exp_path = exp_path + "/"
    for i in hp_dict.keys():
        exp_path += hp_dict[i]+"_"
    os.makedirs(exp_path,exist_ok=True)
    return exp_path

def log_train_data(exp_path,df,tl_algo,hp_dict,unfrozen_blocks):
    print("Saving train-val loss data . . .")
    f_name = "train_val_loss_{}.csv".format(tl_algo)
    df.to_csv(os.path.join(exp_path, f_name),index=False)

    with open(exp_path+"/config.txt",'w') as file:
        file.write('Epochs: {}\nOptimizer: {}\nLearning rate: {}\nWeight decay: {}\nUnfrozen blocks: {}\n'.format(hp_dict['epochs'], hp_dict['optimizer'],hp_dict['lr'], hp_dict['weight_decay'],unfrozen_blocks))
        file.write(f'\nModel: {tl_algo}')

def save_model_state_dict(exp_path,model):
    print("Saving model . . .")
    torch.save(model,os.path.join(exp_path,"best_model.pth"))

def calculate_class_weights(image_dataset):
    targets = [i[1] for i in image_dataset.samples]
    return list(compute_class_weight(class_weight='balanced',classes=np.unique(targets),y=targets))
    
def export_conf_mat(conf_mat,class_names,exp_path):
    res_df = pd.DataFrame(conf_mat.numpy().astype('int'))
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
from cmath import exp
from operator import index
import numpy as np
import torch
import os
import pandas as pd
import time
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(1234)
torch.manual_seed(1234)

def create_exp_path(cfg_data):
    exp_name = ""
    exp_name += time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    exp_name +="_"
    exp_name += cfg_data.TRAIN_MODE
        
    exp_path = "exp/{}/{}".format(cfg_data['TL_ALGO'], exp_name)
    os.makedirs(exp_path,exist_ok=True)
    print(exp_path+" directory created!")

    return exp_path

def create_part_results_exp_path(exp_path,hp_dict):
    exp_path = exp_path + "/"
    for i in hp_dict.keys():
        exp_path += hp_dict[i]+"_"
    os.makedirs(exp_path,exist_ok=True)
    return exp_path

def log_train_data(exp_path,df,tl_algo,hp_dict,unfrozen_blocks):
    print("Saving train-val loss data . . .")
    f_name = "train_val_loss_{}.csv".format(tl_algo)
    df.to_csv(os.path.join(exp_path, f_name),index=False)

    with open(exp_path+"/config.txt",'w') as file:
        file.write('Epochs: {}\nOptimizer: {}\nLearning rate: {}\nWeight decay: {}\nUnfrozen blocks: {}\n'.format(hp_dict['epochs'], hp_dict['optimizer'],hp_dict['lr'], hp_dict['weight_decay'],unfrozen_blocks))
        file.write(f'\nModel: {tl_algo}')

def save_model_state_dict(exp_path,model):
    print("Saving model . . .")
    torch.save(model,os.path.join(exp_path,"best_model.pth"))

def calculate_class_weights(image_dataset):
    targets = [i[1] for i in image_dataset.samples]
    return list(compute_class_weight(class_weight='balanced',classes=np.unique(targets),y=targets))
    
def export_conf_mat(conf_mat,class_names,exp_path):
    res_df = pd.DataFrame(conf_mat.numpy().astype('int'))
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
