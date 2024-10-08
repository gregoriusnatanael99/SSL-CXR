import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from src.misc.utils import *
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK, STATUS_FAIL
import copy
import warnings
from tqdm import tqdm

class Model_Trainer():
    def __init__(self,cfg_data,dataloaders,dataset_sizes):
        self.cfg_data = cfg_data
        self.best_model = None
        self.dataloaders = dataloaders
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_sizes = dataset_sizes
        if cfg_data['training']['save_model']:
            self.exp_path = create_exp_path(self.cfg_data)
        else:
            warnings.warn("Model is not saved!")
    
    def begin_training(self):
        self.global_best_loss = np.inf
        self.global_best_model = None
        if self.cfg_data['training']['train_mode']== "normal":
            print("Initiating training with defined hyperparameters. . .")
            best_params = self.train_model(self.cfg_data["train_hp"])
        elif self.cfg_data['training']['train_mode'] == "grid_search":
            print("Initiating training with hyperparameter grid search. . .")
            bayes_trials = Trials()
            best_params = fmin(
                self.train_model,
                space=self.construct_hp_search_space(),
                algo=tpe.suggest,
                max_evals=self.gs_calculate_max_evals(),
                # max_evals=20,
                trials=bayes_trials,
                rstate=np.random.default_rng(self.cfg_data['hp']['seed'])
            )
        else:
            print("Training mode unrecognized, please check config.py!")
        print(best_params)
        print("Training finished!")

    def train_model(self,params):
        print(params)
        print("Training model . . .")
        self.model = self.initialize_model()
        self.model.to(self.device)

        criterion = nn.CrossEntropyLoss()
        try:
            if self.cfg_data['training']['class_weights']:
                setattr(criterion,'weight',torch.Tensor(self.cfg_data['training']['class_weights']).to(self.device))
                print("Class weights set!")
        except:
            print("Not using class weights")

        if params['optimizer'] == "adam":
            optimizer = optim.Adam(self.model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        since = time.time()

        best_model = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf
        best_acc = 0.0
        history_dict = {'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}

        if self.cfg_data['hp']['early_stopping']:
            early_stopping = EarlyStopping(patience=self.cfg_data['hp']['patience'], verbose=True)

        for epoch in range(params['epochs']):
            print(f"Epoch {epoch+1}/{params['epochs']}")
            print('-' * 20)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
#                        print(outputs)
#                        preds = np.argmax(outputs.cpu().tolist())
#                        print(labels)
                        loss = criterion(outputs, labels)
#                        print(preds,labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            # print(self.model.model.fc.weight.grad)
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    exp_lr_scheduler.step()

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_acc = epoch_acc
                    best_epoch = epoch
                    best_model = copy.deepcopy(self.model.state_dict())
                      
                if phase == 'train':
                    history_dict['train_loss'].append(epoch_loss)
                    history_dict['train_acc'].append(epoch_acc.cpu().numpy())
#                    history_dict['train_acc'].append(epoch_acc)
                elif phase == 'val':
                    history_dict['val_loss'].append(epoch_loss)
                    history_dict['val_acc'].append(epoch_acc.cpu().numpy())
#                    history_dict['val_acc'].append(epoch_acc)
                
                    # Early stopping
                    early_stopping(epoch_loss, self.model)
            if early_stopping.early_stop:
                print(f"Early stopping after {epoch+1} epochs")
                break

        # print(history_dict)
        if self.overwrite_best_model(best_loss,best_model):
            # part_exp_path = create_part_results_exp_path(self.exp_path,params)
            train_val_df = pd.DataFrame.from_dict(history_dict)
            log_train_data(self.exp_path,train_val_df,self.cfg_data,self.cfg_data['model']['tl_algo'])
            if self.cfg_data['training']['save_model']:
                save_model_state_dict(self.exp_path,best_model)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Loss: {best_loss:4f}')
        print(f'Best val Acc: {best_acc:4f}')
        
        return {'loss':best_loss,'acc':best_acc,'params':params,'status': STATUS_OK}

    def gs_calculate_max_evals(self):
        print("Calculating max evals . . .")
        num_evals = 1
        for i in self.cfg_data['train_hp'].keys():
            num_evals*=len(self.cfg_data['train_hp'][i])
        return num_evals
    
    def construct_hp_search_space(self):
        print("Building hyperparameter search space . . .")
        search_space = {}
        for i in self.cfg_data['train_hp'].keys():
            search_space[i] = hp.choice(i,self.cfg_data['train_hp'][i])
        return search_space

    def initialize_model(self):
        torch.manual_seed(self.cfg_data['hp']['seed'])
        if self.cfg_data['model']['backbone'] == "resnet50":
            from src.models.ResNet50 import ResNet50_Model as net
            return net(self.cfg_data['model'])
        elif self.cfg_data['model']['backbone'] == "densenet121":
            from src.models.DenseNet121 import DenseNet121_Model as net
            return net(self.cfg_data['model'])
        elif self.cfg_data['model']['backbone'] == "vit":
            from src.models.VisionTransformer import ViT_Model as net
            return net(self.cfg_data['model'])
        elif self.cfg_data['model']['backbone'] == "convnextv2":
            from src.models.ConvNeXt import ConvNeXt_Model as net
            return net(self.cfg_data['model'])
        else:
            raise TypeError(self.cfg_data['model']['backbone'])

    def overwrite_best_model(self,curr_best_loss,curr_model):
        if curr_best_loss < self.global_best_loss:
            self.global_best_loss = curr_best_loss
            self.global_best_model = curr_model
            return True
        return False
