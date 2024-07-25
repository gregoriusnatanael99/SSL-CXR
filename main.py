import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import os
# from easydict import EasyDict as edict

from src.misc.utils import *
from src.model_trainer import Model_Trainer
from datetime import datetime as dt

def construct_dataset(cfg_ds,batch_size,num_workers):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(cfg_ds['img_size']),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfg_ds['mean'], cfg_ds['std'])
        ]),
        'val': transforms.Compose([
            transforms.Resize(cfg_ds['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg_ds['mean'], cfg_ds['std'])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(cfg_ds['dir'], x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
                for x in ['train', 'val']}

    return image_datasets, dataloaders

def lower_cfg_vals(cfg):
    cfg['training']['train_mode'] = cfg['training']['train_mode'].lower()
    cfg['model']['backbone'] = cfg['model']['backbone'].lower()
    cfg['model']['backbone_arch'] = cfg['model']['backbone_arch'].lower()
    cfg['model']['tl_algo'] = cfg['model']['tl_algo'].lower()
    return cfg

@hydra.main(version_base=None, config_path='config', config_name='train-config')
def init_training(prog_cfg: DictConfig):
    cfg = OmegaConf.to_container(prog_cfg,resolve=True)
    seed = cfg['hp']['seed']

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    cudnn.benchmark = True
    print(torch.cuda.is_available())
    # cfg = map_config(raw_cfg)
    image_datasets, dataloaders = construct_dataset(cfg['dataset'],batch_size=cfg['hp']['batch_size'],num_workers=cfg['training']['num_workers'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(f'Training on classes: {class_names}')
    cfg['model']['num_class'] = len(class_names)

    cfg = lower_cfg_vals(cfg)
    
    if cfg['training']['class_weighting']:
        cfg['training']['class_weights'] = calculate_class_weights(image_datasets['train'])
    #    cfg['class_weights'] = [0.01,0.01,0.01,10]
        print(f"Using class weights: {cfg['training']['class_weights']}")

    if cfg['training']['train_mode'] == 'grid_search':
        cfg['train_hp'] = cfg['hp_configs']['grid-hp']
    elif cfg['training']['train_mode'] == 'normal':
        cfg['train_hp'] = cfg['hp_configs']['normal-hp']
    
    del cfg['hp_configs']

    print(cfg)
    trainer = Model_Trainer(cfg,dataloaders,dataset_sizes)
    trainer.begin_training()

if __name__ == "__main__":
    start = dt.now()
    init_training()
    end = dt.now()
    print(f"Training finished in {end-start}")