import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import os
from misc.utils import *
from easydict import EasyDict as edict
from model_trainer import Model_Trainer

def construct_dataset(cfg):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET_MEAN, cfg.DATASET_STD)
        ]),
        'val': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cfg.DATASET_MEAN, cfg.DATASET_STD)
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(cfg.DATASET_DIR, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=cfg.BATCH_SIZE,
                                                shuffle=True, num_workers=cfg.NUM_WORKERS)
                for x in ['train', 'val']}

    return image_datasets, dataloaders

def map_config(raw_cfg):
    cfg = edict()
    cfg.DATASET_DIR = raw_cfg['dataset']['dir']
    cfg.BATCH_SIZE = raw_cfg['hp']['batch_size']
    cfg.NUM_WORKERS = raw_cfg['training']['num_workers']
    cfg.DATASET_MEAN = raw_cfg['dataset']['mean']
    cfg.DATASET_STD = raw_cfg['dataset']['std']
    cfg.WEIGHTING = raw_cfg['training']['class_weighting']
    cfg.SAVE_MODEL = raw_cfg['training']['save_model']

    if raw_cfg['training']['train_mode'] == 'grid_search':
        cfg['HP'] = raw_cfg['hp_configs']['grid-hp']
    elif raw_cfg['training']['train_mode'] == 'normal':
        cfg['HP'] = raw_cfg['hp_configs']['normal-hp']
    return cfg

@hydra.main(version_base=None, config_path='config', config_name='train-config')
def init_training(prog_cfg: DictConfig):
    raw_cfg = OmegaConf.to_container(prog_cfg,resolve=True)
    seed = raw_cfg['hp']['seed']

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    cudnn.benchmark = True

    cfg = map_config(raw_cfg)
    image_datasets, dataloaders = construct_dataset(cfg)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    print(class_names)
    cfg['num_class'] = len(class_names)
    print(cfg)


if __name__ == "__main__":
    init_training()