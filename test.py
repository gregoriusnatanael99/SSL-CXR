import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import os
import json
from datetime import datetime as dt

from src.model_evaluator import Model_Evaluator

def construct_dataset(cfg_ds,batch_size,num_workers):
    data_transforms = {
        'test': transforms.Compose([
            transforms.Resize(cfg_ds['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(cfg_ds['mean'], cfg_ds['std'])
        ])
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(cfg_ds['dir'], x),
                                          data_transforms[x])
                  for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=num_workers)
                for x in ['test']}

    return image_datasets, dataloaders

def lower_cfg_vals(cfg):
    cfg['model']['backbone'] = cfg['model']['backbone'].lower()
    cfg['model']['backbone_arch'] = cfg['model']['backbone_arch'].lower()
    cfg['model']['tl_algo'] = cfg['model']['tl_algo'].lower()
    return cfg

def overwrite_cfg(old_cfg:dict, loaded_cfg:dict):
    old_cfg['model']['backbone'] = loaded_cfg['model']['backbone'].lower()
    old_cfg['model']['backbone_arch'] = loaded_cfg['model']['backbone_arch'].lower()
    old_cfg['model']['tl_algo'] = loaded_cfg['model']['tl_algo'].lower()
    old_cfg['model']['unfrozen_blocks'] = loaded_cfg['model']['unfrozen_blocks']
    return old_cfg

@hydra.main(version_base=None, config_path='config', config_name='test-config')
def init_testing(prog_cfg: DictConfig):
    print("Begin testing . . .")
    cfg = OmegaConf.to_container(prog_cfg,resolve=True)
    seed = cfg['hp']['seed']

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    cudnn.benchmark = True
    print(torch.cuda.is_available())
    image_datasets, dataloaders = construct_dataset(cfg['dataset'],batch_size=cfg['hp']['batch_size'],num_workers=cfg['testing']['num_workers'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    class_names = image_datasets['test'].classes

    print(f'Testing on classes: {class_names}')
    cfg['model']['num_class'] = len(class_names)
    cfg['class_names'] = class_names

    try:
        with open(os.path.join(cfg['model']['exp_folder'],'config.json')) as json_file:
            data = json.load(json_file)
            cfg = overwrite_cfg(cfg,data)
    except:
        cfg = lower_cfg_vals(cfg)
    
    print(cfg)
    evaluator = Model_Evaluator(cfg,dataloaders,dataset_sizes)
    evaluator.begin_testing()

if __name__ == "__main__":
    start = dt.now()
    init_testing()
    end = dt.now()
    print(f"Testing finished in {end-start}")