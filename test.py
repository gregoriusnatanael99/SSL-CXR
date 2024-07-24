import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, transforms
import os

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

@hydra.main(version_base=None, config_path='config', config_name='test-config')
def init_testing(prog_cfg: DictConfig):
    cfg = OmegaConf.to_container(prog_cfg,resolve=True)
    seed = cfg['hp']['seed']

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    
    cudnn.benchmark = True
    print(torch.cuda.is_available())
    image_datasets, dataloaders = construct_dataset(cfg['dataset'],batch_size=cfg['hp']['batch_size'],num_workers=cfg['testing']['num_workers'])
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['test'].classes
