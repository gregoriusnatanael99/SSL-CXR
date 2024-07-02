import torch
from torchvision import models
from torch import nn
from backbones.VisionTransformer_APS import Vit

class ConvNeXt_Model(nn.Module):
    def __init__(self,model_cfg_data):
        
        return 0
    
    def forward(self,x):
        return 0
    
    def freeze_layers(self,model_cfg_data):
        a_modules = [i for i in dict(self.model.named_modules()) if "transformer" in i and "." in i and len(i) < 15]
        for i in range(len(a_modules) - model_cfg_data['UNFROZEN_BLOCKS'],len(a_modules)):
            for name,param in self.model.named_parameters():
                if a_modules[i] in name:
                    param.requires_grad = True