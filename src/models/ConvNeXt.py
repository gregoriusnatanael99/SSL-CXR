import torch
from torchvision import models
from torch import nn
from backbones.ConvNeXtV2 import ConvNeXtV2
import re

class ConvNeXt_Model(nn.Module):
    def __init__(self,model_cfg_data):
        """
        Creates the ConvNeXt model object

        """
        self.model = self.build_model_convnext(model_cfg_data['backbone_arch'],model_cfg_data['num_class'])
        print(model_cfg_data)
        
        # Freezing the blocks
        try:
            if model_cfg_data['unfrozen_blocks'] > 0:
                self.freeze_layers(model_cfg_data)
                print(f"{model_cfg_data['unfrozen_blocks']} convolution blocks frozen")
        except Exception as e:
            print(e)
            model_cfg_data['unfrozen_blocks'] = 0
    
    def forward(self,x):
        x = self.model(x)
        return x
    
    def freeze_layers(self,model_cfg_data):
        a_modules = [i for i in dict(self.model.named_modules()) if "stages" in i and re.search("[.][0-9]+[.][0-9]+$", i)]
        for i in range(len(a_modules) - model_cfg_data['UNFROZEN_BLOCKS'],len(a_modules)):
            for name,param in self.model.named_parameters():
                if a_modules[i] in name:
                    param.requires_grad = True

    def build_model_convnext(arch_cfg,weights,num_class,**kwargs):
        if arch_cfg == 'convnextv2-atto':
            model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            if weights == 'ssl':
                old_state_dict = torch.load('./pre-trained/ConvNeXt/v2/convnextv2_atto_1k_224_fcmae.pt')
                model.load_state_dict(old_state_dict['model'],strict=False)
            model.head = nn.Linear(num_ftrs, num_class)
        elif arch_cfg == 'convnextv2-femto':
            model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_ftrs, num_class)
        elif arch_cfg == 'convnextv2-pico':
            model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_ftrs, num_class)
        elif arch_cfg == 'convnextv2-nano':
            model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_ftrs, num_class)
        elif arch_cfg == 'convnextv2-tiny':
            model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_ftrs, num_class)
        elif arch_cfg == 'convnextv2-base':
            model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_ftrs, num_class)
        elif arch_cfg == 'convnextv2-large':
            model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_ftrs, num_class)
        elif arch_cfg == 'convnextv2-huge':
            model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
            num_ftrs = model.norm.normalized_shape[0]
            model.head = nn.Linear(num_ftrs, num_class)
        else:
            raise NameError("backbone_arch must be in ['convnextv2-atto', 'convnextv2-femto', 'convnextv2-pico', 'convnextv2-nano', 'convnextv2-tiny', 'convnextv2-base', 'convnextv2-large', 'convnextv2-huge']")
        return model
