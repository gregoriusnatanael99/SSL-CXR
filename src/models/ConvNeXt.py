import torch
from torchvision import models
from torch import nn
from backbones.ConvNeXtV2 import ConvNeXtV2

class ConvNeXt_Model(nn.Module):
    def __init__(self,model_cfg_data):
        self.model = self.build_model_convnext(model_cfg_data['backbone_arch'])
        if model_cfg_data['tl_algo'] == "ssl":
            return 0
        return 0
    
    def forward(self,x):
        return 0
    
    def freeze_layers(self,model_cfg_data):
        a_modules = [i for i in dict(self.model.named_modules()) if "transformer" in i and "." in i and len(i) < 15]
        for i in range(len(a_modules) - model_cfg_data['UNFROZEN_BLOCKS'],len(a_modules)):
            for name,param in self.model.named_parameters():
                if a_modules[i] in name:
                    param.requires_grad = True

    def build_model_convnext(arch_cfg,weights,**kwargs):
        if arch_cfg == 'convnextv2-atto':
            model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
            if weights == 'ssl':
                old_state_dict = torch.load('./pre-trained/ConvNeXt/v2/convnextv2_atto_1k_224_fcmae.pt')
                model.load_state_dict(old_state_dict['model'])
        elif arch_cfg == 'convnextv2-femto':
            model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
        elif arch_cfg == 'convnextv2-pico':
            model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
        elif arch_cfg == 'convnextv2-nano':
            model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
        elif arch_cfg == 'convnextv2-tiny':
            model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
        elif arch_cfg == 'convnextv2-base':
            model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
        elif arch_cfg == 'convnextv2-large':
            model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
        elif arch_cfg == 'convnextv2-huge':
            model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
        else:
            raise NameError("backbone_arch must be in ['convnextv2-atto', 'convnextv2-femto', 'convnextv2-pico', 'convnextv2-nano', 'convnextv2-tiny', 'convnextv2-base', 'convnextv2-large', 'convnextv2-huge']")
        return model
