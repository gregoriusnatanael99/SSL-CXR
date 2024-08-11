import torch
from torchvision import models
from torch import nn
from .backbones.VisionTransformer_APS import build_model_aps


class ViT_Model(nn.Module):
    def __init__(self, model_cfg_data):
        super(ViT_Model, self).__init__()
        if model_cfg_data["tl_algo"] == "aps":
            self.model = build_model_aps(model_cfg_data["backbone_arch"])
        elif model_cfg_data["tl_algo"] == "dinov2":
            self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")

        for param in self.model.parameters():
            param.requires_grad = False

        if hasattr(self.model, "fc"):
            num_ftrs = self.model.fc.in_features
            self.fc_exists = True
        else:
            self.fc_exists = False
            num_ftrs = 384

        self.model.fc = nn.Linear(num_ftrs, model_cfg_data["num_class"])
        print(model_cfg_data)
        try:
            if model_cfg_data["unfrozen_blocks"] > 0:
                self.freeze_layers(model_cfg_data)
                print(f"{model_cfg_data['unfrozen_blocks']} transformer blocks unfrozen")
        except Exception as e:
            print(e)
            model_cfg_data["unfrozen_blocks"] = 0

    def forward(self, x):
        x = self.model(x)
        if not self.fc_exists:
            x = self.model.fc(x)
        # x = nn.functional.softmax(x,dim=1)
        return x

    def freeze_layers(self, model_cfg_data):
        a_modules = [
            i
            for i in dict(self.model.named_modules())
            if "transformer" in i and "." in i and len(i) < 15
        ]
        for i in range(
            len(a_modules) - model_cfg_data["unfrozen_blocks"], len(a_modules)
        ):
            for name, param in self.model.named_parameters():
                if a_modules[i] in name:
                    print(name)
                    param.requires_grad = True
