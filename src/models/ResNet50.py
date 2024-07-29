import torch
from torchvision import models
from torch import nn


class ResNet50_Model(nn.Module):
    def __init__(self, model_cfg_data):
        super(ResNet50_Model, self).__init__()
        if model_cfg_data["tl_algo"] == "swav":
            self.model = torch.hub.load("facebookresearch/swav:main", "resnet50")
            for param in self.model.parameters():
                param.requires_grad = False
        elif model_cfg_data["tl_algo"] == "vicreg":
            self.model = torch.hub.load("facebookresearch/vicreg:main", "resnet50")
            for param in self.model.parameters():
                param.requires_grad = False
            # no fc
        elif model_cfg_data["tl_algo"] == "barlow_twins":
            self.model = torch.hub.load("facebookresearch/barlowtwins:main", "resnet50")
            for param in self.model.parameters():
                param.requires_grad = False
        elif model_cfg_data["tl_algo"] == "simsiam":
            self.model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
            # load locally
            checkpoint = torch.load("./models/simsiam/model_best.pth.tar")
            for name, param in self.model.named_parameters():
                for i in list(checkpoint["state_dict"].keys()):
                    if "module." + name == i:
                        param.data = checkpoint["state_dict"]["module." + name]
            for param in self.model.parameters():
                param.requires_grad = False
        elif model_cfg_data["tl_algo"] == "moco":
            self.model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
            # load locally
            checkpoint = torch.load("./models/moco/moco_v2_800ep_pretrain.pth.tar")
            for name, param in self.model.named_parameters():
                for i in list(checkpoint["state_dict"].keys()):
                    if "module.encoder_q." + name == i:
                        param.data = checkpoint["state_dict"][
                            "module.encoder_q." + name
                        ]
            for param in self.model.parameters():
                param.requires_grad = False
        elif model_cfg_data["tl_algo"] == "supervised":
            self.model = models.resnet50(weights="ResNet50_Weights.DEFAULT")
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.model = models.resnet50(weights=None)

        if hasattr(self.model, "fc"):
            num_ftrs = self.model.fc.in_features
            self.fc_exists = True
        else:
            num_ftrs = 2048
            self.fc_exists = False

        self.model.fc = nn.Linear(num_ftrs, model_cfg_data["num_class"])
        # self.model.fc = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(num_ftrs,128),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(128,model_cfg_data['num_class']),
#            nn.Softmax(dim=1)
        # )

        try:
            if model_cfg_data["unfrozen_blocks"] > 0:
                self.freeze_layers(model_cfg_data)
        except Exception as e:
            print(e)
            model_cfg_data["unfrozen_blocks"] = 0

    def forward(self, x):
        x = self.model(x)
        if not self.fc_exists:
            x = self.model.fc(x)
        x = nn.functional.softmax(x, dim=1)
        return x

    def freeze_layers(self, model_cfg_data):
        a_modules = [
            i for i in dict(self.model.named_modules()) if "layer" in i and "." not in i
        ]
        for i in range(
            len(a_modules) - model_cfg_data["unfrozen_blocks"], len(a_modules)
        ):
            for name, param in self.model.named_parameters():
                if a_modules[i] in name:
                    print(name)
                    param.requires_grad = True

    def build_hidden_dense(self, in_features:int, hidden_dim:list=[], dropout:float=0.2):
        for i in range(hidden_dim):
            print(i)
        return 0
