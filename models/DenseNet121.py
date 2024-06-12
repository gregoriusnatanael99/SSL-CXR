<<<<<<< HEAD
from ast import Try
from operator import mod
import torch
from torchvision import models
from torch import nn

class DenseNet121_Model(nn.Module):
    def __init__(self,model_cfg_data):
        super(DenseNet121_Model, self).__init__()
        self.model = models.densenet121(pretrained=True)

        if hasattr(self.model,"classifier"):
            num_ftrs = self.model.classifier.in_features
            self.fc_exists = True
        else:
            num_ftrs = 1024
            self.fc_exists = False
        self.model.classifier = nn.Linear(num_ftrs, model_cfg_data['num_class'])

    def forward(self,x):
        x = self.model(x)
        if not self.fc_exists:
            x = self.model.classifier(x)
        x = nn.functional.softmax(x,dim=1)
=======
from ast import Try
from operator import mod
import torch
from torchvision import models
from torch import nn

class DenseNet121_Model(nn.Module):
    def __init__(self,model_cfg_data):
        super(DenseNet121_Model, self).__init__()
        self.model = models.densenet121(pretrained=True)

        if hasattr(self.model,"classifier"):
            num_ftrs = self.model.classifier.in_features
            self.fc_exists = True
        else:
            num_ftrs = 1024
            self.fc_exists = False
        self.model.classifier = nn.Linear(num_ftrs, model_cfg_data['num_class'])

    def forward(self,x):
        x = self.model(x)
        if not self.fc_exists:
            x = self.model.classifier(x)
        x = nn.functional.softmax(x,dim=1)
>>>>>>> fe029da0e2d9f65aaa0ee55f0b45fe7781c008ce
        return x