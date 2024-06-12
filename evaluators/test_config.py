
import os
from easydict import EasyDict as edict
import time
import torch

__C = edict()
cfg = __C

__C.SEED = 1234
__C.MODEL_ARCH = 'resnet50'
__C.MODEL_DIR = 'exp/swav/22-06-16_14-41-57_normal/'
__C.MODEL_NAME = "best_model.pth"
__C.GPU_ID = [0]    
__C.BATCH_SIZE = 8
__C.NUM_WORKERS = 4
__C.DATASET_DIR = "../preprocessed_data/"
__C.DATASET_MEAN = [0.5094484686851501, 0.5094484686851501, 0.5094484686851501]
__C.DATASET_STD = [0.2523978352546692, 0.2523978352546692, 0.2523978352546692]

__C.EXP_OUTPUT = True #save outputs if True
__C.OUTPUTS = [
    'results',
    'roc',
    'gradcam',
]
import os
from easydict import EasyDict as edict
import time
import torch

__C = edict()
cfg = __C

__C.SEED = 1234
__C.MODEL_ARCH = 'resnet50'
__C.MODEL_DIR = 'exp/swav/22-06-16_14-41-57_normal/'
__C.MODEL_NAME = "best_model.pth"
__C.GPU_ID = [0]    
__C.BATCH_SIZE = 8
__C.NUM_WORKERS = 4
__C.DATASET_DIR = "../preprocessed_data/"
__C.DATASET_MEAN = [0.5094484686851501, 0.5094484686851501, 0.5094484686851501]
__C.DATASET_STD = [0.2523978352546692, 0.2523978352546692, 0.2523978352546692]

__C.EXP_OUTPUT = True #save outputs if True
__C.OUTPUTS = ['results','roc','gradcam','precision_recall'] #results, roc, gradcam, precision_recall