# SSL for Image Classification 

This program is used for Transfer Learning (TL) in training single-label image classification models. All codes are written in Python using PyTorch framework. Models pre-trained through Self-Supervised Learning are used here. Models supported:

|Architecture |Size  |Pre-training|Dataset|Notes|
|-------------| -----|-----------------|---|---|
|ResNet-50||SwAV|ImageNet|
|ResNet-50||VicReg|ImageNet|
|ResNet-50||Barlow Twins|ImageNet|
|ResNet-50||SimSiam|ImageNet|Requires local pth file|
|ViT|tiny, small, base|APS|ImageNet|Requires local pth file|
|ViT|small-14|DINOV2|ImageNet|
|ConvNeXtV2|Atto|SSL|ImageNet|Requires local pth file|

Local pth files can be saved at `./src/models/pre-trained/`

## Installing Pre-requisites and Basic Execution
```
pip install -r requirements.txt
python main.py
```

### For DINOV2
DINOV2 requires PyTorch>=2.3.0 so please make sure to install it. If you use conda:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
For more information, take a look at https://pytorch.org/get-started/locally/

### For APS
APS requires PyTorch==2.1.0, so you need a different environment from DINOV2. If you use conda:
```
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
For more information, take a look at https://pytorch.org/get-started/previous-versions/

## Using the Program
We used Hydra to enable flexible hyperparameter configuration. 
Example usage: `python main.py training.class_weighting=True training.train_mode='normal' hp.batch_size=8 hp_configs.normal-hp.lr=1e-3 hp_configs.normal-hp.epochs=20 model.unfrozen_blocks=1`

## File Structures

## Contributors:
1. Gregorius Natanael Elwirehardja (https://github.com/gregoriusnatanael99)
2. Farrel Alexander Tjan (https://github.com/XHYPEX)
3. Joselyn Setiawan (https://github.com/joselynss)
4. Steve Marcello Liem (https://github.com/steveee27)
5. Maria Linneke Adjie (https://github.com/linneke17)
6. Muhammad Edo Syahputra (https://github.com/edosyhptra)


