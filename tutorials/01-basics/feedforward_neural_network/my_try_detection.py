#%%
import torch 
from torch import nn as nn 
from torchvision import models

# %%
frcnn = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)

# %%
vgg16 = models.vgg16(pretrained=False)

# %%
