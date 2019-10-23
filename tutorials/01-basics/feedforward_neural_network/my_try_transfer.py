#%%
import torch
import torch.nn as nn
from torchvision import models

#%%
mbv = models.mobilenet_v2(pretrained=True)

#%%
torch.save(mbv,"mvb.pth")

#%%
