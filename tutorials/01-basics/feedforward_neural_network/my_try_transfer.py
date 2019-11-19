#%%
import torch
import torch.nn as nn
from torchvision import models

#%%
vgg16 = models.vgg16(pretrained=True)
class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.vgg = vgg16.features[:16]
        
    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for layer in self.vgg:
            x = layer(x)
            features.append(x)
        return x


#%%
my_vgg = VGGNet()
torch.save(my_vgg,"vgg_b3_c3.pth")

#%%
