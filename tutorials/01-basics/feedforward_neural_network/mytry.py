#%%
import torch
import torch.nn as nn
from torchviz import make_dot

#%%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10,5)
        self.fc2 = nn.Linear(5,2)
        self.relu1 = nn.ReLU()
    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        return x
#%%
n = Net()
input_x = torch.randn(10)
o = n(input_x)

#%%
print(n)
#%%保存模型+参数
torch.save(n,"torch_model.pth")

#%%
g = make_dot(o)
g.render('espnet_model', view=False)