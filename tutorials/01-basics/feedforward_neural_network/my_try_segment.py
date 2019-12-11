#%%
import torch
from torchvision.models import segmentation
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#%%
fcn_res101 = segmentation.fcn_resnet101(pretrained=True)
fcn_res101.to(device)
#%%
print(fcn_res101)

# %%
img = Image.open("1.jpg")
img = np.array(img)
img_tensor = torch.from_numpy(img.astype(np.float32))
img_tensor = torch.unsqueeze(img_tensor,0)
img_tensor = img_tensor.permute(0,3,1,2)
img_tensor = img_tensor.to(device)
#%%
predict = fcn_res101(img_tensor)
#%%
pre_numpy = predict['out'].detach().numpy()
pre_numpy_person = pre_numpy[0,15,:,:]
plt.imshow(pre_numpy_person)
plt.show()
