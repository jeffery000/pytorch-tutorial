#%%
import torch

#%% 反向传播试验
a = torch.ones((2,2))
print(a.requires_grad) #默认是false
a.requires_grad_(True)
print(a.requires_grad)
b = a*2
#如果上面是false，这里是none
print(b.grad_fn)  
c = b.mean()
c.backward()
print(a.grad)

# %%
