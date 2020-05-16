import torch

arr = torch.ones((2, 3))
print(arr.shape)
arr2 = torch.arange(20).unsqueeze(1)
print(arr2.shape)
print(arr.shape)