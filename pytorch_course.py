mport torch
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)

x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")

x_rand.dtype
x_data.dtype
x_ones.dtype


tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

tensor.to('cuda')

tensor = torch.ones(4, 4)
print('First row: ',tensor[1])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1])
tensor[1,1] = 4
tensor[2] = 2
print(tensor)

t1 = torch.cat([tensor, tensor, tensor*2], dim=)
print(t1)


# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

agg = tensor.sum()
agg_item = agg.item()  
print(agg_item, type(agg_item))



#################### proyecto completo


# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)