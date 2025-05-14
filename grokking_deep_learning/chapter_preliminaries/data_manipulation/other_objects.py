import torch
import numpy

# Converting to a Numpy tensor (ndarray), or vice versa. The torch tensor and Numpy array will share their underlying memory.
X = torch.tensor([1.0, 2, 4, 8])
A = X.numpy()
B = torch.from_numpy(A)

print(type(A), type(B))

# convert a size-1 tensor to a Python scalar
a = torch.tensor([5.5])
print(a)
print(a.item())
print(float(a))
print(int(a))
