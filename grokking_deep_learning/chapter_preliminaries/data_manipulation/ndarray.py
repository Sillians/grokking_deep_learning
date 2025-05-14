import torch

array1 = torch.arange(12, dtype=torch.float32)
print(array1)

# inspect total number of elements in a tensor
print(array1.numel())

# access the shape (length along each axis) of the tensor
print(array1.shape)
print(array1.size())

# change the shape of the tensor without altering its size or values (reconfigures the vector into a matrix)
new_array1 = array1.reshape(3, 4)
print(new_array1)

# to automatically infer one component of the shape, we can place a `-1` for the shape component that should be inferred automatically
infer_array1 = array1.reshape(-1, 4)
print(infer_array1)

#---- OR
inferr_array1 = array1.reshape(3, -1)
print(inferr_array1)


# construct a tensor with all elements set to `0` and a shape (2, 3, 4) via the zeros function.
zeros_tensor = torch.zeros((2,3,4))
print(zeros_tensor)


# create a tensor with all `1s` by invoking ones
ones_tensor = torch.ones(2, 3, 4)
print(ones_tensor)


# creates a tensor with elements drawn from standard Gaussian (normal) distribution with mean `0` and standard deviation `1`.
normal_dist = torch.randn(3, 4)
print(normal_dist)


# construct a matrix with a list of lists, where the outermost list corresponds to axis `0`, and the inner list corresponds to axis `1`.
array_llist = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(array_llist)
























