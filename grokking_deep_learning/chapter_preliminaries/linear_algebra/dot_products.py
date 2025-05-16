import torch

# Dot product is one of the fundamental operations in deep learning.
# Given two vectors `x, y`, their dot product (also known as inner product <x, y>) is the sum over the product of the elements at the same position.

y = torch.ones(3, dtype=torch.float32)
x = torch.arange(3, dtype=torch.float32)
print(x)
print(y)

dot_prod = torch.dot(x, y)
print(dot_prod)


# Calculate the dot product of two vectors by performing an elementwise multiplication followed by a sum:
print(torch.sum(x * y))