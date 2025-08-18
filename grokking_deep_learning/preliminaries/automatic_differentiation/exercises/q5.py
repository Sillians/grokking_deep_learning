import torch

# Input tensor
x = torch.tensor([2.0, 3.0])

# Step-by-step computation
a = x ** 2                     # a = x^2
b = torch.log(a)                # b = log(a)
c = torch.sin(x)                # c = sin(x)
d = b * c                       # d = b * c
e = x ** (-1)                   # e = x^{-1}
f = d + e                       # f = d + e

print(f)  # Output tensor