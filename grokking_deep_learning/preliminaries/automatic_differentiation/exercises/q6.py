import torch

# Input tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)

# Forward pass (same as before)
a = x ** 2                     # a = x^2
b = torch.log(a)               # b = log(a)
c = torch.sin(x)               # c = sin(x)
d = b * c                      # d = b * c
e = x ** (-1)                  # e = x^{-1}
f = d + e                      # f = d + e

# Backward pass (autodifferentiation)
f.backward(torch.ones_like(f)) # Compute gradients (df/dx)

# Gradient of f w.r.t. x
print("Gradient (df/dx):", x.grad)