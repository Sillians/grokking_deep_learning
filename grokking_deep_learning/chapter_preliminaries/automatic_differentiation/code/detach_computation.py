import torch

x = torch.arange(4.0, requires_grad=True)
print(x)

y = x * x
print(y)

u = y.detach()
print(u)

z = u * x
print(z)

z.sum().backward()
print(x.grad == u)


# Gradient of y with respect to x
x.grad.zero_()
y.sum().backward()
print(x.grad == 2 * x)

