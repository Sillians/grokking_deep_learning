import torch

# create a tensor with gradient tracking
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()
print(x, y, z)

print(f"Before detach - y.requires_grad: {y.requires_grad}")
print(f"y.grad_fn: {y.grad_fn}")

# Detach y from computational graph
y_detached = y.detach()

print(f"\nAfter detach - y_detached.requires_grad: {y_detached.requires_grad}")
print(f"y_detached.grad_fn: {y_detached.grad_fn}")

# Original tensor still tracks gradients z.backward()
print(f"\nx.grad after backward: {x.grad}")

# Using detached tensor won't track gradients
w = y_detached.sum()