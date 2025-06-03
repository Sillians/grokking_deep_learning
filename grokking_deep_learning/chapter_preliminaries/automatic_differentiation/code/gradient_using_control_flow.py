import torch

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c


# Call the function passing a random value as input.
# When we execute f(a) on a specific input, we realize a specific computational graph and can subsequently run backward
a = torch.randn(size=(), requires_grad=True)
print(a)


# what would happen if we changed the variable "a" to a random vector or a matrix?
# a = torch.randn(20, requires_grad=True).reshape(5, -1)
# print(a)

# a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
# print(a)


d = f(a)
d.backward()
print(a.grad)
print(a.grad == d / a)



# Example 2 (Gradients Through Conditionals)
x = torch.tensor([2.0], requires_grad=True)

def f(x):
    if x.item() > 1:
        return x * x
    else:
        return x + 2

y = f(x)
y.backward()
print(x.grad)  # Output: tensor([4.]) since f(x) = x², and d(x²)/dx = 2x = 4



# Example 3 (Gradients Through Loops)
x = torch.tensor(1.0, requires_grad=True)

y = x
for _ in range(3):
    y = y * 2

y.backward()
print(x.grad)  # Output: tensor(8.) since y = 8x, and dy/dx = 8