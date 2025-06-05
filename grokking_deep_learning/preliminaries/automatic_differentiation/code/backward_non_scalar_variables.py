import torch

# Example 1
x = torch.arange(4.0, requires_grad=True)
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
print(x.grad)




# Example 2
x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x * x  # y is a vector: [x_0^2, x_1^2] = [4, 9]

# Attempting y.backward() here will raise an error because y is not a scalar
# Instead, provide a vector v (gradient) to reduce y into a scalar:
v = torch.tensor([1.0, 1.0])  # represents d(output)/dy

y.backward(gradient=v)

print(x.grad)  # → tensor([4.0, 6.0]) = [∂(x_0^2)/∂x_0, ∂(x_1^2)/∂x_1]