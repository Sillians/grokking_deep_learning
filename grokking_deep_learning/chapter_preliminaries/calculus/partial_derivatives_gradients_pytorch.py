import torch

# Input and target label
x = torch.tensor([2.0], requires_grad=False)
y_true = torch.tensor([7.0], requires_grad=False)

# Initialize parameters (weights and bias)
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

# Forward pass (prediction and loss)
y_pred = w * x + b

# loss: Mean Squared Error
n = 0.5
loss = n * (y_pred - y_true) ** 2

# backward pass (compute gradients)
loss.backward()

# View gradients (partial derivatives)
print("Loss:", loss.item())
print("Gradient w.r.t. w:", w.grad.item())
print("Gradient w.r.t. b:", b.grad.item())
