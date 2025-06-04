"""
This code plots sin(x) and its derivative computed purely through automatic differentiation,
without using the analytical knowledge that the derivative of sin(x) is cos(x).
"""

import torch
import matplotlib.pyplot as plt

# Enable gradient tracking (Create x values from -2π to 2π)
x_vals = torch.linspace(-2 * torch.pi, 2 * torch.pi, 1000, requires_grad=True)
y_vals = torch.sin(x_vals)

# Use backward pass to compute gradients
y_vals.sum().backward()
y_prime_vals = x_vals.grad.clone()


# Plot the graph of `f` and `f_prime`
plt.figure(figsize=(10, 6))
plt.plot(x_vals.detach(), y_vals.detach(), label="f(x) = sin(x)")
plt.plot(x_vals.detach(), y_prime_vals, label="f'(x) via autodiff", linestyle='--')
plt.title("f(x) = sin(x) and Its Derivative (Automatic diﬀerentiation with PyTorch)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.legend()
plt.tight_layout()
plt.show()