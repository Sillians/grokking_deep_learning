"""
This code plots sin(x) and its derivative computed purely through automatic differentiation,
without using the analytical knowledge that the derivative of sin(x) is cos(x).

The key insight is that we create a tensor with requires_grad=True, compute sin(x),
and then use PyTorch's automatic differentiation to find the gradient at each point.
The resulting derivative matches cos(x) perfectly, demonstrating the power and accuracy of automatic differentiation.

"""

import numpy as np
import torch
import matplotlib.pyplot as plt

# Create x values from -2π to 2π
x_vals = torch.linspace(-2 * np.pi, 2 * np.pi, 1000, requires_grad=True)

# Compute f(x) = sin(x)
f_x = torch.sin(x_vals)

# Compute derivative using automatic differentiation
# We need to sum f_x to get a scalar for backward()
f_sum = f_x.sum()
f_sum.backward()

# The gradient of each x_val is the derivative at that point
f_prime_x = x_vals.grad

# Detach and convert to numpy for plotting
x_np = x_vals.detach().numpy()
f_np = f_x.detach().numpy()
f_prime_x_np = f_prime_x.detach().numpy()

# Create the plot
plt.figure(figsize=(12, 8))

# Plot f(x) = sin(x)
plt.subplot(2, 1, 1)
plt.plot(x_np, f_np, 'b-', linewidth=2, label='f(x) = sin(x)')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('f(x) = sin(x)')
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)


# Plot f'(x) computed via automatic differentiation
plt.subplot(2, 1, 2)
plt.plot(x_np, f_prime_x_np, 'r-', linewidth=2, label="f'(x) via autodiff")
plt.plot(x_np, np.cos(x_np), 'g--', linewidth=1, alpha=0.7, label='cos(x) for comparison')
plt.grid(True, alpha=0.3)
plt.xlabel('x')
plt.ylabel("f'(x)")
plt.title("Derivative of sin(x) using Automatic Differentiation")
plt.legend()
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.show()






