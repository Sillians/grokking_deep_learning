import numpy as np
import matplotlib.pyplot as plt

"""Plot the function `y = f(x) = x**3 - 1/x` and plot its tangent line at x = 1"""

# define function
def f(x):
    return x**3 - 1/x

# function derivative
def f_prime(x):
    return 3 * x**2 + 1 / x**2

# tangent line at...
x0 = 1
y0 = f(x0)
slope = f_prime(x0)
# print(slope)

# Equation of the tangent line at x0
def tangent_line(x):
    return slope * (x - x0) + y0

x_vals = np.linspace(0.5, 1.5, 400)
y_vals = f(x_vals)
tangent_vals = tangent_line(x_vals)


plt.figure(figsize=(8, 5))
plt.plot(x_vals, y_vals, label='f(x) = x³ - 1/x', color='blue')
plt.plot(x_vals, tangent_vals, label='Tangent at x = 1', linestyle='--', color='red')
plt.scatter(x0, y0, color='black', zorder=5)
plt.title('Function and Tangent Line at x = 1')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid(True)
plt.show()



