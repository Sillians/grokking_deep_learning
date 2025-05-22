**multivariable chain rule** in the case where:

* $`u = f(x, y, z)`$, a function of three intermediate variables
* Each of $`x, y, z`$ are themselves functions of $`a`$ and $`b`$:

$$
x = x(a, b), \quad y = y(a, b), \quad z = z(a, b)
$$

---

### **Chain Rule for $`\frac{\partial u}{\partial a}`$ and $`\frac{\partial u}{\partial b}`$**

By the **multivariable chain rule**, the partial derivatives of $u$ with respect to $a$ and $b$ are:

$$
\frac{\partial u}{\partial a} = 
\frac{\partial f}{\partial x} \cdot \frac{\partial x}{\partial a} + 
\frac{\partial f}{\partial y} \cdot \frac{\partial y}{\partial a} + 
\frac{\partial f}{\partial z} \cdot \frac{\partial z}{\partial a}
$$

$$
\frac{\partial u}{\partial b} = 
\frac{\partial f}{\partial x} \cdot \frac{\partial x}{\partial b} + 
\frac{\partial f}{\partial y} \cdot \frac{\partial y}{\partial b} + 
\frac{\partial f}{\partial z} \cdot \frac{\partial z}{\partial b}
$$

---

###  **Compact Vector Form (Jacobian Perspective)**

Let $`\nabla f = \left[ \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right]`$,
and define the Jacobians of $`x, y, z`$ with respect to $`a, b`$ as:

$$
J =
\begin{bmatrix}
\frac{\partial x}{\partial a} & \frac{\partial x}{\partial b} \\
\frac{\partial y}{\partial a} & \frac{\partial y}{\partial b} \\
\frac{\partial z}{\partial a} & \frac{\partial z}{\partial b}
\end{bmatrix}
$$

Then:

$$
\nabla_{(a,b)} u = J^\top \cdot \nabla f
$$

---

### ✅ **Summary**

When composing multivariable functions, the chain rule **propagates derivatives through every path** 
from output back to the independent variables, summing all contributions. This rule is essential 
in **automatic differentiation**, **backpropagation**, and **computation graphs** in deep learning.
