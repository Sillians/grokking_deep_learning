## **Backward for Non-Scalar Variables: Jacobians, Gradients, and PyTorch Backpropagation**

---

In deep learning, we frequently compute **derivatives of one vector with respect to another vector**. 
The most mathematically correct representation of such a derivative is the **Jacobian matrix**.

---

### **1. Jacobian Matrix**

If:

* $`y \in \mathbb{R}^m`$ is a **vector-valued function** of $`x \in \mathbb{R}^n`$,
  then the derivative $`\frac{\partial y}{\partial x}`$ is a **Jacobian matrix** of shape $`m \times n`$.

Each entry:

$$
J_{ij} = \frac{\partial y_i}{\partial x_j}
$$

So, the Jacobian contains **all partial derivatives** of each component of $`y`$ with respect to each component of $x$.

---

### **2. Gradients in Deep Learning: Summing Over Batches**

In practice, especially in **training**, we often have a **vector output (e.g., loss per batch example)** 
and want to compute the **gradient of the total loss** with respect to model parameters (a vector $x$).

Rather than tracking full Jacobians, we usually:

* **Sum** the gradients of each output element with respect to $x$
* Get a **single vector** of the same shape as $x$, not a Jacobian

---

###  **3. PyTorch and Non-Scalar Outputs**

PyTorch’s `.backward()` computes gradients automatically **only for scalar outputs**.

If you try:

```python
y.backward()
```

when `y` is **not a scalar** (e.g., a vector), PyTorch throws an error because it doesn’t know how to reduce the output into a scalar.

Instead, you must provide a **gradient vector $v$**, representing how much each component of $y$ contributes to a scalar. PyTorch will then compute:

$$
v^\top \cdot \frac{\partial y}{\partial x}
$$

This is known as the **vector-Jacobian product (VJP)**.

In PyTorch, this `v` is confusingly passed as the `gradient=` argument.

---

###  **4. Example in PyTorch**

```python
import torch

x = torch.tensor([2.0, 3.0], requires_grad=True)
y = x * x  # y is a vector: [x_0^2, x_1^2] = [4, 9]

# Attempting y.backward() here will raise an error because y is not a scalar
# Instead, provide a vector v (gradient) to reduce y into a scalar:
v = torch.tensor([1.0, 1.0])  # represents d(output)/dy

y.backward(gradient=v)

print(x.grad)  # → tensor([4.0, 6.0]) = [∂(x_0^2)/∂x_0, ∂(x_1^2)/∂x_1]
```

Here:

* $y = [x_0^2, x_1^2]$
* The derivative of each is $2x_i$, evaluated at \[2, 3] gives \[4, 6]
* The `gradient` vector $v = [1, 1]$ effectively tells PyTorch: "Just sum the gradients of all components of $y$."

---

###  **Summary**

* The **Jacobian** is the full matrix of partial derivatives for vector-valued outputs.
* In deep learning, we often want a single **gradient vector**, not the full Jacobian.
* **PyTorch requires scalar outputs** for `.backward()`—for vectors, you must supply a **"gradient" vector** to guide reduction.
* This is computing the **vector-Jacobian product**, not just the plain derivative.
 