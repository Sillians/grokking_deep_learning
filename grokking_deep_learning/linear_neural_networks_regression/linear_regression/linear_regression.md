# **Linear Regression**

---

## **1. Basis**

Linear regression assumes a **linear relationship** between input variables $`x \in \mathbb{R}^d`$ and an output $`y \in \mathbb{R}`$.

The model approximates $y$ as a linear combination of basis functions $`\phi\_j(x)`$:

$`y \approx f(x) = w_0 + \sum_{j=1}^{d} w_j\,\phi_j(x)`$

In matrix notation, let

$`\Phi(x) = \begin{bmatrix}1 \\ \phi_1(x) \\ \dots \\ \phi_d(x)\end{bmatrix}, \qquad w = \begin{bmatrix}w_0 \\w_1 \\ \dots \\ w_d\end{bmatrix}`$

Then

$f(x) = w^\top \Phi(x)`$

The basis functions determine the model expressiveness.

* **Standard basis:** $`\phi\_j(x) = x\_j`$
* **Polynomial basis:** $`\phi\_j(x) = x^j`$
* **Other basis (Fourier, Radial):** used for nonlinear structure in $x$

---

## **2. Model Architecture**

Given $n$ data points $`{(x\_i, y\_i)}\_{i=1}^{n}`$, define the **design matrix**

$`X = \begin{bmatrix} \Phi(x_1)^\top \\ \Phi(x_2)^\top\\ \vdots\\ \Phi(x_n)^\top \end{bmatrix} \in \mathbb{R} {n\times (d+1)}`$

and the target vector

$`y = \begin{bmatrix} y_1 \\ y_2 \\ \vdots \\y_n\end{bmatrix}`$

Model output:

$`\hat{y} = X w \in \mathbb{R}^n`$

Each row of $X$ corresponds to the basis-transformed input. Each column corresponds to one parameter $`w\_j`$.

---

## **3. Loss Function**

The standard loss used in linear regression is the **sum of squared errors (SSE)**:

$`\mathcal{L}(w) = \sum_{i=1}^{n} \left(y_i - f(x_i\right)^2 = \|y - Xw\|_2^2`$

Equivalently, the **mean squared error (MSE)**:

$`\mathrm{MSE}(w) = \frac{1}{n} \|y - Xw\|_2^2`$

Both attain minima at the same $w$ (the scaling factor $`1/n`$ is irrelevant for the minimizer).

---

## **4. Analytic Solution**

To solve

$`\min_{w\in \mathbb{R}^{d+1}} \|y - Xw\|_2^2,`$

take the gradient `w.r.t.` $w$:

$`\nabla_w \|y - Xw\|_2^2 = -2X^\top (y - Xw)`$

Set to zero:

$`X^\top X\,w = X^\top y`$

This is the **normal equation** and yields the closed form solution

$`w^\star = (X^\top X)^{-1} X^\top y`$

provided $`X^\top X`$ is invertible.

---

## **5. Minibatch Stochastic Gradient Descent**

Rather than solving the normal equation, one may use **gradient descent** to iteratively update:

$`w^{(t+1)} = w^{(t)} - \eta \nabla_w \mathcal{L}(w^{(t)})`$

For the MSE:

$`\nabla_w \mathcal{L}(w) = -\frac{2}{n} X^\top (y - Xw)`$

**Full-batch** gradient descent uses all $n$ samples.
**Minibatch** SGD uses a subset $`B\_t \subset {1,\dots, n}`$:

$`w^{(t+1)} = w^{(t)} - \eta \left(-\frac{2}{|B_t|} \sum_{i\in B_t} \Phi(x_i)\left(y_i - w^\top \Phi(x_i)\right)\right)`$

This provides stochastic estimates of the gradient and often converges faster in practice for large datasets.

---

## **6. Prediction**

Once $`w^\star`$ is obtained, predictions for a new input $`x\_{\mathrm{new}}`$ use

$`\hat{y}_{\mathrm{new}} = f(x_{\mathrm{new}}) = w^{\star\top} \Phi(x_{\mathrm{new}})`$

Thus prediction is a single dot product between the learned weights and the basis representation of the input.

---

## **7. The Normal Distribution and Squared Loss**

Assume the **data generation model**

$`y_i = w^\top \Phi(x_i) + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0,\sigma^2)`$

The likelihood of observing $y$ given $X$ and $w$ is

$`p(y\,|\,X, w) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}}\,
\exp\!\left(-\frac{(y_i - w^\top \Phi(x_i))^2}{2\sigma^2}\right)`$

Log-likelihood:

$`\ell(w) = - \frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - w^\top \Phi(x_i))^2`$

Maximizing $`\ell(w)`$ is equivalent to **minimizing** the squared loss.
Hence, the MSE loss in linear regression arises as the **maximum likelihood estimator** under a Gaussian noise assumption.

---

## **8. Linear Regression as a Neural Network**

Linear regression can be viewed as a **single-layer neural network**:

* **Input:** $\Phi(x)$
* **Weights:** $w$
* **No hidden layers**
* **Linear activation**

The network function is

$\hat{y} = w^\top \Phi(x)$

The **loss** is the MSE:

$`\mathcal{L}(w) = \|y - \hat{y}\|_2^2`$

Training this neural network via **backpropagation** is equivalent to performing **gradient descent** on the linear regression objective. No nonlinear activation is applied; thus, linear regression is the simplest form of a neural network.

---
