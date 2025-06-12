# **Expectations and Related Concepts in Probability & Statistics (for Deep Learning)**

---

## **1. Expectation of a Random Variable**

The **expectation** (or **expected value**) of a random variable captures the **average value** it takes over many trials.

* **Discrete**:

$$
\mathbb{E}[X] = \sum_{i} x_i \cdot P(X = x_i)
$$

* **Continuous** (with density function $`p(x)`$):

$$
\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x)\,dx
$$

---

## **2. Expectation of a Function of a Random Variable**

If $X$ is a random variable and $`g(X)`$ is a function applied to it:

* **Discrete**:

$$
\mathbb{E}[g(X)] = \sum_{i} g(x_i) \cdot P(X = x_i)
$$

* **Continuous**:

$$
\mathbb{E}[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot p(x)\,dx
$$

### 🔹 Example (ReLU function):

If $`X \sim \mathcal{N}(0, 1)`$, then $`\mathbb{E}[\text{ReLU}(X)] = \mathbb{E}[\max(0, X)] \approx 0.3989`$

This appears in **neural activation expectations**.

---

## **3. Variance**

Variance measures **how spread out** the values of a random variable are:

$$
\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

* A **high variance** indicates values are widely spread.
* A **low variance** means values are tightly clustered around the mean.

---

## **4. Standard Deviation**

The square root of variance:

$$
\text{Std}(X) = \sqrt{\text{Var}(X)}
$$

###  Example:

For $`X \sim \mathcal{N}(\mu, \sigma^2)`$,

* $`\mathbb{E}[X] = \mu`$
* $`\text{Var}(X) = \sigma^2`$
* $`\text{Std}(X) = \sigma`$

---

## **5. Vector-Valued Random Variables**

Let $`\mathbf{X} = (X_1, X_2, \dots, X_n)^\top`$

* **Expectation (mean vector)**:

$$
\mathbb{E}[\mathbf{X}] = (\mathbb{E}[X_1], \mathbb{E}[X_2], \dots, \mathbb{E}[X_n])^\top
$$

* **Covariance matrix**:

$$
\text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X} - \mathbb{E}[\mathbf{X}])(\mathbf{X} - \mathbb{E}[\mathbf{X}])^\top]
$$

Used in **multivariate Gaussian modeling**, **PCA**, and **weight initialization** in deep learning.

---

## **6. Covariance**

Covariance between two random variables $X$ and $Y$ measures how they **vary together**:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

* **Positive** covariance: X and Y increase together.
* **Negative** covariance: X increases, Y decreases.

---

## **7. Correlation**

The **normalized** version of covariance:

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\cdot \text{Var}(Y)}}
$$

* $`\rho \in [-1, 1]`$
* Useful for measuring **linear relationships**

---

## **Real-World Examples and Applications**

| Concept                | Example in Deep Learning                                     | Description                                         |
| ---------------------- | ------------------------------------------------------------ | --------------------------------------------------- |
| **Expectation**        | $`\mathbb{E}_{x,y \sim P}[ \text{loss}(x, y) ]`$               | Average loss over data distribution                 |
| **Variance**           | Weight variance initialization                               | He/Xavier initialization scales weights by variance |
| **ReLU Expectation**   | $`\mathbb{E}[\text{ReLU}(X)]`$ where $`X \sim \mathcal{N}(0,1)`$ | Helps in activation distribution analysis           |
| **Covariance**         | Feature redundancy                                           | High covariance between features may hurt training  |
| **Correlation**        | Multimodal data                                              | Correlating audio and video features                |
| **Vector Expectation** | Output of a stochastic layer                                 | Useful in variational autoencoders (VAEs)           |

---

## **Bonus: Monte Carlo Estimation of Expectation**

Used in practice when direct computation is intractable:

$$
\mathbb{E}_{x \sim p}[f(x)] \approx \frac{1}{n} \sum_{i=1}^n f(x_i)
\quad \text{with } x_i \sim p(x)
$$

Used in:

* Dropout
* Bayesian neural networks
* Policy gradients in RL

---

## **Summary Table**

| Concept              | Formula                                                 | Use Case                       |
| -------------------- | ------------------------------------------------------- | ------------------------------ |
| Expectation          | $`\mathbb{E}[X]`$                                         | Average prediction, loss, etc. |
| Variance             | $`\mathbb{E}[X^2] - (\mathbb{E}[X])^2`$                   | Initialization, uncertainty    |
| Std. Dev.            | $`\sqrt{\text{Var}(X)}`$                                  | Interpretable spread           |
| Function Expectation | $`\mathbb{E}[g(X)]`$                                      | Activation/output modeling     |
| Covariance           | $`\mathbb{E}[(X - \mu_X)(Y - \mu_Y)]`$                    | Feature interaction            |
| Correlation          | $`\frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}`$            | Normalized dependency          |
| Vector Expectation   | $`\mathbb{E}[\mathbf{X}]`$                                | Feature-wise means             |
| Covariance Matrix    | $`\mathbb{E}[(\mathbf{X} - \mu)(\mathbf{X} - \mu)^\top]`$ | Multivariate modeling          |


