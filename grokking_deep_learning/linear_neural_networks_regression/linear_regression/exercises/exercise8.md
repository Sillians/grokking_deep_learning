**Regression for Count Data (Apples Sold)**

---

### **1. Why Gaussian Additive Noise is Inappropriate**

In a standard linear regression model we assume

$`y = f(x) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\sigma^2)`$


This is problematic for **count data** such as the number of apples sold:

| Issue                               | Explanation                                                                                                              |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| **Negative values possible**        | A Gaussian noise model allows $y<0$ with non–zero probability. This is impossible for counts.                            |
| **Real‐valued outputs**             | Gaussian models produce real numbers, whereas counts are **integers** (0,1,2,...)                                        |
| **Variance is independent of mean** | Gaussian noise assumes constant variance, whereas in real count data the variance typically **increases with the mean**. |

Hence additive Gaussian noise violates the key properties of count data.

---

### **2. Poisson Distribution and its Expectation**

The Poisson probability mass function is

$`p(k\mid \lambda)\;=\; \frac{\lambda^k e^{-\lambda}}{k!},\qquad k=0,1,2,\dots`$

To show $` \mathbb{E}[k]=\lambda`$,
compute

$`\mathbb{E}[k] = \sum_{k=0}^{\infty} k \; \frac{\lambda^k e^{-\lambda}}{k!}`$

Rewrite the sum by pulling out one $`\lambda`$:

$`\mathbb{E}[k] = \sum_{k=1}^{\infty} k\;\frac{\lambda^k e^{-\lambda}}{k!} = \sum_{k=1}^{\infty} \frac{\lambda^k e^{-\lambda}}{(k-1)!} = \lambda \sum_{m=0}^{\infty} \frac{\lambda^m e^{-\lambda}}{m!} \quad (m=k-1)`$

But

$`\sum_{m=0}^{\infty}\frac{\lambda^m}{m!} = e^\lambda \Rightarrow \sum_{m=0}^{\infty} \frac{\lambda^m e^{-\lambda}}{m!} = 1`$

Thus

$`\boxed{\mathbb{E}[k] = \lambda}`$

---

### **3. Loss Function (Poisson Negative Log-likelihood)**

Assume

$`k_i \sim \text{Poisson}(\lambda_i),\qquad \lambda_i = f(x_i)`$

The log-likelihood over data is

$`\log P(k\,|\,\lambda) = \sum_{i=1}^{n} \left( k_i\log\lambda_i - \lambda_i - \log(k_i!)\right)`$

The **negative** log-likelihood (loss) is therefore

$`\boxed{\mathcal{L}(\lambda) = \sum_{i=1}^n \left(\lambda_i - k_i \log\lambda_i \right)} \quad (\text{const. term } \sum \log(k_i!) \text{ omitted})`$

---

### **4. Loss Function for Estimating $\log\lambda$**

Let

$`\theta_i := \log \lambda_i \qquad\Rightarrow\qquad \lambda_i = \exp(\theta_i)`$

Plug into the loss:

$`\mathcal{L}(\theta)\;=\; \sum_{i=1}^{n} \bigl(\exp(\theta_i) - k_i\,\theta_i \bigr)`$

$`\boxed{\mathcal{L}(\theta) = \sum_{i=1}^n \left( \exp(\theta_i) \;-\; k_i\,\theta_i \right)}`$

This is a **convex loss** in $`\theta`$ and is used in **Poisson regression** (also known as generalized linear models with log-link).

---
