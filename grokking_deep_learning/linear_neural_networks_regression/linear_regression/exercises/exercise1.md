**1. Analytic Solution (Squared Loss)**

We are solving

$`\min_{b\in \mathbb{R}} \;\sum_{i=1}^n (x_i - b)^2`$

Differentiate w\.r.t.,$b$:

$`\frac{d}{db} \sum_{i=1}^n (x_i - b)^2 = \sum_{i=1}^n 2(b - x_i) = 2\left(nb - \sum_{i=1}^n x_i\right)`$

Set the derivative to zero:

$`nb - \sum_{i=1}^n x_i = 0 \;\Longrightarrow\; b^{\star} = \frac{1}{\,n\,}\sum_{i=1}^n x_i`$

**→ The optimal $`b`$ is the sample mean.**

---

**2. Relation to the Normal Distribution**

Assume that

$`x_i \;=\; b + \varepsilon_i,\qquad \varepsilon_i \sim \mathcal{N}(0,\sigma^2)`$

The likelihood of the samples is

$`p(x_1,\dots,x_n \mid b) =\prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}\; \exp \!\left(-\frac{(x_i-b)^2}{2\sigma^2}\right)`$

The log-likelihood is

$`\ell(b) = -\frac{n}{2}\ln(2\pi\sigma^2) -\frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - b)^2`$

Maximizing $`\ell(b)`$ is equivalent to **minimizing** $`\sum_{i}(x_i-b)^2`$

$`\Longrightarrow\qquad b^{\star} = \frac{1}{n} \sum_{i=1}^n x_i`$

Hence the **sample mean** is the **maximum likelihood estimator of the mean of a normal distribution**.

---

**3. Absolute Loss**

Now consider

$`\min_{b\in\mathbb{R}} \;\sum_{i=1}^n |x_i - b|`$

The minimizer of this objective is the **median** of the data:

$`\boxed{\;b^{\star} = \operatorname{median}(x_1,\dots,x_n)\;}`$

---

**Summary**

| Loss Function                 | Optimal $b$      |   |                    |
| ----------------------------- | ---------------- | - | ------------------ |
| $`\sum_{i}(x_i - b)^2`$       | Mean of the data |   |                    |
| $`(\sum\_{i}\| x\_i - b \| )$`  | Median of the data |

* Squared loss corresponds to **Gaussian** noise assumptions.
* Absolute loss corresponds to **Laplace** noise assumptions (heavy tails).
