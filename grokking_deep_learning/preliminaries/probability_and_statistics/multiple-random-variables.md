### **Multiple Random Variables**

When dealing with **more than one random variable**, we are often interested in how they **relate** to each other. 
This forms the basis of many important concepts in probability, statistics, and deep learning.

---

### **1. Joint Random Variables**

Given two random variables $X$ and $Y$, the **joint distribution** captures the probability of **both** occurring together.

#### Discrete Case:

* $`P(X = x, Y = y)`$: Probability that $`X = x`$ and $`Y = y`$ simultaneously.

#### Continuous Case:

* $`p(x, y)`$: Joint **probability density function** (PDF).
* To find probability over regions:

$$
P(a \leq X \leq b, \, c \leq Y \leq d) = \int_a^b \int_c^d p(x, y)\, dy\, dx
$$

---

### **2. Marginal Distributions**

Marginals are the **individual** distributions of each variable, ignoring the other.

* Discrete:

$$
P(X = x) = \sum_y P(X = x, Y = y)
$$

* Continuous:

$$
p(x) = \int p(x, y) \, dy
$$

---

### **3. Conditional Distributions**

These express the probability of one variable **given** the other.

* Discrete:

$$
P(X = x \mid Y = y) = \frac{P(X = x, Y = y)}{P(Y = y)}
$$

* Continuous:

$$
p(x \mid y) = \frac{p(x, y)}{p(y)}
$$

Conditional distributions are essential in **Bayesian inference** and **deep generative models**.

---

### **4. Independence**

Two random variables $X$ and $Y$ are **independent** if:

$$
P(X = x, Y = y) = P(X = x) \cdot P(Y = y)
$$

Or for continuous variables:

$$
p(x, y) = p(x) \cdot p(y)
$$

This implies knowing one variable gives **no information** about the other.

---

### **5. Correlation and Dependence**

Even if two variables are **not independent**, they may be **correlated**:

* **Covariance**:

$$
\text{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

* **Correlation coefficient**:

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$$

Values range from -1 (perfect inverse) to +1 (perfect alignment).

---

### **6. Applications in Deep Learning**

| Use Case                      | Description                                                         |
| ----------------------------- | ------------------------------------------------------------------- |
| **Multivariate input/output** | Networks process multiple features at once.                         |
| **Latent variable models**    | Use multiple RVs to model observed + hidden structure (e.g. VAEs).  |
| **Conditional models**        | Predict $Y$ given $X$, e.g., in supervised learning.                |
| **Attention mechanisms**      | Measure dependencies between elements (e.g., tokens in a sequence). |

---

### **Summary Table**

| Concept                 | Description                                               |
| ----------------------- | --------------------------------------------------------- |
| Joint Distribution      | Probability of two RVs occurring together                 |
| Marginal Distribution   | Probability of one RV, summing/integrating over the other |
| Conditional Probability | Probability of one RV given the value of another          |
| Independence            | No influence between two RVs                              |
| Correlation             | Quantifies linear relationship between two RVs            |

Understanding multiple random variables is foundational for working with **real-world data** and **deep learning architectures** where many features interact simultaneously.
