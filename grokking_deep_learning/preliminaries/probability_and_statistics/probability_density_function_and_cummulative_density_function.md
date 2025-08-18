## **Probability Density Function (PDF) and Cumulative Distribution Function (CDF)**

---

## **1. Probability Density Function (PDF)**

### **Definition**

A **Probability Density Function (PDF)** describes the **likelihood** of a continuous random variable taking 
on a specific value. For continuous variables, the **probability of any exact value is zero**; instead, we compute the **probability over intervals**.

### **Mathematical Form**

Let $X$ be a continuous random variable with PDF $`f_X(x)`$. Then:

* Probability over an interval:

$$
P(a \leq X \leq b) = \int_{a}^{b} f_X(x) \, dx
$$

* Properties:

* $`f_X(x) \geq 0`$ for all $x$
* $`\int_{-\infty}^{\infty} f_X(x) \, dx = 1`$

---

## **2. Cumulative Distribution Function (CDF)**

### **Definition**

The **Cumulative Distribution Function (CDF)** gives the **probability that a random variable takes on a value less than or equal to** a specific value.

$$
F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t)\,dt
$$

### **Properties**:

* $`F_X(x)`$ is **monotonically non-decreasing**
* $`\lim_{x \to -\infty} F_X(x) = 0`$
* $`\lim_{x \to \infty} F_X(x) = 1`$

### **Relationship Between PDF and CDF**:

If $`f_X(x)`$ is differentiable, then:

$$
f_X(x) = \frac{d}{dx} F_X(x)
$$

---

## **3. Discrete vs Continuous Summary**

| Aspect      | Discrete Random Variable            | Continuous Random Variable               |
| ----------- | ----------------------------------- | ---------------------------------------- |
| Function    | Probability Mass Function (PMF)     | Probability Density Function (PDF)       |
| Meaning     | Exact value probability             | Area under curve over interval           |
| Prob(X = x) | Nonzero                             | Zero                                     |
| CDF         | $`F_X(x) = \sum_{t \leq x} P(X = t)`$ | $`F_X(x) = \int_{-\infty}^{x} f_X(t)\,dt`$ |

---

## **4. Examples**

### **Normal Distribution** (continuous)

Let $`X \sim \mathcal{N}(\mu, \sigma^2)`$

* PDF:

$$
f_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

* CDF:

$$
F_X(x) = \int_{-\infty}^{x} f_X(t)\,dt
$$

  (No closed form; uses error function or numerical tables)

### **Uniform Distribution** $`X \sim \mathcal{U}(a, b)`$

* PDF:

$$
f_X(x) = \begin{cases}
\frac{1}{b - a} & \text{if } x \in [a, b] \\
0 & \text{otherwise}
\end{cases}
$$

* CDF:

$$
F_X(x) = \begin{cases}
0 & x < a \\
\frac{x - a}{b - a} & a \leq x \leq b \\
1 & x > b
\end{cases}
$$

---

## **5. Application in Deep Learning**

### **PDFs and CDFs appear in:**

| Area                                | Description                                                                              |
| ----------------------------------- | ---------------------------------------------------------------------------------------- |
| **Variational Autoencoders (VAEs)** | KL divergence between latent Gaussian distributions uses PDFs                            |
| **Normalizing Flows**               | Learn invertible transformations using Jacobians and PDF manipulation                    |
| **Uncertainty Estimation**          | Bayesian deep learning uses posterior PDFs                                               |
| **Sampling**                        | CDF inversion method for sampling from arbitrary distributions                           |
| **Loss Functions**                  | Negative log-likelihood uses the PDF of the predicted distribution                       |
| **Activation Distributions**        | Understanding how activations (e.g., ReLU outputs) distribute under different input PDFs |

---

## **6. Visualization Summary**

### PDF (Normal):

* Bell-shaped curve centered at $`\mu`$
* Area under curve = 1
* Probability is `area under curve` between two points

### CDF (Normal):

* S-shaped sigmoid curve
* Approaches 1 as $`x \to \infty`$
* $`F_X(\mu) = 0.5`$

---

## **7. Summary Table**

| Concept          | Definition                          | Formula                                | Notes                  |
| ---------------- | ----------------------------------- | -------------------------------------- | ---------------------- |
| **PDF**          | Density of probability at value $x$ | $`f_X(x)`$                               | Area = probability     |
| **CDF**          | Cumulative probability up to $x$    | $`F_X(x) = \int_{-\infty}^x f_X(t)\,dt`$ | Always increases       |
| **PDF from CDF** | Derivative                          | $`f_X(x) = \frac{d}{dx} F_X(x)`$         | If differentiable      |
| **CDF from PDF** | Integral                            | $`F_X(x) = \int_{-\infty}^x f_X(t)\,dt`$ | Used for probabilities |

