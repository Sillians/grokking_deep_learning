## Probability and Statistics
Probability is the mathematical field concerned with reasoning under uncertainty. Given a
probabilistic model of some process, we can reason about the likelihood of various events.

It forms the **mathematical backbone of deep learning**, providing the language to model **uncertainty**, **variability**, and **inference**.

---
Some key concepts and their roles;

### **1. Probability Theory: Foundation of Uncertainty**

#### **Random Variables & Distributions**

* Model inputs, outputs, noise, and weights.
* Examples:

  * Input features: $`x \sim p(x)`$
  * Model uncertainty: $`y \sim p(y|x)`$
  * Dropout: Bernoulli distribution

####  **Common Distributions**

| Distribution              | Use in DL                                   |
| ------------------------- | ------------------------------------------- |
| **Bernoulli**             | Binary classification (sigmoid output)      |
| **Categorical**           | Multi-class classification (softmax output) |
| **Normal (Gaussian)**     | Weight initialization, Bayesian inference   |
| **Exponential / Poisson** | Time-series, event modeling                 |

####  **Conditional Probability**

* Core to supervised learning:
  $`p(y|x)`$: probability of label given input

####  **Bayes’ Theorem**

$$
p(\theta|x) = \frac{p(x|\theta)p(\theta)}{p(x)}
$$

* Basis of **Bayesian Neural Networks**
* Enables **uncertainty estimation**

---

### **2. Statistical Inference: Learning from Data**

####  **Maximum Likelihood Estimation (MLE)**

* Learn model parameters $`\theta`$ that maximize data likelihood:

$$
\theta^* = \arg\max_\theta \prod_i p(y_i|x_i; \theta)
$$

* Used in classification, regression, and generative models.

####  **Loss Functions as Negative Log-Likelihoods**

| Task           | Distribution | Loss Function            |
| -------------- | ------------ | ------------------------ |
| Classification | Categorical  | Cross-entropy            |
| Regression     | Gaussian     | Mean Squared Error (MSE) |

####  **Expectation & Variance**

* Measure the predicted value’s mean and spread.
* Applied in uncertainty modeling, dropout, and ensembles.

---

### **3. Information Theory: Measuring Uncertainty**

####  **Entropy (H):**

$$
H(X) = -\sum p(x) \log p(x)
$$

* Measures **uncertainty** in predictions.
* Used in **decision trees**, **regularization**, **variational methods**.

####  **KL Divergence:**

$$
D_{\text{KL}}(P \| Q) = \sum P(x) \log \frac{P(x)}{Q(x)}
$$

* Used in **variational inference**, **VAE loss**, **distillation**

---

### **4. Probabilistic Models in Deep Learning**

####  **Bayesian Neural Networks (BNNs)**

* Weights are treated as distributions: $`w \sim p(w)`$
* Capture **model uncertainty**, improve **robustness**

####  **Variational Autoencoders (VAEs)**

* Use **variational inference** to approximate posterior:

  * Optimize ELBO:

$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x) \| p(z))
$$

####  **Generative Models (GANs, Flow Models)**

* Model **data distribution** $`p(x)`$
* Use statistical principles for density estimation

---

### **5. Statistical Learning Theory**

####  **Bias–Variance Tradeoff**

* Key to understanding overfitting/underfitting.
* Variance increases with model complexity.

####  **PAC Learning, VC Dimension**

* Theoretical bounds on **generalization**
* Important for model capacity and design

---

####  Summary Table

| Concept                     | Role in Deep Learning                        |
| --------------------------- | -------------------------------------------- |
| Probability distributions   | Model data, noise, and uncertainty           |
| Conditional probabilities   | Supervised learning and generative models    |
| MLE / MAP                   | Parameter estimation                         |
| Entropy & KL divergence     | Loss functions, regularization               |
| Bayesian inference          | Uncertainty-aware models (BNNs, VAEs)        |
| Statistical learning theory | Generalization, regularization, model design |

---

#### 📘 Recommended Study Topics (in order):

1. **Basic probability theory** (discrete, continuous)
2. **Conditional probability & Bayes' Rule**
3. **Common distributions** (Bernoulli, Gaussian, Categorical)
4. **MLE and MAP estimation**
5. **KL divergence, entropy, mutual information**
6. **Probabilistic graphical models**
7. **Bayesian methods and variational inference**
8. **Information theory for deep learning**
9. **Uncertainty in neural networks**

---
