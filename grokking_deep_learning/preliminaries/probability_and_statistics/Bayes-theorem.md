## **Understanding Bayes' Theorem**

Bayes' Theorem is a fundamental principle in probability theory that allows **updating beliefs** (probabilities) based on new evidence.

---

### **1. The Formula**

$$
P(H \mid E) = \frac{P(E \mid H) \cdot P(H)}{P(E)}
$$

Where:

* $`P(H \mid E)`$: **Posterior** – probability of hypothesis $H$ given evidence $E$
* $`P(E \mid H)`$: **Likelihood** – probability of evidence $E$ assuming $H$ is true
* $`P(H)`$: **Prior** – initial belief about hypothesis $H$ before seeing evidence
* $`P(E)`$: **Evidence** – total probability of the evidence under all possible hypotheses

---

### **2. Intuition Behind Each Term**

| Term           | Meaning                                           | Role                                             |
| -------------- | ------------------------------------------------- | ------------------------------------------------ |
| **Prior**      | Belief about a situation before new data          | Incorporates background knowledge                |
| **Likelihood** | How well the evidence supports a given hypothesis | Measures compatibility between hypothesis & data |
| **Posterior**  | Updated belief after observing the evidence       | The refined probability we're most interested in |
| **Evidence**   | Total probability of observing the data           | Normalizes the posterior                         |

---

### **3. Real-World Examples**

#### **a. Medical Diagnosis**

* $`H`$: Patient has disease (e.g., cancer)
* $`E`$: Test result is positive

$$
P(\text{Cancer} \mid \text{Positive}) = \frac{P(\text{Positive} \mid \text{Cancer}) \cdot P(\text{Cancer})}{P(\text{Positive})}
$$

* **Prior**: Prevalence of cancer in population
* **Likelihood**: Probability the test correctly detects cancer (sensitivity)
* **Posterior**: Probability the patient has cancer **given** a positive result

#### **b. Spam Detection**

* $`H`$: Email is spam
* $`E`$: Email contains the word "free"

$$
P(\text{Spam} \mid \text{"free"}) = \frac{P(\text{"free"} \mid \text{Spam}) \cdot P(\text{Spam})}{P(\text{"free"})}
$$

* **Prior**: Probability any email is spam
* **Likelihood**: How often "free" appears in spam emails
* **Posterior**: Updated belief the email is spam given the word "free" appears

#### **c. Machine Learning (Naive Bayes)**

In classification:

* $`H = \text{Class}`$, $`E = \text{Feature vector}`$
* Use Bayes' theorem to compute posterior probabilities of classes given input features
* Classifier picks class with highest posterior

---

### **4. Why It Matters**

Bayes' Theorem is foundational in:

* **Bayesian inference**: Updating model parameters using data
* **Decision-making under uncertainty**
* **Probabilistic models**: E.g., Bayesian networks, Hidden Markov Models

It provides a **principled framework** for learning from data and adjusting beliefs in light of new information.

---

### **5. Summary Table**

| Component      | Definition                                  | Function                                |
| -------------- | ------------------------------------------- | --------------------------------------- |
| **Prior**      | Belief before data                          | Encodes initial assumptions             |
| **Likelihood** | Chance of seeing data if hypothesis is true | Connects hypothesis to data             |
| **Posterior**  | Updated belief after observing data         | Guides final decisions or inferences    |
| **Evidence**   | Overall probability of the data             | Normalizes posteriors across hypotheses |

Bayes' Theorem is a bridge between what we **already know** and what we **learn from data**.
