**Numerical breakdown** of the Bayes's theorem examples provided using **arbitrary but realistic values**:

---

### **1. Medical Diagnosis Example**

#### Problem:

A disease affects **1 in 1,000 people**. A test is:

* **99% sensitive**: $`P(\text{Positive} \mid \text{Disease}) = 0.99`$
* **5% false positive rate**: $`P(\text{Positive} \mid \text{No Disease}) = 0.05`$

What’s the probability a person has the disease **given** they test positive?

---

#### Step-by-step using Bayes' Theorem:

Let:

* $`H = \text{Has Disease}`$
* $`E = \text{Tests Positive}`$

Given:

* $`P(H) = 0.001`$
* $`P(\text{Positive} \mid H) = 0.99`$
* $`P(\text{Positive} \mid \text{No Disease}) = 0.05`$
* $`P(\text{No Disease}) = 0.999`$

First, compute the total probability of a positive test (the **evidence**):

$$
P(\text{Positive}) = P(\text{Positive} \mid H) \cdot P(H) + P(\text{Positive} \mid \text{No Disease}) \cdot P(\text{No Disease}) 
$$

$$
= 0.99 \cdot 0.001 + 0.05 \cdot 0.999 = 0.00099 + 0.04995 = 0.05094
$$

Now, compute **posterior**:

$$
P(H \mid \text{Positive}) = \frac{0.99 \cdot 0.001}{0.05094} \approx \frac{0.00099}{0.05094} \approx 0.0194 \text{ or } 1.94\%
$$

#### 🔎 Interpretation:

Even with a **positive test**, the chance the person actually has the disease is **only \~1.94%**. This shows the importance of accounting for **base rates (priors)**.

---

### **2. Spam Detection Example**

Suppose:

* 20% of all emails are spam → $`P(\text{Spam}) = 0.2`$
* The word **“free”** appears in 40% of spam emails → $`P(\text{"free"} \mid \text{Spam}) = 0.4`$
* “Free” appears in 5% of non-spam emails → $`P(\text{"free"} \mid \text{Not Spam}) = 0.05`$

What is the probability an email is spam **given** it contains “free”?

---

#### Step-by-step:

Let:

* $`H = \text{Spam}`$
* $`E = \text{"free"} \text{ appears}`$

Calculate total probability that an email contains "free":

$$
P(\text{"free"}) = P(\text{"free"} \mid \text{Spam}) \cdot P(\text{Spam}) + P(\text{"free"} \mid \text{Not Spam}) \cdot P(\text{Not Spam})
$$

$$
= 0.4 \cdot 0.2 + 0.05 \cdot 0.8 = 0.08 + 0.04 = 0.12
$$

Now apply Bayes' theorem:

$$
P(\text{Spam} \mid \text{"free"}) = \frac{0.4 \cdot 0.2}{0.12} = \frac{0.08}{0.12} = 0.6667 \text{ or } 66.67\%
$$

#### 🔎 Interpretation:

Given an email contains “free”, there's about a **66.67%** chance it’s spam.

---

These examples show how **prior knowledge and data likelihood** combine to update our belief in a hypothesis, a core concept in 
Bayesian thinking and many real-world systems, including **medical diagnostics**, **email filtering**, and **machine learning inference**.
