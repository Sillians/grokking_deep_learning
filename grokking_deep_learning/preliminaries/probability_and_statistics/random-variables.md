## **Random Variables — Simplified and Explained**

A **random variable (RV)** is a function that maps outcomes from the **sample space** to numerical values, enabling us to quantify uncertainty.

---

### **1. What Is a Random Variable?**

A **random variable** is not just a value, but a **function**:

* From: **Sample space** (all possible outcomes)
* To: **Values** (real numbers or categories)

#### Examples:

| Random Process          | Sample Space (Ω)             | Random Variable (X)         | Possible Values    |
| ----------------------- | ---------------------------- | --------------------------- | ------------------ |
| Flip a coin             | {Heads, Tails}               | X = 1 if Heads, 0 if Tails  | {0, 1}             |
| Roll a die              | {1, 2, 3, 4, 5, 6}           | Y = value shown             | {1, 2, 3, 4, 5, 6} |
| Random point in \[0, 1] | Infinite outcomes in \[0, 1] | Z = 1 if > 0.5, 0 otherwise | {0, 1}             |

---

### **2. Why Random Variables Matter**

* They allow us to **summarize outcomes** with meaningful numbers.
* We can compute **probabilities**, expectations, and other statistics.
* Multiple random variables can **depend** on the same sample space, helping us model real-world interactions.

#### Example:

* RV A = "Alarm goes off" (1 = yes, 0 = no)
* RV B = "House was burgled" (1 = yes, 0 = no)

If we observe A = 1, the probability that B = 1 may increase → **correlated random variables**.

---

### **3. Events and Random Variables**

* The event **X = x** is a **subset** of the sample space.
* The **probability** of this event is written as:

  * $`P(X = x)`$
  * $`P(x)`$ (abuse of notation when context is clear)

#### General Form:

* $`P(X \in [a, b])`$: probability that X falls in interval $`[a, b]`$

---

### **4. Discrete vs. Continuous Random Variables**

| Type           | Description                  | Example             | Probability Type                       |
| -------------- | ---------------------------- | ------------------- | -------------------------------------- |
| **Discrete**   | Countable values             | Die roll, coin flip | $`P(X = x)`$                             |
| **Continuous** | Infinite, uncountable values | Height, weight      | $`P(a \leq X \leq b)`$ via **integrals** |

#### Note:

* For **continuous RVs**, $`P(X = x) = 0`$ for any exact value.
* We use **probability density functions (PDFs)** instead.
* To find real probabilities:
  $`P(a \leq X \leq b) = \int_a^b p(x)\,dx`$

---

### **5. Summary Table**

| Concept              | Meaning                                                   |
| -------------------- | --------------------------------------------------------- |
| Random Variable      | Function from outcomes to values                          |
| $`P(X = x)`$           | Probability that X takes value x (discrete case)          |
| $`P(a \leq X \leq b)`$ | Probability X falls in interval \[a, b] (continuous case) |
| PDF $`p(x)`$           | Describes **density** of values in continuous case        |
| Event $`X = x`$        | A subset of the sample space                              |

---

Random variables are central to **deep learning**, especially in:

* **Loss functions** (expectations over data)
* **Bayesian models**
* **Stochastic gradient descent**
* **Variational inference**

Understanding them is key to modeling and reasoning about uncertainty.
