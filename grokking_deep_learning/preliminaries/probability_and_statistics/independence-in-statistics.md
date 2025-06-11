## **Independence in Statistics**

**Independence** describes a fundamental relationship between two events or random variables: the occurrence or value of one gives **no information** about the other.

---

### **1. Independence of Events**

Two events **A** and **B** are **independent** if:

$$
P(A \cap B) = P(A) \cdot P(B)
$$

This means the probability of both happening equals the product of their individual probabilities.

**Example**:

* Tossing a fair coin twice:

  * Let A = first toss is heads
  * Let B = second toss is heads
* Then:

$$
P(A) = \frac{1}{2}, \quad P(B) = \frac{1}{2}, \quad P(A \cap B) = \frac{1}{4}
$$

  * Since $`\frac{1}{2} \cdot \frac{1}{2} = \frac{1}{4}`$, A and B are independent.

---

### **2. Independence of Random Variables**

Two random variables **X** and **Y** are **independent** if their **joint distribution** factorizes:

$$
P(X = x, Y = y) = P(X = x) \cdot P(Y = y) \quad \text{for all } x, y
$$

In continuous settings:

$$
f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y)
$$

---

### **3. Conditional Independence**

Two variables **X** and **Y** are **conditionally independent given Z** if:

$$
P(X, Y \mid Z) = P(X \mid Z) \cdot P(Y \mid Z)
$$

This means that once Z is known, learning X gives no extra information about Y.

**Example**:

* Let Z = whether it's raining,
* X = carrying an umbrella,
* Y = street is wet.
  Given Z (raining or not), X and Y are conditionally independent.

---

### **4. Independence vs. Uncorrelated**

* **Uncorrelated** means $`\text{Cov}(X, Y) = 0`$
* **Independent** ⇒ Uncorrelated
* But **Uncorrelated ≠⇒ Independent** (except for some special cases like jointly Gaussian variables)

---

### **5. Importance in Deep Learning & ML**

| Area                   | Role of Independence                                                       |
| ---------------------- | -------------------------------------------------------------------------- |
| **Naive Bayes**        | Assumes feature independence given class label                             |
| **Bayesian networks**  | Model dependencies/independencies between variables                        |
| **Sampling methods**   | Use independence for tractable joint sampling                              |
| **Data preprocessing** | Avoid information leakage by ensuring independence of training/test splits |

---

### **Summary**

* **Independence** means no statistical relationship between events/variables.
* Crucial for simplifying probability computations and modeling assumptions.
* Always check whether independence is **assumed** or **needs to be tested** in probabilistic models.
