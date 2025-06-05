## **Probabilities vs. Statistics — A Clear Distinction**

* **Probability** is **theoretical**: it describes the **underlying likelihood** of an event in a data-generating process.
  Example: A fair coin has a **true** probability of heads,
  $`P(\text{heads}) = \frac{1}{2}`$.

* **Statistics** are **empirical**: they are **computed from observed data**.
  Example: Toss a coin 100 times, get 53 heads →
  $`\hat{P}(\text{heads}) = \frac{53}{100} = 0.53`$

---

### **Key Concepts**

| Concept         | Description                                                                          |
| --------------- | ------------------------------------------------------------------------------------ |
| **Probability** | Theoretical value, like $`P(\text{heads}) = 0.5`$. Reflects true, but unseen, process. |
| **Statistic**   | Computed from data. E.g., $`\frac{n_h}{n}`$, where $`n_h`$ is observed heads.            |
| **Estimator**   | A function of data used to estimate model parameters (like probabilities).           |
| **Consistency** | A good estimator will converge to the true parameter as sample size → ∞.             |

---

### **Why Both Matter in Machine Learning**

* We assume data is sampled from some **unknown distribution**.
* **Probabilities** define models (e.g., likelihood functions, priors).
* **Statistics** (sample mean, variance, frequencies) help us **learn** from data.
* **Estimators** bridge them: e.g., using sample frequency to estimate class probabilities.

---

### **Example**

A biased coin:

* True probability: $`P(\text{heads}) = 0.6`$ → **probability**.
* Toss 10 times: heads = 7 → $`\hat{P}(\text{heads}) = 0.7`$ → **statistic**.
* Use this 0.7 as an **estimate** for future predictions → **estimator**.

---

### **Conclusion**

* **Probabilities** describe the **world**.
* **Statistics** describe the **data**.
* **Estimators** help us use data to uncover the world's hidden probabilities.
