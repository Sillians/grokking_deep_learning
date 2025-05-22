### **Common Loss Functions Used in Neural Networks**

Loss functions (also called objective functions or cost functions) quantify the **difference between predicted outputs and actual targets**. In neural networks, they guide learning by providing a **signal to optimize weights during training**.

---

### 🔹 **1. Mean Squared Error (MSE)**

**Use Case**: Regression problems
**Formula**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

* Penalizes large errors more than small ones.
* Smooth and differentiable, suitable for gradient-based optimization.

---

### 🔹 **2. Mean Absolute Error (MAE)**

**Use Case**: Regression problems
**Formula**:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

* More robust to outliers than MSE.
* Less sensitive to large deviations.

---

### 🔹 **3. Binary Cross-Entropy (Log Loss)**

**Use Case**: Binary classification
**Formula**:

$$
\text{BCE} = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \right]
$$

* Measures the distance between true labels and predicted probabilities.
* Used with a sigmoid activation in the final layer.

---

### 🔹 **4. Categorical Cross-Entropy**

**Use Case**: Multi-class classification (one-hot encoded labels)
**Formula**:

$$
\text{CCE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
$$

* Often used with softmax in the final layer.
* Penalizes confident but wrong predictions.

---

### 🔹 **5. Sparse Categorical Cross-Entropy**

**Use Case**: Multi-class classification (integer labels)

* Similar to categorical cross-entropy, but works directly with class indices.
* More efficient when labels aren’t one-hot encoded.

---

### 🔹 **6. Hinge Loss**

**Use Case**: Support Vector Machines, sometimes in deep learning
**Formula**:

$$
\text{Hinge} = \sum \max(0, 1 - y_i \hat{y}_i)
$$

* Encourages correct classification with a margin.
* Used in "max-margin" classifiers.

---

### 🔹 **7. Kullback-Leibler Divergence (KL Divergence)**

**Use Case**: Probabilistic models, variational autoencoders
**Formula**:

$$
\text{KL}(P || Q) = \sum P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$$

* Measures the divergence between two distributions.
* Often used as a regularization term.

---

### 🔹 **8. Huber Loss**

**Use Case**: Robust regression
**Formula**:

$$
L_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| < \delta \\
\delta(|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
$$

* Combines MSE and MAE advantages.
* Less sensitive to outliers than MSE.

---

### ✅ **Choosing the Right Loss Function**

| Problem Type               | Recommended Loss Function        |
| -------------------------- | -------------------------------- |
| Regression                 | MSE, MAE, Huber                  |
| Binary Classification      | Binary Cross-Entropy             |
| Multi-class Classification | Categorical/Sparse Cross-Entropy |
| Probabilistic Modeling     | KL Divergence                    |

---

Loss functions are critical to how neural networks learn. The right choice depends on the **type of task**, **data characteristics**, and **desired sensitivity to errors**.
