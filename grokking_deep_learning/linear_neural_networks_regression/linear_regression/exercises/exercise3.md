To represent a **quadratic function**

$`f(x)\;=\; b + \sum_i w_i\,x_i \;+\; \sum_{j\le i} w_{ij}\,x_i x_j,`$

in a **deep network**, the idea is to **augment the input** with all **second–order interaction terms** and then apply a **linear layer** on top of this augmented vector.

---

### **Step 1 — Build an Expanded Input Vector**

Define the **original input**

$`x = (x_1,\dots,x_d)`$

Create an **expanded feature vector**

$`\Phi(x) =  \Big(x_1,\dots,x_d,\;x_1x_1,\;x_2x_1,\;\dots,\;x_d x_d\Big)`$

i.e.,

$`\Phi(x) =  \Big( \underbrace{x_i}_{\text{linear terms}}, \; \underbrace{x_i x_j}_{\text{all pairs } j\le i} \Big)`$

This vector contains:

* all $`d`$ **first-order** terms $`x_i`$
* all $`\frac{d(d+1)}{2}`$ **second-order** terms $`x_i x_j`$

---

### **Step 2 — Apply a Linear Layer**

Let

$`W = (\,w_1,\dots,w_d,\; w_{11}, w_{21}, \dots, w_{dd} \,)\qquad\text{and}\qquad b\in \mathbb{R}.`$

Then define a **one-layer neural network** (no activation) as

$`f(x) = W^{\top}\,\Phi(x) + b`$

---

### **Final Form**

$`\boxed{f(x) \;=\; b + \sum_{i=1}^{d} w_i\,x_i + \sum_{j\le i} w_{ij}\,x_i x_j} \quad =\quad\boxed{W^{\top}\Phi(x) + b}`$

---

### **Interpretation**

* The “network” has **input dimension** $`d`$

* A **hidden layer** computes all second-order features $`x_i x_j`$
  
* The **output layer** is linear (no activation)

This effectively turns a quadratic model into a **linear model in an expanded feature space**, which is exactly how such polynomial functions are represented in neural networks.
