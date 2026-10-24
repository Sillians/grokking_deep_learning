**Exponential (Laplace) Noise Model**

Assume

$`y_i = w^\top x_i + \varepsilon_i,\qquad \varepsilon_i \sim \text{Laplace}(0,1) \quad\Rightarrow\quad p(\varepsilon_i) = \frac{1}{2} \exp\!\left(-\,|\varepsilon_i|\right).`$

---

### **1. Negative Log-likelihood**

The likelihood of the full dataset is

$`P(y\,|\,X,w) = \prod_{i=1}^n \frac{1}{2}\exp\big(-\,|\,y_i - w^{\top}x_i\,|\big)`$

Taking the negative log:

$`-\log P(y\,|\,X,w) = -\sum_{i=1}^n\left[ \log \frac{1}{2} - |y_i - w^\top x_i|\right] = \sum_{i=1}^n |\,y_i - w^\top x_i\,| + n\log 2`$


The constant term $`n\log 2`$ does not affect the optimizer, so the relevant objective is

$`\boxed{ \;\mathcal{L}(w) = \sum_{i=1}^n |\,y_i - w^\top x_i\,| \;}`$

---

### **2. Closed Form Solution**

No closed-form solution in general (except in the 1-dimensional case, where the optimal $w$ is the **median** of the data projected along the input).

For general $`w\in\mathbb{R}^d`$, the optimization involves **absolute values** and leads to a **convex but non-differentiable** problem → must be solved numerically.

---

### **3. Minibatch Stochastic Gradient Descent**

We approximate the sub-gradient:

For each data point,

$`g_i(w) \;:=\; \partial_{w}\;|\,y_i - w^\top x_i\,| = \begin{cases} -\,x_i & \text{if }\,y_i - w^\top x_i < 0 \\[4pt] \;\;\;x_i & \text{if }\,y_i - w^\top x_i > 0 \\[4pt] \in[-x_i,\;x_i] & \text{if }\,y_i - w^\top x_i = 0 \end{cases}`$


The minibatch update at iteration $t$ is

$`w^{(t+1)} \;=\; w^{(t)} -\eta_t\;\frac{1}{|B_t|}\sum_{i\in B_t} g_i\big(w^{(t)}\big)`$

---

### **What Can Go Wrong?**

Near the **stationary point**, many residuals satisfy

$`y_i - w^\top x_i \approx 0\Rightarrow g_i(w)\in [-x_i,x_i]`$


i.e. the subgradient becomes **discontinuous** and can jump in different directions → SGD continues to **oscillate** instead of converging smoothly.

---

