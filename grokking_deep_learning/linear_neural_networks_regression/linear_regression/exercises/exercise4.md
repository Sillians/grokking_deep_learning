**1. What happens if $`X^\top X`$ does *not* have full rank?**

If $`X^\top X`$ is **not invertible**, then the normal equation

$`X^\top X\,w = X^\top y`$

has **infinitely many** solutions.
The reason is that the columns of $X$ are **linearly dependent**, so some weights can be changed without affecting $`Xw`$; the least-squares objective therefore has multiple minimizers and the analytic solution

$`w^\star = (X^\top X)^{-1}X^\top y`$

is **not defined**.

---

**2. How could you fix it?**

There are two standard approaches:

| Method             | Idea                                           |
| ------------------ | ---------------------------------------------- |
| **Regularization** | Add a small positive multiple of the identity: |

$`(X^\top X + \lambda I)^{-1}X^\top y, \qquad \lambda>0`$


This is **Ridge Regression** and guarantees invertibility. |
\| **Noise Injection**       | Add a small amount of random noise to $X$, which breaks exact dependencies among columns and makes the matrix full rank with probability one. |

---

**What happens if you add coordinate-wise independent Gaussian noise to $`X`$?**

Let $`\tilde{X} = X + \varepsilon`$, where each entry $`\varepsilon_{ij} \sim \mathcal{N}(0,\sigma^2)`$ **independently**.

Then **with probability one**, $`\tilde{X}^\top\tilde{X}`$ is **full rank**, because the added noise almost surely destroys any linear dependencies between the columns.

---

**3. Expected value of the design matrix**

$`\mathbb{E}[\tilde{X}^\top \tilde{X}] = \mathbb{E}\Big[(X+\varepsilon)^\top(X+\varepsilon)\Big]`$

Expand:

$`= X^\top X + \mathbb{E}[\varepsilon^\top X] + \mathbb{E}[X^\top \varepsilon] + \mathbb{E}[\varepsilon^\top \varepsilon]`$

* $`\mathbb{E}[\varepsilon^\top X] = 0`$
* $`\mathbb{E}[X^\top \varepsilon] = 0`$
* $`\mathbb{E}[\varepsilon^\top \varepsilon] = n\,\sigma^2\,I`$

Hence

$`\boxed{\mathbb{E}[\tilde{X}^\top \tilde{X}] = X^\top X + n\,\sigma^2 I}`$

This looks exactly like **ridge regularization in expectation**.

---

**4. What happens with SGD when $`X^\top X`$ does *not* have full rank?**

Stochastic gradient descent **can still converge** even if the matrix is not full rank.
However, in that case:

* It converges to **one of the many minimizers** of the loss.
  
* The specific solution depends on the **initialization** and the **updates**.
  
* The system is **underdetermined** and SGD typically converges to the **minimum-norm** solution in the column space of $X$ (assuming small step sizes).

---

**Summary**

| Condition                                 | Consequence                                                |
| ----------------------------------------- | ---------------------------------------------------------- |
| $`X^\top X`$ singular                     | infinitely many solutions / normal equation not invertible |
| Add noise or regularization               | matrix becomes invertible                                  |
| $`\mathbb{E}[\tilde{X}^\top \tilde{X}]`$  | $`=X^\top X + n\sigma^2I`$                                   |
| SGD when $`X^\top X`$ not full rank       | still converges, but to *one* of many minimizers           |
