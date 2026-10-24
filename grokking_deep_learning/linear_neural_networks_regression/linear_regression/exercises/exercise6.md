Composing two purely **linear** layers results in **another linear function**.

---

Let

$`h(x) = W_1 x + b_1 \qquad\text{(first layer)}`$


$`f(x) = W_2\,h(x) + b_2 = W_2(W_1x + b_1) + b_2`$

Expand:

$`f(x) = (W_2 W_1) \, x + (W_2 b_1 + b_2)`$

Both terms are **linear** in $`x`$.

Thus the whole **two-layer network collapses into a single affine layer**

$`f(x) = \tilde{W}x + \tilde{b} \quad\text{with}\quad \tilde{W} = W_2W_1,\;\;\tilde{b}= W_2 b_1 + b_2`$

---

### **Conclusion**

A composition of two linear layers **does not increase expressiveness** —
it is **still a linear model**.

To make the network *nonlinear* and more expressive, you must add a **nonlinear activation** (e.g. $`\sigma(\cdot)\`$) between the layers.
