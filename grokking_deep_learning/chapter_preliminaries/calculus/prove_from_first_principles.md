Here’s a **first principles proof** of the **sum**, **product**, and **quotient rules** in differentiation using limits.

---

## ✅ **1. Sum Rule:**

Let

$$
f(x) = u(x) + v(x)
$$

We want to show:

$$
f'(x) = u'(x) + v'(x)
$$

### **Proof:**

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
= \lim_{h \to 0} \frac{[u(x + h) + v(x + h)] - [u(x) + v(x)]}{h}
$$

$$
= \lim_{h \to 0} \left( \frac{u(x + h) - u(x)}{h} + \frac{v(x + h) - v(x)}{h} \right)
$$

$$
= \lim_{h \to 0} \frac{u(x + h) - u(x)}{h} + \lim_{h \to 0} \frac{v(x + h) - v(x)}{h}
= u'(x) + v'(x)
$$

---

## ✅ **2. Product Rule:**

Let

$$
f(x) = u(x) \cdot v(x)
$$

We want to show:

$$
f'(x) = u'(x) v(x) + u(x) v'(x)
$$

### **Proof:**

$$
f'(x) = \lim_{h \to 0} \frac{u(x + h) v(x + h) - u(x) v(x)}{h}
$$

Add and subtract $u(x + h)v(x)$:

$$
= \lim_{h \to 0} \frac{[u(x + h) v(x + h) - u(x + h) v(x)] + [u(x + h) v(x) - u(x) v(x)]}{h}
$$

$$
= \lim_{h \to 0} \left( u(x + h) \cdot \frac{v(x + h) - v(x)}{h} + v(x) \cdot \frac{u(x + h) - u(x)}{h} \right)
$$

Take the limit:

$$
= u(x) \cdot v'(x) + v(x) \cdot u'(x)
$$

---

## ✅ **3. Quotient Rule:**

Let

$$
f(x) = \frac{u(x)}{v(x)}, \quad v(x) \ne 0
$$

We want to show:

$$
f'(x) = \frac{u'(x)v(x) - u(x)v'(x)}{[v(x)]^2}
$$

### **Proof:**

$$
f'(x) = \lim_{h \to 0} \frac{\frac{u(x + h)}{v(x + h)} - \frac{u(x)}{v(x)}}{h}
$$

Combine into a single fraction:

$$
= \lim_{h \to 0} \frac{u(x + h) v(x) - u(x) v(x + h)}{h \cdot v(x + h) v(x)}
$$

Expand numerator:

$$
= \lim_{h \to 0} \frac{[u(x + h) - u(x)] v(x) - u(x) [v(x + h) - v(x)]}{h \cdot v(x + h) v(x)}
$$

Split into two terms:

$$
= \lim_{h \to 0} \left( \frac{u(x + h) - u(x)}{h} \cdot \frac{v(x)}{v(x + h) v(x)} - \frac{v(x + h) - v(x)}{h} \cdot \frac{u(x)}{v(x + h) v(x)} \right)
$$

Take the limit:

$$
= \frac{u'(x) v(x) - u(x) v'(x)}{[v(x)]^2}
$$

---

## ✅ **Summary of All Three Rules**

| Rule              | Formula                                             |
| ----------------- | --------------------------------------------------- |
| **Sum Rule**      | $`(u + v)' = u' + v'`$                                |
| **Product Rule**  | $`(uv)' = u'v + uv'`$                                 |
| **Quotient Rule** | $`\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}`$ |

Each derived from the **limit definition of the derivative** using **algebraic manipulation**.
