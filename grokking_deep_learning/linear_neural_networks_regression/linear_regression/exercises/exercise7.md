**Regression for Realistic Price Estimation**

---

### **1. Why the additive Gaussian noise assumption is inappropriate**

In linear regression we typically assume

$`y = f(x) + \varepsilon,\qquad \varepsilon \sim \mathcal{N}(0,\sigma^2)`$

which implies that
`$`\varepsilon`$ is **symmetric** and has **unbounded support**.

* This means that $`y`$ (price) could take **negative values** with non-zero probability.
  
* Real house prices or stock prices are strictly **positive**.
  
* Price fluctuations tend to be **multiplicative** (e.g., “+3%” or “−2%”), not additive.

**Hence**:
An additive Gaussian noise model can produce *negative* or *unrealistically large* values, and therefore does **not** reflect real price dynamics.

---

### **2. Why regression on $y = \log(\text{price})$ is preferable**

Log-pricing assumes

$`\log(\text{price}) = f(x) + \varepsilon`$

which implies

$`\text{price} = \exp(f(x) + \varepsilon)\;=\;\exp(f(x)) \cdot \exp(\varepsilon)`$

→ the noise becomes **multiplicative** in the original scale.
This is consistent with:

| Characteristic of real prices                | Captured by log-regression?    |
| -------------------------------------------- | ------------------------------ |
| Always positive                              | ✅ (exponential always >0)      |
| Percentage changes                           | ✅ (log differences ≈ % change) |
| Skewed distribution (long tail on the right) | ✅                              |

Thus, modelling the **log-price** with additive Gaussian noise better matches real-world behaviour of prices.

---

### **3. What to worry about for penny stocks**

Penny stocks have very **low nominal prices** (e.g. \$0.01 – \$1).
In this regime:

* **Price grid is coarse** → you *cannot* trade at arbitrary prices (e.g. \$0.01143).
  Only discrete jumps are possible.

* **A small absolute change corresponds to a large *relative* change**
  (e.g. \$0.01 → \$0.02 is +100%).


Therefore:

| Problem              | Effect                                                               |
| -------------------- | -------------------------------------------------------------------- |
| Discrete tick size   | The price cannot vary smoothly → continuous models become inaccurate |
| Large relative jumps | Log-returns become erratic and heavy-tailed                          |
| Liquidity constraint | Harder to execute trades at quoted prices                            |

These effects are **more pronounced** for cheap stocks and violate the assumptions of classical models such as **Black–Scholes**, which assume a *continuous diffusion* (i.e., continuous Brownian motion of log-price).

---

**In summary**

* Additive Gaussian noise → inappropriate because prices are positive and move multiplicatively.
  
* log-price regression → converts multiplicative noise into additive form, consistent with financial reality.
  
* Penny stocks → discrete tick sizes and large relative jumps break the smooth log-noise assumptions, requiring more careful or discrete-state models (e.g., jump-diffusions).
