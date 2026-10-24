To show the equivalence, introduce an **augmented input vector**

$`\tilde{x} :=  \begin{bmatrix} x \\[3pt] 1  \end{bmatrix} \in \mathbb{R}^{d+1}.`$

Also define an **augmented weight vector**

$` \tilde{w} := \begin{bmatrix} w\\[3pt] b \end{bmatrix} \in \mathbb{R}^{\,d+1}.`$

Now compute the **inner product** of these two augmented vectors:

$`\tilde{x}^{\top}\tilde{w} = \begin{bmatrix} x^{\top} & 1 \end{bmatrix} \begin{bmatrix} \,w\,\\[3pt] \,b\, \end{bmatrix} = x^{\top}w + b·1 = x^{\top}w + b.`$

---

### **Conclusion**

* The function $`f(x) = x^{\top}w + b`$ is **affine** in $`x`$.
  
* By augmenting the input with a constant $`1`$, the same function becomes a **linear** function in the augmented input $`\tilde{x}`$:

$`f(x) = \tilde{x}^{\top}\tilde{w}.`$

Hence,

$`\boxed{\text{Affine functions in }x \text{ are equivalent to linear functions in } (x,1).}`$

This equivalence is why in machine learning, bias terms are often absorbed into the weight vector via input augmentation.
