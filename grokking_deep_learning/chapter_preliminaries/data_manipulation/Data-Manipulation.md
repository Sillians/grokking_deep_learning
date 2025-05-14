### **Tensors**
**Tensors** are multi-dimensional arrays that generalize `scalars (0D)`, `vectors (1D)`, and `matrices (2D)` 
to **higher dimensions**. They are a core data structure in machine learning and deep learning, 
used to represent and manipulate data in numerical computations.



### **Tensor Dimensions at a Glance:**

| Dimension | Name   | Example                                            |
| --------- | ------ | -------------------------------------------------- |
| 0D        | Scalar | $x = 7$                                            |
| 1D        | Vector | $x = [1, 2, 3]$                                    |
| 2D        | Matrix | $x = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| 3D+       | Tensor | Stack of matrices (e.g., image batches)            |



### **Key Properties:**

* **Rank**: Number of dimensions (axes).
* **Shape**: Size along each dimension (e.g., $(3, 2, 4)$).
* **Data type**: Type of values (e.g., `float32`, `int64`).


### **In Practice:**

Tensors are the primary data containers in frameworks like **TensorFlow**, **PyTorch**, and **JAX**, 
used for storing:

* Images
* Time series
* Word embeddings
* Model parameters

They enable efficient computation on CPUs and GPUs.




---



