import torch


# matrix-vector products are useful.
# we can represent rotations as multiplications by certain square matrices.
# we can also use matrix-vector products to describe the key calculation involved in computing the outputs of each layer in a neural network given the outputs from the previous layer.

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
x = torch.arange(3, dtype=torch.float32)

print(A.shape)
print(x.shape)


# we can use the `mv` function to express a matrix-vector product in code.
# The column dimension of A (its length along axis 1) must be the same as the dimension of x (its length)
matrix_vector = torch.mv(A, x)
print(matrix_vector)
print(matrix_vector.shape)

# OR
print(A@x)