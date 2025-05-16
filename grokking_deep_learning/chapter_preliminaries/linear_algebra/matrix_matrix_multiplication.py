import torch

# matrix-matrix multiplication is straight-forward
# In the example below, `A` is a matrix with two rows and three columns, and
# B is a matrix with three rows and four columns, after multiplication we obtain a matrix with two rows and four columns

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = torch.ones(3, 4)
matrix_matrix = torch.mm(A, B)
print(matrix_matrix)
print(matrix_matrix.shape)

# OR
print(A@B)