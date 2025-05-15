import torch

# conversion of any appropriately sized `m x n` tensor into an `m x n` matrix by passing the desired shape to reshape
A = torch.arange(6).reshape(3, -1)
print(A)

# flip the axes by exchanging the matrix's rows and columns, the result is called its transpose.
print(A.T)

# symmetric matrices are the subset of square matrices that are equal to their own transposes.
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)