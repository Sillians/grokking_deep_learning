import torch

# expressing the sum of elements in a vector x of length n
x = torch.arange(3, dtype=torch.float32)
print(x)
print(x.sum())


# express sums over the elements of tensors of arbitrary shape, we simply sum over all its axes
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print(A.shape)
print(A.sum())

# summation over all elements along the rows (axis 0), specify axis=0 in `sum`.
print(A.shape)
sum_rows = A.sum(axis=0)
print(sum_rows)
print(sum_rows.shape)

# specifying `axis=1` will reduce the column dimension (axis 1) by summing up elements of all the columns
sum_cols = A.sum(axis=1)
print(sum_cols)
print(sum_cols.shape)


# Reducing a matrix along both rows and columns via summation is equivalent to summing up all the elements of the matrix
print(A.sum(axis=[0, 1]) == A.sum())


# related quantity is the mean, also called the average
print(A.mean())
print(A.sum() / A.numel())


# Can also use the function for calculating the mean to reduce a tensor along specific axes
# * along row axis
print(A.mean(axis=0))
print(A.sum(axis=0) / A.shape[0])

# *along column axis
print(A.mean(axis=1))
print(A.sum(axis=1) / A.shape[1])




