import torch

# Keep the number of axes unchanged when invoking the function for calculating the sum or mean.
# Matters when we want to use the broadcast mechanism.

A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
sum_A = A.sum(axis=1, keepdim=True)
print(sum_A)
print(sum_A.shape)


# since `sum_A` keeps its two axes after summing each row, we can divide `A` by `sum_A` with broadcasting to create a matrix where each row sums up to 1.
result = A / sum_A
print(result)
print(result.sum(axis=1))


# calculate the cumulative sum of elements of A along some axis, say `axis=0` (row by row), we can use the `cumsum` function to achieve this.
cummsum = A.cumsum(axis=0)
print(cummsum)