import torch

A = torch.arange(6, dtype=torch.float32).reshape(2, -1)
B = A.clone()      # Assign a copy of A to B by allocating new memory

print(A)
print(A + B)

# Product of two matrices is called their Hadamard product.
print(A * B)


# Adding or multiplying a scalar and a tensor produces a result with the same shape as the original tensor.
a = 2
X = torch.arange(24).reshape(2, 3, 4)

print(a + X)
print((a * X).shape)