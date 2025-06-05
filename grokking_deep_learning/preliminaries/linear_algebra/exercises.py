import torch

# Using the `torch.linalg.norm` to apply the Frobenius norm across the last two axes of the tensor.
# Tensor of shape [2, 3, 4]
x = torch.randn(2, 3, 4)

# Apply linalg.norm
norm = torch.linalg.norm(x)

print("Shape:", x.shape)
print("Norm:", norm)

#9
# For tensors of Shape (Batch of Matrices)
x = torch.randn(10, 3, 3)  # 10 matrices
norms = torch.linalg.norm(x, dim=(1,2))  # Frobenius norm per matrix
print(norms.shape)


#12
# Contruct a Tensor with three axes by stacking matrices A, B, and C.
A = torch.randn(100, 200)
B = torch.randn(100, 200)
C = torch.randn(100, 200)

# Stack the Matrices
T = torch.stack([A, B, C])
print(T[0])
print(T[1])

print(T.shape)
print(A.shape)

