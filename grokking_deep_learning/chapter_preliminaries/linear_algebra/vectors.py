import torch

"""
In Python, as in most programming languages, vector indices start at `0`, also known as `zero-based indexing`,
whereas in linear algebra subscripts begin at `1` (one-dimensional indexing)
"""

x = torch.arange(5)
print(x)

# access a tensor's elements via indexing
print(x[2])

# dimensionality of the vector
print(len(x))

# access the length via shape attribute
print(x.shape)