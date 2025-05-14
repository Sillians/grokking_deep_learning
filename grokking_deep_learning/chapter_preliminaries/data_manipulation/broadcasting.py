import torch

"""
Under certain conditions, even when shapes differ, we can still perform elementwise binary operations 
by invoking the broadcasting mechanism. 

Broadcasting works according to the following two-step procedure: 

i. expand one or both arrays by copying elements along axes with length 1 so that after this transformation, the two tensors have the same shape; 
ii. perform an elementwise operation on the resulting arrays.
"""

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a, b)


# Broadcasting produces a larger 3 × 2 matrix by replicating matrix `a` along the columns and
# matrix `b` along the rows before adding them elementwise.
new = a + b
print(new)

