import torch


array_tensor = torch.arange(12, dtype=torch.float32)
print(array_tensor)

# unary scalar operation
print(torch.exp(array_tensor))

# Binary scalar operators, which maps pairs of real numbers to a (single) real number
"""
he common standard arithmetic operators for addition (+), subtraction (-), multiplication (*), 
division (/), and exponentiation (**) have all been lifted to elementwise operations 
for identically-shaped tensors of arbitrary shape.
"""

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

# addition
print(x + y)
# subtraction
print(x - y)
# multiplication
print(x * y)
# division
print(x / y)
# exponentiation
print(x ** y)



# we can concatenate multiple tensors, stacking them end-to-end to form a larger one.
# Just provide a list of tensors and tell the system along which axis to concatenate.
# to concatenate along the row, set `dim=0` and along the column set `dim=1`
X = torch.arange(12, dtype=torch.float32).reshape(3,4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# Concatenate along rows (dim=0)
result_row = torch.cat((X, Y), dim=0)
print(result_row)

# Concatenate along columns (dim=1)
result_col = torch.cat((X, Y), dim=1)
print(result_col)


# construct a binary tensor via logical statements
print(X == Y)

# Summing all the elements in the tensor yields a tensor with only one element
print(X.sum())


















































