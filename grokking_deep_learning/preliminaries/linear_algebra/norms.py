import torch

# the norm of a vector tells us how big it is.
# for instance, the `l2` norm measures the (Euclidean) length of a vector.

# the method `norm` calculates the `l2` norm
u = torch.tensor([3.0, -4.0])
l2_norm = torch.norm(u)
print(l2_norm)



# the `l1` norm is also common and its associated measure is called the Manhattan distance.
# `l1` norm sums the absolute values of a vector's element.
# `l1` norm is less sensitive to outliers.
# compose the absolute value of the sum operation.
l1_norm = torch.abs(u).sum()
print(l1_norm)


# Frobenius norms
print(torch.norm(torch.ones((4, 9))))
