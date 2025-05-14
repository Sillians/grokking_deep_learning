import torch

array1 = torch.arange(12, dtype=torch.float32)
new_array1 = array1.reshape(3, 4)
print(new_array1)

# we can access tensor elements by indexing (starting from 0).
# access whole ranges of indices via slicing (array[start:stop])
# returned value includes the first index (start) but not the last (stop)
print(new_array1[-1], new_array1[1:3])


# write elements of a matrix by specifying indices
new_array1[1, 2] = 30
print(new_array1)


# assign multiple elements the same value
new_array1[:2 :] = 23
print(new_array1)
