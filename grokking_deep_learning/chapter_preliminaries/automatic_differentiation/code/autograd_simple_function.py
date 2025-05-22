import torch

x = torch.arange(4.0)
print(x)

x.requires_grad_(True) # can also create x = torch.arange(4.0, requires_grad=True
print(x.grad) # None gradient by default

# calculate the function of x and assign the result to y
y = 2 * torch.dot(x, x)
print(y)

# we can now take the gradient of y with respect to x by calling its backward method.
y.backward()
print(x.grad) # can now access the gradient

# verify that the automatic gradient computation and the expected result are identical
print(x.grad == 4 * x)





# Calculate another function of x and take its gradient
# reset the gradient buffer
x.grad.zero_() # Reset the gradient
y = x.sum()
y.backward()
print(x.grad)