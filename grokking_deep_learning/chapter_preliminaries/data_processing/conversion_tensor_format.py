import torch
import pandas as pd

# Load the inputs and target data
inputs = pd.read_csv("prepped_inputs.csv")
targets = pd.read_csv("prepped_targets.csv")

print(inputs)

# Now that all the entries in inputsand targetsare numerical, we can load them into a tensor
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))

# input
print(X)

# target
print(y)