from ucimlrepo import fetch_ucirepo
import torch

# fetch dataset
heart_disease = fetch_ucirepo(id=45)

# data (as pandas dataframes)
X = heart_disease.data.features
y = heart_disease.data.targets

# print the inputs
print(X)
print(X.shape)

# print the target
print(y)
print(y.shape)

# metadata
print(heart_disease.metadata)

# variable information
print(heart_disease.variables)


# convert to the tensor format
X_tensor = torch.tensor(X.to_numpy(dtype=float))
y_tensor = torch.tensor(y.to_numpy(dtype=float))
print(X_tensor)
print(y_tensor)