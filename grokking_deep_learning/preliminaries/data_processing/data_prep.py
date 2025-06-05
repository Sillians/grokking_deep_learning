import pandas as pd
from io import StringIO

"""
For categorical input fields, we can treat NaN
as a category. Since the RoofTypecolumn takes values Slateand NaN, pandascan convert
this column into two columns RoofType_Slateand RoofType_nan. 


A row whose roof type
is Slate will set values of RoofType_Slate and RoofType_nan to 1 and 0, respectively.
The converse holds for a row with a missing RoofType value.
"""

# Load the data
data = pd.read_csv('../data/house_tiny.csv', na_values=["NA"])
print(data)

# Get the input and output columns
inputs = data.drop(columns=["RoofType", "Price"])
targets = data.iloc[:, 2]

# Derive the new columns for the inputs
inputs["RoofType_Slate"] = (data["RoofType"] == "Slate").astype(bool)
inputs["RoofType_Nan"] = data["RoofType"].isna().astype(bool)
print(inputs)


# For missing numerical values, one common heuristic is to replace the NaN entries with the
# mean value of the corresponding column.

# Replace NaNs in 'NumRooms' with the column mean, and ensure the column is numeric
inputs["NumRooms"] = pd.to_numeric(inputs["NumRooms"], errors='coerce')
inputs["NumRooms"].fillna(inputs["NumRooms"].mean(), inplace=True)
print(inputs)


# save
inputs.to_csv('prepped_inputs.csv', index=False)
targets.to_csv('prepped_targets.csv', index=False)