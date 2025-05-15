import os
import pandas as pd

df = '''NumRooms,RoofType,Price
        NA,NA,127500
        2,NA,106000
        4,Slate,178100
        NA,NA,140000'''

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')

with open(data_file, 'w') as f:
    f.write(df)

# load dataset
data = pd.read_csv(data_file)
print(data)
