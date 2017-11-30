import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Source: https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python
# Load in the data
filepath = "/Users/Jostein/Grad School/SMU/6372/project3/bank-marketing/data/bank-full.csv"
bank = pd.read_csv(filepath, header='infer', delimiter=';')

# Check if data loaded in properly
print(bank)

# Check how clean the data is
# If there is missing data, we could use fillna(mean) for imputation
pd.isnull(bank).sum()

# Snapshot of the data
bank.head()
# Summary statistics for variables
bank.describe()

# Turn the data into a data frame
# Source: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
df = pd.DataFrame(data=bank)

# Check the data types
df.dtypes

#labels, levels = pd.factorize(df.Class)
