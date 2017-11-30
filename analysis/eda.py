import pandas as pd
import numpy as np
import matplotlib.pyplot as mt
import scipy as sp

# Source: https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python
# Load in the data
filepath = "/Users/Jostein/Grad School/SMU/6372/project3/bank-marketing/data/bank-full.csv"
bank = pd.read_csv(filepath, header='infer', delimiter=';')
bank.describe()
print(bank)