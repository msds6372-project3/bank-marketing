import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns

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
# TO DO: No N/A's, but figure out what to do with Unknown category

# Summary statistics for variables
bank.describe()

# Turn the data into a data frame
# Source: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
df = pd.DataFrame(data=bank)

# Check the data types
df.dtypes

# https://seaborn.pydata.org/generated/seaborn.countplot.html
sns.set(style="darkgrid")
sns.countplot(x='y', data=bank, palette='hls')


# Factorize categorical variables
# labels, levels = pd.factorize(df.Class)
# Source: http://www.data-mania.com/blog/logistic-regression-example-in-python/

# For loop to encode dummy variables
bank_new = pd.DataFrame()
for column in bank.columns:
    if bank[column].dtypes != 'int64' and column != 'y':
        temp = column + '_id'
        temp = pd.get_dummies(bank[column], drop_first=True)
        print(temp.head())
        #result = pd.concat(temp)
        data1 = bank.join(temp)

data1.head()

# Asha
%matplotlib inline
pd.crosstab(df.education,df.y).plot(kind='bar')
plt.title('Purchase Frequency for education type')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_education')
