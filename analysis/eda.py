import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import seaborn as sns
import statistics as st

# TO DO: Assess the fit- Hosmer and Lemeshow Goodness of fit
# TO DO: ROC Curve

# Load in the data
# Source:
# https://www.datacamp.com/community/tutorials/exploratory-data-analysis-python
filepath = "/Users/Jostein/Grad School/SMU/6372/project3/bank-marketing/data/bank-full.csv"
bank = pd.read_csv(filepath, header='infer', delimiter=';')

# Check if data loaded in properly
print(bank)

# Check how clean the data is
# If there is missing data, we could use fillna(mean) for imputation
pd.isnull(bank).sum()

# Deal with unknown values
# No NA's, but there are unknowns
# Let's plot each variable and see in which the unknowns occur
sns.set(style="darkgrid")
sns.countplot(y='job', data=bank, palette='hls') # Has unknowns
sns.countplot(y='marital', data=bank, palette='hls')
sns.countplot(y='education', data=bank, palette='hls') # Has unknowns
sns.countplot(x='default', data=bank, palette='hls')
#sns.countplot(y='balance', data=bank, palette='hls')
sns.countplot(x='housing', data=bank, palette='hls')
sns.countplot(x='loan', data=bank, palette='hls')
sns.countplot(x='contact', data=bank, palette='hls') # Unknowns, silly variable
sns.countplot(x='poutcome', data=bank, palette='hls') # Has bigly unknowns

# Return the number of unknowns for variables:
# job, education, contact, and poutcome
print('The number of unknowns in job variable is: '
      + str(sum(bank['job'] == 'unknown')))

print('The number of unknowns in education variable is: '
      + str(sum(bank['education'] == 'unknown')))

print('The number of unknowns in contact variable is: '
      + str(sum(bank['contact'] == 'unknown')))

print('The number of unknowns in poutcome variable is: '
      + str(sum(bank['poutcome'] == 'unknown')))

# Imputate the unknown values with the mode of their respective variable
bank.replace({'job': {'unknown': 'blue-collar'}}, inplace=True)
bank.replace({'education': {'unknown': 'secondary'}}, inplace=True)
bank.replace({'contact': {'unknown': 'cellular'}}, inplace=True)
bank.replace({'poutcome': {'unknown': 'failure'}}, inplace=True)

# Check to see if unknowns were imputed
if sum(bank['job'] == 'unknown') == 0:
    print('Unknowns in job variable imputed successfully!')
if sum(bank['education'] == 'unknown') == 0:
    print('Unknowns in education variable imputed successfully!')
if sum(bank['contact'] == 'unknown') == 0:
    print('Unknowns in contact variable imputed successfully!')
if sum(bank['poutcome'] == 'unknown') == 0:
    print('Unknowns in poutcome variable imputed successfully!')

sns.countplot(y='job', data=bank, palette='hls')
sns.countplot(y='education', data=bank, palette='hls')
sns.countplot(x='contact', data=bank, palette='hls')
sns.countplot(x='poutcome', data=bank, palette='hls')

sum(bank['education'] == 'unknown')
sns.countplot(y='job', data=bank, palette='hls')

# Snapshot of the data
bank.head()
# TO DO: No N/A's, but figure out what to do with Unknown category

# Summary statistics for variables
bank.describe()

# Turn the data into a data frame
# Source:
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
df = pd.DataFrame(data=bank)

# Check the data types
df.dtypes

# https://seaborn.pydata.org/generated/seaborn.countplot.html
sns.set(style="darkgrid")
sns.countplot(x='y', data=bank, palette='hls')

# Default, housing, and loan are unkown

# Factorize categorical variables
# labels, levels = pd.factorize(df.Class)
# Source: http://www.data-mania.com/blog/logistic-regression-example-in-python/
# Python forgive me, for I have sinned
# Yes I know the following is awful, but the for loop wasn't working
job_dmy = pd.get_dummies(bank['job'], drop_first=True)
marital_dmy = pd.get_dummies(bank['marital'], drop_first=True)
education_dmy = pd.get_dummies(bank['education'], drop_first=True)
default_dmy = pd.get_dummies(bank['default'], drop_first=True)
housing_dmy = pd.get_dummies(bank['housing'], drop_first=True)
loan_dmy = pd.get_dummies(bank['loan'], drop_first=True)
contact_dmy = pd.get_dummies(bank['contact'], drop_first=True)
month_dmy = pd.get_dummies(bank['month'], drop_first=True)
poutcome_dmy = pd.get_dummies(bank['poutcome'], drop_first=True)
y_dmy = pd.get_dummies(bank['y'], drop_first=True)

# Check the structure of the dummy variables
job_dmy.head()
marital_dmy.head()
education_dmy.head()
default_dmy.head()
housing_dmy.head()
loan_dmy.head()
contact_dmy.head()
month_dmy.head()
poutcome_dmy.head()
y_dmy.head()

bank.drop(['job', 'marital', 'education', 'default', 'housing',
           'loan', 'contact', 'month', 'poutcome', 'y'], axis=1, inplace=True)

# Check if variables were dropped
bank.head()

# Create new data frame with int and dummy variables
bank_dmy = pd.concat([bank, job_dmy, marital_dmy, education_dmy, default_dmy,
                      housing_dmy, loan_dmy, contact_dmy, month_dmy,
                      poutcome_dmy, y_dmy], axis=1)
bank_dmy.head()
print(bank_dmy)


# Asha
%matplotlib inline
pd.crosstab(df.education,df.y).plot(kind='bar')
plt.title('Purchase Frequency for education type')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_education')
