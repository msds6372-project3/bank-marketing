
# coding: utf-8

# In[207]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

import patsy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import statsmodels.discrete.discrete_model as sm

''' Reading the csv dataset '''

df = pd.read_csv("bank-additional-full.csv", header='infer', delimiter=';')
#df = df.dropna()
print("Dimension of the data {0}".format(data.shape))
print("Dataset has following columns \n {0}".format(list(data.columns)))

'''print(df.head(5))
print("\nNumber of rows {0}".format(len(df.index)))
print(df.index)
print("\n****dtypes in the df**** \n {0} \n \n******list of row and column axis**** \n\n{1}\n\n"
      .format(df.dtypes, df.axes))
'''

#print(df['job'].unique())

''' Exploratory Data set '''

#default - has credit yes = 1, no = 0
df['education']=np.where(df['education'] =='basic.9y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.4y', 'Basic', df['education'])
df['education']=np.where(df['education'] =='basic.6y', 'Basic', df['education'])

df['y'] = np.where(df['y'] == 'yes', '1', '0')
print(df.head(5))
df['y'].value_counts()


#Rename some columns with . as column name
df = df.rename(index=str, columns={"emp.var.rate"   : "emp_var_rate", 
                              "cons.price.idx" : "cons_price_idx",
                              "cons.conf.idx"  : "cons_conf_idx",
                              "nr.employed"    : "nr_employed" })

print(df.head(5))

print("\n***********NULL values in Bank data********\n")
print(df.isnull().sum())

#convert categorical into numeric 
'''
cleanup_nums = {"education":     {"unknown": 0, "illiterate": 1, "Basic": 2,
                                   "high.school" : 3, "professional.course" : 4, "university.degree" : 5},
                "marital": {"unknown": 0, "single": 1, "married": 2, "divorced": 3 },
                 "y" : {"no" : 0, "yes" :1}}

df.replace(cleanup_nums, inplace=True)
df.head()
'''

#frequency distribution plot of how many in each EDUCATION category got credits yes/no
get_ipython().magic('matplotlib inline')
'''
pd.crosstab(df.education,df.y).plot(kind='bar')
plt.title('Purchase Frequency for education type')
plt.xlabel('Education')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_education')

#MARITAL category got credits yes/no
pd.crosstab(df.marital,df.y).plot(kind='bar')
plt.title('Purchase Frequency for marital type')
plt.xlabel('Marital Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_marital')

#JOB category got credits yes/no
pd.crosstab(df.job,df.y).plot(kind='bar')
plt.title('Purchase Frequency for JOB type')
plt.xlabel('JOB Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_job')

#AGE
pd.crosstab(df.age,df.y).plot(kind='bar')
plt.title('Purchase Frequency for AGE type')
plt.xlabel('AGE ')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_age')

#DEFAULT has credit yes or no
pd.crosstab(df.default,df.y).plot(kind='bar')
plt.title('Purchase Frequency for DEFAULT type')
plt.xlabel('DEFAULT CREDIT Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_default')


#Housing, personal, contact method does not realy
#influence the credit purchase or not.

#HOUSING LOAN
pd.crosstab(df.housing,df.y).plot(kind='bar')
plt.title('Purchase Frequency for HOUSING type')
plt.xlabel('HOUSING LOAN Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_housing_loan')

#PERSONAL LOAD
pd.crosstab(df.loan,df.y).plot(kind='bar')
plt.title('Purchase Frequency for PERSONAL LOAN type')
plt.xlabel('PERSONAL LOAN Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_personal_loan')

#CONTACT METHOD for the client
pd.crosstab(df.contact,df.y).plot(kind='bar')
plt.title('Purchase Frequency for CONTACT METHOD type')
plt.xlabel('CONACT Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_personal_contact')



#MONTH Influence on CREDIT purchase
pd.crosstab(df.month,df.y).plot(kind='bar')
plt.title('Purchase Frequency for MONTH type')
plt.xlabel('MONTH Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_personal_month')

#DURATION
pd.crosstab(df.duration,df.y).plot(kind='bar')
plt.title('Purchase Frequency for DURATION ')
plt.xlabel('DURATION Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_personal_duration')

#DAY OF WEEK - does not matter 
pd.crosstab(df.day_of_week,df.y).plot(kind='bar')
plt.title('Purchase Frequency for DAY OF WEEK ')
plt.xlabel('DAY_OF_WEEK Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_personal_day_of_week')

#CAMPAIGN - does not matter 
pd.crosstab(df.campaign,df.y).plot(kind='bar')
plt.title('Purchase Frequency for CAMPAIGN ')
plt.xlabel('campaign Status')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_campaign')

#pdays - does not matter 
pd.crosstab(df.pdays,df.y).plot(kind='bar')
plt.title('Purchase Frequency for PDAYS ')
plt.xlabel('PDAY Status')
plt.ylabel('Frequency of PDAYS')
plt.savefig('purchase_pdays')

#previous - does not matter 
pd.crosstab(df.previous,df.y).plot(kind='bar')
plt.title('Purchase Frequency for PREVIOUS ')
plt.xlabel('PREVIOUS Status')
plt.ylabel('Frequency of PREVIOUS')
plt.savefig('purchase_previous')

#poutcome - does not matter 
pd.crosstab(df.poutcome,df.y).plot(kind='bar')
plt.title('Purchase Frequency for POUTCOME ')
plt.xlabel('POUTCOME Status')
plt.ylabel('Frequency of POUTCOME')
plt.savefig('purchase_poutcome')


#emp_var_rate
pd.crosstab(df.emp_var_rate,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Employemnt varaite rate ')
plt.xlabel('Employemnt varaite rate Status')
plt.ylabel('Frequency of Employemnt varaite rate')
plt.savefig('purchase_employ_var_rate')


#Consumer price index
pd.crosstab(df.cons_price_idx,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Consumer price index ')
plt.xlabel('Consumer price index Status')
plt.ylabel('Frequency of Consumer price index')
plt.savefig('purchase_cons_price_idx')

#Consumer confidence index
pd.crosstab(df.cons_conf_idx,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Consumer confidence index ')
plt.xlabel('Consumer confidence index Status')
plt.ylabel('Frequency of Consumer confidence index')
plt.savefig('purchase_cons_conf_idx')

#Euro Interbank Offered Rate 
pd.crosstab(df.euribor3m,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Euro Interbank Offered Rate  ')
plt.xlabel('Euro Interbank Offered Rate  Status')
plt.ylabel('Frequency of Euro Interbank Offered Rate ')
plt.savefig('purchase_euribor3m')

#Number of employees
pd.crosstab(df.nr_employed,df.y).plot(kind='bar')
plt.title('Purchase Frequency for Number of employees ')
plt.xlabel('Number of employees Status')
plt.ylabel('Frequency of Number of employees')
plt.savefig('purchase_nr_employed')
'''

#After looking at if each of the variables have a role in the response
#heres a scatter plot matrix of all the varaibles against each other to check for multicolinearity

#s = pd.tools.plotting.scatter_matrix(df, alpha=0.2, figsize=(21, 21), diagonal='hist')
#plt.title('Scatter Plot Matrix')
#plt.savefig('Scatter_plot_bank_target_marketing_dependent_varaibles')


#Keep age, job, marital, education, housing, loan, campaign, poutcome, emp_var_rate
#cons_price_idx, cons_conf_idx, euribor3m

#Pick the final varaibles you wanted to consider for the model.
categorical_variables = ['job','marital','education', 'default',
                         'housing','loan','contact','month','day_of_week','poutcome']
for var in categorical_variables:
    mylist = pd.get_dummies(df[var], prefix=var)
    data = df.join(mylist)
    df = data
    

data_vars = df.columns.values.tolist()
variables_to_keep = [i for i in data_vars if i not in categorical_variables]

data_final = df[variables_to_keep]
print("\n **************This is the final selected varaibles ***********\n")
print(data_final.info())


#data_final_vars=data_final.columns.values.tolist()
#print(data_final_vars)


y = ['y']
X = [i for i in variables_to_keep if i not in y]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state=25)

model = LogisticRegression()
mdl = model.fit(data_final[X], data_final[y])
print(mdl.score(data_final[X], data_final[y]))


# In[ ]:

############### Older Work #######################

#Using patsy to create the dataframe of response and independent varaibles
#function = 'y~duration+poutcome+month+contact+age+job';
#y,X = patsy.dmatrices(function, df,return_type='dataframe')

#print(X)

'''
bank_new = pd.DataFrame()
for column in df.columns:
    if df[column].dtypes != 'int64' and column != 'y':
        temp = column + '_id'
        temp = pd.get_dummies(df[column], drop_first=True)
        print(temp.head())
        #result = pd.concat(temp)
        data1 = df.join(temp)

data1.head()
'''

print("\n\n ******* Bank Telemarketing input data *********\n ")
#print(data1.head())
#There is no way to switch off regularization in scikit-learn, 
#but you can make it ineffective by setting the tuning parameter C to a large number.

#model = LogisticRegression(fit_intercept = True, C = 1e9)
#mdl = model.fit(X, y)

#plot of education Vs y
'''
x = df['education']
y = df['y']

plt.scatter(x, y, c="g", alpha=0.5)
plt.xlabel("Education")
plt.ylabel("Credits")
plt.legend(loc=2)
plt.show()
'''

