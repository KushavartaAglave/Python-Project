#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv(r"C:\Users\hp\Desktop\Churn_Modelling (1).csv")


# In[5]:


df.info()


# #### The data set contains 10,000 rows and 14 columns,with no missing values.

# In[15]:


df.head()


# In[16]:


df.tail()


# 1.RowNumber:sequential row numbers. 2.CustomerID: Unique Id for each customer. 3.Surname: Customers surname. 4.CreditScore:Credit score of the customer. 5.Geography:CUstomers Country(France,spain,germany). 6.Gender:Customers gender.7.Age: Customers Age. 8.Tenure:NUmber of Years the customer has been with the bank. 9.Balance:Customers account balance.10.NUmberofProducts:Number of products the customers uses. 11.HascrCard:Wheather the customers has a creditcard(1=Yes,0=NO).12.IsActiveMember:Wheather the customer is an active member(1=Yes,0=No). 13.EstimatedSalary:Estimated salary of the customers 14.Exited: weather the customers has exited the bank (1=Yes ,0=No)

# In[18]:


df.describe()


# In[25]:


df['Exited'].value_counts(normalize=True)


# In[26]:


df['Geography'].value_counts()


# In[27]:


df['Gender'].value_counts()


# In[34]:


plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Exited',palette='viridis')
plt.title('Customer churn Distribution')
plt.xlabel('Exited (1=yes, 0=NO)')
plt.ylabel('count')
plt.xticks([0,1],['NO','Yes'])
plt.show()


# In[36]:


plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Gender',palette='viridis')
plt.title('Customer churn Distribution by gender')
plt.xlabel('Gender')
plt.ylabel('count')
plt.show()


# In[37]:


plt.figure(figsize=(6,4))
sns.countplot(data=df,x='Geography',palette='viridis')
plt.title('Customer churn Distribution by Geography')
plt.xlabel('Geography')
plt.ylabel('count')
plt.show()


# In[39]:


df.head()


# In[45]:


numric=df.drop(['Surname','Geography','Gender'],axis=1)
numric


# In[46]:


numric.corr()


# In[47]:


sns.heatmap(numric.corr(),cmap='Blues')


# In[53]:


plt.figure(figsize=(6,4))
sns.histplot(df['Age'],kde=True)
plt.show()


# In[56]:


sns.distplot(df['Balance'])
plt.show()


# Descriptive Statistics : 
# 
#     Credit score - Ranges from 350 to 850, with a mean of ~650.
#     
#     Age - Majority of customers are aged between 18 and 92, with a mean age of ~39.
#     
#     Balance - Acount Balances range from $0 to ~$250,000, with  a mean of ~$76,485.
#     
#     EstimatedSalary - Ranges widely between ~$12 and ~$200,000.
#     
#     
# Churn Insights:
#    Exited( Churn Rate) :
#    
#         ~20.37% of customers have exited(1)
#         ~79.63% of customers are retained(0)
#         
# Geography :
# 
#     Customers are primarily from france (50.14%),followed by Germany(25.09%) and Spain(24.77%).
#     
# Gender :
# 
#     Slightly more male customers(54.57%) compared to females(45.43%).
#     
#     

# Observations:
# 
#     1)The churn rate is relatively low (~20%),but this can vary across demographics like geography,gender,and age.
#     
#     2)Germany might need additional focus due to its moderate customers base size and possible churn risks.

# In[ ]:




