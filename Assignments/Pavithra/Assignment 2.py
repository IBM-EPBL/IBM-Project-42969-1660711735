#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[8]:


#load the dataset
df=pd.read_csv("Churn_Modelling.csv")


# In[9]:


#display first 5 rows
df.head()


# In[10]:


#descriptive statistical
df.describe()


# In[27]:


#split dependent and independent varaibles
x=df.iloc[:,3:13].values
y=df.iloc[:,13:14].values
x.shape


# In[28]:


y.shape


# In[34]:


#split the data into the training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)


# In[35]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[36]:


#missing values
df.isna()


# In[17]:


df.isnull().sum()


# In[18]:


# finding the outliers
df.skew()


# In[29]:


#Categorical colums and encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer([("oh",OneHotEncoder(),[1,2])],remainder="passthrough")
x=ct.fit_transform(x)
x.shape


# In[30]:


df["Gender"].unique()


# In[ ]:





# In[37]:


#Univariate Analysis
df["Balance"].plot(kind='hist');


# In[38]:


#Bi-Variate Analysis
cy=df[df.Exited==1].Tenure
cn=df[df.Exited==0].Tenure
plt.title("Churn prediction")
plt.xlabel("Tenure")
plt.ylabel("No of customers")
plt.hist([cy,cn],color=['green','red'],label=["churn=yes"])
plt.show()


# In[39]:


#Multi-variate Analysis
sns.pairplot(df)


# In[40]:


#Scale the independent variables
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[41]:


import joblib
joblib.dump(ct,"churn.pkl")


# In[42]:


joblib.dump(sc,"churnsc.pkl")


# In[ ]:




