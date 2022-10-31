#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os


# In[3]:


os.chdir("C:\\Users\\india\\OneDrive\\Desktop\\data sets")


# In[4]:


rawdata=pd.read_csv('startup.csv')


# In[5]:


rawdata.head()


# In[6]:


rawdata.shape


# In[7]:


rawdata.describe


# In[8]:


target=rawdata['Profit']


# In[9]:


target


# In[10]:


from statsmodels.graphics.gofplots import qqplot
qqplot(target,line='s')


# In[11]:


sns.boxplot(target)


# In[12]:


sns.distplot(target)


# In[13]:


target.isna().sum()


# In[14]:


q1=target.quantile(0.25)
q3=target.quantile(0.75)


# In[15]:


q1


# In[16]:


q3


# In[17]:


IQR=q3-q1


# In[18]:


IQR


# In[19]:


var=[]
for i in target:
    var.append(i)


# In[20]:


var


# In[21]:


infence=q1-(1.5*IQR)
outfence=q3+(1.5*IQR)


# In[22]:


infence


# In[23]:


outfence


# In[24]:


location=[]
for i in var:
    if i<infence or i>outfence:
        out=var.index(i)
        location.append(out)


# In[25]:


location


# In[26]:


rawdata.drop(49,axis=0,inplace=True)


# In[27]:


rawdata.shape


# In[28]:


rawdata.reset_index(inplace=True)


# In[29]:


rawdata.drop('index',axis=1,inplace=True)


# In[30]:


rawdata.head()


# In[31]:


target1=rawdata["Profit"]


# In[32]:


target1


# In[33]:


target1.shape


# In[34]:


sns.distplot(target1)


# In[35]:


sns.boxplot(target1)


# In[36]:


plt.figure(figsize=(8,8))
sns.heatmap(rawdata.corr(),annot=True,cmap='coolwarm')


# In[37]:


idv=rawdata.drop(['Profit'],axis=1)


# In[38]:


idv


# In[58]:


sns.boxplot(idv)


# In[59]:


from sklearn.preprocessing import MinMaxScaler


# In[61]:


scalar1=MinMaxScaler()


# In[62]:


col=idv.columns
scalar_var_independent=pd.DataFrame(scalar1.fit_transform(idv),columns=col)


# In[63]:


scalar_var_independent


# In[64]:


sns.boxplot(scalar_var_independent)


# In[66]:


target1.shape


# In[68]:


scalar_var_independent.shape


# In[69]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(scalar_var_independent,target1,test_size=0.3,random_state=0)


# In[70]:


x_train.shape


# In[71]:


y_train.shape


# In[72]:


y_test.shape


# In[73]:


x_test.shape


# In[74]:


import statsmodels.api as sm


# In[75]:


model2=sm.OLS(y_train,x_train).fit()


# In[76]:


model2.summary()


# In[77]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(scalar_var_independent,target1,test_size=0.2,random_state=0)


# In[78]:


x_train.shape


# In[79]:


y_train.shape


# In[80]:


x_test.shape


# In[81]:


y_test.shape


# In[82]:


import statsmodels.api as sm


# In[83]:


model2=sm.OLS(y_train,x_train).fit()


# In[84]:


model2.summary()


# In[ ]:




