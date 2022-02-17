#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd


# In[32]:


#importing the csv file and creating a dataframe
df = pd.read_csv("dataset.csv")


# ### EDA and Statistical Analysis

# In[33]:


df


# In[34]:


df[df.duplicated(['id'], keep=False)]


# In[35]:


# To display no. of rows and columns in the dataset
df.shape


# In[36]:


# To calculate basic statistical data except id
df.loc[:, df.columns != 'id'].describe()


# In[37]:


# To check no. of unique values in the dataset.
df.nunique()


# In[38]:


# To check how many values are null
df.isnull().sum()


# In[39]:


# another way to check missing values
import missingno as msno
msno.bar(df);


# No missing Values

# In[40]:


# To show different datatypes of the columns
df.dtypes


# In[18]:


# Using sweetviz to create further visualizations 
import sweetviz as sv


# In[19]:


report= sv.analyze(df)
report.show_html()


# In[41]:


import seaborn as sns
sns.countplot(df['diagnosis'])


# ### Outlier Detection and Handling

# In[42]:


# Outlier detection for radius_mean column
Q1 = df["radius_mean"].quantile(0.25)
Q3 = df["radius_mean"].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR


# In[43]:


df[(df["radius_mean"] > upper)]


# In[44]:


# dropping outlier values
df = df.drop(df[df["radius_mean"] > upper].index)


# In[45]:


# no outliers left for radius_mean
df[(df["radius_mean"] > upper)]


# In[46]:


# no outliers left for radius_mean
df[(df["radius_mean"] < lower)]


# ### Cross Validation to check accuracy of Decision Tree, Random Forest and KNN algorithms.

# In[47]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# In[48]:


models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('DT',  DecisionTreeClassifier()))
models.append(('RF',  RandomForestClassifier(n_estimators=100)))


# In[49]:


# all the columns except diagnosis (target value / dependent variable) and id are set as independent variables.
X=df2 = df[df.columns.difference(['diagnosis', 'id'])]


# In[50]:


# diagnosis is the target variable
y= df[['diagnosis']]


# ### Normalization

# In[51]:


X.hist(figsize=(35,25),rwidth=0.8,grid=False)


# It can be interepreted with the histograms that most of the data is not normal.

# To make the data normal, sklearn preprocesssing library's normalize function is used.

# In[52]:


from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)


# Normalized data is then used for modeling.

# ### Cross Validation and Modeling

# In[53]:


# Train/Test split
X_train_cross, X_test_cross, y_train_cross, y_test_cross = train_test_split(normalized_X, y, stratify = df.diagnosis,random_state=1)


# ### Model 1

# In[54]:


rf = RandomForestClassifier(n_estimators=500, random_state=17, n_jobs= -1)
rf.fit(X_train_cross,y_train_cross)
y_predicted_rf = rf.predict(X_test_cross) 
acc=accuracy_score(y_test_cross, y_predicted_rf)
acc


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




