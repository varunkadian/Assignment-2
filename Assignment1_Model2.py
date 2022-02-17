#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


#importing the csv file and creating a dataframe
df = pd.read_csv("dataset.csv")


# ### EDA and Statistical Analysis

# In[4]:


df


# In[5]:


df[df.duplicated(['id'], keep=False)]


# In[6]:


# To display no. of rows and columns in the dataset
df.shape


# In[7]:


# To calculate basic statistical data except id
df.loc[:, df.columns != 'id'].describe()


# In[8]:


# To check no. of unique values in the dataset.
df.nunique()


# In[9]:


# To check how many values are null
df.isnull().sum()


# In[10]:


# another way to check missing values
import missingno as msno
msno.bar(df);


# No missing Values

# In[11]:


# To show different datatypes of the columns
df.dtypes


# In[12]:


# Using sweetviz to create further visualizations 
import sweetviz as sv


# In[13]:


report= sv.analyze(df)
report.show_html()


# In[14]:


import seaborn as sns
sns.countplot(df['diagnosis'])


# ### Outlier Detection and Handling

# In[15]:


# Outlier detection for radius_mean column
Q1 = df["radius_mean"].quantile(0.25)
Q3 = df["radius_mean"].quantile(0.75)
IQR = Q3-Q1
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR


# In[16]:


df[(df["radius_mean"] > upper)]


# In[17]:


# dropping outlier values
df = df.drop(df[df["radius_mean"] > upper].index)


# In[18]:


# no outliers left for radius_mean
df[(df["radius_mean"] > upper)]


# In[19]:


# no outliers left for radius_mean
df[(df["radius_mean"] < lower)]


# ### Cross Validation to check accuracy of Decision Tree, Random Forest and KNN algorithms.

# In[20]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


# In[21]:


models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('DT',  DecisionTreeClassifier()))
models.append(('RF',  RandomForestClassifier(n_estimators=100)))


# In[22]:


# all the columns except diagnosis (target value / dependent variable) and id are set as independent variables.
X=df2 = df[df.columns.difference(['diagnosis', 'id'])]


# In[23]:


# diagnosis is the target variable
y= df[['diagnosis']]


# ### Normalization

# In[24]:


X.hist(figsize=(35,25),rwidth=0.8,grid=False)


# It can be interepreted with the histograms that most of the data is not normal.

# To make the data normal, sklearn preprocesssing library's normalize function is used.

# In[25]:


from sklearn import preprocessing
normalized_X = preprocessing.normalize(X)


# Normalized data is then used for modeling.

# ### Cross Validation and Modeling

# In[26]:


# Train/Test split
X_train_cross, X_test_cross, y_train_cross, y_test_cross = train_test_split(normalized_X, y, stratify = df.diagnosis,random_state=1)


# ### Model 2

# In[32]:


dt = DecisionTreeClassifier(max_depth = 8, criterion = 'entropy')
dt.fit(X_train_cross,y_train_cross)
y_predicted_rf = dt.predict(X_test_cross) 
acc=accuracy_score(y_test_cross, y_predicted_rf)
acc


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




