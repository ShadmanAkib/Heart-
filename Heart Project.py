#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Regular EDA and plotting libraries
import numpy as np # np is short for numpy
import pandas as pd # pandas is so commonly used, it's shortened to pd
import matplotlib.pyplot as plt
import seaborn as sns # seaborn gets shortened to sns

# We want our plots to appear in the notebook
get_ipython().run_line_magic('matplotlib', 'inline')

## Models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

## Model evaluators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve


# In[2]:


df =pd.read_csv("heart-disease.csv")
df


# In[3]:


df.head


# In[4]:


df.tail()


# In[5]:


df["target"].value_counts()


# In[6]:


df["target"].value_counts().plot(kind="bar", color=["red", "lightblue"]);


# In[7]:


df.info()


# In[8]:


df.isna().sum()


# In[9]:


df.describe()


# In[10]:


# Compare target column with sex column
pd.crosstab(df.target, df.sex)


# In[11]:


# Create a plot
pd.crosstab(df.target, df.sex).plot(kind="bar", 
                                    figsize=(10,6), 
                                    color=["salmon", "lightblue"]);


# In[12]:


# Create a plot
pd.crosstab(df.target, df.sex).plot(kind="bar", figsize=(10,6), color=["salmon", "lightblue"])

# Add some attributes to it
plt.title("Heart Disease Frequency for Sex")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Amount")
plt.legend(["Female", "Male"])
plt.xticks(rotation=0); # keep the labels on the x-axis vertical


# In[13]:


import requests


# In[14]:


url = '"ANITA (@farzana_anita) • Instagram photos and videos.html"'
crypto_url = requests.get(url)
crypto_url


# In[ ]:


crypto_url.shape()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




