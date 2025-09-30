#!/usr/bin/env python
# coding: utf-8

# ## Q1. Pandas version
# 

# What version of Pandas did you install?
# 
# You can get the version information using the __version__ field:
# 

# In[1]:


import pandas as pd
pd.__version__


# 
# Getting the data
# For this homework, we'll use the Car Fuel Efficiency dataset. Download it from here.
# 
# You can do it with wget:
# 

# In[2]:



# 
# 
# 
# Or just open it with your browser and click "Save as...".
# 
# Now read it with Pandas.

# ## Q2. Records count
# 

# How many records are in the dataset?
# 
# 

# In[3]:


df = pd.read_csv("../car_fuel_efficiency.csv")


# In[4]:


print(df.shape)


# ## Q3. Fuel types
# 

# How many fuel types are presented in the dataset?

# In[5]:


df.head()


# In[6]:


print("Unique fuel types:", df['fuel_type'].nunique())


# ## Q4. Missing values
# 

# How many columns in the dataset have missing values?

# In[7]:


print("Number of columns with missing values:", df.isnull().any().sum())


# ## Q5. Max fuel efficiency
# 

# What's the maximum fuel efficiency of cars from Asia?

# In[8]:


asia_max_eff = df[df['origin'] == 'Asia']['fuel_efficiency_mpg'].max()


# In[9]:


print("Max fuel efficiency (Asia):", asia_max_eff)


# ## Q6. Median value of horsepower
# 

# 1. Find the median value of the horsepower column in the dataset.
# 2. Next, calculate the most frequent value of the same horsepower column.
# 3. Use the fillna method to fill the missing values in the horsepower column with the most frequent value from the previous step.
# 4. Now, calculate the median value of horsepower once again.
# 
# Has it changed?

# In[10]:


median_hp_before = df['horsepower'].median()
mode_hp = df['horsepower'].mode()[0]

# Fill missing values with most frequent value
df['horsepower'] = df['horsepower'].fillna(mode_hp)
median_hp_after = df['horsepower'].median()


# In[11]:


mode_hp


# In[12]:


print("Median horsepower (before):", median_hp_before)
print("Most frequent horsepower:", mode_hp)
print("Median horsepower (after fillna):", median_hp_after)
print("Has median changed?:", 
      "Yes, it increased" if median_hp_after > median_hp_before 
      else "Yes, it decreased" if median_hp_after < median_hp_before 
      else "No")


# ## Q7. Sum of weights
# 

# In[13]:


import numpy as np

df_asia = df[df['origin'] == 'Asia'][['vehicle_weight', 'model_year']].head(7)
X = df_asia.values
XTX = X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv.dot(X.T).dot(y)
print("w:", w)
print("Sum of w:", w.sum())


