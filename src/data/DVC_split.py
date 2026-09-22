#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
print(os.getcwd())


# In[2]:


import os
os.chdir("/home/pitipillon/examen-dvc")
print(os.getcwd())


# In[3]:


import pandas as pd


# In[4]:


from sklearn.model_selection import train_test_split


# In[5]:


df = pd.read_csv("data/raw/raw.csv")

X = df.drop(columns=["silica_concentrate", "date"])
y = df["silica_concentrate"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train.to_csv("data/processed/X_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)

print("Split termine.")



# In[ ]:




