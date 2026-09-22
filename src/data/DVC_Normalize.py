#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("/home/pitipillon/examen-dvc")
print(os.getcwd())


# In[2]:


import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

pd.DataFrame(X_train_scaled, columns=X_train.columns).to_csv(
    "data/processed/X_train_scaled.csv", index=False
)
pd.DataFrame(X_test_scaled, columns=X_test.columns).to_csv(
    "data/processed/X_test_scaled.csv", index=False
)

joblib.dump(scaler, "models/scaler.pkl")

print("Normalisation terminee")


# In[3]:


print(X_train.dtypes)
print(X_train.isna().sum())


# In[ ]:




