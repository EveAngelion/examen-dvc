#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("/home/pitipillon/examen-dvc")
print(os.getcwd())


# In[ ]:


import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

best_params = joblib.load("models/best_params.pkl")

model = RandomForestRegressor(random_state=42, **best_params)
model.fit(X_train, y_train)

joblib.dump(model, "models/gbr_model.pkl")

print("Modele entraine")

