#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("/home/pitipillon/examen-dvc")
print(os.getcwd())


# In[2]:


import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2,5],
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1,
)
grid.fit(X_train, y_train)

joblib.dump(grid.best_params_, "models/best_params.pkl")

print("Meilleurs parametres :", grid.best_params_)


