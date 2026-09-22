#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("/home/pitipillon/examen-dvc")
print(os.getcwd())


# In[ ]:


import json
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

model = joblib.load("models/gbr_model.pkl")
predictions = model.predict(X_test)

pd.DataFrame({"predictions": predictions}).to_csv(
    "data/predictions.csv", index=False
)

scores = {
    "mse": mean_squared_error(y_test, predictions),
    "r2": r2_score(y_test, predictions),
}

with open("metrics/scores.json", "w") as f:
    json.dump(scores, f, indent=4)


