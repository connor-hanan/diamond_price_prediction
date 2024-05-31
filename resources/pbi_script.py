# 'dataset' holds the input data for this script
import pandas as pd
import pickle

# Loading random forest model & scaler
file_path = r"C:\Users\conno\workspace\projects\diamond_price_prediction\resources\random_forest_model.pkl"
scaler_path = r"C:\Users\conno\workspace\projects\diamond_price_prediction\resources\scaler.pkl"
with open(file_path, 'rb') as file:
    model = pickle.load(file)
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Feature Engineering
d_dataset = pd.get_dummies(dataset)
d_dataset = d_dataset.drop(['price', 'x', 'y', 'z'], axis=1)
X = scaler.transform(d_dataset)

# Make predictions
dataset['predictions'] = model.predict(X)