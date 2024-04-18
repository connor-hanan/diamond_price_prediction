import pandas as pd
import pickle

# Loading random forest model
file = open(r"C:\Users\connor.hanan\workspace\Python\Projects\diamond_price_prediction\Resources\random_forest_model.pkl", 'rb')
model = pickle.load(file)

# Feature Engineering
d_dataset = pd.get_dummies(dataset)
d_dataset = d_dataset.dropna(axis=1)
X = d_dataset.drop(['price', 'x', 'y', 'z'], axis=1)
dataset['predictions'] = model.predict(X)