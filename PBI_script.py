import pandas as pd
import pickle

file = open(r"C:\Users\connor.hanan\workspace\Python\Projects\diamond_price_prediction\Resources\random_forest_model.pkl", 'rb')
model = pickle.load(file)

df = dataset
d_df = pd.get_dummies(df)
X = d_df.drop(['price', 'x', 'y', 'z'], axis=1)
df['predictions'] = model.predict(X)