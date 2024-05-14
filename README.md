# Project Overview
---
Using everyones favorite Diamond dataset, I was able to predict the selling price of the Diamonds in the data with a high degree of accuracy wiht the variables I was given. 
### Tools
- Python was used to transform, explore, evaluate the data, and to build the machine learning model.
  - Powershell was used to create a custom 'Conda' virtual enviroment and to install all the nessecary packages.
- PowerBI was used to visualize the data and the model results.
## Python
In the following section I'll walkthrough key points of the project as well as some of the code.
---
To give you an idea of what the data looks like I'll first display the first 5 rows out of the 53,940 in the entire dataset.
```python
df.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>carat</th>
      <th>cut</th>
      <th>color</th>
      <th>clarity</th>
      <th>depth</th>
      <th>table</th>
      <th>price</th>
      <th>x</th>
      <th>y</th>
      <th>z</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.23</td>
      <td>Ideal</td>
      <td>E</td>
      <td>SI2</td>
      <td>61.5</td>
      <td>55.0</td>
      <td>326</td>
      <td>3.95</td>
      <td>3.98</td>
      <td>2.43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.21</td>
      <td>Premium</td>
      <td>E</td>
      <td>SI1</td>
      <td>59.8</td>
      <td>61.0</td>
      <td>326</td>
      <td>3.89</td>
      <td>3.84</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.23</td>
      <td>Good</td>
      <td>E</td>
      <td>VS1</td>
      <td>56.9</td>
      <td>65.0</td>
      <td>327</td>
      <td>4.05</td>
      <td>4.07</td>
      <td>2.31</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29</td>
      <td>Premium</td>
      <td>I</td>
      <td>VS2</td>
      <td>62.4</td>
      <td>58.0</td>
      <td>334</td>
      <td>4.20</td>
      <td>4.23</td>
      <td>2.63</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.31</td>
      <td>Good</td>
      <td>J</td>
      <td>SI2</td>
      <td>63.3</td>
      <td>58.0</td>
      <td>335</td>
      <td>4.34</td>
      <td>4.35</td>
      <td>2.75</td>
    </tr>
  </tbody>
</table>
</div>

## Feature Engineering
In order to reduce the complexity of the data I create a ratio of two of two demensions and elect to drop the original columns
```Python
df['xy'] = df['x']/df['y'] 

# Encode categorical variables into machine readable values
d_df = pd.get_dummies(df)
X = d_df.drop(['price', 'x', 'y', 'z'], axis=1) # Dropping target variable & highly correlated columns
y = d_df['price']

corr_heatmap(d_df.corr()) # Calling my correlation variable (see notebook)

```
### Correlation Heatmap
![alt text](correlation_heatmap)

## Model Results
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RMSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>KNN</th>
      <td>1170.659452</td>
    </tr>
    <tr>
      <th>MLR</th>
      <td>1120.830488</td>
    </tr>
    <tr>
      <th>RF</th>
      <td>552.995054</td>
    </tr>
    <tr>
      <th>Lasso</th>
      <td>1120.723179</td>
    </tr>
    <tr>
      <th>Null</th>
      <td>3942.168776</td>
    </tr>
  </tbody>
</table>
</div>

As you can see the Random Forest has the best score which is the model I choose
```Python
# Save ML model to disk
import pickle

directory_path = r"..\workspace\projects\Diamond_Price_Prediction\Resources"
file_name = 'random_forest_model.pkl'
full_path = f"{directory_path}\\{file_name}"

with open(full_path, 'wb') as file:
    pickle.dump(knn, file)

# Saving the processed data as a csv
processed_data = 'processed_diamond_data'

df.to_csv(f"{directory_path}\\{processed_data}", index=False)
