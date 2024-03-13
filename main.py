"""
Original file is located at
    https://colab.research.google.com/drive/1X7gbTNt1UC0pjr8NMq-45yZiqcBon7_A
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# data already converted to csv file
data_train = pd.read_csv("./data/train/train.csv")

data_train.columns= data_train.columns.str.replace(".underwritingdataclarity.clearfraud.clearfraudinquiry.", "")
data_train.columns= data_train.columns.str.replace(".underwritingdataclarity.clearfraud.clearfraudidentityverification.", "")
data_train.columns= data_train.columns.str.replace(".underwritingdataclarity.clearfraud.clearfraudindicator.", "")

data_train = data_train.drop("underwritingid", axis = 1)

data_train = data_train.drop(["phonetype","ssndobreasoncode","ssnnamereasoncode","nameaddressreasoncode", "ssnnamereasoncodedescription","nameaddressreasoncodedescription","driverlicenseinconsistentwithonfile"], axis = 1)

data_train = data_train.drop(["workphonepreviouslylistedascellphone","workphonepreviouslylistedashomephone"], axis = 1)

data_train.replace("unavailable", np.nan, inplace=True)

data_train = data_train.drop("phonematchresult", axis = 1)

# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the column 'class_column' to replace classes with numbers
data_train['ssnnamematch'] = label_encoder.fit_transform(data_train['ssnnamematch'])
data_train['phonematchtype'] = label_encoder.fit_transform(data_train['phonematchtype'])
data_train['phonematchtypedescription'] = label_encoder.fit_transform(data_train['phonematchtypedescription'])
data_train['ssndobmatch'] = label_encoder.fit_transform(data_train['ssndobmatch'])
data_train['nameaddressmatch'] = label_encoder.fit_transform(data_train['ssndobmatch'])


matchResult = pd.get_dummies(data_train["overallmatchresult"])
data_train = pd.concat([data_train.drop("overallmatchresult", axis = 1), matchResult], axis=1)

data_train = data_train.fillna(data_train.mean())

X = data_train.drop('clearfraudscore', axis=1)
y = data_train["clearfraudscore"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



## methode 1 using xgboost

import xgboost as xgb
from sklearn.metrics import r2_score, mean_squared_error

params = {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'objective': 'reg:squarederror'}

dtrain = xgb.DMatrix(X_train, label=y_train)



dtest = xgb.DMatrix(X_test)
model = xgb.train(params, dtrain, num_boost_round=10)
model.save_model('./your_model_path.model')
predictions = model.predict(dtest)
model_boost = xgb.Booster()
model_boost.load_model('./your_model_path.model')


# Calculate the R^2 score
r2_score = r2_score(y_test, predictions)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, predictions))

# Print the score and RMSE
print("Results for first model")
print("Score:", r2_score)
print("RMSE:", rmse)




## the second method using ensemble learning

from sklearn.ensemble import BaggingRegressor

# Initialize the model
model = BaggingRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error

# Calculate the R^2 score
r2_score = r2_score(y_test, y_pred)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the score and RMSE
print("Results for second model")
print("Score:", r2_score)
print("RMSE:", rmse)



# the third method using RandomForestRegression

from sklearn.ensemble import RandomForestRegressor

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error

# Calculate the R^2 score
r2_score = r2_score(y_test, y_pred)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the score and RMSE
print("Results for third model")
print("Score:", r2_score)
print("RMSE:", rmse)



# the fourth method using GridSearchCV to determine the best parms

from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
model = RandomForestRegressor(random_state=42)

# Define the grid search object
grid_search = GridSearchCV(model, param_grid, cv=5)

# Fit the grid search object to the data
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Print the best parameters
print("Best parameters:", best_params)

# Train the model with the best parameters
model = RandomForestRegressor(**best_params)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the R^2 score
r2_score = r2_score(y_test, y_pred)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print the score and RMSE
print("Results for fourth model")
print("Score:", r2_score)
print("RMSE:", rmse)