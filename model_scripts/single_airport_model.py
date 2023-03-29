#
# Kyler Robison
#

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

import math

df = pd.read_csv("../train_tables/main.csv")

input_features = ['minutes_until_etd', 'departure_runway', 'delay_3hr', 'delay_30hr', 'standtime_3hr', 'standtime_30hr']
# input_features = ['minutes_until_etd']

# add column representing standard and improved baselines
df['baseline'] = df.apply(lambda row: max(row['minutes_until_etd'] - 15, 0), axis=1)
df['improved_baseline'] = df.apply(lambda row: round(max(row['minutes_until_etd'] - 13, 0)), axis=1)

# print performance of baseline estimates
mae = mean_absolute_error(df['minutes_until_pushback'], df['baseline'])
print(f"\nMAE with baseline: {mae:.4f}")
mae = mean_absolute_error(df['minutes_until_pushback'], df['improved_baseline'])
print(f"MAE with improved baseline: {mae:.4f}\n")

# df['minutes_until_etd'] = df['minutes_until_etd'].apply(lambda x: max(x, 0))

data_train, data_test = train_test_split(df, test_size=0.2)

X_train = data_train[input_features]
X_test = data_test[input_features]

y_train = data_train['minutes_until_pushback']
y_test = data_test['minutes_until_pushback']

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# print('Intercept:', model.intercept_)
# print('Slope:', model.coef_)

# Predict the target values for the test data using the trained model
y_pred = model.predict(X_test)

# Evaluate the model performance on the test data using MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"\nMAE on test data: {mae:.4f}")

# k-fold cross val
X = df[input_features]
y = df['minutes_until_pushback']

# Impute missing values with mean
X = imputer.fit_transform(X)

model = DecisionTreeRegressor()
mae_scores = -cross_val_score(model, X, y, cv=10, scoring='neg_mean_absolute_error')

print(f"\nk-fold MAE scores: {mae_scores}")
print(f"k-fold MAE: {mae_scores.mean():.4f}\n")
