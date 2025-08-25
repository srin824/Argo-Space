import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Step 1: Load Data
dataset1 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user1_data.csv', delimiter=',')
dataset2 = pd.read_csv('https://raw.githubusercontent.com/chidaksh/CosmosocClub/master/Parsec2023/user2_data.csv', delimiter=',')

# Combine datasets
df = pd.concat([dataset1, dataset2], ignore_index=True)
print(df)
# Convert 'ttime' to timestamp format
df['ttime'] = pd.to_datetime(df['ttime'])
df['ttime'] = df['ttime'].apply(lambda x: x.timestamp())

# Step 2: Splitting data into Features (X) and Targets (Y)
x = df.iloc[:, 0:8].values
y = df.iloc[:, 8:12].values

# Step 3: Split into Train and Test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Step 4: Handle missing data using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
x_train = imputer.fit_transform(x_train)
x_test = imputer.transform(x_test)
y_train = imputer.fit_transform(y_train)
y_test = imputer.transform(y_test)

# Step 5: Feature Scaling using StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 6: Initialize and train the RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(x_train, y_train)

# Step 7: Save the trained components (Imputer, Scaler, and Regressor)
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(regressor, 'random_forest_regressor.pkl')

# Step 8: Model Evaluation
y_pred = regressor.predict(x_test)

# Comparing actual vs predicted results for Temperature, Humidity, and Moisture
df1 = pd.DataFrame({
    'Actual Temp': y_test[:, 0],
    'Predicted Temp': y_pred[:, 0],
    'Actual Humidity': y_test[:, 1],
    'Predicted Humidity': y_pred[:, 1],
    'Actual Moisture': y_test[:, 2],
    'Predicted Moisture': y_pred[:, 2]
})

print('Training Accuracy =', regressor.score(x_train, y_train))
print('Test Accuracy =', regressor.score(x_test, y_test))
print(df1.head())  # Display a few rows of the comparison DataFrame
