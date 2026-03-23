# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and separate it into input features (e.g., humidity, wind speed) and target variables (temperature, PM2.5, energy).

2.Generate several decision trees using random subsets of the data and features (bootstrap sampling).

3.Each tree makes its own prediction; combine all predictions (average for regression problems).

4.Compare predicted values with actual values using metrics like Mean Squared Error (MSE) to check accuracy.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Ashwath p
RegisterNumber: 212224220012
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

data = {
    'Humidity': [60, 65, 70, 75, 80],
    'WindSpeed': [5, 7, 6, 8, 7],
    'Temperature': [30, 32, 31, 29, 28],
    'PM25': [120, 130, 125, 140, 150],
    'Energy': [200, 210, 205, 220, 230]
}

df = pd.DataFrame(data)

X = df[['Humidity', 'WindSpeed']]
y_temp = df['Temperature']
y_pm25 = df['PM25']
y_energy = df['Energy']

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
_, _, y_pm25_train, y_pm25_test = train_test_split(X, y_pm25, test_size=0.2, random_state=42)
_, _, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)

rf.fit(X_train, y_temp_train)
temp_pred = rf.predict(X_test)

rf.fit(X_train, y_pm25_train)
pm25_pred = rf.predict(X_test)

rf.fit(X_train, y_energy_train)
energy_pred = rf.predict(X_test)

print("Temperature MSE:", mean_squared_error(y_temp_test, temp_pred))
print("PM2.5 MSE:", mean_squared_error(y_pm25_test, pm25_pred))
print("Energy MSE:", mean_squared_error(y_energy_test, energy_pred))
*/
```

## Output:
<img width="474" height="93" alt="Screenshot 2026-03-20 102839" src="https://github.com/user-attachments/assets/76e1651e-0631-49bb-891b-3d46f4be7115" />


## Result:
The Random Forest model was successfully implemented and used to predict temperature, PM2.5 level, and energy consumption. The model produced accurate predictions with low error, showing good performance.
