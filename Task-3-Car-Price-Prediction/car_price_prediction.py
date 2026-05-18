# Car Price Prediction with Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("car_data.csv")

# Display first 5 rows
print("First 5 rows of dataset:")
print(df.head())

# Check dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Create car age column
current_year = 2025
df["Car_Age"] = current_year - df["Year"]

# Drop unnecessary columns
df = df.drop(["Car_Name", "Year"], axis=1)

# Encode categorical columns
df = pd.get_dummies(df, drop_first=True)

# Split data into features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Linear Regression model
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Predict using Linear Regression
linear_pred = linear_model.predict(X_test)

print("\nLinear Regression Results")
print("MAE:", mean_absolute_error(y_test, linear_pred))
print("MSE:", mean_squared_error(y_test, linear_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_pred)))
print("R2 Score:", r2_score(y_test, linear_pred))

# Train Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

# Predict using Random Forest
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Regressor Results")
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("MSE:", mean_squared_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))

# Compare actual and predicted prices
comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": rf_pred
})

print("\nActual vs Predicted Prices:")
print(comparison.head(10))

# Actual vs Predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()

# Feature importance
importance = pd.Series(rf_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importance.plot(kind="bar")
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()