import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

housing = fetch_california_housing()
print(housing.keys())

print("\nDetails about the feature matrix")
print(housing.data.shape)
print(housing.feature_names)
print(housing.data)

print("\nDetails about the target array")
print(housing.target.shape)
print(housing.target)

# Convert the feature matrix into a Pandas DataFrame, and add feature names as column names for convenience.
X = pd.DataFrame(housing.data)
print(X.head())
X.columns = housing.feature_names
print(X.head())

# Assign the target array to a variable named y, for readability.
y = housing.target

# Split the feature matrix and target array into "training" and "testing" datasets.
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    train_size = 0.80)
    
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create a LinearRegression model object, and fit it to the "training" dataset.
model = LinearRegression()
model.fit(X_train, y_train)

# Use the model to predict labels for the "testing" dataset.
y_pred = model.predict(X_test)

# Print the actual vs. predicted house prices.
y_sideBySide = list(zip(y_test, y_pred))
print(y_sideBySide)

# Plot the actual vs. predicted house prices.
plt.scatter(y_test, y_pred)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title("Actual prices vs. predicted prices")
plt.show()

# Determine the root mean squared error.
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
print("Root mean squared error %f" % rmse)