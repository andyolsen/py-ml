import pandas as pd
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
print(housing.keys())

print("\nDetails about the feature matrix")
print(housing.data.shape)
print(housing.feature_names)
print(housing.data)

print("\nDetails about the target array")
print(housing.target.shape)
print(housing.target)

# Convert the feature matrix into a pandas DataFrame, and add feature names as column names for convenience.
X = pd.DataFrame(housing.data)
print(X.head())
X.columns = housing.feature_names
print(X.head())

# Assign the target array to a variable named y, for readability.
y = housing.target