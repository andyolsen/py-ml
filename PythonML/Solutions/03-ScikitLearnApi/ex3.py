import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

# Assign the target array to a variable named Y, for readability.
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