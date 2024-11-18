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