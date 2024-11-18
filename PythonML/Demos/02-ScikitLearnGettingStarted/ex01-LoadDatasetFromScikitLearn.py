from sklearn.datasets import load_diabetes, load_iris

diabetes = load_diabetes()

print("-----------------------------------------------------------------")
print("diabetes")
print("-----------------------------------------------------------------")
print("Description    ", diabetes.DESCR)
print("Feature names  ", diabetes.feature_names)
print("Data shape     ", diabetes.data.shape)
print("Data           ", diabetes.data)

iris = load_iris()

print("\n-----------------------------------------------------------------")
print("iris")
print("-----------------------------------------------------------------")
print("Description    ", iris.DESCR)
print("Feature names  ", iris.feature_names)
print("Data shape     ", iris.data.shape)
print("Data           ", iris.data)