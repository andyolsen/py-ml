import pandas as pd 
from sklearn.datasets import load_diabetes, load_iris

diabetes = load_diabetes()
diabetesDataFrame = pd.DataFrame(diabetes.data)

print("-----------------------------------------------------------------")
print("diabetes")
print("-----------------------------------------------------------------")
print("Data shape         ",  diabetes.data.shape)
print("Feature names      ",  diabetes.feature_names)
print("DataFrame shape    ",  diabetesDataFrame.shape)
print("DataFrame describe\n", diabetesDataFrame.describe())
print("DataFrame head\n",     diabetesDataFrame.head())

iris = load_iris()
irisDataFrame = pd.DataFrame(iris.data)

print("\n-----------------------------------------------------------------")
print("iris")
print("-----------------------------------------------------------------")
print("Data shape         ",  iris.data.shape)
print("Feature names      ",  iris.feature_names)
print("DataFrame shape    ",  irisDataFrame.shape)
print("DataFrame describe\n", irisDataFrame.describe())
print("DataFrame head\n",     irisDataFrame.head())
