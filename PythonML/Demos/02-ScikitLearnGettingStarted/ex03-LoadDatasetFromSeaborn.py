import seaborn as sns

irisDataFrame = sns.load_dataset('iris')

print("-----------------------------------------------------------------")
print("iris")
print("-----------------------------------------------------------------")
print("DataFrame shape    ",  irisDataFrame.shape)
print("DataFrame describe\n", irisDataFrame.describe())
print("DataFrame head\n",     irisDataFrame.head())

exerciseDataFrame = sns.load_dataset('exercise')
print("\n-----------------------------------------------------------------")
print("exercise")
print("-----------------------------------------------------------------")
print("DataFrame shape    ",  exerciseDataFrame.shape)
print("DataFrame describe\n", exerciseDataFrame.describe())
print("DataFrame head\n",     exerciseDataFrame.head())

tipsDataFrame = sns.load_dataset('tips')
print("\n-----------------------------------------------------------------")
print("tips")
print("-----------------------------------------------------------------")
print("DataFrame shape    ",  tipsDataFrame.shape)
print("DataFrame describe\n", tipsDataFrame.describe())
print("DataFrame head\n",     tipsDataFrame.head())