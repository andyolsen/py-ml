import seaborn as sns

iris = sns.load_dataset('iris')

print(iris.head(20))

print(iris.info())

print(iris['species'])

print(iris['species'].value_counts())