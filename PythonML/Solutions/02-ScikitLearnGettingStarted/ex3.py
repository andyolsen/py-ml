import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

sns.pairplot(data=iris, hue='species', kind='scatter')
print(iris)

plt.show()