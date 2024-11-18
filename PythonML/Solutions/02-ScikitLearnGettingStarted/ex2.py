import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

sns.scatterplot(x='petal_length', y='petal_width', data=iris)
plt.show()