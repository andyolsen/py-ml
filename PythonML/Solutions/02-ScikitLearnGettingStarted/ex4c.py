import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

# A violin plot is similar to a box plot, with the addition of a density plot on each side.
# For more info, see https://en.wikipedia.org/wiki/Violin_plot

sns.violinplot(x='species', y='petal_length', data=iris)
plt.show()