import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

# A box plot shows the spread of numerical data through their quartiles:
#   - Minimum (0th percentile)       - Lowest data point in the dataset, excluding any outliers.
#   - Maximum (100th percentile)     - Highest data point in the dataset, excluding any outliers.
#   - Median  (50th percentile)      - Middle value in the dataset.
#   - 1st quartile (25th percentile) - Median of the lower half of the dataset.
#   - 3rd quartile (75th percentile) - Median of the upper half of the dataset.
# For more info, see https://en.wikipedia.org/wiki/Box_plot

sns.boxplot(x='species', y='petal_length', data=iris)
plt.show()