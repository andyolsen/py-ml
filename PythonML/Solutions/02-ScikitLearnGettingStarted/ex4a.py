import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset('iris')

# A joint plot shows the distribution and relationship of bivariate data. 
# In our joint plot:
#   - The top edge will show the distribution of petal_length.
#   - The RHS edge will show the distribution of petal_width.
#   - The centre plot will show a scatter diagram of petal_length vs petal_width.
# For more info, see https://en.wikipedia.org/wiki/Joint_probability_distribution

sns.jointplot(x='petal_length', y='petal_width', data=iris)
plt.show()