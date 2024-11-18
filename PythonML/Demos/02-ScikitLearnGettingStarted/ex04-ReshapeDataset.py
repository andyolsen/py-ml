import seaborn as sns

iris = sns.load_dataset('iris')

print("-----------------------------------------------------------------")
print("x_iris")
print("-----------------------------------------------------------------")

x_iris = iris.drop('species', axis=1)
print(x_iris.shape)

# For each numeric column, describe() prints:
#   count - number of values
#   mean  - arithmetic mean
#   std   - standard deviation
#   min   - minimum value
#   25%   - value at the 25th percentile
#   50%   - value at the 50th percentile (i.e. the median value)
#   75%   - value at the 75th percentile
#   max   - maximum value
print(x_iris.describe())   

# Print the top 20 rows in the DataFrame (default is 5 rows).
print(x_iris.head(20))


print("-----------------------------------------------------------------")
print("y_iris")
print("-----------------------------------------------------------------")

y_iris = iris['species']
print(y_iris.shape)

# For non-numeric columns (e.g. strings), describe() prints:
#   count  - number of values
#   unique - number of unique values
#   top    - most common value (i.e. statistical mode)
#   freq   - frequency of the most common value
print(y_iris.describe())  

# Print the top 20 rows in the Series (default is 5 rows).
print(y_iris.head(20))