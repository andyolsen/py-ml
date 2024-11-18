import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Set aesthetic parameters for Seaborn in one convenient step, to make the graphs look nice.
sns.set_theme()

# Create a numpy RandomState object, which has convenience methods for generating random numbers and matrices.
rng = np.random.RandomState(1)

# Generate 100 random (x,y) points as follows:
#   rng.rand(2, 2)    - Creates a 2x2 random matrix with values in the range [0, 1).
#   rng.randn(2, 100) - Creates a 2x100 random matrix with values (mean=0, stdev=1).
#   dot()             - Performs a dot product on the two matrices, creates a 2x100 result matrix.
#   transpose()       - Transposes the 2x100 matrix into a 100x2 matrix, i.e. 100 (x,y) points.
X = np.dot(rng.rand(2, 2), rng.randn(2, 100)).transpose()
print("All 100 (x,y) points\n", X)
print("\nColumn 0 (x values)\n", X[:, 0])  
print("\nColumn 1 (y values)\n", X[:, 1])  

# Draw a scatterplot graph showing the 100 (x,y) points.
plt.scatter(X[:, 0], X[:, 1], alpha=0.4)
plt.axis('equal')
plt.show()

#---------------------------------------------------------------------------------------
# Code for Exercise 2...
#---------------------------------------------------------------------------------------

# Import the PCA class from Scikit-Learn.
from sklearn.decomposition import PCA

# Create a Scikit-Learn PCA model object to find 2 principal components.
pca = PCA(n_components=2)

# Fit the PCA model object to the data. The PCA object will compute the 2 principle components for the data.
pca.fit(X)

# After you've fitted the PCA model object to the data, the PCA object has two useful attributes:
#
#   components_          - Array of shape (n_components, n_features), indicating how much each feature influences each principal component. 
#                          The bigger the number, the bigger the influence of that feature on that component axis. 
#
#   explained_variance_  - Array of shape (n_components), indicating the amount of variance explained by each component.
#
print("PCA components_\n", pca.components_)
print("PCA explained_variance_\n", pca.explained_variance_)