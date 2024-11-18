from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import GaussianNB 
from sklearn import metrics 
import pandas as pd

# Load the iris dataset 
iris = load_iris() 
 
# Store the feature matrix in X, and the target array in y. 
X = pd.DataFrame(iris.data)
X.columns = iris.feature_names
y = iris.target 

print("\nFeature matrix\n", X)  
print("\nTarget array\n", y)
  
# Split X and y into "training" and "testing" datasets. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4) 
  
# Create a GaussianNB model object, and train the model on the "training" dataset. 
gnb = GaussianNB() 
gnb.fit(X_train, y_train) 
  
# Predict species for the "testing" dataset. 
y_pred = gnb.predict(X_test) 
  
# Print actual vs. predicted species.
y_sideBySide = list(zip(y_test, y_pred))
print("\nActual vs. predicted species\n", y_sideBySide)
  
# Comparing predicted species (y_pred) against actual species (y_test). 
print("GaussianNB model accuracy: ", metrics.accuracy_score(y_test, y_pred))

# Create a confusion matrix, to see how well the predicted data shaped against the actual data. Then draw in a heatmap.
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, square=True, annot=True, cbar=False)
plt.xlabel('Predicted value')
plt.ylabel('Actual value');
plt.show()