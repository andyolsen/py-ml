import matplotlib.pyplot as plt
import numpy as np
import random 

# Step 1: Import the LinearRegression model class, which can model a line of best fit for a feature matrix and a target array.
from sklearn.linear_model import LinearRegression

# Create random (x,y) datapoints, same code as in ex01-CreateData.py.
x = 10 * np.random.sample(50)
y = [2 * ii - 5 + random.random() for ii in x]

# Step 2: Create an instance of the LinearRegression model class, and set hyperparameters as appropriate.
model = LinearRegression()

# Step 3: Arrange the data (here, we have to convert x from a 1D array into a 2D features matrix).
x = x[:, np.newaxis]
print(x)

# Step 4: Fit the model to the data, and examine the learnt parameters.
model.fit(x, y)
print("Model coef_      %f" % model.coef_)
print("Model intercept_ %f" % model.intercept_)

# Plot the line of best fit via linear regression.
plt.title("Line of best fit via linear regression")
plt.scatter(x, y)
plt.plot(x, model.intercept_ + model.coef_ * x, color="orange", lw=1.5)
plt.show()

# Step 5: Predict labels for new data.
xnew = 10 * np.random.sample(20)
Xnew = xnew[:, np.newaxis]
ynew = model.predict(Xnew)

# Plot the predicted y values for new x values. 
plt.title("Predicted y values for new x values")
plt.scatter(xnew, ynew)
plt.plot(xnew, model.intercept_ + model.coef_ * xnew, color="orange", lw=1.5)
plt.show()