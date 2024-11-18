import matplotlib.pyplot as plt
import numpy as np
import random 

x = 10 * np.random.sample(50)
print(x)

y = [2 * ii - 5 + random.random() for ii in x]
print(y)

plt.scatter(x, y)
plt.show()