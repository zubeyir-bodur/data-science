import os
import numpy as np
import matplotlib.pyplot as plt
from defs import *

# Read the data set
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = read_data()

print("Plot the data first:")

plt.figure(figsize=(6, 8), dpi=300)
plt.scatter(x=X_TRAIN, y=Y_TRAIN, s=10, c='#00FF00')
plt.scatter(x=X_TEST, y=Y_TEST, s=10, c='#FF0000')
plt.title("Scatter plot of the data set")
plt.ylabel("X axis")
plt.xlabel("Y axis")
plt.legend(labels=["Training data", "Test data"])
plt.show()

print("Now, regressors: ")

# TODO Regress the training data using
#   1 - ANN with no hidden layers
#   2 - ANN with a single hidden
#   3 - Then plot the smoothed regressions separately

