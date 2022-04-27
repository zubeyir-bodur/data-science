import operator
import os
import numpy as np
import matplotlib.pyplot as plt
from defs import *
import model
from scipy.interpolate import make_interp_spline, BSpline

# Read the data set
X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = read_data()

"""
print("Plot the data first:")

plt.figure(figsize=(6, 8), dpi=300)
plt.scatter(x=X_TRAIN, y=Y_TRAIN, s=10, c='#00FF00')
plt.scatter(x=X_TEST, y=Y_TEST, s=10, c='#FF0000')
plt.title("Scatter plot of the data set")
plt.ylabel("X axis")
plt.xlabel("Y axis")
plt.legend(labels=["Training data", "Test data"])
plt.show()
"""
print("Now, regressors: ")

"""
# Regress the training data using normalization or not
artificial_nn = model.ANN(num_units=512, epochs=500000, learning_rate=5e-5, is_normalized=False, stop_M=3)
artificial_nn.train(X_TRAIN, Y_TRAIN)
y_predict_train, train_loss = artificial_nn.predict(X_TRAIN, Y_TRAIN)
y_predict_test, test_loss = artificial_nn.predict(X_TEST, Y_TEST)
plot_predictions(X_TRAIN, Y_TRAIN, y_predict_train, is_test=False, is_norm=False)
plot_predictions(X_TEST, Y_TEST, y_predict_test, is_test=True, is_norm=False)
"""
artificial_nn_normalized = model.ANN(num_units=0, epochs=100000, learning_rate=5e-4, is_normalized=False, stop_M=3)
artificial_nn_normalized.train(X_TRAIN, Y_TRAIN)
y_predict_train_norm, train_loss_norm = artificial_nn_normalized.predict(X_TRAIN, Y_TRAIN)
y_predict_test_norm, test_loss_norm = artificial_nn_normalized.predict(X_TEST, Y_TEST)
plot_predictions(X_TRAIN, Y_TRAIN, y_predict_train_norm, is_test=False, is_norm=False)
plot_predictions(X_TEST, Y_TEST, y_predict_test_norm, is_test=True, is_norm=False)


