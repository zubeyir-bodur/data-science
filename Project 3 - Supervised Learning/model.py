import os
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from random import random
from defs import *


class ANN:
    """
    Artificial Neural Network w/ some hidden units in a single layer
    If num_units  is 0, then there are no hidden units.
    """

    def __init__(self, num_units, epochs, learning_rate, is_normalized, stop_M):
        self.num_units = num_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.is_normalized = is_normalized

        # 3 point moving averager can assure convergence
        self.stop_M = stop_M
        if num_units != 0:
            # Initialize random weights
            self.weights = np.zeros(shape=(num_units, 2))
            # Weights = [w_xh, w_hy, wh]
            """
            if not self.is_normalized:
                self.weights[:, 0] = random_weights(num_units)
                self.weights[:, 1] = random_weights(num_units)
            else:
                self.weights[:, 0] = 1
                self.weights[:, 1] = 1
            """
            # Pick weights from a gaussian distribution in [0, 1]
            self.weights[:, 0] = random_weights(num_units)
            self.weights[:, 1] = random_weights(num_units)
            # Output emitted by the hidden units
            self.h = np.array([(1 / num_units) for _ in range(num_units)])
        else:
            # We will just have a single weight
            # that determines the linear regression
            # which is the slope
            # and a bias value
            self.slope = 1  # Let the initial slope be 1
            self.bias = 1   # amd bias be nonzero

    def train(self, x_train, y_train):
        losses = [0.0 for _ in range(self.stop_M)]
        if self.is_normalized:
            print("Using normalization...")
            # normalize the x and y train values to the interval [0, 1]
            x_train_norm, scaler_x = min_max_scale(x_train)
            # y_train_norm, scaler_y = min_max_scale(y_train)
        if self.num_units == 0:
            print("Using no hidden layers...")
            for i in range(self.epochs):
                index = np.random.randint(0, len(x_train))

                if not self.is_normalized:
                    f_of_x = x_train[index] * self.slope + self.bias
                    delta_w = self.learning_rate*(y_train[index] - f_of_x)*x_train[index]
                    delta_bias = self.learning_rate*(y_train[index] - f_of_x)
                    self.slope += delta_w
                    self.bias += delta_bias
                    loss = (1/2)*((f_of_x - y_train[index]) ** 2)
                else:
                    f_of_x = x_train_norm[index] * self.slope + self.bias
                    delta_w = self.learning_rate * (y_train[index] - f_of_x) * x_train_norm[index]
                    delta_bias = self.learning_rate * (y_train[index] - f_of_x)
                    self.slope += delta_w
                    self.bias += delta_bias
                    loss = (1 / 2) * ((f_of_x - y_train[index]) ** 2)
                losses.append(loss)
                ma_losses = int(np.sum(np.array(losses[-self.stop_M:-1]))) / self.stop_M

                # If the mth point moving average has less loss than the current
                # Then the network has converged
                if ma_losses >= losses[-1]:
                    print(f"Epoch {i} have converged w/ Loss = {loss}")
                    break
        else:
            print(f"NN w/ 1 Hidden Layer w/ {self.num_units} units...")
            for i in range(self.epochs):
                index = np.random.randint(0, len(x_train))
                w0 = self.weights[:, 0]
                w1 = self.weights[:, 1]

                if not self.is_normalized:
                    forward = w0 + w1 * x_train[index]
                    hx = sigmoid(forward)
                    derivative_sigmoid = sigmoid_der(forward)
                    output = np.dot(hx, self.h)
                    self.weights[:, 0] += self.learning_rate * (y_train[index] - output) * self.h * derivative_sigmoid
                    self.weights[:, 1] += self.learning_rate * (y_train[index] - output) * self.h * derivative_sigmoid \
                                       * x_train[index]
                    self.h += self.learning_rate * (y_train[index] - output) * hx
                    forward = w0 + x_train.reshape(len(x_train), 1) * w1
                    hx = sigmoid(forward)
                    predictions = np.dot(hx, self.h)
                    loss = float(np.sum((predictions - y_train) ** 2))
                else:
                    forward = w0 + w1 * x_train_norm[index]
                    hx = sigmoid(forward)
                    derivative_sigmoid = sigmoid_der(forward)
                    output = np.dot(hx, self.h)
                    self.weights[:, 0] += self.learning_rate * (y_train[index] - output) * self.h * derivative_sigmoid
                    self.weights[:, 1] += self.learning_rate * (y_train[index] - output) * self.h * derivative_sigmoid \
                                          * x_train_norm[index]
                    self.h += self.learning_rate * (y_train[index] - output) * hx
                    forward = w0 + x_train_norm.reshape(len(x_train_norm), 1) * w1
                    hx = sigmoid(forward)
                    predictions = np.dot(hx, self.h)
                    loss = float(np.sum((predictions - y_train) ** 2))

                losses.append(loss)
                # print(f"Epoch {i}/{self.epochs} -- Loss = {loss}")
                # Compute the m-point moving average of losses
                ma_losses = int(np.sum(np.array(losses[-self.stop_M:-1]))) / self.stop_M

                # If the mth point moving average has less loss than the current
                # Then the network has converged
                if ma_losses >= losses[-1]:
                    print(f"Epoch {i} have converged w/ Loss = {loss}")
                    break
        print(f"Max epochs have been reached... w/ Loss = {loss}")

    def predict(self, x_test, y_test):
        if self.num_units != 0:
            w0 = self.weights[:, 0]
            w1 = self.weights[:, 1]
        else:
            w0 = self.bias
            w1 = self.slope

        if self.is_normalized:
            print("Using normalization while predicting...")
            # normalize the x and y train values to the interval [0, 1]
            x_test_norm, scaler_x = min_max_scale(x_test)
            # y_test_norm, scaler_y = min_max_scale(y_test)

        if not self.is_normalized:
            forward = w0 + x_test.reshape(len(x_test), 1) * w1
            if self.num_units == 0:
                y_pred = forward
            else:
                hx = sigmoid(forward)
                y_pred = np.dot(hx, self.h)
            loss = np.sum((y_pred - y_test) ** 2)
        else:
            forward = w0 + x_test_norm.reshape(len(x_test_norm), 1) * w1
            if self.num_units == 0:
                y_pred = forward
            else:
                hx = sigmoid(forward)
                y_pred = np.dot(hx, self.h)
            loss = np.sum((y_pred - y_test) ** 2)
        return y_pred, loss
