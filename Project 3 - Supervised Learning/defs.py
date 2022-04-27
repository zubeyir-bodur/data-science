import os
import numpy as np
import matplotlib.pyplot as plt
from random import random


def read_data():
    """
    :return: Reads the variables from files
    Then returns X & Y values for test and train
    sets
    """
    train_file = open("train1", "r")
    test_file = open("test1", "r")
    files = [train_file, test_file]
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    for i in range(len(files)):
        for line in files[i]:
            x_, y_ = line[:-1].split("\t")
            x_ = float(x_)
            y_ = float(y_)
            if i == 0:
                x_train.append(x_)
                y_train.append(y_)
            else:
                x_test.append(x_)
                y_test.append(y_)
    return [x_train, y_train, x_test, y_test]


def sigmoid(x):
    """
    Compute the sigmoid for the vector x
    :param x:
    :return:
    """
    return 1 / (1 + (np.exp(-x)))


def sum_of_squared(y_pred, y_truth):
    """
    Compute the sum of squared error,
    given the predictions and the ground truth
    :param y_pred:
    :param y_truth:
    :return:
    """
    return np.sum((y_pred - y_truth) ** 2)


def sigmoid_der(x):
    """
    Compute the sigmoid' for the vector x
    :param x:
    :return:
    """
    return sigmoid(x) * (1 - sigmoid(x))


def random_weights(num_units):
    return np.array([random() for _ in range(num_units)])
