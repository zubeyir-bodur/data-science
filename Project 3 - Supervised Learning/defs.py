import io
import operator
import os
import numpy as np
import matplotlib.pyplot as plt
from random import random
from scipy.interpolate import make_interp_spline
from sklearn.preprocessing import MinMaxScaler
from PIL import Image


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
    return [np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)]


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


def min_max_scale(ndarray):
    """
    Min-max normalization meets the needs.
    However, better approaches can be used
    such as L1, L2, max-abs, gaussian normalization ...
    :param ndarray:
    :return:
    """
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(np.transpose(ndarray.reshape(1, -1))).flatten()
    return transformed, scaler


def inverse_transform(transformed, scaler):
    return scaler.inverse_transform(np.transpose(transformed.reshape(1, -1))).flatten()


def uniques(array):
    unique_indices = []
    unique_values = []
    for i in range(len(array)):
        if array[i] not in unique_values:
            unique_indices.append(i)
            unique_values.append(array[i])
    return np.array(unique_indices), np.array(unique_values)


def smooth_predictions(x, y_predictions):
    """
    Smooth the predictions by normalizing points into 300 samples
    :param x:
    :param y_predictions:
    :return:
    """
    # Get the unique x_test values
    test_unique_indices, x_test_unique = uniques(x)
    y_predict_unique = y_predictions[test_unique_indices]

    # Enumerate the coordinate system and sort in ascending order
    # Return the old indexes
    enumerate_object_xtest = enumerate(x_test_unique)
    sorted_pairs = sorted(enumerate_object_xtest, key=operator.itemgetter(1))
    sorted_indices = [index for index, element in sorted_pairs]

    # Find the sorted version of y_predict
    x_test_sorted = x_test_unique[sorted_indices]
    y_predict_sorted = y_predict_unique[sorted_indices]
    x_test_sorted_new = np.linspace(x_test_sorted.min(), x_test_sorted.max(), 300)
    spl = make_interp_spline(x_test_sorted, y_predict_sorted, k=3)  # type: BSpline
    y_predict_sorted_smooth = spl(x_test_sorted_new)
    return x_test_sorted_new, y_predict_sorted_smooth


def plot_predictions(x, y, y_predictions, is_test, is_norm, show=True):
    """
    Given predictions and the test data
    Plot the predictions smoothly
    so that they look just like a
    polynomial regression
    :param show:
    :param is_norm:
    :param is_test:
    :param x:
    :param y:
    :param y_predictions:
    :return: PIL object for the plot
    """
    plt.figure(figsize=(6, 4), dpi=120)
    x_test_sorted_new, y_predict_sorted_smooth = smooth_predictions(x, y_predictions)
    if is_test:
        plt.scatter(x=x, y=y, s=10, c='#FF0000')
        plt.plot(x_test_sorted_new, y_predict_sorted_smooth, c='#FFA500')
        plt.title(f"Scatter plot of the test values vs Predicted function is_norm={is_norm}")
        plt.legend(labels=["Test data", "Predictions"])
    else:
        plt.scatter(x=x, y=y, s=10, c='#00FF00')
        plt.plot(x_test_sorted_new, y_predict_sorted_smooth, c='#FFA500')
        plt.title(f"Scatter plot of the training values vs Predicted function is_norm={is_norm}")
        plt.legend(labels=["Training data", "Predictions"])
    plt.ylabel("X axis")
    plt.xlabel("Y axis")
    if show:
        plt.show()
        return None
    else:
        img_buf = io.BytesIO()
        plt.savefig(img_buf, format='png')
        im = Image.open(img_buf)
        im_matrix = np.array(im)
        img_buf.close()
        return im_matrix


def display_blocks(plots,
                   disp_rows=2,
                   disp_cols=3):
    """
    Given a set of plots, combine them into a single image
    assuming we have 5 plots.
    """
    plot_height = plots[0].shape[0]
    plot_width = plots[0].shape[1]
    channels = plots[0].shape[2]
    out = np.zeros(shape=(plot_height * disp_rows, plot_width * disp_cols, channels), dtype=float)
    for a in range(disp_rows):
        for b in range(disp_cols):
            if (disp_cols*a + b) < len(plots):
                I = plots[disp_cols*a + b, :, :, :]
                if disp_cols*a + b < 4:
                    out[plot_height * a: plot_height*(a+1), plot_width * b: plot_width*(b+1), :] = I
                else:
                    gap = int((plot_width * disp_cols - plot_width * 2) / 2)
                    out[plot_height * a: plot_height * (a + 1), plot_width * b + gap: plot_width * (b + 1) + gap, :] = I
    # Image.fromarray(out).show()
    return out

