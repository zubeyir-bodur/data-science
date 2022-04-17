import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import metrics

"""
source: https://github.com/RobinKarlsson/Sammon-Mapping
"""


def C(distances):
    c = 0
    for i in range(distances.shape[0]):
        for j in range(distances.shape[0]):
            if i < j:
                c += distances[i, j]

    return c


def sammonStress(in_distances, out_distances, c):
    E = 0
    for i in range(in_distances.shape[0]):
        for j in range(in_distances.shape[0]):
            if i < j:
                E += ((out_distances[i, j] - in_distances[i, j]) ** 2) / in_distances[i, j]
    return E / c


def sammon(X, max_iter=500, epsilon=1e-7, alpha=0.3):
    rows = X.shape[0]

    # random two-dimensional layout of points
    y = np.random.normal(0.0, 1.0, [rows, 2])

    in_distances = metrics.pairwise_distances(X)
    c = C(in_distances)
    print(f'c={c}')

    stress_old = np.inf

    for epoch in range(1, max_iter + 1):
        out_distances = metrics.pairwise_distances(y)

        stress = sammonStress(in_distances, out_distances, c)

        print(f'Epoch {epoch} Sammon stress: {stress}')
        if stress_old - stress < epsilon:
            break

        stress_old = stress

        partial_der1 = np.array([0, 0])
        partial_der2 = np.array([0, 0])

        for i in range(rows):
            # calculate sum part of the partial derivatives
            for j in range(rows):
                if i != j:
                    denominator = out_distances[i, j] * in_distances[i, j]
                    difference = out_distances[i, j] - in_distances[i, j]

                    y_difference = np.subtract(y[i], y[j])

                    if denominator < 0.000001:
                        print(f'denominator = {denominator} set to 0.000001')
                        denominator = 0.000001

                    partial_der1 = partial_der1 + np.multiply(difference / denominator, y_difference)
                    partial_der2 = partial_der2 + (1 / denominator) * (
                            difference - np.divide(np.square(y_difference), in_distances[i, j]) * (
                            1 + difference / in_distances[i, j]))

            partial_der1 = (-2 / c) * partial_der1
            partial_der2 = (-2 / c) * partial_der2

            # update y[i]
            y[i] = y[i] + alpha * (partial_der1 / np.abs(partial_der2))

    return y
