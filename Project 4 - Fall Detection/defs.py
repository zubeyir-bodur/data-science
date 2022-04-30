from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_data():
    """
    Read the data from the .csv file.
    :return:
    """
    DATA = pd.read_csv("falldetection_dataset.csv", encoding="ISO-8859-2", header=None)
    X_TRAIN = np.array(DATA[[i + 2 for i in range(DATA.shape[1] - 2)]])
    Y_TRAIN = np.array(DATA[1])
    fall_detector = lambda x: (x != "N") & (x == "F")
    Y_TRAIN = fall_detector(Y_TRAIN).astype(int)
    return X_TRAIN, Y_TRAIN


def min_max_scale(ndarray):
    """
    Min-max normalization meets the needs.
    However, better approaches can be used
    such as L1, L2, max-abs, gaussian normalization ...
    :param ndarray:
    :return:
    """
    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(ndarray)
    return transformed, scaler


def inverse_transform(transformed, scaler):
    return scaler.inverse_transform(transformed)


def plot_pca(x_norm_reduced, y, show_labels=True):
    x_norm_reduced_f = []
    x_norm_reduced_nf = []
    for i in range(x_norm_reduced.shape[0]):
        if y[i]:
            x_norm_reduced_f.append(x_norm_reduced[i])
        else:
            x_norm_reduced_nf.append(x_norm_reduced[i])
    x_norm_reduced_f = np.array(x_norm_reduced_f)
    x_norm_reduced_nf = np.array(x_norm_reduced_nf)
    plt.figure(figsize=(8, 6), dpi=90)
    if show_labels:
        plt.scatter(x=x_norm_reduced_f[:, 0], y=x_norm_reduced_f[:, 1], c="#FF0000", s=1.5)
        plt.scatter(x=x_norm_reduced_nf[:, 0], y=x_norm_reduced_nf[:, 1], c="#00FF00", s=1.5)
    else:
        plt.scatter(x=x_norm_reduced[:, 0], y=x_norm_reduced[:, 1], c="#0000FF", s=1.5)
    plt.xlabel("First PCA component")
    plt.ylabel("Second PCA component")
    plt.legend(labels=["Fall", "Non-Fall"])
    plt.title("PCA Visualization of Telehealth Data")


def plot_clustered_data(x_norm_reduced, y_predict, n_clusters):
    # Initialize k empty arrays
    x_norm_reduced_j = [[] for _ in range(n_clusters)]

    # Colors
    c = ['#00FF00', '#FF0000', '#0000FF', '#F0F000',
              '#00F0F0', '#F000F0', '#FFA500', '#FFC0CB',
              '#000000', '#851E1E']
    for i in range(x_norm_reduced.shape[0]):
        for j in range(n_clusters):
            if j == y_predict[i]:
                x_norm_reduced_j[j].append(x_norm_reduced[i])

    plt.figure(figsize=(8, 6), dpi=90)
    for j in range(n_clusters):
        x_norm_reduced_j_first_pca = []
        x_norm_reduced_j_second_pca = []
        for m in range(len(x_norm_reduced_j[j])):
            x_norm_reduced_j_first_pca.append(x_norm_reduced_j[j][m][0])
            x_norm_reduced_j_second_pca.append(x_norm_reduced_j[j][m][1])
        plt.scatter(x=x_norm_reduced_j_first_pca, y=x_norm_reduced_j_second_pca, c=c[j], s=1.5)
    plt.xlabel("First PCA component")
    plt.ylabel("Second PCA component")
    plt.legend(labels=[f"Cluster {i}" for i in range(n_clusters)])
    plt.title(f"K-Means Visualization of Telehealth Data w/ K={n_clusters}")

