import numpy as np
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt

DIGITS = loadmat('digits.mat')['digits']
LABELS = loadmat('digits.mat')['labels']
DIM = len(DIGITS[0])
WIDTH = int(math.sqrt(DIM))
HEIGHT = DIM // WIDTH
SIZE = len(DIGITS)
DISP_WIDTH = 11
DISP_HEIGHT = 19
TRAIN_SIZE = SIZE // 2


def display_blocks(eigenvectors,
                   disp_rows=DISP_WIDTH,
                   disp_cols=DISP_HEIGHT,
                   ind_img_height=HEIGHT,
                   ind_img_width=WIDTH,
                   code="mnist"):
    """
    Given a set of images (eigenvectors), returns the merged image
    that will be displayed in PIL, where the top left image will be the
    first image, bottom right image will be the last
    :return: A 2D numpy array
    """
    out = np.zeros((disp_rows * ind_img_height, disp_cols * ind_img_width), dtype=float)
    for a in range(disp_rows):
        for b in range(disp_cols):
            if (disp_cols*a + b) < len(eigenvectors):
                I = []
                if code == "mnist":
                    I = pack(eigenvectors[disp_cols*a + b])
                elif code == "eig":
                    I = squash(eigenvectors[:, disp_cols*a + b], ind_img_height, ind_img_width)
                elif code == "lda":
                    I = squash(eigenvectors[disp_cols * a + b, :], ind_img_height, ind_img_width)
                out[ind_img_height * a: ind_img_height*(a+1), ind_img_width * b: ind_img_width*(b+1)] = I
    return out


def pack(vector):
    """
    Given a MINST handwritten digit vector ranging from [0,1]
    Return a 20x20 grayscale image that can be displayed in Python
    :param vector:
    :return:
    """
    temp = np.reshape(vector, (20, 20))
    # Transpose, for Python only
    temp = np.transpose(temp)
    return np.array(temp * 255, dtype=float)


def squash(vector, ind_img_height=20, ind_img_width=20):
    """
    Given a MINST handwritten digit vector, can be negative or positive, can have any dimensions
    Return a 20x20 grayscale image that can be displayed in Python
    :param ind_img_height:
    :param ind_img_width:
    :param vector:
    :return:
    """
    temp = np.reshape(vector, (ind_img_height, ind_img_width))
    # Transpose, for Python only
    temp = np.transpose(temp)
    maximum = np.amax(temp)
    minimum = np.amin(temp)
    temp = ((temp - minimum) / (maximum - minimum)) * 255
    return np.array(temp, dtype=float)


def split(dataset=DIGITS, labels=LABELS):
    """
    Splits the data into two randomly
    Also returns the indexes of the training set, which is
    necessary for labels
    :param: dataset is the dataset to be split
    :return: train_data the training partition
    :return: test_data the test partition
    :return: train_label labels of the training partition
    :return: test_label labels of the test partition
    """
    perm = np.random.permutation(SIZE)
    train_idx = perm[0:SIZE//2]
    test_idx = perm[SIZE//2:SIZE]
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    train_data = dataset[train_idx]
    test_data = dataset[test_idx]
    train_label = labels[train_idx]
    test_label = labels[test_idx]
    return train_data, test_data, train_label.flatten(), test_label.flatten()


def partition(digits, labels):
    partitioned = []
    for i in range(10):
        partitioned.append([])
    for j in range(len(digits)):
        partitioned[int(labels[j])].append(digits[j])
    return partitioned


def plot_scatter(y, labels, mode):
    for i in range(10):
        plt.scatter(y[labels == i, 0], y[labels == i, 1], s=20, c='r', marker='o', label=i)
    if mode == "tsne":
        plt.title('t-SNE Visualization of MNIST Handwritten Digit Dataset')
    elif mode == "sammon":
        plt.title('Sammon\'s Mapping of MNIST Handwritten Digit Dataset')
    plt.legend(loc=2)
    plt.show()
    return
