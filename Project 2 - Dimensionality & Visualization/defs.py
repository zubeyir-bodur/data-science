import numpy as np
import math
from scipy.io import loadmat


DIGITS = loadmat('digits.mat')['digits']
LABELS = loadmat('digits.mat')['labels']
DIM = len(DIGITS[0])
WIDTH = int(math.sqrt(DIM))
HEIGHT = DIM // WIDTH
SIZE = len(DIGITS)
DISP_WIDTH = 10
DISP_HEIGHT = 12
TRAIN_SIZE = SIZE // 2


def display_blocks(eigenvectors, disp_rows=DISP_WIDTH, disp_cols=DISP_HEIGHT, code="mnist"):
    """
    Given a set of images (eigenvectors), returns the merged image
    that will be displayed in PIL, where the top left image will be the
    first image, bottom right image will be the last
    :return: A 2D numpy array
    """
    out = np.zeros((disp_rows * HEIGHT, disp_cols * WIDTH), dtype=float)
    for a in range(disp_rows):
        for b in range(disp_cols):
            if (disp_cols*a + b) < len(eigenvectors):
                I = []
                if code == "mnist":
                    I = pack(eigenvectors[disp_cols*a + b])
                elif code == "eig":
                    I = squash(eigenvectors[:, disp_cols*a + b])
                out[HEIGHT * a: HEIGHT*(a+1), WIDTH * b: WIDTH*(b+1)] = I
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


def squash(vector):
    """
    Given a MINST handwritten digit vector ranging from [0,1]
    Return a 20x20 grayscale image that can be displayed in Python
    :param vector:
    :return:
    """
    temp = np.reshape(vector, (20, 20))
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
    train_data = dataset[train_idx]
    test_data = dataset[test_idx]
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    train_label = labels[train_idx]
    test_label = labels[test_idx]
    return train_data, test_data, train_label, test_label

