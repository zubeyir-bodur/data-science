import numpy as np
import math
import scipy.io
import sklearn as sk


DATASET = scipy.io.loadmat('digits.mat')['digits']
LABELS = scipy.io.loadmat('digits.mat')['labels']
DIM = len(DATASET[0])
WIDTH = int(math.sqrt(DIM))
HEIGHT = DIM // WIDTH
SIZE = len(DATASET)
DISP_WIDTH = 10
DISP_HEIGHT = 12
TRAIN_SIZE = SIZE / 2


def display_blocks(eigenvectors, disp_rows=DISP_WIDTH, disp_cols=DISP_HEIGHT):
    """
    Given a set of images (eigenvectors), returns the merged image
    that will be displayed in PIL, where the top left image will be the
    first image, bottom right image will be the last
    :return: A 2D numpy array
    """
    out = np.zeros((disp_rows * HEIGHT, disp_cols * WIDTH), dtype=int)
    for a in range(disp_rows):
        for b in range(disp_cols):
            if (disp_cols * (a - 1) + b) < DIM:
                out[HEIGHT * a: HEIGHT*(a+1) - 1][WIDTH * b: WIDTH*(b+1) - 1] = np.\
                    reshape(eigenvectors[disp_cols*a + b], (20, 20))
    return out


def split(dataset=DATASET, labels=LABELS):
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
    train_idx = perm[0:SIZE//2 - 1]
    test_idx = perm[SIZE//2:SIZE - 1]
    train_data = dataset[train_idx]
    test_data = dataset[test_idx]
    train_label = labels[train_idx]
    test_label = labels[test_idx]
    return

