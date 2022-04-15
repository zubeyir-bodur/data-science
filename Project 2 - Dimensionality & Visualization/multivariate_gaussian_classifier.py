import numpy as np
from scipy.stats import multivariate_normal


class MultivariateGaussianClassifier:
    """
    Multivariate Gaussian Classifier
    source: https://medium.com/swlh/understanding-gaussian-classifier-6c9f3452358f
    """

    def __init__(self):
        # Create initial dummy data
        # Initial dummy data assumes no dimensionality reduction
        self.train_data = [np.zeros(shape=(2500, 400), dtype=float), np.ones(2500)]
        self.means = [0.0 for _ in range(10)]

        # We don't know the sizes of the covariance matrices
        # So, they will be a 2D, 0x0, empty array for now
        # The main reason why Python is a better alternative
        # for classification tasks than MATLAB...
        # Initial dummy classifier will assume independency
        self.cov_matrices = [np.eye(400, dtype=float) for _ in range(10)]

        # Class prior probabilities, dummy declaration
        # self.clas_priors = [0.0 for _ in range(10)]

        # Initialize pdfs, allow singular covariance matrices for now
        self.pdfs = [multivariate_normal(mean=self.means[i],
                                         cov=self.cov_matrices[i],
                                         allow_singular=True)
                     for i in range(10)]

    def train(self, train_digits, train_labels):
        """
        Train the data, "learn" the pdf for each class
        Dimensionality reduction should be done before using this function
        :param train_digits: Digits with dimensions d_prime.
        d_prime should be reduced using PCA or LDA.
        :param train_labels: Label information
        :return:
        """
        self.train_data = [train_digits, train_labels]
        num_data = len(train_labels)
        for i in range(10):
            # Fetch the training data
            digit_i_digits = train_digits[train_labels == i]
            # digit_i_labels is trivial, won't be stored

            # MLE estimates for each class
            self.means[i] = np.average(digit_i_digits)
            self.cov_matrices[i] = np.cov(digit_i_digits, rowvar=False)

            # Class prior probabilities - basically the ratios of labels in the train data
            # Will not be used -
            # We assume number of 0's 1's... in our dataset
            # have no effect on the ground truth
            # num_digit_i_s = len(digit_i_digits)
            # self.clas_priors[i] = num_digit_i_s / num_data

            # Multivariate class-conditional density functions
            # We will assign the patterns to the highest pdf
            self.pdfs[i] = multivariate_normal(x=digit_i_digits,
                                               mean=self.means[i],
                                               cov=self.cov_matrices[i],
                                               allow_singular=False)

        return

    def predict(self, test_digits):
        """
        Make a set of predictions, using the test set.
        :param: Test digits, they also must be projected using PCA/LDA
        :return: Predicted values. Ground truth is already at hand, no need
        """
        predictions = []
        for j in range(len(test_digits)):
            probabilities = np.array([])

            # Iterate through pdfs and save
            # the estimated probs at that "location" (digit)
            for i in range(10):
                # Prob that this test digit belongs to i
                prob_i = self.pdfs[i].pdf(test_digits[j])
                np.append(probabilities, prob_i)

            # Find the label with the highest prob
            max_prob = -1.0
            max_prob_idx = -1
            for i in range(10):
                if probabilities[i] > max_prob:
                    max_prob_idx = i
                    max_prob = probabilities[i]

            # Argmax is found, append it
            predictions.append(max_prob_idx)
        return predictions

    def clear(self):
        """
        Restart the classifier. Removes all the data
        regarding the previous "training"
        :return:
        """
        self.train_data = [np.zeros(2500, 400, dtype=float), np.ones(2500)]
        self.means = [0.0 for _ in range(10)]
        self.cov_matrices = [np.eye(400, dtype=float) for _ in range(10)]
        self.pdfs = [multivariate_normal(mean=self.means[i],
                                         cov=self.cov_matrices[i],
                                         allow_singular=False)
                     for i in range(10)]


def classification_error(predict_labels, test_labels):
    conf_matrix = confusion_matrix(predict_labels=predict_labels,
                                   test_labels=test_labels)
    return results(conf_matrix)


def confusion_matrix(predict_labels, test_labels):
    """
    Compute the confusion matrix
    :param predict_labels:
    :param test_labels:
    :return: True class (test labels) are the column indexes
            Predictions are row indexes
    """
    conf_matrix = np.zeros(shape=(10, 10), dtype=int)
    for i in range(len(predict_labels)):
        conf_matrix[predict_labels[i]][test_labels[i]] += 1
    return conf_matrix


def results(conf_matrix):
    """
    Compute the classification error
    given a confusion matrix
    :param conf_matrix:
    :return: Wrong classifications / all classifications
    """
    # True classifications are diagonals
    trace = np.trace(conf_matrix)
    sum_of = np.sum(conf_matrix)
    return (sum_of - trace) / sum_of
