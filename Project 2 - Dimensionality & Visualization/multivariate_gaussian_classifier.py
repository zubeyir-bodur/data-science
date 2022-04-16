import numpy as np
from scipy.stats import multivariate_normal
import defs
from PIL import Image


class MultivariateGaussianClassifier:
    """
    Multivariate Gaussian Classifier
    source: https://medium.com/swlh/understanding-gaussian-classifier-6c9f3452358f
    """
    means = []
    cov_matrices = []
    random_variables = []
    class_priors = []
    dim = 0

    def __init__(self):
        self.dim = 0
        self.means = []
        self.cov_matrices = []
        self.random_variables = []
        self.class_priors = []

    def train(self, train_digits, train_labels):
        """
        Train the data, "learn" the pdf for each class
        Dimensionality reduction should be done before using this function
        :param train_digits: Digits with dimensions d_prime.
        d_prime should be reduced using PCA or LDA.
        :param train_labels: Label information
        :return:
        """
        self.means = []
        self.cov_matrices = []
        self.dim = len(train_digits[0])
        self.random_variables = []
        self.class_priors = []

        num_data = len(train_labels)

        for i in range(10):
            # Fetch the training data
            digit_i_digits = np.zeros(shape=(len(train_digits), len(train_digits[0])), dtype=float)
            digit_i_count = 0
            for j in range(len(train_digits)):
                if train_labels[j] == i:
                    digit_i_digits[digit_i_count, :] = train_digits[j, :]
                    digit_i_count += 1
            digit_i_digits = np.array(digit_i_digits[0:digit_i_count, :])
            # MLE estimates for each class
            self.means.append(np.mean(digit_i_digits, axis=0))
            self.cov_matrices.append(np.cov(digit_i_digits, rowvar=False))

            # Uncomment to display the sample mean of each digit class
            # display_mean = defs.squash(self.means[i])
            # Image.fromarray(display_mean).show()

            # Class prior probabilities - basically the ratios of labels in the train data
            num_digit_i_s = len(digit_i_digits)
            self.class_priors.append(num_digit_i_s / num_data)

            # Multivariate class-conditional density functions
            # We will assign the patterns to the highest class posterior
            self.random_variables.append(multivariate_normal(mean=self.means[i],
                                                             cov=self.cov_matrices[i],
                                                             allow_singular=True))

        return

    def predict(self, test_digits):
        """
        Make a set of predictions, using the test set.
        :param: Test digits, they also must be projected using PCA/LDA
        :return: Predicted values. Ground truth is already at hand, no need
        """
        predictions = []
        for j in range(len(test_digits)):
            probabilities = []

            # Iterate through pdfs and save
            # the estimated probs at that "location" (digit)
            for i in range(10):
                # Prob that this test digit belongs to i
                class_posterior_i = self.random_variables[i].pdf(test_digits[j]) * self.class_priors[i]
                probabilities.append(class_posterior_i)

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
        regarding the previous "training" - Not used
        :return:
        """
        self.means = [0.0 for _ in range(10)]
        self.cov_matrices = [np.eye(400, dtype=float) for _ in range(10)]
        self.dim = 0


def multivariate_normal_distribution(x, d, mean, covariance):
    """pdf of the multivariate normal distribution.
    source: https://peterroelants.github.io/posts/multivariate-normal-primer/
    Not used
    """
    x_m = x - mean
    covariance_inverse = np.linalg.inv(covariance)
    x_m_transpose = np.transpose(x_m)
    return (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(covariance)))) *\
        np.exp(-(np.matmul(np.matmul(x_m_transpose, covariance_inverse), x_m)) / 2)


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
    return 100 * (sum_of - trace) / sum_of
