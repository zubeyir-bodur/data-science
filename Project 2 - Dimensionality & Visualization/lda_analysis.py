from defs import *
from numpy import linalg
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
from multivariate_gaussian_classifier import *
from sklearn.decomposition import PCA


def main():
    # Split the data into two, get the labels
    train_data, test_data, train_label, test_label = split(DIGITS)

    pca = PCA()
    pca.fit(train_data)
    cov_matrix = pca.get_covariance()

    # Covariance matrix of the train_data
    [eigenvalues, eigenvectors] = linalg.eig(cov_matrix)

    # Display all of the eigen-digits
    eigen_digits = display_blocks(eigenvectors, 20, 20, code="eig")
    Image.fromarray(eigen_digits).show()

    # TODO:
    #   Project the training data and test data to those subspaces
    #       Estimate a transformation matrix from the training data
    #       Transform both data using this matrix and save them

    # Train a Gaussian classifier for each saved subspace
    subspaces_train = []
    subspaces_test = []
    classification_errors = []
    num_components = []
    for i in range(20):
        classifier = MultivariateGaussianClassifier()
        classifier.train(train_digits=subspaces_train, train_labels=train_label)
        predictions = classifier.predict(test_digits=subspaces_test)
        classification_errors.append(
            classification_error(predict_labels=predictions,
                                 test_labels=test_label))
        num_components.append(np.shape(subspaces_train)[2])
        classifier.clear()

    # TODO: Plot the classification error vs Num components used
    return

    return


if __name__ == "__main__":
    main()
