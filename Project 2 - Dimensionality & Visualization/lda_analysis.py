from defs import *
from numpy import linalg
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib.pyplot import figure
from multivariate_gaussian_classifier import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def main():
    # Split the data into two, get the labels
    train_data, test_data, train_label, test_label = split(DIGITS)

    # Uncomment to display the bases as images
    lda = LinearDiscriminantAnalysis()

    # Shared covariance matrix of the train_data
    lda.fit(train_data, train_label)
    bases = np.transpose(lda.scalings_)
    bases_img = display_blocks(lda.scalings_, 3, 3, 20, 20, code="eig")
    Image.fromarray(bases_img).show()

    # Train a Gaussian classifier for each subspace
    classification_errors_test = []
    classification_errors_train = []
    num_components = []
    for i in range(9):
        # Project the training data and test data to those subspaces
        #   The subspaces are chosen from first 1-9 LDA components
        num_components.append(i+1)
        lda = LinearDiscriminantAnalysis(n_components=num_components[i], solver='eigen')
        subspace_train = lda.fit_transform(train_data, train_label)
        subspace_test = lda.transform(test_data)

        # Train the classifier with the PCA components
        classifier = MultivariateGaussianClassifier()
        classifier.train(train_digits=subspace_train, train_labels=train_label)

        # Test the classifier
        predictions = classifier.predict(test_digits=subspace_test)

        # Gather results - on the test data
        classification_errors_test.append(
            classification_error(predict_labels=predictions,
                                 test_labels=test_label))

        # Gather results - on the train data
        predictions = classifier.predict(test_digits=subspace_train)
        classification_errors_train.append(
            classification_error(predict_labels=predictions,
                                 test_labels=train_label))

    # Plot the classification error vs Num components used - test data
    figure(figsize=(8, 6), dpi=300)
    plt.plot(num_components, classification_errors_test)
    plt.title("Classification Error Plot for LDA - Test Set")
    plt.xlabel("# of First LDA Components Used")
    plt.ylabel("Classification Error (%)")
    plt.show()

    # Plot the classification error vs Num components used - train data
    figure(figsize=(8, 6), dpi=300)
    plt.plot(num_components, classification_errors_train)
    plt.title("Classification Error Plot for LDA - Training Set")
    plt.xlabel("# of First LDA Components Used")
    plt.ylabel("Classification Error (%)")
    plt.show()
    return


if __name__ == "__main__":
    main()
