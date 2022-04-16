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

    # Uncomment to display all training digits
    # img_train = display_blocks(train_data, 50, 50)
    # Image.fromarray(img_train).show()

    # Uncomment to display all test digits
    # img_test = display_blocks(test_data, 50, 50)
    # Image.fromarray(img_test).show()

    # Uncomment to show partitioned digits
    # Currently, 3 will be displayed
    # partitioned_train_data = partition(train_data, train_label)
    # partitioned_test_data = partition(test_data, test_label)
    # Image.fromarray(display_blocks(partitioned_train_data[3], 20, 12)).show()

    pca = PCA()
    pca.fit(train_data)
    cov_matrix = pca.get_covariance()

    # Covariance matrix of the train_data
    [eigenvalues, eigenvectors] = linalg.eig(cov_matrix)

    # Eigenvalues are sorted in descending order by default
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors)
    """
    # Uncomment to plot the eigenvalues in descending order
    figure(figsize=(8, 6), dpi=300)
    plt.bar(x=[i for i in range(len(eigenvalues))]
            , height=eigenvalues
            , width=1.5
            , align='center'
            , color='#4444FF'
            , edgecolor='#000000'
            , linewidth=0.1)
    plt.show()
    """

    """
    # Display the sample mean of the whole train_data
    mean_img = pack(pca.mean_)
    Image.fromarray(mean_img).show()
    """


    # Display top 205 eigenvectors
    eigen_digits = display_blocks(eigenvectors, code="eig")
    Image.fromarray(eigen_digits).show()


    # Train a Gaussian classifier for each subspace
    classification_errors = []
    num_components = []
    for i in range(40):
        # Project the training data and test data to those subspaces
        #   The subspaces are chosen from first 60, 65, .. , 160 PCA components
        num_components.append(5*i + 5)
        pca = PCA(n_components=num_components[i])
        subspace_train = pca.fit_transform(train_data)
        subspace_test = pca.transform(test_data)

        # Train the classifier with the PCA components
        classifier = MultivariateGaussianClassifier()
        classifier.train(train_digits=subspace_train, train_labels=train_label)

        # Test the classifier
        predictions = classifier.predict(test_digits=subspace_test)

        # Gather results
        classification_errors.append(
            classification_error(predict_labels=predictions,
                                 test_labels=test_label))

    # Plot the classification error vs Num components used
    figure(figsize=(8, 6), dpi=300)
    plt.plot(num_components, classification_errors)
    plt.title("Classification Error Plot for Principal Component Analysis")
    plt.xlabel("# of First PCA Components Used")
    plt.ylabel("Classification Error (%)")
    plt.show()
    return


if __name__ == "__main__":
    main()
