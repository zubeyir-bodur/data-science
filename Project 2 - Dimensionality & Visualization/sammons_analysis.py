from defs import *
import sammon
import sammon_naive
from sklearn import datasets


def main():
    """
    Same algo on iris dataset:
    iris = datasets.load_iris()
    (X, index) = np.unique(iris.data, axis=0, return_index=True)

    target = iris.target[index]

    y = sammon_naive.sammon(X, alpha=0.3)
    plot_scatter(y, target, "sammon")
    Now, for MNIST dataset
    """
    labels = LABELS.ravel()
    # Run sammon() function to map the dataset into two dimensions
    y, E = sammon.sammon(DIGITS, 2, init="pca",  maxhalves=100, maxiter=10000, tolfun=1e-11)
    # y = sammon_naive.sammon(DIGITS, max_iter=500, epsilon=1e-7, alpha=0.3)
    # Plot the image space WITH their class information
    plot_scatter(y, labels, "sammon")
    return


if __name__ == "__main__":
    main()
