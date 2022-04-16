from defs import *
from sklearn.manifold import TSNE


def main():

    # Run t-SNE to map the dataset into two dimensions
    digits_embedded = TSNE(n_components=2,
                           learning_rate='auto',
                           init='random').fit_transform(X=DIGITS, y=LABELS)
    # Plot the image space WITH their class information
    plot_scatter(digits_embedded, LABELS, "tsne")
    return


if __name__ == "__main__":
    main()
