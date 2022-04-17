from defs import *
from sklearn.manifold import TSNE


def main():
    labels = LABELS.ravel()
    # Run t-SNE to map the dataset into two dimensions
    digits_embedded = TSNE(n_components=2,
                           perplexity=50.0,
                           early_exaggeration=30.0,
                           n_iter=10000,
                           learning_rate=30.0,
                           metric='manhattan',
                           init='pca').fit_transform(X=DIGITS, y=labels)
    # Plot the image space WITH their class information
    plot_scatter(digits_embedded, labels, "tsne")
    return


if __name__ == "__main__":
    main()
