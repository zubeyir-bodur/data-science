import defs


def main():

    # Split the data into two, get the labels

    # Run PCA on the train_data

    # Plot the  eigenvalues in descending order
    # latent(i) is the eigenvalue:
    #   Sum of squared distances of projections
    #   with respect to the origin
    # The eigenvalues were already sorted by the pca() function

    # Display the sample mean of the whole train_data

    # Compute the covariance matrix of the training set

    # Eigenvectors & values of the covariance matrix

    # Display base images just like in Figure 1 :)

    # Project the training data and test data to those subspaces
    #   Estimate a transformation matrix from the training data
    #   Transform both data using this matrix and save them
    # Train a Gaussian classifier for each saved subspace

    # For each saved subspace, whose dimensionality is reduced from training
    # set or test set:
    #   Compute the classification error
    #   Plot the classification error vs Num components used

    return


if __name__ == "__main__":
    main()
