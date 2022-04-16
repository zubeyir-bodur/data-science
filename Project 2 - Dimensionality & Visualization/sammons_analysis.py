from sammon import *
from defs import *
import matplotlib.pyplot as plt


def main():

    # Run sammon() function to map the dataset into two dimensions
    y = sammon(DIGITS, 2)
    # Plot the image space WITH their class information
    plot_scatter(y, LABELS, "sammon")
    return


if __name__ == "__main__":
    main()
