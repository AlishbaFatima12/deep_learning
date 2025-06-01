import numpy as np


class Dataset:
    def __init__(self):
        pass

    # Generate random data
    def generate_dataset():
        np.random.seed(0)
        X = np.random.rand(100, 1) * 10  # Feature X
        Y = 2 * X + 1 + np.random.randn(100, 1)  # Linear relationship with some noise

        # Save the dataset to a text file
        data = np.hstack((X, Y))
        np.savetxt("dataset.txt", data, delimiter=",", header="X,Y", comments="")
