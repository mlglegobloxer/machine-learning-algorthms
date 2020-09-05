import numpy as np


def PCA(x, var_to_retain = 0.99):
    Sigma = (1 / np.shape(x)[0]) * np.matmul(np.transpose(x), x)

    # Singular value decomposition of sigma
    results = np.linalg.svd(Sigma)
    U, S = results[0:2]

    # Find min number of principal components to preserve (var_to_retain) varience
    n_pc = np.where((np.cumsum(S) / np.sum(S)) > var_to_retain)[0][0]

    # Return data mapped to a lower dimention
    return np.matmul(x, U[:, 0:n_pc])
