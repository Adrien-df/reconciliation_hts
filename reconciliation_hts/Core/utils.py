import numpy as np


def cross_product(matrix):
    n = matrix.shape[0]
    output = np.zeros([matrix.shape[1], matrix.shape[1]])
    for i in range(n):
        mat = matrix[i].T@matrix[i]
        output += mat
    return(output)


def scale(matrix, scales):
    p, n = (matrix.shape[1], matrix.shape[0])
    scaled = np.zeros([n, p])
    for j in range(p):
        for i in range(n):
            scaled[i, j] = matrix[i, j]/scales[j]
    return(scaled)
