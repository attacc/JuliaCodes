import numpy as np

def W2H_rotate(M, eigenvec):
    return np.conjugate(eigenvec).T @ M @ eigenvec

def H2W_rotate(M, eigenvec):
    return eigenvec @ M @ np.conjugate(eigenvec).T

