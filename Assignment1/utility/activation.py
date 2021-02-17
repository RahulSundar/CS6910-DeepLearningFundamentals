import numpy as np


def sigmoid(z):
    return 1.0/ (1 + np.exp(-(z)))


def tanh(z):
    return np.tanh(z)
    
    
def sin(z)
    return np.sin(z)
    
    
def reLu(z)
    return np.max(z,0)
