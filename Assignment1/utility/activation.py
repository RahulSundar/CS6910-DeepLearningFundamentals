import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-(z)))


def tanh(z):
    return np.tanh(z)


def sin(z):
    return np.sin(z)


def relu(z):
    return (z>=0)*z + (z<0)*0.001 


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def der_sigmoid(z):
    #return sigmoid(z)*(1 - sigmoid(z))
    return  (1.0 / (1 + np.exp(-(z))))*(1 -  1.0 / (1 + np.exp(-(z))))

def der_tanh(z):
    return 1 - np.tanh(z) ** 2

def der_relu(z):
    return (z>=0)*1 + (z<0)*0.001 
