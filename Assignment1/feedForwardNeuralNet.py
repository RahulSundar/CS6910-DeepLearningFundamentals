import numpy as np
import scipy as sp

from dataPreprocessing import *
from utility.optimiser import Optimiser
from utility.lossFunction import *
from utility.activation import *
from utility.gradientCalculations import *

class FeedForwardNeuralNetwork():
    def __init__(self):
        pass

    def Xavier_initialiser(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        # modify the 
        return np.random.normal([in_dim, out_dim], stddev=xavier_stddev),
            dtype=tf.float32,
        )

        
    def initialiseNeuralNet(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_initialiser(size=[layers[l], layers[l + 1]])
            b = np.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def feedForwardNeuralNet(Input_data, weights, biases, activation):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                 : X - input matrix
                 : Weights  - Weights matrix
                 : biases - Bias vectors 
        """
        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        # H - Activation units
        # X - Input
        X = Input_data
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = activation(np.add(np.matmul(H, W), b))

        # Here the last layer is not activated as it is a regression problem
        W = weights[-1]
        b = biases[-1]
        Y = np.add(np.matmul(H, W), b)
        return Y    

