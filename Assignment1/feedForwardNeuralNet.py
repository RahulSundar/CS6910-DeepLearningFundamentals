import numpy as np
import scipy as sp

from dataPreprocessing import *
from utility.optimiser import Optimiser
from utility.lossFunction import *
from utility.activation import *
from utility.gradientCalculations import *

class FeedForwardNeuralNetwork():
    def __init__(self, layers, epochs, X_train,Y_train_raw):
        self.layers = layers
        self.X_train = X_train
        
        self.num_classes = np.max(Y_train)
        self.Y_train = self.oneHotEncode(Y_train_raw)
        
        self.epochs = epochs
        self.weights, self.biases = self.initialiseNeuralNet(layers)
        


    def oneHotEncode(self, Y_train_raw):
        Ydata = np.zeros((self.num_classes,Y_train_raw.shape[0]))
        for i in range(Y_train_raw.shape[0]): 
            value = Y_train_raw[i] 
            Ydata[int(value)][i] = 1.0 
        return Ydata
    
    
    def Xavier_initialiser(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return np.random.normal( 0, xavier_stddev, size = (in_dim, out_dim))

            
    def initialiseNeuralNet(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.Xavier_initialiser(size=[layers[l], layers[l + 1]])
            b = np.zeros((1, layers[l + 1]))
            weights.append(W)
            biases.append(b)
        return weights, biases  

    def forwardPropagate(self, Input_data, weights, biases, activation):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                 : X - input matrix
                 : Weights  - Weights matrix
                 : biases - Bias vectors 
        """
        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        #A - Preactivations
        #H - Activations
        X = Input_data
        H = []
        A = []
        H.append(X)
        A.append(X)
        for l in range(0, num_layers - 2):
            if l == 0:
                W = weights[l]
                b = biases[l]
                A.append(np.add(np.matmul(X, W), b))
                H.append(activation(A[l+1]))
            else:
                W = weights[l]
                b = biases[l]
                A.append(np.add(np.matmul(H[l], W), b))
                H.append(activation(A[l+1]))

        # Here the last layer is not activated as it is a regression problem
        W = weights[-1]
        b = biases[-1]
        A.append(np.add(np.matmul(H[-1], W), b))
        #Y = softmax(A[-1])
        Y = A[-1]
        return Y, H, A
            
            
    #Back prop trial code. Lot of cleanup to be done. May be a dictionary/ simple list of arrays should work. 
    def backPropagate(Y,H,A,Y_train, der_activation): 
      
        gradients_weights = [] 
        gradients_biases = [] 
        num_layers = len(layers) 
        globals()["grad_a"+str(len(layers)-1)] = -np.array(Y_train - Y).transpose() 
        for l in range(num_layers - 2, -1,-1): 
             globals()["grad_W"+str(l+1)] = np.outer(globals()["grad_a"+str(l+1)], H[l]) 
             globals()["grad_b"+str(l+1)] = globals()["grad_a"+str(l+1)] 
             gradients_weights.append(globals()["grad_W"+str(l+1)]) 
             gradients_biases.append(globals()["grad_b"+str(l+1)]) 
             globals()["grad_h"+str(l)] = np.matmul(weights[l], globals()["grad_a"+str(l+1)]) 
             globals()["grad_a"+str(l)] = np.multiply(globals()["grad_h"+str(l)], der_activation(A[l]).transpose()) 
        return gradients_weights, gradients_biases
        
    def gradientDescent(self, epochs, )
        pass
