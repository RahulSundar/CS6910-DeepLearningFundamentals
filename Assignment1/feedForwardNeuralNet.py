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
        self.Y_shape = Y_train.shape
        self.epochs = epochs
        self.weights, self.biases = self.initialiseNeuralNet(layers)
        
        self.Optimiser = Optimiser()
        
    def oneHotEncode(self, Y_train_raw):
        Ydata = np.zeros((self.num_classes,Y_train_raw.shape[0]))
        for i in range(Y_train_raw.shape[0]): 
            value = Y_train_raw[i] 
            Ydata[int(value)][i] = 1.0 
        return Ydata


    def Xavier_initialiser(self, size):
        in_dim = size[1]
        out_dim = size[0]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return np.random.normal( 0, xavier_stddev, size = (out_dim, in_dim))

            

    def initialiseNeuralNet(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.Xavier_initialiser(size=[layers[l+1], layers[l]])
            b = np.zeros((layers[l + 1],1))
            weights.append(W)
            biases.append(b)
        return weights, biases  



    def forwardPropagate(self, X_train_batch, weights, biases, activation):
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
        X = X_train_batch
        H = []
        A = []
        H.append(X)
        A.append(X)
        for l in range(0, num_layers - 2):
            if l == 0:
                W = weights[l]
                b = biases[l]
                A.append(np.add(np.matmul(W,X), b))
                H.append(activation(A[l+1]))
            else:
                W = weights[l]
                b = biases[l]
                A.append(np.add(np.matmul(W, H[l]), b))
                H.append(activation(A[l+1]))

        # Here the last layer is not activated as it is a regression problem
        W = weights[-1]
        b = biases[-1]
        A.append(np.add(np.matmul( W, H[-1]), b))
        #Y = softmax(A[-1])
        Y = softmax(A[-1])
        return Y, H, A
        

        
    #Back prop trial code. Lot of cleanup to be done. May be a dictionary/ simple list of arrays should work.     
    def backPropagate(self, Y,H,A,Y_train_batch):
     
        gradients_preactivations = []
        gradients_activations = []
        gradients_weights = [] 
        gradients_biases = [] 
        num_layers = len(layers) 
        globals()["grad_a"+str(len(layers)-1)] = np.array([-(Y_train - Y)]).transpose()
        for l in range(num_layers - 2, -1,-1): 
             globals()["grad_W"+str(l+1)] = np.outer(globals()["grad_a"+str(l+1)], H[l]) 
             globals()["grad_b"+str(l+1)] = globals()["grad_a"+str(l+1)] 
             gradients_weights.append(globals()["grad_W"+str(l+1)]) 
             gradients_biases.append(globals()["grad_b"+str(l+1)]) 
             globals()["grad_h"+str(l)] = np.matmul(weights[l].transpose(), globals()["grad_a"+str(l+1)]) 
             globals()["grad_a"+str(l)] = np.multiply(globals()["grad_h"+str(l)], der_activation(A[l])) 
        return gradients_weights, gradients_biases


        
    
    def batchGradientDescent(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers)
        
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train_batch, weights, biases, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train_batch)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
                num_points_seen +=1
                if int(num_points_seen) % batch_size == 0:
                    weights = [weights[i] - deltaw[i] for i in range(len(weights))] 
                    biases = [biases[i] - deltab[i] for i in range(len(biases))]
                    #resetting gradient updates
                    deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
                    deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        return weights, biases
                             
                             
                             
    def gradientDescent(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), weights, biases, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
            weights = [weights[i] - learning_rate*deltaw[i] for i in range(len(weights))] 
            biases = [biases[i] - learning_rate*deltab[i] for i in range(len(biases))]

        return weights, biases


    
    def momentumGradientDescent(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        gamma = 0.9
        prev_v_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        prev_v_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), weights, biases, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
            
            v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
            v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
            
            weights = [weights[i] - v_w[i] for i in range(len(weights))] 
            biases = [biases[i] - v_b[i] for i in range(len(biases))]
            prev_v_w = v_w
            prev_v_b = v_b
        return weights, biases





    def stochasticMomentumGradientDescent(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        gamma = 0.9
        prev_v_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        prev_v_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), weights, biases, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]
            
                v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
            
                weights = [weights[i] - v_w[i] for i in range(len(weights))] 
                biases = [biases[i] - v_b[i] for i in range(len(biases))]
                prev_v_w = v_w
                prev_v_b = v_b
        
        return weights, biases




    def nesterovGradientDescent(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        gamma = 0.9
        prev_v_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        prev_v_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            winter = [weights[i] - gamma*prev_v_w[i] for l in range(0, len(layers)-1)]  
            binter = [biases[i] - gamma*prev_v_b[i] for l in range(0, len(layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), winter, binter, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
            
            v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
            v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
        
            weights = [weights[i] - v_w[i] for i in range(len(weights))] 
            biases = [biases[i] - v_b[i] for i in range(len(biases))]
            prev_v_w = v_w
            prev_v_b = v_b
    
        return weights, biases
    
    
    
    
    
    def stochasticNesterovGradientDescent(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        gamma = 0.9
        prev_v_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        prev_v_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            winter = [weights[i] - gamma*prev_v_w[i] for l in range(0, len(layers)-1)]  
            binter = [biases[i] - gamma*prev_v_b[i] for l in range(0, len(layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), winter, binter, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]
            
                v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
            
                weights = [weights[i] - v_w[i] for i in range(len(weights))] 
                biases = [biases[i] - v_b[i] for i in range(len(biases))]
                prev_v_w = v_w
                prev_v_b = v_b
    
        return weights, biases
    
    def stochasticGradientDescent(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), weights, biases, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]
                weights = [weights[i] - learning_rate*deltaw[i] for i in range(len(weights))] 
                biases = [biases[i] - learning_rate*deltab[i] for i in range(len(biases))]

        return weights, biases
    
    
    

    def rmsProp(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        eps, beta = 1e-8, 0.9
        v_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        v_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), weights, biases, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
            
            v_w = [beta*v_w[i] + (1-beta)*deltaw[i]**2 for i in range(num_layers - 1)]
            v_b = [beta*v_b[i] + (1-beta)*deltab[i]**2 for i in range(num_layers - 1)]
            
            weights = [weights[i] - learning_rate/np.sqrt(v_w[i]+eps) for i in range(len(weights))] 
            biases = [biases[i] - learning_rate/np.sqrt(v_b[i]+eps) for i in range(len(biases))]

        return weights, biases
    
    

    def adam(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        eps, beta1, beta2 = 1e-8, 0.9, 0.99
        m_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        m_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        v_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        v_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]        

        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), weights, biases, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

            m_w = [beta*m_w[i] + (1-beta1)*deltaw[i] for i in range(num_layers - 1)]
            m_b = [beta*m_b[i] + (1-beta1)*deltab[i] for i in range(num_layers - 1)]
            v_w = [beta*v_w[i] + (1-beta2)*deltaw[i]**2 for i in range(num_layers - 1)]
            v_b = [beta*v_b[i] + (1-beta2)*deltab[i]**2 for i in range(num_layers - 1)]
            
            m_w = [m_w[i]/(1-beta1**(epoch+1)) for i in range(num_layers - 1)]
            m_b = [m_b[i]/(1-beta1**(epoch+1)) for i in range(num_layers - 1)]            
            v_w = [v_w[i]/(1-beta2**(epoch+1)) for i in range(num_layers - 1)]
            v_b = [v_b[i]/(1-beta2**(epoch+1)) for i in range(num_layers - 1)]
            weights = [weights[i] - learning_rate*m_w[i]/np.sqrt(v_w[i]+eps) for i in range(len(weights))] 
            biases = [biases[i] - learning_rate*m_b[i]/np.sqrt(v_b[i]+eps) for i in range(len(biases))]

        return weights, biases
    

    def nadam(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases)
        eps, beta1, beta2 = 1e-8, 0.9, 0.99
        m_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        m_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
        v_w = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
        v_b = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]        

        for epoch in range(epochs):
            deltaw = [np.zeros((layers[l+1], layers[l])) for l in range(0, len(layers)-1)]
            deltab = [np.zeros((layers[l+1], 1)) for l in range(0, len(layers)-1)]
            winter = [weights[i] - gamma*prev_v_w[i] for l in range(0, len(layers)-1)]  
            binter = [biases[i] - gamma*prev_v_b[i] for l in range(0, len(layers)-1)]
            for i in range(length_dataset):
                Y,H,A = forwardPropagate(X_train[i,:].reshape(784,1), winter, binter, activation) 
                grad_weights, grad_biases = backpropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

            m_w = [beta*m_w[i] + (1-beta1)*deltaw[i] for i in range(num_layers - 1)]
            m_b = [beta*m_b[i] + (1-beta1)*deltab[i] for i in range(num_layers - 1)]
            v_w = [beta*v_w[i] + (1-beta2)*deltaw[i]**2 for i in range(num_layers - 1)]
            v_b = [beta*v_b[i] + (1-beta2)*deltab[i]**2 for i in range(num_layers - 1)]
            
            m_w = [m_w[i]/(1-beta1**(epoch+1)) for i in range(num_layers - 1)]
            m_b = [m_b[i]/(1-beta1**(epoch+1)) for i in range(num_layers - 1)]            
            v_w = [v_w[i]/(1-beta2**(epoch+1)) for i in range(num_layers - 1)]
            v_b = [v_b[i]/(1-beta2**(epoch+1)) for i in range(num_layers - 1)]
            weights = [weights[i] - learning_rate*m_w[i]/np.sqrt(v_w[i]+eps) for i in range(len(weights))] 
            biases = [biases[i] - learning_rate*m_b[i]/np.sqrt(v_b[i]+eps) for i in range(len(biases))]

        return weights, biases
    
    
    
    
    
    
    
    
    
    
    
    
    
