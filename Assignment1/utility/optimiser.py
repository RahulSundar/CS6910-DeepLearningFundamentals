import numpy as np
init_w, init_b = -2.0 , -2.0
X = [2.5, 0.9]
Y = [0.5, 0.1]

from lossFunction import *
from activations import *
from gradientCalculations import *

class Optimiser():

    def __init__(self, max_epochs) 
        self.max_epochs = max_epochs
        
    
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
            
