import numpy as np
import scipy as sp

#from dataPreprocessing import *
#from utility.optimiser import Optimiser
#from utility.lossFunction import *
from utility.activation import *
#from utility.gradientCalculations import *

class FeedForwardNeuralNetwork():
    def __init__(self, num_hidden_layers, num_hidden_neurons, X_train_raw, Y_train_raw):
    
        '''
        Here, we initialise the FeedForwardNeuralNetwork class with the number of hidden layers, number of hidden neurons, raw training data. 
        '''
        self.data_size = Y_train_raw.shape[0] #[NTRAIN]
        
        self.X_train = np.transpose(X_train_raw.reshape(X_train_raw.shape[0], X_train_raw.shape[1]*X_train_raw.shape[2])) # [IMG_HEIGHT*IMG_WIDTH X NTRAIN]
        self.X_train = self.X_train/255
        self.num_classes = np.max(Y_train_raw) + 1 
        self.Y_train = self.oneHotEncode(Y_train_raw) #[NUM_CLASSES X NTRAIN]
                

        self.Y_shape = self.Y_train.shape 
        
        self.output_layer_size = self.num_classes
        self.img_height = X_train_raw.shape[1]
        self.img_width = X_train_raw.shape[2]
        self.img_flattened_size = self.img_height*self.img_width
        
        #self.layers = layers
        self.layers = [self.img_flattened_size] + num_hidden_layers*[num_hidden_neurons] + [self.output_layer_size]
        
        self.weights, self.biases = self.initialiseNeuralNet(self.layers)
        self.activation = np.tanh
        self.der_activation = der_tanh
    
    #helper functions
    def oneHotEncode(self, Y_train_raw):
        Ydata = np.zeros((self.num_classes,Y_train_raw.shape[0]))
        for i in range(Y_train_raw.shape[0]): 
            value = Y_train_raw[i] 
            Ydata[int(value)][i] = 1.0 
        return Ydata
        
    def meanSquaredErrorLoss(self,Y_train,Y_pred):
        MSE = np.mean((Y_train - Y_pred)**2)
        return MSE
        
    def crossEntropyLoss(self, expected, predicted):
        CE = [-expected[i]*np.log(predicted[i]) for i in range(len(predicted)) ]
        crossEntropy = np.sum(CE)
        return crossEntropy

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
    def backPropagate(self, Y,H,A, Y_train_batch, der_activation):
     
        gradients_weights = [] 
        gradients_biases = [] 
        num_layers = len(self.layers) 
        globals()["grad_a"+str(len(self.layers)-1)] = -(Y_train_batch - Y)
        for l in range(num_layers - 2, -1,-1): 
             globals()["grad_W"+str(l+1)] = np.outer(globals()["grad_a"+str(l+1)], H[l]) 
             globals()["grad_b"+str(l+1)] = globals()["grad_a"+str(l+1)] 
             gradients_weights.append(globals()["grad_W"+str(l+1)]) 
             gradients_biases.append(globals()["grad_b"+str(l+1)]) 
             globals()["grad_h"+str(l)] = np.matmul(self.weights[l].transpose(), globals()["grad_a"+str(l+1)]) 
             globals()["grad_a"+str(l)] = np.multiply(globals()["grad_h"+str(l)], der_activation(A[l])) 
        return gradients_weights, gradients_biases


    def predict(self):
        Y_pred = []
        for i in range(self.data_size):
            Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation)
            Y_pred.append(Y.reshape(10,))
        Y_pred = np.array(Y_pred).transpose()
        return Y_pred    
    
    def batchGradientDescent(self, epochs,length_dataset, batch_size, learning_rate):
        loss = []
        num_layers = len(self.layers)
        for epoch in range(epochs):
            CE = []
            Y_pred = []
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
                num_points_seen +=1
                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
                if int(num_points_seen) % batch_size == 0:
                    #print(num_points_seen)
                    self.weights = [self.weights[i] -learning_rate*deltaw[i]/batch_size for i in range(len(self.weights))] 
                    self.biases = [self.biases[i] - learning_rate*deltab[i]/batch_size for i in range(len(self.biases))]
                    #resetting gradient updates
                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            print(learning_rate, epoch, np.sum(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
            #weights, biases = self.weights, self.biases
        return self.weights, self.biases, loss, Y_pred
                             
    def stochasticGradientDescent(self, epochs,length_dataset, learning_rate):
        loss = []
        num_layers = len(self.layers)
        for epoch in range(epochs):
            CE = []
            Y_pred = []
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))

                #print(num_points_seen)
                self.weights = [self.weights[i] -learning_rate*deltaw[i] for i in range(len(self.weights))] 
                self.biases = [self.biases[i] - learning_rate*deltab[i] for i in range(len(self.biases))]
            
            print(learning_rate, epoch, np.sum(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))

        return self.weights, self.biases, loss, Y_pred
                              
                             
    def gradientDescent(self, epochs, length_dataset, learning_rate):
        num_layers = len(self.layers)
        loss = []
        for epoch in range(epochs):
            CE = []
            Y_pred = []
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))

            self.weights = [self.weights[i] - learning_rate*deltaw[i]/length_dataset for i in range(len(self.weights))] 
            self.biases = [self.biases[i] - learning_rate*deltab[i]/length_dataset for i in range(len(self.biases))]
            
            print(learning_rate, epoch, np.sum(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        return self.weights, self.biases, loss, Y_pred

 
    def momentumGradientDescent(self, epochs,length_dataset, learning_rate):
        gamma = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        for epoch in range(epochs):
            CE = []
            Y_pred = []
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
            
            v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i]/length_dataset for i in range(num_layers - 1)]
            v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i]/length_dataset for i in range(num_layers - 1)]
            
            self.weights = [self.weights[i] - v_w[i] for i in range(len(self.weights))] 
            self.biases = [self.biases[i] - v_b[i] for i in range(len(self.biases))]
            prev_v_w = v_w
            prev_v_b = v_b

            print(learning_rate, epoch, np.sum(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        return self.weights, self.biases, loss, Y_pred


    def batchMomentumGradientDescent(self, epochs,length_dataset, batch_size, learning_rate):
        gamma = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        for epoch in range(epochs):
            CE = []
            Y_pred = []
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:

                    v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]
                    
                    self.weights = [self.weights[i] - v_w[i] for i in range(len(self.weights))] 
                    self.biases = [self.biases[i] - v_b[i] for i in range(len(self.biases))]

                    prev_v_w = v_w
                    prev_v_b = v_b

                    #resetting gradient updates
                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            print(learning_rate, epoch, np.sum(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        return self.weights, self.biases, loss, Y_pred



    def stochasticMomentumGradientDescent(self, epochs,length_dataset, learning_rate):
        gamma = 0.9

        loss = []
        num_layers = len(self.layers)
        
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        for epoch in range(epochs):
            CE = []
            Y_pred = []            
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            for i in range(length_dataset):
                
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))            

                v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
            
                self.weights = [self.weights[i] - v_w[i] for i in range(len(self.weights))] 
                self.biases = [self.biases[i] - v_b[i] for i in range(len(self.biases))]
                
                prev_v_w = v_w
                prev_v_b = v_b
        
            print(learning_rate, epoch, np.sum(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        
        return self.weights, self.biases, loss, Y_pred




    def nesterovGradientDescent(self,epochs,length_dataset, learning_rate):
        gamma = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        for epoch in range(epochs):
            CE = []
            Y_pred = []  
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            winter = [self.weights[i] - gamma*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            binter = [self.biases[i] - gamma*prev_v_b[i] for i in range(0, len(self.layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), winter, binter, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
                            
            v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i]/length_dataset for i in range(num_layers - 1)]
            v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i]/length_dataset for i in range(num_layers - 1)]
        
            self.weights = [self.weights[i] - v_w[i] for i in range(len(self.weights))] 
            self.biases = [self.biases[i] - v_b[i] for i in range(len(self.biases))]
            prev_v_w = v_w
            prev_v_b = v_b
    
            print(learning_rate, epoch, np.sum(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        
        return self.weights, self.biases, loss, Y_pred
    
    
    def stochasticNesterovGradientDescent(self,epochs,length_dataset, learning_rate):
        gamma = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        for epoch in range(epochs):
            CE = []
            Y_pred = []  
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            winter = [self.weights[i] - gamma*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            binter = [self.biases[i] - gamma*prev_v_b[i] for i in range(0, len(self.layers)-1)]
            
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), winter, binter, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
                            
                v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
        
                self.weights = [self.weights[i] - v_w[i] for i in range(len(self.weights))] 
                self.biases = [self.biases[i] - v_b[i] for i in range(len(self.biases))]
                prev_v_w = v_w
                prev_v_b = v_b
    
            print(learning_rate, epoch, np.sum(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        
        return self.weights, self.biases, loss, Y_pred
    

    def batchNesterovGradientDescent(self,epochs,length_dataset, batch_size,learning_rate):
        gamma = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        for epoch in range(epochs):
            CE = []
            Y_pred = []  
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            winter = [self.weights[i] - gamma*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            binter = [self.biases[i] - gamma*prev_v_b[i] for i in range(0, len(self.layers)-1)]
            num_points_seen = 0
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), winter, binter, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))

                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:                            

                    v_w = [gamma*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [gamma*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]
        
                    self.weights = [self.weights[i] - v_w[i] for i in range(len(self.weights))] 
                    self.biases = [self.biases[i] - v_b[i] for i in range(len(self.biases))]
                    prev_v_w = v_w
                    prev_v_b = v_b

                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

    
            print(learning_rate, epoch, np.sum(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        
        return self.weights, self.biases, loss, Y_pred
    
      
    
    

    def rmsProp(self, epochs,length_dataset, batch_size, learning_rate):

        loss = []
        num_layers = len(self.layers)
        eps, beta = 1e-8, 0.9
        
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        for epoch in range(epochs):
            CE = []
            Y_pred = []
                        
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            for i in range(length_dataset):
            
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
            
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
                
                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))            
            
            v_w = [beta*v_w[i] + (1-beta)*(deltaw[i])**2 for i in range(num_layers - 1)]
            v_b = [beta*v_b[i] + (1-beta)*(deltab[i])**2 for i in range(num_layers - 1)]
            print(v_w[0].shape)
            self.weights = [self.weights[i] - deltaw[i]*(learning_rate/np.sqrt(v_w[i]+eps)) for i in range(len(self.weights))] 
            self.biases = [self.biases[i] - deltab[i]*(learning_rate/np.sqrt(v_b[i]+eps)) for i in range(len(self.biases))]
    
            print(learning_rate, epoch, np.sum(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.sum(CE))
        
        return self.weights, self.biases, loss, Y_pred
    
  

    def adam(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases):
        eps, beta1, beta2 = 1e-8, 0.9, 0.99
        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        

        for epoch in range(epochs):
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
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
  
    def adamMiniBatch(self, epochs,length_dataset, batch_size, learning_rate):
        eps, beta1, beta2 = 1e-8, 0.9, 0.99
        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        

        for epoch in range(epochs):
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
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
'''
    def nadam(X_train,Y_train, epochs,length_dataset, batch_size, learning_rate, num_layers, layers, weights, biases):
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
                grad_weights, grad_biases = backPropagate(Y,H,A,Y_train[:,i].reshape(10,1), der_activation)
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
    
    

    def train(self):
        pass
        
        
    def predict(self):
        pass
        
    
'''    
    
    
    
    
    
    
    
    
    
