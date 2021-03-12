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
        
        #self.weights, self.biases = self.initialiseNeuralNet(self.layers)

        self.weights, self.biases = self.initialiseNeuralNet_dict(self.layers)

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
        crossEntropy = np.mean(CE)
        return crossEntropy

    def accuracy(self, Y_train, Y_pred, data_size): 
        Y_true_label = [] 
        Y_pred_label = [] 
        ctr = 0 
        for i in range(data_size): 
           Y_true_label.append(np.argmax(Y_train[:,i])) 
           Y_pred_label.append(np.argmax(Y_pred[:,i])) 
           if Y_true_label[i] == Y_pred_label[i]: 
               ctr +=1 
        accuracy = ctr/data_size 
        return accuracy, Y_true_label, Y_pred_label


    def Xavier_initialiser(self, size):
        in_dim = size[1]
        out_dim = size[0]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return np.random.normal( 0, xavier_stddev, size = (out_dim, in_dim))



    def random_initialiser(self, size):
        in_dim = size[1]
        out_dim = size[0]
        return np.random.normal( 0, 1, size = (out_dim, in_dim))/np.sqrt(2 / (in_dim))
     
     

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

    def initialiseNeuralNet_dict(self, layers):
        weights = {}
        biases = {}
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.Xavier_initialiser(size=[layers[l+1], layers[l]])
            b = np.zeros((layers[l + 1],1))
            weights[str(l+1)] = W
            biases[str(l+1)] = b
        return weights, biases  

    
    def forwardPropagate(self,X_train_batch, weights, biases, activation):
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
        H = {}
        A = {}
        H["0"] = X
        A["0"] = X
        for l in range(0, num_layers - 2):
            if l == 0:
                W = weights[str(l+1)]
                b = biases[str(l+1)]
                A[str(l+1)] = np.add(np.matmul(W,X), b)
                H[str(l+1)] = activation(A[str(l+1)])
            else:
                W = weights[str(l+1)]
                b = biases[str(l+1)]
                A[str(l+1)] = np.add(np.matmul(W,H[str(l)]), b)
                H[str(l+1)] = activation(A[str(l+1)])

        # Here the last layer is not activated as it is a regression problem
        W = weights[str(num_layers-1)]
        b = biases[str(num_layers-1)]
        A[str(num_layers - 1)] = np.add(np.matmul( W, H[str(num_layers - 2)]), b)
        #Y = softmax(A[-1])
        Y = softmax(A[str(num_layers - 1)] )
        H[str(num_layers - 1)] = Y
        return Y, H, A


    #Back prop trial code. Lot of cleanup to be done. May be a dictionary/ simple list of arrays should work.     
    def backPropagate(self, Y,H,A, Y_train_batch, der_activation, loss_function = "CrossEntropy"):
     
        gradients_weights = [] 
        gradients_biases = []
        num_layers = len(self.layers) 
        
        
        # Gradient with respect to the output layer is absolutely fine.
        if loss_function == "CrossEntropy":
            globals()["grad_a"+str(num_layers-1)] = - (Y_train_batch - Y)
        elif loss_function == "MeanSquaredError":
            globals()["grad_a"+str(num_layers-1)] = ( Y - Y_train_batch)
        
        for l in range(num_layers - 2, -1,-1): 

            globals()["grad_W"+str(l+1)] = np.outer(globals()["grad_a"+str(l+1)],H[str(l)]) 
            globals()["grad_b"+str(l+1)] = globals()["grad_a"+str(l+1)]            
            gradients_weights.append(globals()["grad_W"+str(l+1)]) 
            gradients_biases.append(globals()["grad_b"+str(l+1)]) 
            if l != 0:
                globals()["grad_h"+str(l)] = np.matmul(self.weights[str(l+1)].transpose(), globals()["grad_a"+str(l+1)]) 
                globals()["grad_a"+str(l)] = np.multiply(globals()["grad_h"+str(l)], der_activation(A[str(l)])) 
            elif l==0:

                globals()["grad_h"+str(l)] = np.matmul(self.weights[str(l+1)].transpose(), globals()["grad_a"+str(l+1)]) 
                globals()["grad_a"+str(l)] = np.multiply(globals()["grad_h"+str(l)], (A[str(l)])) 
        return gradients_weights, gradients_biases

    #Back prop trial code. Lot of cleanup to be done. May be a dictionary/ simple list of arrays should work.     
    def backPropagate_l2regular(self, Y,H,A, Y_train_batch, der_activation, loss_function = "CrossEntropy"):
        ALPHA = 0.01
        gradients_weights = [] 
        gradients_biases = []
        num_layers = len(self.layers) 
        
        
        # Gradient with respect to the output layer is absolutely fine.
        if loss_function == "CrossEntropy":
            globals()["grad_a"+str(num_layers-1)] = - (Y_train_batch - Y)
        elif loss_function == "MeanSquaredError":
            globals()["grad_a"+str(num_layers-1)] = ( Y - Y_train_batch)
        
        for l in range(num_layers - 2, -1,-1): 

            globals()["grad_W"+str(l+1)] = np.outer(globals()["grad_a"+str(l+1)],H[str(l)]) + ALPHA*self.weights[str(l+1)]
            globals()["grad_b"+str(l+1)] = globals()["grad_a"+str(l+1)]          
            gradients_weights.append(globals()["grad_W"+str(l+1)]) 
            gradients_biases.append(globals()["grad_b"+str(l+1)]) 
            if l != 0:
                globals()["grad_h"+str(l)] = np.matmul(self.weights[str(l+1)].transpose(), globals()["grad_a"+str(l+1)]) 
                globals()["grad_a"+str(l)] = np.multiply(globals()["grad_h"+str(l)], der_activation(A[str(l)])) 
            elif l==0:

                globals()["grad_h"+str(l)] = np.matmul(self.weights[str(l+1)].transpose(), globals()["grad_a"+str(l+1)]) 
                globals()["grad_a"+str(l)] = np.multiply(globals()["grad_h"+str(l)], (A[str(l)])) 
        return gradients_weights, gradients_biases

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
                self.weights = {str(i+1):(self.weights[str(i+1)] - learning_rate*deltaw[i]) for i in range(len(self.weights))}
                self.biases = {str(i+1):(self.biases[str(i+1)] - learning_rate*deltab[i]) for i in range(len(self.biases))}
            
            print(learning_rate, epoch, np.mean(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))

        return self.weights, self.biases, loss, Y_pred
        
    def batchGradientDescent(self, epochs,length_dataset, batch_size, learning_rate):
        loss = []
        num_layers = len(self.layers)
        num_points_seen = 0
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
                num_points_seen +=1
                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
                if int(num_points_seen) % batch_size == 0:
                    #print(num_points_seen)
                    self.weights = {str(i+1):(self.weights[str(i+1)] - learning_rate*deltaw[i]/batch_size) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):(self.biases[str(i+1)] - learning_rate*deltab[i]) for i in range(len(self.biases))}
                    #resetting gradient updates
                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            print(learning_rate, epoch, np.mean(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
            #weights, biases = self.weights, self.biases
        return self.weights, self.biases, loss, Y_pred



    def momentumGradientDescent(self, epochs,length_dataset, learning_rate):
        GAMMA = 0.9

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
            
            v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/length_dataset for i in range(num_layers - 1)]
            v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/length_dataset for i in range(num_layers - 1)]
            
            self.weights = {str(i+1):(self.weights[str(i+1)] - v_w[i]) for i in range(len(self.weights))}
            self.biases = {str(i+1):(self.biases[str(i+1)] - v_b[i]) for i in range(len(self.biases))}
            prev_v_w = v_w
            prev_v_b = v_b

            print(learning_rate, epoch, np.mean(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        return self.weights, self.biases, loss, Y_pred


    def batchMomentumGradientDescent(self, epochs,length_dataset, batch_size, learning_rate):
        GAMMA = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        num_points_seen = 0
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
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]
                    
                    self.weights = {str(i+1) : (self.weights[str(i+1)] - v_w[i]) for i in range(len(self.weights))}
                    self.biases = {str(i+1): (self.biases[str(i+1)] - v_b[i]) for i in range(len(self.biases))}

                    prev_v_w = v_w
                    prev_v_b = v_b

                    #resetting gradient updates
                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            print(learning_rate, epoch, np.mean(CE))
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        return self.weights, self.biases, loss, Y_pred


    def nesterovGradientDescent(self,epochs,length_dataset, learning_rate):
        GAMMA = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        num_points_seen = 0
        for epoch in range(epochs):
            CE = []
            Y_pred = []  
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(self.layers)-1)]

            for i in range(length_dataset):
                winter = {str(i+1) : self.weights[str(i+1)] - v_w[i] for i in range(0, len(self.layers)-1)}
                binter = {str(i+1) : self.biases[str(i+1)] - v_b[i] for i in range(0, len(self.layers)-1)}

                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), winter, binter, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
                            
            v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/length_dataset for i in range(num_layers - 1)]
            v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/length_dataset for i in range(num_layers - 1)]
        
            self.weights = {str(i+1) :self.weights[str(i+1)] - v_w[i] for i in range(len(self.weights))}
            self.biases = {str(i+1) :self.biases[str(i+1)] - v_b[i] for i in range(len(self.biases))}
            prev_v_w = v_w
            prev_v_b = v_b
    
            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred
 
 
    def stochasticNesterovGradientDescent(self,epochs,length_dataset, learning_rate):
        GAMMA = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        for epoch in range(epochs):
            CE = []
            Y_pred = []  
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(self.layers)-1)]
                        
            for i in range(length_dataset):
                winter = {str(i+1) : self.weights[str(i+1)] - v_w[i] for i in range(0, len(self.layers)-1)}
                binter = {str(i+1) : self.biases[str(i+1)] - v_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), winter, binter, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))
                            
                v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
        
                self.weights = {str(i+1):self.weights[str(i+1)] - v_w[i] for i in range(len(self.weights))} 
                self.biases = {str(i+1):self.biases[str(i+1)] - v_b[i] for i in range(len(self.biases))}
                prev_v_w = v_w
                prev_v_b = v_b
    
            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred
    

    def batchNesterovGradientDescent(self,epochs,length_dataset, batch_size,learning_rate):
        GAMMA = 0.9

        loss = []
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        num_points_seen = 0
        for epoch in range(epochs):
            CE = []
            Y_pred = []  
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(self.layers)-1)]

            for i in range(length_dataset):
                winter = {str(i+1) : self.weights[str(i+1)] - v_w[i] for i in range(0, len(self.layers)-1)}
                binter = {str(i+1) : self.biases[str(i+1)] - v_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), winter, binter, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))

                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:                            

                    v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i]/batch_size for i in range(num_layers - 1)]
                    v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i]/batch_size for i in range(num_layers - 1)]
        
                    self.weights ={str(i+1):self.weights[str(i+1)]  - v_w[i] for i in range(len(self.weights))}
                    self.biases = {str(i+1):self.biases[str(i+1)]  - v_b[i] for i in range(len(self.biases))}
                    prev_v_w = v_w
                    prev_v_b = v_b

                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

    
            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred
    
    def rmsProp(self, epochs,length_dataset, learning_rate):

        loss = []
        num_layers = len(self.layers)
        EPS, BETA = 1e-8, 0.9
        
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
            
            v_w = [BETA*v_w[i] + (1-BETA)*(deltaw[i])**2 for i in range(num_layers - 1)]
            v_b = [BETA*v_b[i] + (1-BETA)*(deltab[i])**2 for i in range(num_layers - 1)]

            self.weights = {str(i+1):self.weights[str(i+1)]  - deltaw[i]*(learning_rate/(np.sqrt(v_w[i])+EPS)) for i in range(len(self.weights))} 
            self.biases = {str(i+1):self.biases[str(i+1)]  - deltab[i]*(learning_rate/(np.sqrt(v_b[i])+EPS)) for i in range(len(self.biases))}
    
            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred
    
    def batchRmsProp(self, epochs,length_dataset, batch_size, learning_rate):

        loss = []
        num_layers = len(self.layers)
        EPS, BETA = 1e-8, 0.9
        
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        num_points_seen = 0        
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
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:
                
                    v_w = [BETA*v_w[i] + (1-BETA)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA*v_b[i] + (1-BETA)*(deltab[i])**2 for i in range(num_layers - 1)]

                    self.weights = {str(i+1):self.weights[str(i+1)]  - deltaw[i]*(learning_rate/np.sqrt(v_w[i]+EPS)) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):self.biases[str(i+1)]  - deltab[i]*(learning_rate/np.sqrt(v_b[i]+EPS)) for i in range(len(self.biases))}

                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
    
            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred  

    def adam(self,epochs,length_dataset,learning_rate):

        loss = []
        num_layers = len(self.layers)
        
        EPS, BETA1, BETA2 = 1e-8, 0.9, 0.99
        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        

        m_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)] 
        
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
                
            m_w = [BETA1*m_w[i] + (1-BETA1)*(deltaw[i]) for i in range(num_layers - 1)]
            m_b = [BETA1*m_b[i] + (1-BETA1)*(deltab[i]) for i in range(num_layers - 1)]
            v_w = [BETA2*v_w[i] + (1-BETA2)*(deltaw[i]**2) for i in range(num_layers - 1)]
            v_b = [BETA2*v_b[i] + (1-BETA2)*(deltab[i])**2 for i in range(num_layers - 1)]
            
            m_w_hat = [m_w[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]
            m_b_hat = [m_b[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]            
            v_w_hat = [v_w[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
            v_b_hat = [v_b[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
            
            self.weights ={str(i+1):self.weights[str(i+1)]  - m_w_hat[i]*(learning_rate/(np.sqrt(v_w_hat[i])+EPS)) for i in range(len(self.weights))}
            self.biases = {str(i+1):self.biases[str(i+1)]  - m_b_hat[i]*(learning_rate/(np.sqrt(v_b_hat[i])+EPS)) for i in range(len(self.biases))}

            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred



    def adamMiniBatch(self, epochs,length_dataset, batch_size, learning_rate):
        
        loss = []
        num_layers = len(self.layers)
        EPS, BETA1, BETA2 = 1e-8, 0.9, 0.99
        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        
        m_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]   
        num_points_seen = 0 
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

                num_points_seen += 1
                ctr = 0
                if int(num_points_seen) % batch_size == 0:
                    ctr += 1
                    m_w = [BETA1*m_w[i] + (1-BETA1)*deltaw[i] for i in range(num_layers - 1)]
                    m_b = [BETA1*m_b[i] + (1-BETA1)*deltab[i] for i in range(num_layers - 1)]
                    v_w = [BETA2*v_w[i] + (1-BETA2)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA2*v_b[i] + (1-BETA2)*(deltab[i])**2 for i in range(num_layers - 1)]
                    
                    m_w_hat = [m_w[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]
                    m_b_hat = [m_b[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]            
                    v_w_hat = [v_w[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                    v_b_hat = [v_b[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                    self.weights = {str(i+1):self.weights[str(i+1)] - (learning_rate/np.sqrt(v_w[i]+EPS))*m_w_hat[i] for i in range(len(self.weights))} 
                    self.biases = {str(i+1):self.biases[str(i+1)] - (learning_rate/np.sqrt(v_b[i]+EPS))*m_b_hat[i] for i in range(len(self.biases))}

                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred


    def nadam(self, epochs,length_dataset, learning_rate):
        
        loss = []
        num_layers = len(self.layers)
        
        GAMMA, EPS, BETA1, BETA2 = 0.9, 1e-8, 0.9, 0.99

        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        

        m_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)] 

        num_points_seen = 0 
        
        for epoch in range(epochs):

            CE = []
            Y_pred = []

            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(length_dataset):

#                winter = {str(i+1):self.weights[str(i+1)] - m_w[i] for i in range(0, len(self.layers)-1)}  
#                binter = {str(i+1):self.biases[str(i+1)] - m_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)

                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))   
                num_points_seen += 1
                
                #if num_points_seen % batch_size == 0
            m_w = [BETA1*m_w[i] + (1-BETA1)*deltaw[i] for i in range(num_layers - 1)]
            m_b = [BETA1*m_b[i] + (1-BETA1)*deltab[i] for i in range(num_layers - 1)]
            v_w = [BETA2*v_w[i] + (1-BETA2)*deltaw[i]**2 for i in range(num_layers - 1)]
            v_b = [BETA2*v_b[i] + (1-BETA2)*deltab[i]**2 for i in range(num_layers - 1)]
            
            m_w_hat = [m_w[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]
            m_b_hat = [m_b[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]            
            v_w_hat = [v_w[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
            v_b_hat = [v_b[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
            
            self.weights = {str(i+1):self.weights[str(i+1)] - (learning_rate/(np.sqrt(v_w_hat[i])+EPS))*(BETA1*m_w_hat[i]+ (1-BETA1)*deltaw[i]) for i in range(len(self.weights))} 
            self.biases = {str(i+1):self.biases[str(i+1)] - (learning_rate/(np.sqrt(v_b_hat[i])+EPS))*(BETA1*m_b_hat[i] + (1-BETA1)*deltab[i]) for i in range(len(self.biases))}

            #self.weights = {str(i+1):self.weights[str(i+1)] - m_w[i]*(learning_rate/(np.sqrt(v_w[i])+EPS)) for i in range(len(self.weights))} 
            #self.biases = {str(i+1):self.biases[str(i+1)] - m_b[i]*(learning_rate/(np.sqrt(v_b[i])+EPS)) for i in range(len(self.biases))}
             
            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred
    
    def nadamMinibatch(self, epochs,length_dataset, batch_size, learning_rate):
        
        loss = []
        num_layers = len(self.layers)
        
        GAMMA, EPS, BETA1, BETA2 = 0.9, 1e-8, 0.9, 0.99

        m_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]        

        m_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        m_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        v_w_hat = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b_hat = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)] 

        num_points_seen = 0 
        
        for epoch in range(epochs):

            CE = []
            Y_pred = []

            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(length_dataset):

#                winter = {str(i+1):self.weights[str(i+1)] - m_w[i] for i in range(0, len(self.layers)-1)}  
#                binter = {str(i+1):self.biases[str(i+1)] - m_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(784,1), self.weights, self.biases, self.activation) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(10,1), self.der_activation)

                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                Y_pred.append(Y.reshape(10,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(10,1), Y))   
                num_points_seen += 1
                
                if num_points_seen % batch_size == 0:
                    m_w = [BETA1*m_w[i] + (1-BETA1)*deltaw[i] for i in range(num_layers - 1)]
                    m_b = [BETA1*m_b[i] + (1-BETA1)*deltab[i] for i in range(num_layers - 1)]
                    v_w = [BETA2*v_w[i] + (1-BETA2)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA2*v_b[i] + (1-BETA2)*(deltab[i])**2 for i in range(num_layers - 1)]
                    
                    m_w_hat = [m_w[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]
                    m_b_hat = [m_b[i]/(1-BETA1**(epoch+1)) for i in range(num_layers - 1)]            
                    v_w_hat = [v_w[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                    v_b_hat = [v_b[i]/(1-BETA2**(epoch+1)) for i in range(num_layers - 1)]
                    
                    self.weights = {str(i+1):self.weights[str(i+1)] - (learning_rate/(np.sqrt(v_w_hat[i])+EPS))*(BETA1*m_w_hat[i]+ (1-BETA1)*deltaw[i]) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):self.biases[str(i+1)] - (learning_rate/(np.sqrt(v_b_hat[i])+EPS))*(BETA1*m_b_hat[i] + (1-BETA1)*deltab[i]) for i in range(len(self.biases))}

                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]


            #self.weights = {str(i+1):self.weights[str(i+1)] - m_w[i]*(learning_rate/(np.sqrt(v_w[i])+EPS)) for i in range(len(self.weights))} 
            #self.biases = {str(i+1):self.biases[str(i+1)] - m_b[i]*(learning_rate/(np.sqrt(v_b[i])+EPS)) for i in range(len(self.biases))}
             
            print(learning_rate, epoch, np.mean(CE))
        
            Y_pred = np.array(Y_pred).transpose()
            loss.append(np.mean(CE))
        
        return self.weights, self.biases, loss, Y_pred  
    
    
