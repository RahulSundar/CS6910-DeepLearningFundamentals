import numpy as np
import scipy as sp
import wandb
import time

from utility.activation import *


class FeedForwardNeuralNetwork:
    def __init__(
        self, 
        num_hidden_layers, 
        num_hidden_neurons, 
        X_train_raw, 
        Y_train_raw,  
        N_train, 
        X_val_raw, 
        Y_val_raw, 
        N_val,
        X_test_raw, 
        Y_test_raw, 
        N_test,        
        optimizer,
        batch_size,
        weight_decay,
        learning_rate,
        max_epochs,
        activation,
        initializer,
        loss

    ):

        """
        Here, we initialize the FeedForwardNeuralNetwork class with the number of hidden layers, number of hidden neurons, raw training data. 
        """
        
        self.num_classes = np.max(Y_train_raw) + 1  # NUM_CLASSES
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_neurons = num_hidden_neurons
        self.output_layer_size = self.num_classes
        self.img_height = X_train_raw.shape[1]
        self.img_width = X_train_raw.shape[2]
        self.img_flattened_size = self.img_height * self.img_width

        # self.layers = layers
        self.layers = (
            [self.img_flattened_size]
            + num_hidden_layers * [num_hidden_neurons]
            + [self.output_layer_size]
        )

        self.N_train = N_train
        self.N_val = N_val
        self.N_test = N_test
        


        self.X_train = np.transpose(
            X_train_raw.reshape(
                X_train_raw.shape[0], X_train_raw.shape[1] * X_train_raw.shape[2]
            )
        )  # [IMG_HEIGHT*IMG_WIDTH X NTRAIN]
        self.X_test = np.transpose(
            X_test_raw.reshape(
                X_test_raw.shape[0], X_test_raw.shape[1] * X_test_raw.shape[2]
            )
        )  # [IMG_HEIGHT*IMG_WIDTH X NTRAIN]
        self.X_val = np.transpose(
            X_val_raw.reshape(
                X_val_raw.shape[0], X_val_raw.shape[1] * X_val_raw.shape[2]
            )
        )  # [IMG_HEIGHT*IMG_WIDTH X NTRAIN]


        self.X_train = self.X_train / 255
        self.X_test = self.X_test / 255
        self.X_val = self.X_val / 255
        
        self.Y_train = self.oneHotEncode(Y_train_raw)  # [NUM_CLASSES X NTRAIN]
        self.Y_val = self.oneHotEncode(Y_val_raw)
        self.Y_test = self.oneHotEncode(Y_test_raw)
        #self.Y_shape = self.Y_train.shape




        # self.weights, self.biases = self.initializeNeuralNet(self.layers)



        self.Activations_dict = {"SIGMOID": sigmoid, "TANH": tanh, "RELU": relu}
        self.DerActivation_dict = {
            "SIGMOID": der_sigmoid,
            "TANH": der_tanh,
            "RELU": der_relu,
        }

        self.Initializer_dict = {
            "XAVIER": self.Xavier_initializer,
            "RANDOM": self.random_initializer,
            "HE": self.He_initializer
        }

        self.Optimizer_dict = {
            "SGD": self.sgdMiniBatch,
            "MGD": self.mgd,
            "NAG": self.nag,
            "RMSPROP": self.rmsProp,
            "ADAM": self.adam,
            "NADAM": self.nadam,
        }
        
        self.activation = self.Activations_dict[activation]
        self.der_activation = self.DerActivation_dict[activation]
        self.optimizer = self.Optimizer_dict[optimizer]
        self.initializer = self.Initializer_dict[initializer]
        self.loss_function = loss
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.weights, self.biases = self.initializeNeuralNet(self.layers)


        
        
    # helper functions
    def oneHotEncode(self, Y_train_raw):
        Ydata = np.zeros((self.num_classes, Y_train_raw.shape[0]))
        for i in range(Y_train_raw.shape[0]):
            value = Y_train_raw[i]
            Ydata[int(value)][i] = 1.0
        return Ydata

    # Loss functions
    def meanSquaredErrorLoss(self, Y_true, Y_pred):
        MSE = np.mean((Y_true - Y_pred) ** 2)
        return MSE

    def crossEntropyLoss(self, Y_true, Y_pred):
        CE = [-Y_true[i] * np.log(Y_pred[i]) for i in range(len(Y_pred))]
        crossEntropy = np.mean(CE)
        return crossEntropy

    def L2RegularisationLoss(self, weight_decay):
        ALPHA = weight_decay
        return ALPHA * np.sum(
            [
                np.linalg.norm(self.weights[str(i + 1)]) ** 2
                for i in range(len(self.weights))
            ]
        )


    def accuracy(self, Y_true, Y_pred, data_size):
        Y_true_label = []
        Y_pred_label = []
        ctr = 0
        for i in range(data_size):
            Y_true_label.append(np.argmax(Y_true[:, i]))
            Y_pred_label.append(np.argmax(Y_pred[:, i]))
            if Y_true_label[i] == Y_pred_label[i]:
                ctr += 1
        accuracy = ctr / data_size
        return accuracy, Y_true_label, Y_pred_label

    def Xavier_initializer(self, size):
        in_dim = size[1]
        out_dim = size[0]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return np.random.normal(0, xavier_stddev, size=(out_dim, in_dim))

    def random_initializer(self, size):
        in_dim = size[1]
        out_dim = size[0]
        return np.random.normal(0, 1, size=(out_dim, in_dim))


    def He_initializer(self,size):
        in_dim = size[1]
        out_dim = size[0]
        He_stddev = np.sqrt(2 / (in_dim))
        return np.random.normal(0, 1, size=(out_dim, in_dim)) * He_stddev


    def initializeNeuralNet(self, layers):
        weights = {}
        biases = {}
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.initializer(size=[layers[l + 1], layers[l]])
            b = np.zeros((layers[l + 1], 1))
            weights[str(l + 1)] = W
            biases[str(l + 1)] = b
        return weights, biases

    def forwardPropagate(self, X_train_batch, weights, biases):
        """
        Returns the neural network given input data, weights, biases.
        Arguments:
                 : X - input matrix
                 : Weights  - Weights matrix
                 : biases - Bias vectors 
        """
        # Number of layers = length of weight matrix + 1
        num_layers = len(weights) + 1
        # A - Preactivations
        # H - Activations
        X = X_train_batch
        H = {}
        A = {}
        H["0"] = X
        A["0"] = X
        for l in range(0, num_layers - 2):
            if l == 0:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, X), b)
                H[str(l + 1)] = self.activation(A[str(l + 1)])
            else:
                W = weights[str(l + 1)]
                b = biases[str(l + 1)]
                A[str(l + 1)] = np.add(np.matmul(W, H[str(l)]), b)
                H[str(l + 1)] = self.activation(A[str(l + 1)])

        # Here the last layer is not activated as it is a regression problem
        W = weights[str(num_layers - 1)]
        b = biases[str(num_layers - 1)]
        A[str(num_layers - 1)] = np.add(np.matmul(W, H[str(num_layers - 2)]), b)
        # Y = softmax(A[-1])
        Y = softmax(A[str(num_layers - 1)])
        H[str(num_layers - 1)] = Y
        return Y, H, A

    def backPropagate(
        self, Y, H, A, Y_train_batch, weight_decay=0
    ):

        ALPHA = weight_decay
        gradients_weights = []
        gradients_biases = []
        num_layers = len(self.layers)

        # Gradient with respect to the output layer is absolutely fine.
        if self.loss_function == "CROSS":
            globals()["grad_a" + str(num_layers - 1)] = -(Y_train_batch - Y)
        elif self.loss_function == "MSE":
            globals()["grad_a" + str(num_layers - 1)] = np.multiply(
                2 * (Y - Y_train_batch), np.multiply(Y, (1 - Y))
            )

        for l in range(num_layers - 2, -1, -1):

            if ALPHA != 0:
                globals()["grad_W" + str(l + 1)] = (
                    np.outer(globals()["grad_a" + str(l + 1)], H[str(l)])
                    + ALPHA * self.weights[str(l + 1)]
                )
            elif ALPHA == 0:
                globals()["grad_W" + str(l + 1)] = np.outer(
                    globals()["grad_a" + str(l + 1)], H[str(l)]
                )
            globals()["grad_b" + str(l + 1)] = globals()["grad_a" + str(l + 1)]
            gradients_weights.append(globals()["grad_W" + str(l + 1)])
            gradients_biases.append(globals()["grad_b" + str(l + 1)])
            if l != 0:
                globals()["grad_h" + str(l)] = np.matmul(
                    self.weights[str(l + 1)].transpose(),
                    globals()["grad_a" + str(l + 1)],
                )
                globals()["grad_a" + str(l)] = np.multiply(
                    globals()["grad_h" + str(l)], self.der_activation(A[str(l)])
                )
            elif l == 0:

                globals()["grad_h" + str(l)] = np.matmul(
                    self.weights[str(l + 1)].transpose(),
                    globals()["grad_a" + str(l + 1)],
                )
                globals()["grad_a" + str(l)] = np.multiply(
                    globals()["grad_h" + str(l)], (A[str(l)])
                )
        return gradients_weights, gradients_biases


    def predict(self,X,length_dataset):
        Y_pred = []        
        for i in range(length_dataset):

            Y, H, A = self.forwardPropagate(
                X[:, i].reshape(self.img_flattened_size, 1),
                self.weights,
                self.biases,
            )

            Y_pred.append(Y.reshape(self.num_classes,))
        Y_pred = np.array(Y_pred).transpose()
        return Y_pred

    def sgd(self, epochs, length_dataset, learning_rate, weight_decay=0):
        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]

        for epoch in range(epochs):
            start_time = time.time()
            # perm = np.random.permutation(N)
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            CE = []
            #Y_pred = []
            deltaw = [
                np.zeros((self.layers[l + 1], self.layers[l]))
                for l in range(0, len(self.layers) - 1)
            ]
            deltab = [
                np.zeros((self.layers[l + 1], 1))
                for l in range(0, len(self.layers) - 1)
            ]

            for i in range(length_dataset):

                Y, H, A = self.forwardPropagate(
                    X_train[:, i].reshape(self.img_flattened_size, 1),
                    self.weights,
                    self.biases,
                )
                grad_weights, grad_biases = self.backPropagate(
                    Y, H, A, Y_train[:, i].reshape(self.num_classes, 1)
                )
                deltaw = [
                    grad_weights[num_layers - 2 - i] for i in range(num_layers - 1)
                ]
                deltab = [
                    grad_biases[num_layers - 2 - i] for i in range(num_layers - 1)
                ]

                #Y_pred.append(Y.reshape(self.num_classes,))

                CE.append(
                    self.crossEntropyLoss(
                        self.Y_train[:, i].reshape(self.num_classes, 1), Y
                    )
                    + self.L2RegularisationLoss(weight_decay)
                )

                # print(num_points_seen)
                self.weights = {
                    str(i + 1): (self.weights[str(i + 1)] - learning_rate * deltaw[i])
                    for i in range(len(self.weights))
                }
                self.biases = {
                    str(i + 1): (self.biases[str(i + 1)] - learning_rate * deltab[i])
                    for i in range(len(self.biases))
                }

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])
            
            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch, })
        # data = [[epoch, loss[epoch]] for epoch in range(epochs)]
        # table = wandb.Table(data=data, columns = ["Epoch", "Loss"])
        # wandb.log({'loss':wandb.plot.line(table, "Epoch", "Loss", title="Loss vs Epoch Line Plot")})
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred


      
    def sgdMiniBatch(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        

        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        num_points_seen = 0


        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)
            
            CE = []
            #Y_pred = []
            
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(length_dataset):
                
                Y,H,A = self.forwardPropagate(X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,Y_train[:,i].reshape(self.num_classes,1))
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
                
                #Y_pred.append(Y.reshape(self.num_classes,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(self.num_classes,1), Y) + self.L2RegularisationLoss(weight_decay))
                
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:
                    
                    
                    self.weights = {str(i+1):(self.weights[str(i+1)] - learning_rate*deltaw[i]/batch_size) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):(self.biases[str(i+1)] - learning_rate*deltab[i]) for i in range(len(self.biases))}
                    
                    #resetting gradient updates
                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
            
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred



    def mgd(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        GAMMA = 0.9

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        

        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            CE = []
            #Y_pred = []
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            

            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(self.num_classes,1), Y) + self.L2RegularisationLoss(weight_decay))
                
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

            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })


        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred


 
 
    def stochasticNag(self,epochs,length_dataset, learning_rate, weight_decay = 0):
        GAMMA = 0.9

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        

        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            CE = []
            #Y_pred = []  
            
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(self.layers)-1)]
                        
            for i in range(length_dataset):
                winter = {str(i+1) : self.weights[str(i+1)] - v_w[i] for i in range(0, len(self.layers)-1)}
                binter = {str(i+1) : self.biases[str(i+1)] - v_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), winter, binter) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))
                
                deltaw = [grad_weights[num_layers-2 - i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(self.num_classes,1), Y) + self.L2RegularisationLoss(weight_decay))
                            
                v_w = [GAMMA*prev_v_w[i] + learning_rate*deltaw[i] for i in range(num_layers - 1)]
                v_b = [GAMMA*prev_v_b[i] + learning_rate*deltab[i] for i in range(num_layers - 1)]
        
                self.weights = {str(i+1):self.weights[str(i+1)] - v_w[i] for i in range(len(self.weights))} 
                self.biases = {str(i+1):self.biases[str(i+1)] - v_b[i] for i in range(len(self.biases))}
                
                prev_v_w = v_w
                prev_v_b = v_b
    
            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred
    

    def nag(self,epochs,length_dataset, batch_size,learning_rate, weight_decay = 0):
        GAMMA = 0.9

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        


        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        
        prev_v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        prev_v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        num_points_seen = 0
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            CE = []
            #Y_pred = []  
            
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
            v_w = [GAMMA*prev_v_w[i] for i in range(0, len(self.layers)-1)]  
            v_b = [GAMMA*prev_v_b[i] for i in range(0, len(self.layers)-1)]

            for i in range(length_dataset):
                winter = {str(i+1) : self.weights[str(i+1)] - v_w[i] for i in range(0, len(self.layers)-1)}
                binter = {str(i+1) : self.biases[str(i+1)] - v_b[i] for i in range(0, len(self.layers)-1)}
                
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), winter, binter) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(self.num_classes,1), Y) + self.L2RegularisationLoss(weight_decay))

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

    
            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )

            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred
    

    
    def rmsProp(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):


        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        

        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
        
        num_layers = len(self.layers)
        EPS, BETA = 1e-8, 0.9
        
        v_w = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
        v_b = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
        
        num_points_seen = 0        
        for epoch in range(epochs):
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)


            CE = []
            #Y_pred = []
                        
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(length_dataset):
            
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))
            
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]
                
                #Y_pred.append(Y.reshape(self.num_classes,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(self.num_classes,1), Y) + self.L2RegularisationLoss(weight_decay))            
                num_points_seen +=1
                
                if int(num_points_seen) % batch_size == 0:
                
                    v_w = [BETA*v_w[i] + (1-BETA)*(deltaw[i])**2 for i in range(num_layers - 1)]
                    v_b = [BETA*v_b[i] + (1-BETA)*(deltab[i])**2 for i in range(num_layers - 1)]

                    self.weights = {str(i+1):self.weights[str(i+1)]  - deltaw[i]*(learning_rate/np.sqrt(v_w[i]+EPS)) for i in range(len(self.weights))} 
                    self.biases = {str(i+1):self.biases[str(i+1)]  - deltab[i]*(learning_rate/np.sqrt(v_b[i]+EPS)) for i in range(len(self.biases))}

                    deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
                    deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
    
            
            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred  



    def adam(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):
        
        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        

        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
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
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)


            CE = []
            #Y_pred = []
            
            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]
            
           
            for i in range(length_dataset):
                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))
                
                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(self.num_classes,1), Y) + self.L2RegularisationLoss(weight_decay))                 

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


            elapsed = time.time() - start_time
            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )
                    
            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
        
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred


    
    def nadam(self, epochs,length_dataset, batch_size, learning_rate, weight_decay = 0):

        X_train = self.X_train[:, :length_dataset]
        Y_train = self.Y_train[:, :length_dataset]        

        
        trainingloss = []
        trainingaccuracy = []
        validationaccuracy = []
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
            start_time = time.time()
            idx = np.random.shuffle(np.arange(length_dataset))
            X_train = X_train[:, idx].reshape(self.img_flattened_size, length_dataset)
            Y_train = Y_train[:, idx].reshape(self.num_classes, length_dataset)

            CE = []
            #Y_pred = []

            deltaw = [np.zeros((self.layers[l+1], self.layers[l])) for l in range(0, len(self.layers)-1)]
            deltab = [np.zeros((self.layers[l+1], 1)) for l in range(0, len(self.layers)-1)]

            for i in range(length_dataset):

                Y,H,A = self.forwardPropagate(self.X_train[:,i].reshape(self.img_flattened_size,1), self.weights, self.biases) 
                grad_weights, grad_biases = self.backPropagate(Y,H,A,self.Y_train[:,i].reshape(self.num_classes,1))

                deltaw = [grad_weights[num_layers-2 - i] + deltaw[i] for i in range(num_layers - 1)]
                deltab = [grad_biases[num_layers-2 - i] + deltab[i] for i in range(num_layers - 1)]

                #Y_pred.append(Y.reshape(self.num_classes,))
                CE.append(self.crossEntropyLoss(self.Y_train[:,i].reshape(self.num_classes,1), Y) + self.L2RegularisationLoss(weight_decay))   
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
             
            elapsed = time.time() - start_time

            #Y_pred = np.array(Y_pred).transpose()
            Y_pred = self.predict(self.X_train, self.N_train)
            trainingloss.append(np.mean(CE))
            trainingaccuracy.append(self.accuracy(Y_train, Y_pred, length_dataset)[0])
            validationaccuracy.append(self.accuracy(self.Y_val, self.predict(self.X_val, self.N_val), self.N_val)[0])

            print(
                        "Epoch: %d, Loss: %.3e, Training accuracy:%.2f, Validation Accuracy: %.2f, Time: %.2f, Learning Rate: %.3e"
                        % (
                            epoch,
                            trainingloss[epoch],
                            trainingaccuracy[epoch],
                            validationaccuracy[epoch],
                            elapsed,
                            self.learning_rate,
                        )
                    )
            wandb.log({'loss':np.mean(CE), 'trainingaccuracy':trainingaccuracy[epoch], 'validationaccuracy':validationaccuracy[epoch],'epoch':epoch })
            
        return trainingloss, trainingaccuracy, validationaccuracy, Y_pred  

