import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import wandb

from feedForwardNeuralNet import FeedForwardNeuralNetwork

((trainIn,trainOut), (testIn,testOut)) = mnist.load_data()

N_train_full = trainOut.shape[0]
N_train = int(0.9*N_train_full)
N_validation = int(0.1 * trainOut.shape[0])
N_test = testOut.shape[0]


idx  = np.random.choice(trainOut.shape[0], N_train_full, replace=False)
idx2 = np.random.choice(testOut.shape[0], N_test, replace=False)

trainInFull = trainIn[idx, :]
trainOutFull = trainOut[idx]

trainIn = trainInFull[:N_train,:]
trainOut = trainOutFull[:N_train]

validIn = trainInFull[N_train:, :]
validOut = trainOutFull[N_train:]    

testIn = testIn[idx2, :]
testOut = testOut[idx2]


sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
  'early_terminate': {
        'type':'hyperband',
        'min_iter': [3],
        's': [2]
  },
  "parameters": {
        
        "activation":{
            "values": ["RELU", "SIGMOID"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        },
        "initializer": {
            "values": ["XAVIER", "HE"]
        }
        
        
        
    }
}

sweep_id = wandb.sweep(sweep_config,project='CS6910-DeepLearningFundamentals-Assignment1', entity='rahulsundar')



def train():    

     
    config_defaults = dict(
            max_epochs=2,
            num_hidden_layers=3,
            num_hidden_neurons=128,
            weight_decay=0,
            learning_rate=1e-3,
            optimizer="NADAM",
            batch_size=16,
            activation="SIGMOID",
            initializer="XAVIER",
            loss="CROSS",
        ) 
        
    wandb.init(config = config_defaults)
    

    wandb.run.name = "MNIST_hl_" + str(wandb.config.num_hidden_layers) + "_hn_" + str(wandb.config.num_hidden_neurons) + "_opt_" + wandb.config.optimizer + "_act_" + wandb.config.activation + "_bs_"+str(wandb.config.batch_size) 
    
    CONFIG = wandb.config
  

    FFNN = FeedForwardNeuralNetwork(
        num_hidden_layers=CONFIG.num_hidden_layers,
        num_hidden_neurons=CONFIG.num_hidden_neurons,
        X_train_raw=trainIn,
        Y_train_raw=trainOut,
        N_train = N_train,
        X_val_raw = validIn,
        Y_val_raw = validOut,
        N_val = N_validation,
        X_test_raw = testIn,
        Y_test_raw = testOut,
        N_test = N_test,
        optimizer = CONFIG.optimizer,
        batch_size = CONFIG.batch_size,
        weight_decay = CONFIG.weight_decay,
        learning_rate = CONFIG.learning_rate,
        max_epochs = CONFIG.max_epochs,
        activation = CONFIG.activation,
        initializer = CONFIG.initializer,
        loss = CONFIG.loss
        )



    #training_loss, trainingaccuracy, validationaccuracy, Y_pred_train = FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, FFNN.batch_size, FFNN.learning_rate)
    
    FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, FFNN.batch_size, FFNN.learning_rate)
    wandb.finish()
    #Y_pred_test =  FFNN.predict(FFNN.X_test, FFNN.N_test)
    #train_accuracy, Y_true_train, Y_pred_train = FFNN.accuracy(FFNN.Y_train, Y_pred_train, FFNN.N_train)
    #test_accuracy, Y_true_test, Y_pred_test = FFNN.accuracy(FFNN.Y_test, Y_pred_test, FFNN.N_test)
    #train_pred = (train_accuracy, Y_true_train, Y_pred_train)
    #test_pred = (test_accuracy, Y_true_test, Y_pred_test)

    #return train_pred, test_pred
    
    
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    wandb.agent(sweep_id, train, count = 5)
    '''
    config1 = dict(
            max_epochs=10,
            num_hidden_layers=3,
            num_hidden_neurons=128,
            weight_decay=0,
            learning_rate=1e-3,
            optimizer="NADAM",
            batch_size=16,
            activation="SIGMOID",
            initializer="XAVIER",
            loss="CROSS",
        )
        
    config2 = dict(
            max_epochs=10,
            num_hidden_layers=3,
            num_hidden_neurons=128,
            weight_decay=0,
            learning_rate=1e-3,
            optimizer="NADAM",
            batch_size=32,
            activation="SIGMOID",
            initializer="XAVIER",
            loss="CROSS",
        )
        
    config3 = dict(
            max_epochs=10,
            num_hidden_layers=3,
            num_hidden_neurons=64,
            weight_decay=0,
            learning_rate=1e-3,
            optimizer="NADAM",
            batch_size=32,
            activation="SIGMOID",
            initializer="XAVIER",
            loss="CROSS",
        )
    
    '''
    #config_dict = {"1":config1, "2":config2, "3":config3}
    

    #Results = {}
    
    
    #for i in range(3):
    #    Results["train_pred"+str(i+1)], Results["test_pred"+str(i+1)] = train(config_dict[str(i+1)])
        
        
        

