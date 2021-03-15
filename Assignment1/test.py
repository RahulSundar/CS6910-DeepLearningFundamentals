import wandb

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


from feedForwardNeuralNet import FeedForwardNeuralNetwork

(trainIn, trainOut), (testIn, testOut) = fashion_mnist.load_data()

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


def trainandtest():    
    config_best = dict(
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
        
    wandb.init(project='CS6910-DeepLearningFundamentals-Assignment1', entity='rahulsundar', config = config_best)
    

    wandb.run.name = "hl_" + str(wandb.config.num_hidden_layers) + "_hn_" + str(wandb.config.num_hidden_neurons) + "_opt_" + wandb.config.optimizer + "_act_" + wandb.config.activation + "_lr_" + str(wandb.config.learning_rate) + "_bs_"+str(wandb.config.batch_size) + "_init_" + wandb.config.initializer + "_ep_"+ str(wandb.config.max_epochs)+ "_l2_" + str(wandb.config.weight_decay) 
    CONFIG = wandb.config

  

    FFNN = FeedForwardNeuralNetwork(
        num_hidden_layers=CONFIG.num_hidden_layers,
        num_hidden_neurons=CONFIG.num_hidden_neurons,
        X_train_raw=trainInFull,
        Y_train_raw=trainOutFull,
        N_train = N_train_full,
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



    

    training_loss, trainingaccuracy, validationaccuracy, Y_pred_train = FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, FFNN.batch_size, FFNN.learning_rate)
    wandb.finish()
    Y_pred_test =  FFNN.predict(FFNN.X_test, FFNN.N_test)
    train_accuracy, Y_true_train, Y_pred_train = FFNN.accuracy(FFNN.Y_train, Y_pred_train, FFNN.N_train)
    test_accuracy, Y_true_test, Y_pred_test = FFNN.accuracy(FFNN.Y_test, Y_pred_test,FFNN.N_test)
    train_pred = (train_accuracy, Y_true_train, Y_pred_train)
    test_pred = (test_accuracy, Y_true_test, Y_pred_test)

    return train_pred, test_pred

    
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"

    Results = {}
    Results["train_pred_best"], Results["test_pred_best"] = trainandtest()
    
    data = [[label, val] for (label, val) in zip(["test_pred_best"],[Results['test_pred_best'][0]])] 
    table = wandb.Table(data=data, columns = ["Configuration", "Test accuracy"])
    wandb.log({"my_bar_chart_id" : wandb.plot.bar(table, "Configuration", "Test accuracy",title="Test accuracy for the best configuration chosen for Fashion MNIST classification")})
    
    wandb.finish()
    
    wandb.init(project='CS6910-DeepLearningFundamentals-Assignment1', entity='rahulsundar')
    wandb.sklearn.plot_confusion_matrix(Results["train_pred_best"][1], Results["train_pred_best"][2], labels =[0,1,2,3,4,5,6,7,8,9])    
    wandb.finish()
    
    wandb.init(project='CS6910-DeepLearningFundamentals-Assignment1', entity='rahulsundar')    
    wandb.sklearn.plot_confusion_matrix(Results["train_pred_best"][1], Results["train_pred_best"][2], labels =[0,1,2,3,4,5,6,7,8,9])
    wandb.finish()
    
