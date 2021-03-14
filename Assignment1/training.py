import numpy as np
import matplotlib.pyplot as plt
import wandb

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


from feedForwardNeuralNet import FeedForwardNeuralNetwork

#if __name__ == "__main__":

    # Load the data in predefined train and test split ratios:
#    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
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


sweep_config = {
  "name": "Random Sweep",
  "method": "random",
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  }
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER"]
        },

        "num_layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": ['RELU', 'SIGMOID', 'TANH']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}

sweep_id = wandb.sweep(sweep_config,project='CS6910-DeepLearningFundamentals-Assignment1', entity='rahulsundar')

def train():    
    config_defaults = dict(
            max_epochs=5,
            num_hidden_layers=3,
            num_hidden_neurons=32,
            weight_decay=0,
            learning_rate=1e-3,
            optimizer="MGD",
            batch_size=16,
            activation="TANH",
            initializer="RANDOM",
            loss="CROSS",
        )
        
    #wandb.init(project='CS6910-DeepLearningFundamentals-Assignment1', entity='rahulsundar', config = config_defaults)
    wandb.init(config = config_defaults)


    wandb.run.name = "hl_" + str(wandb.config.num_hidden_layers) + "_hn_" + str(wandb.config.num_hidden_neurons) + "_opt_" + wandb.config.optimizer + "_act_" + wandb.config.activation + "_lr_" + str(wandb.config.learning_rate) + "_bs_"+str(wandb.config.batch_size) + "_init_" + wandb.config.initializer + "_ep_"+ str(wandb.config.max_epochs)+ "_l2_" + str(wandb.config.weight_decay) 
    CONFIG = wandb.config


    
    #sweep_id = wandb.sweep(sweep_config)
  

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



    training_loss, trainingaccuracy, validationaccuracy, Y_pred_train = FFNN.optimizer(FFNN.max_epochs, FFNN.N_train, FFNN.batch_size, FFNN.learning_rate)
    
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    wandb.agent(sweep_id, train, count = 2)
    #wandb.finish()
    
    #Y_pred_test =  FFNN.predict(FFNN.X_test, FFNN.N_test)
    
    #test_accuracy, Y_true_test, Y_pred_test = FFNN.accuracy(FFNN.X_test, Y_pred_test)
    
    #trainingaccuracy, true_label, predicted_label = FFNN.accuracy(FFNN.Y_train, Y_pred_train, N_train)

"""
# Flexible integration for any Python script
# 1. Start a W&B run
wandb.init(project='deeplearningfundamentals-cs6910', entity='rahulsundar')

# 2. Save model inputs and hyperparameters
config = wandb.config
config.learning_rate = 0.01

# Model training here

# 3. Log metrics over time to visualize performance
wandb.log({"loss": loss})
    
    
# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
hyperparameter_defaults = dict(
    dropout = 0.5,
    channels_one = 16,
    channels_two = 32,
    batch_size = 100,
    learning_rate = 0.001,
    epochs = 2,
    )

# Pass your defaults to wandb.init
wandb.init(config=hyperparameter_defaults)
config = wandb.config

# Your model here ...

# Log metrics inside your training loop

metrics = {'accuracy': accuracy, 'loss': loss}
wandb.log(metrics)



    
    


sweep_config = {
  "name": "Random Sweep",
  "method": "random",
  "metric":{
  "name": "accuracy",
  "goal": "maximize"
  }
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER", "HE"]
        },

        "num_layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": ['RELU', 'SIGMOID', 'TANH']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}


sweep_config = {
  "name": "Random Sweep",
  "method": "random",
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  }
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER", "HE"]
        },

        "num_layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": ['RELU', 'SIGMOID', 'TANH']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}


    sweep_config = {
  "name": "Random Sweep",
  "method": "random",
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  }
  "parameters": {
        "max_epochs": {
            "values": [5, 10]
        },

        "initializer": {
            "values": ["RANDOM", "XAVIER"]
        },

        "num_layers": {
            "values": [2, 3, 4]
        },
        
        
        "num_hidden_neurons": {
            "values": [32, 64, 128]
        },
        
        "activation": {
            "values": ['RELU', 'SIGMOID', 'TANH']
        },
        
        "learning_rate": {
            "values": [0.001, 0.0001]
        },
        
        
        "weight_decay": {
            "values": [0, 0.0005,0.5]
        },
        
        "optimizer": {
            "values": ["SGD", "MGD", "NAG", "RMSPROP", "ADAM","NADAM"]
        },
                    
        "batch_size": {
            "values": [16, 32, 64]
        }
        
        
    }
}
sweep_id = wandb.sweep(sweep_config)

"""
