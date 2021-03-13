import numpy as np
import matplotlib.pyplot as plt
import wandb as wb

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist


from feedForwardNeuralNet import FeedForwardNeuralNet

if __name__ == "__main__":

    # Load the data in predefined train and test split ratios:

    (trainIn, trainOut), (testIn, testOut) = fashion_mnist.load_data()

    N_train = trainOut.shape[0]
    N_validation = int(0.1 * trainOut.shape[0])
    N_test = testOut.shape[0]

    config_defaults = dict(
        max_epochs=5,
        num_hidden_layers=3,
        num_hidden_neurons=32,
        weight_decay=0,
        learning_rate=1e-3,
        optimizer="SGD",
        batch_size=16,
        activation="TANH",
        initializer="XAVIER",
        loss="CROSS",
    )

    idx1 = np.random.choice(trainOut.shape[0], N_train, replace=False)
    idx2 = np.random.choice(testOut.shape[0], N_test, replace=False)
    idx3 = np.random.choice(trainOut.shape[0], N_validation, replace=False)

    trainIn = trainIn[idx1, :]
    trainOut = trainOut[idx1, :]
    testIn = trainIn[idx2, :]
    testOut = trainOut[idx2, :]
    validIn = trainIn[idx2, :]
    validOut = trainOut[idx2, :]

    FFNN = FeedForwardNeuralNetwork(
        num_hidden_layers=4,
        num_hidden_neurons=32,
        X_train_raw=trainIn,
        Y_train_raw=trainOut,
    )

    weights, biases, training_loss, Y_pred_train = FFNN.train(optimizer=optimizer)
    test_loss, Y_pred_test = FFNN.predict()
    accuracy, true_label, predicted_label = FFNN.accuracy(FFNN.Y_train, Y_pred, N_train)

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

    sweep_id = wandb.sweep(sweep_config)
    
    """
