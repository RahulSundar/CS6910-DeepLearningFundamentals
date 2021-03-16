# **Fashion MNIST classification using a "numpy" Bare Bones Feed forward Neural Network**

This folder contains the code base for Assignment 1 as part of CS6910: Deep Learning Fundamentals course at IIT Madras during the Spring of 2021.

The problem statement involves building and training a 'plain vanilla' Feed Forward Neural Network from scratch using primarily Numpy package in Python.  

The code base now has the following features:
1. Forward and backward propagation are hard coded using Matrix operations. The weights and biases are stored separately as dictionaries to go hand in hand with the notation used in class.
2. A neural network class to instantiate the neural network object for specified set of hyperparameters, namely the number of layers, hidden neurons, activation function, optimizer, weight decay,etc.
3. The optimisers, activations and their gradients are passed through dictionaries configured as attributed within the FeedForwardNeuralNetwork class. 
4. Activation functions are defined separately in the utility/activations.py file. 
5. A colab notebook containing the entire code to train and validate the model from scratch. 

## Dataset

Fashion MNIST data set has been used here in this assignment instead of the traditional MNIST hand written digits dataset. 
Train  - 60000
Test - 20000
Validation - 6000

For the hyper parameter optimisation stage, 10% of the randomly shuffled training data set (around 6000 images and corresponding labels) are kept aside for validation for each hyperparameter configuration while the model is trained on the remaining 54000 images from the randomly shuffled training data set. 

Once the best configuration is identified with the help of wandb wither using Random search or Bayesian optimisation, the full training dataset is used to train the best model configuration and the test accuracy is calculated. The resulting confusion matrix is plotted therafter.  

## Code base structure

Utility/activations.py - contains all the activation functions and its allied derivatives.

feedForwardNeuralNet.py - The FeedForwardNeuralNetwork class is defined within this file

training.py - dataset download, splitting and preprocessing along with training and hyper parameter sweep using wandb agent.

datapreprocessing.py - data set download and plotting of sample images. 

Assignment1_training_sweep_Fashion_MNIST.ipynb - Google colab note book to carry out training and hyperparameter search using Wandb for various hyper parameter combinations. 

Assignment1_training_MSE_Loss.ipynb - Google colab notebook to carry out training for MSE loss function. This was done to run simultaneously multiple seeps and wandb runs. But otherwise a single code should suffice.
## Training, Validation and Hyperparameter optimisation


```
python training.py
```
Or if you want to run interactively, you could use Ipython console and later postprocess the data on the console itself. 

Inorder to train and then test on the Fashion MNIST dataset:
```
ipython
In [1]: run training.py 
In [2]: run test.py 
```
Inorder to train on the MNIST hand written digits:
```
ipython
In [1]: run MNIST_training.py 
``` 
 
Once the training is done, you can check all the runs on the wandb dashboards corresponding to each run as the metrics are logged automatically during the runs.  


A template optimiser is provided in feedForwardNeuralNet.py to code the gradient based optimiser of choice. 

Also, the sweep configurations for wandb based Random and Bayesian hyperparameter search can be configured in the following manner(in the training/ test scripts based on the user's choice):

```
sweep_config = {
  "name": "Random Sweep", #(or) Bayesian Sweep (or) Grid search
  "method": "random", #(or) bayes (or) grid
  "metric":{
  "name": "validationaccuracy",
  "goal": "maximize"
  },
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
```

One can choose to select / modify/omit any of the hyperparameters above in the config dictionary.


## Results:
For the plain vanilla feed forward neural network implemented, the maximum test accuracy reported was 88.08% on the Fashion MNIST dataset and ~96.3% on the MNIST hand written datasets.
One of the model configuration chosen to be  the best is as follows:

- Number of Hidden Layers - 3
- Number of Hidden Neurons - 128
- L2 Regularisation - No
- Activation - Sigmoid
- Initialisation - Xavier
- Optimiser - NADAM
- Learning Rate - 0.001
- Batch size - 32
