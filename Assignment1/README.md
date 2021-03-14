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

## Training, Validation and Hyperparameter optimisation


```
python training.py
```
Or if you want to run interactively, 

```
ipython
> run training.py 
```

Once the training is done, you can run the model on the test data. 
## Testing

## Transfer learning to MNIST hand written digits data set



