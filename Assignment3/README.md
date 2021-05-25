# Overview
The purpose of this assignment was:
1. Building and training a RNN model from scratch for seq2seq character level Neural machine transliteration.
2. Implement attention based model.

The link to the wandb project runs:
https://wandb.ai/rahulsundar/CS6910-Assignment-3?workspace=user-rahulsundar

The link to the wandb report:
https://wandb.ai/rahulsundar/CS6910-Assignment-3/reports/CS6910-Assignment-3-Seq2seq-Character-level-Neural-Machine-Transliteration--Vmlldzo3MjM4Mjc
## Dataset:

The dakshina dataset released by google was used for 
In this assignment the Dakshina dataset(https://github.com/google-research-datasets/dakshina) released by Google has been used. This dataset contains pairs of the following form: 
﻿xxx.      yyy﻿
ajanabee अजनबी.
i.e., a word in the native script and its corresponding transliteration in the Latin script (the way we type while chatting with our friends on WhatsApp etc). Given many such (xi,yi)i=1n(x_i, y_i)_{i=1}^n(xi​,yi​)i=1n​ pairs your goal is to train a model y=f^(x)y = \hat{f}(x)y=f^​(x) which takes as input a romanized string (ghar) and produces the corresponding word in Devanagari (घर). 
These blogs were used as references to understand how to build neural sequence to sequence models: 

https://keras.io/examples/nlp/lstm_seq2seq/
https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/

By default the implemented model uses telugu as the target language. 

## Building and training a RNN model with and without attention from scratch for sequence to sequence character level neural machine transliteration:

A model class has been implemented in ```modelClass.py``` which provides a variety of options to build RNN models and add dropout,hidden layers, neurons, etc. 
The training, and testing scripts can be found in ```train.py```, ```test.py```
System requirements are as follows:
- ```CUDA == 11.0```
- ```CUDNN >= v8.0```
- ```Python >= 3.6 ```

The python package dependencies are provided in ```requirements.txt``` which can be installed on the system using:

```
pip install -r requirements.txt
```

A Bahdanau based attention has been implemented by adapting the code for the bahdanau attention layer class from : https://github.com/thushv89/attention_keras. 
It is advised to setup a virtual environment if running locally using virtualenv/venv and pyenv for python version handling. Or even better, use Conda. But in this assignent I have not used anaconda package manager. 
### Training:
Wandb framework is used to track the loss and accuracy metrics of training and validation. Moreover, bayesian sweeps have been performed for various hyper parameter configurations. 
The sweep configuration and default configurations of hyperparameters are specficied as follows:
```
sweep_config = {
    "name"﻿: "Bayesian Sweep without attention - 2"﻿,
    "method"﻿: "bayes"﻿,
    "metric"﻿: {﻿"name"﻿: "val_accuracy"﻿, "goal"﻿: "maximize"﻿}﻿,
    "parameters"﻿: {
        
        "cell_type"﻿: {﻿"values"﻿: [﻿"RNN"﻿, "GRU"﻿, "LSTM"﻿]﻿}﻿,
        
        "latentDim"﻿: {﻿"values"﻿: [﻿256﻿, 128﻿, 64﻿, 32﻿]﻿}﻿,
        
        "hidden"﻿: {﻿"values"﻿: [﻿128﻿, 64﻿, 32﻿, 16﻿]﻿}﻿,
        
        "optimiser"﻿: {﻿"values"﻿: [﻿"rmsprop"﻿, "adam"﻿]﻿}﻿,
        
        "numEncoders"﻿: {﻿"values"﻿: [﻿1﻿, 2﻿, 3﻿]﻿}﻿,
        
        "numDecoders"﻿: {﻿"values"﻿: [﻿1﻿, 2﻿, 3﻿]﻿}﻿,
        
        "dropout"﻿: {﻿"values"﻿: [﻿0.1﻿, 0.2﻿, 0.3﻿]﻿}﻿,
        
        "epochs"﻿: {﻿"values"﻿: [﻿5﻿,﻿10﻿,﻿15﻿, 20﻿]﻿}﻿,
        
        "batch_size"﻿: {﻿"values"﻿: [﻿32﻿, 64﻿]﻿}﻿,
    }﻿,
}

sweep_id = wandb.sweep(sweep_config, project='CS6910-Assignment3', entity='rahulsundar')

# The following is placed within the train() function. 
config_defaults = {
        "cell_type": "GRU",
        "latentDim": 256,
        "hidden": 128,
        "optimiser": "rmsprop",
        "numEncoders": 1,
        "numDecoders": 1,
        "dropout": 0.2,
        "epochs": 1,
        "batch_size": 64,
    }
 
```

The user can either run the colab notebooks directly or use the python script:
The commands to run the training script is simply:
```python3 train_balanced.py```
or 
```python3 trian.py```


### Testing:

In order to test the best trained model on the test data set, a test script has been written that:
1. Evaluates the test accuracy
2. Saves the predicitons in a csv file
The commands to run the testing script is simply:

```python3 test.py```

If one aspires to do further analysis, then it is advised that the test script is run in the ipython console:

```run train.py```

```run test.py```


### Hyperparameter sweeps:

One can find two colab notebooks which are self contained and they can be run on a GPU based runtime session and the results will be logged accordingly in the user entity's wandb account which alone needs to be changed in the notebook before beginning the run. 

Note: ```Assignment3.csv``` file contains the various configurations sweeped over by wandb and the obtained validation accuracies, losses, etc. 
It is  better if the system dependencies 1 and 2 mentioned in the beginning of this readme are satisfied.  
