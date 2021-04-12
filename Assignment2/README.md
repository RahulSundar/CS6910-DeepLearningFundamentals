# Overview
The purpose of this assignment was three fold
1. Building and training a CNN model from scratch for iNaturalist image data classification.
2. Fine tune a pretrained model on the iNaturalist dataset.
3. Use a pretrained Object Detection model for a cool application

## Part 1: Building and training a CNN model from scratch for classification:

A model class has been implemented in ```modelClass.py``` which provides a variety of options to build CNN models and add dropout, batchnormalisation, etc. 
The training, transfer learning and testing scripts can be found in ```train.py```, ```test.py``` and ```transferlearn_test.py ```
System requirements are as follows:
- ```CUDA == 11.0```
- ```CUDNN >= v8.0```
- ```Python >= 3.6 ```

The python package dependencies are provided in ```requirements.txt``` which can be installed on the system using:

```
pip install -r requirements.txt
```

### Training:
Wandb framework is used to track the loss and accuracy metrics of training and validation. Moreover, bayesian sweeps have been performed for various hyper parameter configurations. 
The sweep configuration and default configurations of hyperparameters are specficied as follows:
```
sweep_config = {
  "name": "Bayesian Sweep",
  "method": "bayes",
  "metric":{
  "name": "val_accuracy",
  "goal": "maximize"
  },
  'early_terminate': {
        'type':'hyperband',
        'min_iter': [3],
        's': [2]
  },
  "parameters": {
                    

        "base_model": {
            "values": [ "XCPTN", "IV3", "RN50", "IRV2"]
        },
        "epochs": {
            "values": [ 5, 10, 3]
        }, 
        "dense_neurons": {
            "values": [ 128, 256]
        } 
              
    }
}

sweep_id = wandb.sweep(sweep_config, project='CS6910-Assignment2-CNNs', entity='rahulsundar')

# The following is placed within the train() function. 
config_defaults = dict(
                dense_neurons =256 ,
                activation = 'relu',
                num_classes = 10,
                optimizer = 'adam',
                epochs = 5,
                batch_size = 32, 
                img_size = (224,224),
                base_model = "RN50"
            ) 
```
