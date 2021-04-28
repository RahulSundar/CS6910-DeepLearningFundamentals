# Overview
The purpose of this assignment was three fold
1. Building and training a CNN model from scratch for iNaturalist image data classification.
2. Fine tune a pretrained model on the iNaturalist dataset.
3. Use a pretrained Object Detection model for a cool application

The link to the wandb project runs:
https://wandb.ai/rahulsundar/CS6910-Assignment2-CNNs?workspace=user-rahulsundar

The link to the wandb report:
https://wandb.ai/rahulsundar/CS6910-Assignment2-CNNs/reports/CS6910-Assignment-2-Image-classification-and-Object-detection-using-Convolutional-neural-Networks--Vmlldzo2MjI1NDY

## Part A: Building and training a CNN model from scratch for classification:

A model class has been implemented in ```modelClass.py``` which provides a variety of options to build CNN models and add dropout, batchnormalisation, etc. 
The training, transfer learning and testing scripts can be found in ```train.py```, ```test.py``` and ```transferlearn_test.py ```
System requirements are as follows:
- ```CUDA == 11.0```
- ```CUDNN >= v8.0```
- ```Python >= 3.6 ```

The python package dependencies for Part A and Part B are provided in ```requirements.txt``` which can be installed on the system using:

```
pip install -r requirements.txt
```

For Part C, if one is using a local system, it is advised that one setup a virtual environment with pyenv and virtual env.
pyenv to install python >= 3.8 and virtual env to install all the package dependencies as required by the YOLOv5 model 
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
Inorder to use balanced data, randomly move 100 files from each class folder in the train folder of the data set to the validation folder under the same class name. Once done, change the path names in the validation generator part of the training script ```train_balanced.py```.
 
The user can either run the colab notebooks directly or use the python script:
The commands to run the training script is simply:
```python3 train_balanced.py```
or 
```python3 trian.py```


### Testing:

In order to test the best trained model on the test data set, a test script has been written that:
1. Evaluates the test accuracy
2. Plots a confusion matrix
3. Plots sample images, the associated predictions and true labels.
 
The commands to run the testing script is simply:

```python3 test.py```

If one aspires to do further analysis, then it is advised that the test script is run in the ipython console:

```run train.py```

### Visualisation of CNNs:

In order to visualise how the CNNs learn, the following have been implemented through standalone scripts that use the best trained model or any trained keras compatible model for that matter:
1. ```filterVisualisation.py``` - filters, the associated feature maps of a specified layer. In our case, it is the first convolutional layer "conv2d"
2. ```guided_backprop.py``` - guided backpropagation on a sample of test images. A guided backpropagation function is implemented that can be generalised to any model that's loaded in keras. 

Both these scripts can be run from the working directory either on the terminal or on the ipython console.


## Part B: Fine tuning a pretrained image classification model.
For this problem, pretrained models such as Xception, ResNet50, InceptionV3, InceptionResnetV2 are used as base models and the user can choose between these models.
The user can also choose to freeze all the layers and make them non trainable and only train the newly added dense layers compatible with the number of classes in the dataset. 
In our case the dense layers were swapped with the output layer having 10 softmax neurons.

```transferlearn.py``` script can be run either on the terminal or from the ipython console in a similar manner described earlier. The trained models in all the Parts are saved in "TrainedModel" folder in the directory of the Assignment2. However, the trained models can be saved in any folder of the user's choice by changing the save path in 
```model.save(<model save path>)```

### Part C: Real time object detection application using YOLOV5

In this task, YOLOV5s pretrained model was fine tuned to solve two problems:
1. Mask detection
2. Wildfire detection

The youtube links are provided in the wandb report mentioned above. 

The python dependencies required for YOLOV5 as given in the [official repository](https://github.com/ultralytics/yolov5)'s requirements.txt can be installed in a virtual environment with python 3.8 and above installed. 
It is  better if the system dependencies 1 and 2 mentioned in the beginning of this readme are satisfied.  
