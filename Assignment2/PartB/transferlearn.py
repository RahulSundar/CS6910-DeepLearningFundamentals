import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras

# keras pre-trained models
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2 as IRV2
from keras.applications.resnet50 import ResNet50
from keras.applications.xception import Xception


from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Activation, GlobalAveragePooling2D 
from tensorflow.keras.models import Sequential,  Model

import wandb

# data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib

#
import tensorflow as tf
from modelClass import ImageClassification


#wandb logging
from wandb.keras import WandbCallback


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass




#data pre processing

data_augmentation = False

IMG_SIZE = (224,224)
BATCH_SIZE = 32


if data_augmentation == True:

#Faster Alternative
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False
            )
else:
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    '../Data/inaturalist_12K/train2',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle = True,
     seed = 123)
    
validation_generator = train_datagen.flow_from_directory(
        '../Data/inaturalist_12K/val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
         seed = 123)


        
test_generator = test_datagen.flow_from_directory(
        '../Data/inaturalist_12K/test',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
         seed = 123)

BASE_MODELS = {
                  "IRV2" : IRV2,
                  "IV3" : InceptionV3,
                  "RN50" : ResNet50,
                  "XCPTN" : Xception
              } 

'''
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
'''

def load_pretrained_model():
        tf.keras.backend.clear_session()
        pretrained_model = BASE_MODELS["RN50"]
        new_input = Input(shape=(224, 224, 3), name="input")
        base = pretrained_model(weights='imagenet', input_tensor=new_input)
        #model = Model(inputs=base.input)
        model = Sequential([base, Flatten(), Dense(1000, activation='relu', kernel_initializer="he_uniform"), Dense(10, activation='softmax')])
        # freeze all base model's layers
        for layer in base.layers:
            layer.trainable = False

        return model

def load_pretrained_model_configurable(config):
        tf.keras.backend.clear_session()
        pretrained_model = BASE_MODELS[config["base_model"]]
        base = pretrained_model(weights='imagenet', include_top=False)
        X = base.output
        X = GlobalAveragePooling2D()(X)
        X = Dense(config["dense_neurons"], activation='sigmoid')(X)
        predictions = Dense(config["num_classes"], activation='softmax')(X)
        model = Model(inputs=base.input, outputs=predictions)

        # freeze all base model's layers
        for layer in base.layers:
            layer.trainable = False

        return model


def transfer_learn():
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
            
    wandb.init(project='CS6910-Assignment2-CNNs', entity='rahulsundar', config = config_defaults) 

    CONFIG = wandb.config

    wandb.run.name = "OBJDET_TransferLearn_" + CONFIG.base_model + "_dn_" + str(CONFIG.dense_neurons) + "_opt_" + CONFIG.optimizer + "_ep_" + str(CONFIG.epochs) + "_bs_"+str(CONFIG.batch_size) + "_act_" + CONFIG.activation


    model = load_pretrained_model_configurable(CONFIG)
    #model = load_pretrained_model()
    model.summary()

    model.compile(
    optimizer=CONFIG.optimizer,  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.CategoricalCrossentropy(),#from_logits=True),#'categorical_crossentropy',
    # List of metrics to monitor
    metrics=['accuracy'],
    )

    history = model.fit(
                    train_generator,
                    steps_per_epoch = train_generator.samples // CONFIG.batch_size,
                    validation_data = validation_generator, 
                    validation_steps = validation_generator.samples // CONFIG.batch_size,
                    epochs = CONFIG.epochs, 
                    callbacks=[WandbCallback()]
                    )
    model.evaluate(
                    test_generator,
                    batch_size = 32,
                    callbacks=[WandbCallback()]
                  )
                    
    model.save('./TrainedModel/'+wandb.run.name)                
    wandb.finish()

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    transfer_learn()
    
    
    #wandb.agent(sweep_id,transfer_learn, count = 15)
