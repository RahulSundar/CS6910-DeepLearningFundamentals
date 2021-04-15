# data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib

#
import tensorflow as tf
from modelClass import ObjectDetection


#wandb logging
import wandb
from wandb.keras import WandbCallback
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass


IMG_SIZE = (128,128)

'''
#sweep config
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
        
        "activation":{
            "values": ["relu", "elu", "selu"]
        },
        "filter_size": {
            "values": [(2,2), (3,3), (4,4)]
        },
        "batch_size": {
            "values": [32, 64]
        },
        "padding": {
            "values": ["same","valid"]
        },
        "data_augmentation": {
            "values": [True, False]
        },
        "optimizer": {
            "values": ["sgd", "adam", "rmsprop", "nadam"]
        },
        "batch_normalization": {
            "values": [True, False]
        },
        "batch_normalisation_location": {
            "values": ["Before", "After"]
        },
        "number_of_filters_base": {
            "values": [32, 64]
        },
        "dense_neurons": {
            "values": [32, 64, 128]
        },   
        "dropout_location": {
            "values": ["conv","dense","all"]
        },
        "dropout_fraction": {
            "values": [None, 0.2,0.3]
        },  
        "global_average_pooling": {
            "values": [False,True]
        },        
    }
}

sweep_id = wandb.sweep(sweep_config,project='CS6910-Assignment2-CNNs', entity='rahulsundar')
'''


#train function
def train():

        
    config_defaults = dict(
            num_hidden_cnn_layers = 5 ,
            activation = 'relu',
            batch_normalization = True,
            batch_normalisation_location = "After",
            filter_distribution = "double" ,
            filter_size = (3,3),
            number_of_filters_base  = 32,
            dropout_fraction = None,
            dropout_location = "dense",
            pool_size = (2,2),
            padding = 'same',
            dense_neurons = 128,
            num_classes = 10,
            optimizer = 'adam',
            epochs = 5,
            batch_size = 32, 
            data_augmentation = False,
            global_average_pooling = True,
            img_size = IMG_SIZE
        ) 



    #wandb.init( config = config_defaults)
    wandb.init(project = 'CS6910-Assignment2-CNNs', config = config_defaults,entity='rahulsundar')
    CONFIG = wandb.config


    wandb.run.name = "OBJDET_" + str(CONFIG.num_hidden_cnn_layers) + "_dn_" + str(CONFIG.dense_neurons) + "_opt_" + CONFIG.optimizer + "_dro_" + str(CONFIG.dropout_fraction) + "_bs_"+str(CONFIG.batch_size) + "_fd_" + CONFIG.filter_distribution + "_bnl_" + CONFIG.batch_normalisation_location + "_dpl_" + CONFIG.dropout_location

    data_augmentation = CONFIG.data_augmentation

    
    BATCH_SIZE = CONFIG.batch_size


    if data_augmentation == True:

    #Faster Alternative
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1./255,
                validation_split = 0.1,
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
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split = 0.1)

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)


    train_generator = train_datagen.flow_from_directory(
        './Data/inaturalist_12K/train',
        subset='training',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
        seed = 123)
        
    validation_generator = train_datagen.flow_from_directory(
            './Data/inaturalist_12K/train',
            target_size=IMG_SIZE,
            subset = 'validation',
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle = True,
            seed = 123)


            
    test_generator = test_datagen.flow_from_directory(
            './Data/inaturalist_12K/test',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle = True,
            seed = 123)



       
    objDetn = ObjectDetection(CONFIG.img_size, CONFIG )
    if CONFIG.dropout_location == "all":
        model = objDetn.build_cnnmodel_all()
    elif CONFIG.dropout_location == "conv":
        model = objDetn.build_cnnmodel_conv()
    elif CONFIG.dropout_location == "dense":
        model = objDetn.build_cnnmodel_dense()
    
    model.summary()



    model.compile(
    optimizer=CONFIG.optimizer,  # Optimizer
    # Loss function to minimize
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),#'categorical_crossentropy',
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
    return model, history
        
 
    
if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    
    train()
    #wandb.agent(sweep_id, train, count = 24)
    #Model, History = train()
    

