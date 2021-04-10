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
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Activation 
from tensorflow.keras.models import Sequential,  Model

import wandb


class ObjectDetection():

    def __init__(self, IMG_SIZE, modelConfigDict, using_pretrained_model = False, base_model = "IRV2" ):
        
        self.num_hidden_cnn_layers= modelConfigDict["num_hidden_cnn_layers"]
        self.activation = modelConfigDict["activation"]
        self.batch_normalization = modelConfigDict["batch_normalization"]
        self.filter_distribution = modelConfigDict["filter_distribution"]
        self.filter_size = modelConfigDict["filter_size"]
        self.number_of_filters_base  = modelConfigDict["number_of_filters_base"]
        self.initializer = modelConfigDict["initializer"]
        self.dropout_fraction = modelConfigDict["dropout_fraction"]
        self.pool_size = modelConfigDict["pool_size"]
        self.padding = modelConfigDict["padding"]
        self.dense_neurons = modelConfigDict["dense_neurons"]
        self.num_classes = modelConfigDict["num_classes"]
        self.optimizer = modelConfigDict["optimizer"]

        BASE_MODELS = {
                          "IRV2" : IRV2,
                          "IV3" : InceptionV3,
                          "RN50" : ResNet50,
                          "XCPTN" : Xception
                      }      
        
        if using_pretrained_model == True:
            self.base_model = base_model
            if self.base_model == "RN50":
                self.IMG_HEIGHT = 224
                self.IMG_WIDTH = 224
            else:
                self.IMG_HEIGHT = IMG_SIZE[0]
                self.IMG_WIDTH = IMG_SIZE[1]        

        self.IMG_HEIGHT = IMG_SIZE[0]
        self.IMG_WIDTH = IMG_SIZE[1]        
         
        self.input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)


    def build_cnndropmodel(self):
        keras.backend.clear_session()
        model = Sequential()
        
        #First CNN layer connecting to input layer
        model.add(Conv2D(self.number_of_filters_base, self.filter_size, kernel_regularizer='l2',padding = self.padding, input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)))
        model.add(Activation(self.activation))
        
        #batch_normalisation
        if self.batch_normalization: model.add(BatchNormalization())
        #max pooling
        model.add(MaxPooling2D(pool_size=self.pool_size))  
        if self.dropout_fraction != None:
            model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        for i in range(self.num_hidden_cnn_layers-1):
            #i+2th Convolutional Layer
        
            ## Standard filter distribution - same number of filters in all Convolutional layers
            if self.filter_distribution == "standard":
                model.add(Conv2D(self.number_of_filters_base, self.filter_size,kernel_regularizer='l2', padding = self.padding, kernel_initializer = self.initializer))
        
            ## Double filter distribution - double number of filters in each Convolutional layers
            elif self.filter_distribution == "double":
                model.add(Conv2D(2**(i+1)*self.number_of_filters_base, self.filter_size, kernel_regularizer='l2',padding = self.padding, kernel_initializer = self.initializer))
        
            ## Halve the filter size in each successive convolutional layers
            elif self.filter_distribution == "half":
                model.add(Conv2D(int(self.number_of_filters_base/2**(i+1)), self.filter_size, kernel_regularizer='l2',padding = self.padding, kernel_initializer = self.initializer))
        
            model.add(Activation(self.activation))
        
            if self.batch_normalization: model.add(BatchNormalization())
        
            model.add(MaxPooling2D(pool_size=self.pool_size))
            if self.dropout_fraction != None:
                model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        
        #Final densely connected layers
        model.add(Flatten())
        model.add(Dense(self.dense_neurons, activation = self.activation, kernel_regularizer='l2', kernel_initializer = self.initializer))
        model.add(Dense(self.num_classes, activation = 'softmax'))
        
        #model.compile(optimizer=self.optimizer,
        #      loss='categorical_crossentropy',
        #      metrics=['accuracy'])
        return model      
        
      
        
    def build_cnnmodel(self):
        
        model = Sequential()
        
        #First CNN layer connecting to input layer
        model.add(Conv2D(self.number_of_filters_base, self.filter_size, kernel_regularizer='l2',padding = self.padding, input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)))
        model.add(Activation(self.activation))
        
        #batch_normalisation
        if self.batch_normalization: model.add(BatchNormalization())
        #max pooling
        model.add(MaxPooling2D(pool_size=self.pool_size))  
        for i in range(self.num_hidden_cnn_layers-1):
            #i+2th Convolutional Layer
        
            ## Standard filter distribution - same number of filters in all Convolutional layers
            if self.filter_distribution == "standard":
                model.add(Conv2D(self.number_of_filters_base, self.filter_size,kernel_regularizer='l2', padding = self.padding, kernel_initializer = self.initializer))
        
            ## Double filter distribution - double number of filters in each Convolutional layers
            elif self.filter_distribution == "double":
                model.add(Conv2D(2**(i+1)*self.number_of_filters_base, self.filter_size, kernel_regularizer='l2',padding = self.padding, kernel_initializer = self.initializer))
        
            ## Halve the filter size in each successive convolutional layers
            elif self.filter_distribution == "half":
                model.add(Conv2D(int(self.number_of_filters_base/2**(i+1)), self.filter_size, kernel_regularizer='l2',padding = self.padding, kernel_initializer = self.initializer))
        
            model.add(Activation(self.activation))
        
            if self.batch_normalization: model.add(BatchNormalization())
        
            model.add(MaxPooling2D(pool_size=self.pool_size))
        
        #Final densely connected layers
        model.add(Flatten())
        model.add(Dense(self.dense_neurons, activation = 'sigmoid', kernel_regularizer='l2'))
        if self.dropout_fraction != None:
            model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        model.add(Dense(self.num_classes, activation = 'softmax'))
        
        #model.compile(optimizer=self.optimizer,
        #      loss='categorical_crossentropy',
        #      metrics=['accuracy'])
        return model      
        
        
    def load_pretrained_model(self):
        base_model = BASE_MODELS[self.base_model_name]
        base = base_model(weights='imagenet', include_top=False)
        x = base.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(self.dense_neurons, activation='relu')(x)
        guesses = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=base.input, outputs=guesses)

        # freeze all base layers
        for layer in base.layers:
            layer.trainable = False

        #model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model
                
'''
                
       
def build_model(dropout_fraction = None,batch_normalization=True, activation="relu", initializer="he_uniform", num_hidden_cnn_layers=5, filter_distribution="standard"):
    model = Sequential()
    #First CNN layer connecting to input layer
    model.add(Conv2D(32, (3, 3), padding = 'same', input_shape = (32, 32, 3)))
    model.add(Activation('relu'))
    #batch_normalisation
    if batch_normalization: model.add(BatchNormalization())
    #max pooling
    model.add(MaxPooling2D(pool_size=(2,2)))  
    for i in range(num_hidden_cnn_layers-1):
        #i+2th Convolutional Layer
        ## Standard filter distribution - same number of filters in all Convolutional layers
        if filter_distribution == "standard":
            model.add(Conv2D(32, (3, 3), activation = activation, padding = 'same', kernel_initializer = initializer))
        ## Double filter distribution - double number of filters in each Convolutional layers
        elif filter_distribution == "double":
            model.add(Conv2D(2**(i+1)*32, (3, 3), activation = activation, padding = 'same', kernel_initializer = initializer))
        ## Halve the filter size in each successive convolutional layers
        elif filter_distribution == "half":
            model.add(Conv2D(int(32/2**(i+1)), (3, 3), activation = activation, padding = 'same', kernel_initializer = initializer))
        model.add(Activation('relu'))
        if batch_normalization: model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        if dropout_fraction != None:
            model.add(tf.keras.layers.Dropout(dropout_fraction))
    
    #Final densely connected layers
    model.add(Flatten())
    model.add(Dense(128, activation = activation, kernel_initializer = initializer))
    model.add(Dense(10, activation = 'softmax'))
    
    return model   
'''
