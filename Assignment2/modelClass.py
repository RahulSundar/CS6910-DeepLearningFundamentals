import os

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import keras

# keras pre-trained models
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 as IRV2
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception


from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, InputLayer, Flatten, Conv2D, BatchNormalization, MaxPooling2D, Activation , GlobalAveragePooling2D
from tensorflow.keras.models import Sequential,  Model

import wandb


class ImageClassification():

    def __init__(self, IMG_SIZE, modelConfigDict, using_pretrained_model = False, base_model = "IRV2" ):
        
        self.num_hidden_cnn_layers= modelConfigDict["num_hidden_cnn_layers"]
        self.activation = modelConfigDict["activation"]
        self.batch_normalization = modelConfigDict["batch_normalization"]
        self.filter_distribution = modelConfigDict["filter_distribution"]
        self.filter_size = modelConfigDict["filter_size"]
        self.number_of_filters_base  = modelConfigDict["number_of_filters_base"]
        self.dropout_fraction = modelConfigDict["dropout_fraction"]
        self.pool_size = modelConfigDict["pool_size"]
        self.padding = modelConfigDict["padding"]
        self.dense_neurons = modelConfigDict["dense_neurons"]
        self.num_classes = modelConfigDict["num_classes"]
        self.optimizer = modelConfigDict["optimizer"]
        self.global_average_pooling = modelConfigDict["global_average_pooling"]
        self.batch_normalisation_location = modelConfigDict["batch_normalisation_location"]
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


    def build_cnnmodel_conv(self):
        tf.keras.backend.clear_session()
        model = Sequential()
        
        #First CNN layer connecting to input layer
        model.add(Conv2D(self.number_of_filters_base, self.filter_size, padding = self.padding,kernel_initializer = "he_uniform", input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)))
        if self.batch_normalisation_location == "Before" and self.batch_normalization: model.add(BatchNormalization())
        model.add(Activation(self.activation))
        
        #batch_normalisation
        if self.batch_normalisation_location == "After" and self.batch_normalization: model.add(BatchNormalization())
        #max pooling
        model.add(MaxPooling2D(pool_size=self.pool_size))  
        if self.dropout_fraction != None:
            model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        for i in range(self.num_hidden_cnn_layers-1):
            #i+2th Convolutional Layer
        
            ## Standard filter distribution - same number of filters in all Convolutional layers
            if self.filter_distribution == "standard":
                model.add(Conv2D(self.number_of_filters_base, self.filter_size,kernel_initializer = "he_uniform",padding = self.padding))
        
            ## Double filter distribution - double number of filters in each Convolutional layers
            elif self.filter_distribution == "double":
                model.add(Conv2D(2**(i+1)*self.number_of_filters_base, self.filter_size,kernel_initializer = "he_uniform", padding = self.padding))
        
            ## Halve the filter size in each successive convolutional layers
            elif self.filter_distribution == "half":
                model.add(Conv2D(int(self.number_of_filters_base/2**(i+1)), self.filter_size,kernel_initializer = "he_uniform", padding = self.padding))
        
            if self.batch_normalisation_location == "Before" and self.batch_normalization: model.add(BatchNormalization())
            model.add(Activation(self.activation))
        
            if self.batch_normalisation_location == "After" and self.batch_normalization: model.add(BatchNormalization())
        
            model.add(MaxPooling2D(pool_size=self.pool_size))
            if self.dropout_fraction != None:
                model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        
        #Final densely connected layers
        if self.global_average_pooling == True:
            model.add(GlobalAveragePooling2D())
        else:
            model.add(Flatten())

        model.add(Dense(self.dense_neurons, activation = 'sigmoid'))
        model.add(Dense(self.num_classes, activation = 'softmax'))

        return model      
        
    def build_cnnmodel_all(self):

        
        tf.keras.backend.clear_session()
        model = Sequential()
        
        #First CNN layer connecting to input layer
        model.add(Conv2D(self.number_of_filters_base, self.filter_size, padding = self.padding,kernel_initializer = "he_uniform", input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)))
        model.add(Activation(self.activation))
        
        #batch_normalisation
        if self.batch_normalisation_location == "After" and self.batch_normalization: model.add(BatchNormalization())
        #max pooling
        model.add(MaxPooling2D(pool_size=self.pool_size))  
        if self.dropout_fraction != None:
            model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        for i in range(self.num_hidden_cnn_layers-1):
            #i+2th Convolutional Layer
        
            ## Standard filter distribution - same number of filters in all Convolutional layers
            if self.filter_distribution == "standard":
                model.add(Conv2D(self.number_of_filters_base, self.filter_size,kernel_initializer = "he_uniform", padding = self.padding))
        
            ## Double filter distribution - double number of filters in each Convolutional layers
            elif self.filter_distribution == "double":
                model.add(Conv2D(2**(i+1)*self.number_of_filters_base, self.filter_size,kernel_initializer = "he_uniform",padding = self.padding))
        
            ## Halve the filter size in each successive convolutional layers
            elif self.filter_distribution == "half":
                model.add(Conv2D(int(self.number_of_filters_base/2**(i+1)), self.filter_size,kernel_initializer = "he_uniform", padding = self.padding))
        
            model.add(Activation(self.activation))
        
            if self.batch_normalisation_location == "After" and self.batch_normalization: model.add(BatchNormalization())
        
            model.add(MaxPooling2D(pool_size=self.pool_size))
            if self.dropout_fraction != None:
                model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        
        #Final densely connected layers
        if self.global_average_pooling == True:
            model.add(GlobalAveragePooling2D())
        else:
            model.add(Flatten())

        model.add(Dense(self.dense_neurons, activation = 'sigmoid'))
        if self.dropout_fraction != None:
            model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        model.add(Dense(self.num_classes, activation = 'softmax'))
        
        return model      
      
    def build_cnnmodel_dense(self):

        tf.keras.backend.clear_session()
        model = Sequential()
        
        #First CNN layer connecting to input layer
        model.add(Conv2D(self.number_of_filters_base, self.filter_size ,kernel_initializer = "he_uniform",padding = self.padding,input_shape = (self.IMG_HEIGHT, self.IMG_WIDTH, 3)))
        if self.batch_normalisation_location == "Before" and self.batch_normalization: model.add(BatchNormalization())
        model.add(Activation(self.activation))
        
        #batch_normalisation
        if self.batch_normalisation_location == "After" and self.batch_normalization: model.add(BatchNormalization())
        #max pooling
        model.add(MaxPooling2D(pool_size=self.pool_size))  
        for i in range(self.num_hidden_cnn_layers-1):
            #i+2th Convolutional Layer
        
            ## Standard filter distribution - same number of filters in all Convolutional layers
            if self.filter_distribution == "standard":
                model.add(Conv2D(self.number_of_filters_base, self.filter_size,kernel_initializer = "he_uniform",padding = self.padding))
        
            ## Double filter distribution - double number of filters in each Convolutional layers
            elif self.filter_distribution == "double":
                model.add(Conv2D(2**(i+1)*self.number_of_filters_base, self.filter_size,kernel_initializer = "he_uniform",padding = self.padding))
        
            ## Halve the filter size in each successive convolutional layers
            elif self.filter_distribution == "half":
                model.add(Conv2D(int(self.number_of_filters_base/2**(i+1)),self.filter_size, kernel_initializer = "he_uniform"))
        
            if self.batch_normalisation_location == "Before" and self.batch_normalization: model.add(BatchNormalization())
            model.add(Activation(self.activation))
        
            if self.batch_normalisation_location == "After" and self.batch_normalization: model.add(BatchNormalization())
            
            model.add(MaxPooling2D(pool_size=self.pool_size))
        
        #Final densely connected layers
        if self.global_average_pooling == True:
            model.add(GlobalAveragePooling2D())
        else:
            model.add(Flatten())
        model.add(Dense(self.dense_neurons, activation = 'sigmoid'))
        if self.dropout_fraction != None:
            model.add(tf.keras.layers.Dropout(self.dropout_fraction))
        model.add(Dense(self.num_classes, activation = 'softmax'))
      
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

        return model
 
