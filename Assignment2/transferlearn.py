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
  # Invalid device or cannot modify virtual devices once initialized.
  pass


#data pre processing

data_augmentation = False

IMG_SIZE = (128,128)
BATCH_SIZE = 32


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
        './Data/inaturalist_12K/val',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle = True,
         seed = 123)


def transferlearn():
    pass
