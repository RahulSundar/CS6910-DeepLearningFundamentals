import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
'''
class DataSet():

    def __init__(self, IMG_SIZE,dataset_directory, color_mode='rgb')
        self.IMG_HEIGHT = IMG_SIZE[0]
        self.IMG_WIDTH = IMG_SIZE[1]
        self.DATAPATH = dataset_directory
        if color_mode == 'rgb':
            self.num_channels = 3

img_data, class_names = load_dataset(DATAPATH, IMG_HEIGHT, IMG_WIDTH)

target_dict={k: v for v, k in enumerate(np.unique(class_names))}

#Plot sample image from each class
plt.figure(figsize=(20,20))
ctr = 0
for dir1 in os.listdir(DATAPATH):
    
    if dir1[0] != ".":
        dirpath = os.path.join(DATAPATH, dir1)
        for i in range(5):
            file = np.random.choice(os.listdir(dirpath))
            image_path= os.path.join(dirpath, file)
            img=mpimg.imread(image_path)
            ax=plt.subplot(len(os.listdir(DATAPATH))-1,5,ctr*5 + (i+1))
            ax.title.set_text(dir1)
            plt.imshow(img)
        
        ctr += 1

'''
#Example:
data_augmentation = False
IMG_SIZE = (128,128)

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
        batch_size=32,
        class_mode='categorical',
        shuffle = True,
         seed = 123)
        
validation_generator = train_datagen.flow_from_directory(
        './Data/inaturalist_12K/train',
        target_size=IMG_SIZE,
        subset = 'validation',
        batch_size=32,
        class_mode='categorical',
        shuffle = True,
         seed = 123)
        
test_generator = test_datagen.flow_from_directory(
        './Data/inaturalist_12K/val',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle = True,
         seed = 123)
