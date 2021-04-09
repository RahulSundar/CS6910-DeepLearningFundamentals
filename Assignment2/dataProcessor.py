import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2

from  matplotlib import pyplot as plt
import matplotlib.image as mpimg

class DataSet():

    def __init__(self, IMG_SIZE,dataset_directory, color_mode='rgb')
        self.IMG_HEIGHT = IMG_SIZE[0]
        self.IMG_WIDTH = IMG_SIZE[1]
        self.DATAPATH = dataset_directory
        if color_mode == 'rgb':
            self.num_channels = 3
    def load_dataset(self):
       
        img_data_array=[]
        class_name=[]
       
        for dir1 in os.listdir(self.DATAPATH):
            for file in os.listdir(os.path.join(self.DATAPATH, dir1)):
           
                image_path= os.path.join(self.DATAPATH, dir1,  file)
                image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, (self.IMG_HEIGHT, self.IMG_WIDTH, self.num_channels),interpolation = cv2.INTER_AREA)
                image=np.array(image)
                image = image.astype('float32')
                image /= 255 
                img_data_array.append(image)
                class_name.append(dir1)
        return img_data_array, class_name

def load_dataset(DATAPATH, IMG_HEIGHT, IMG_WIDTH):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(DATAPATH):
        for file in os.listdir(os.path.join(DATAPATH, dir1)):
       
            image_path= os.path.join(DATAPATH, dir1,  file)
            image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
            image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

def load_dataset_batch(DATAPATH, IMG_HEIGHT, IMG_WIDTH, batch_size = None):
   
    img_data_array=[]
    class_name=[]
   
    for dir1 in os.listdir(DATAPATH):
        if dir1[0] != ".":
            ctr = 0
            for file in os.listdir(os.path.join(DATAPATH, dir1)):
                ctr += 1
                if batch_size == None:    
                    image_path= os.path.join(DATAPATH, dir1,  file)
                    image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                    image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                    image=np.array(image)
                    image = image.astype('float32')
                    image /= 255 
                    img_data_array.append(image)
                    class_name.append(dir1)
                else:
                    if ctr <= batch_size:
                        image_path= os.path.join(DATAPATH, dir1,  file)
                        image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)
                        image=cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
                        image=np.array(image)
                        image = image.astype('float32')
                        image /= 255 
                        img_data_array.append(image)
                        class_name.append(dir1)
    return img_data_array, class_name

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
#Example:
'''
IMG_SIZE = (256,256)


'''


#Alternative
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        'data/inaturalist/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
        'data/inaturalist/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')
