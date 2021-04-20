import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

from sklearn import metrics

import pandas as pd
import os
import cv2


import wandb
from wandb.keras import WandbCallback

#To let the gpu memory utilisiation grow as per requirement
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass

#Note: Wandb logging has been removed from the script for easier running
    

# Dataset loading function
def load_dataset_batch(DATAPATH, IMG_HEIGHT, IMG_WIDTH, batch_size = None): 
             
    img_data_array=[] 
    class_name=[] 

    for dir1 in np.sort(os.listdir(DATAPATH)): 
     if dir1[0] != ".": 
         ctr = 0 
         for file in np.sort(os.listdir(os.path.join(DATAPATH, dir1))): 
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
          
#class names:
def class_names(DATAPATH):
   
    class_name=[]
    
    for dir1 in np.sort(os.listdir(DATAPATH)):
        class_name.append(dir1)
    return class_name


class_names = class_names("../Data/inaturalist_12K/test/")
target_dict = {k: v for v, k in enumerate(np.unique(class_names))} 
class_label_names_dict = {str(k): v for k, v in enumerate(np.unique(class_names))} 

# The dimensions of our input image
img_width,img_height = 128, 128

img_array, class_labels = load_dataset_batch("../Data/inaturalist_12K/test/", 128,128) 
img_array = np.array([img_array]).reshape(len(img_array), img_width, img_height,3)
class_labels_num = [target_dict[class_labels[i]] for i in range(len(class_labels))] 


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
            '../Data/inaturalist_12K/test',
            target_size=(img_width,img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle = False)
# Our target layer: we will visualize the filters from this layer.

#Model trained from scratch
source_model = keras.models.load_model("../TrainedModel/Best_Model") #Load the best trained model
#Transfer learned model
#source_model = keras.models.load_model("./TrainedModel/Best_TransferlearntModel") #Load the best trained model

# See `model.summary()` for list of layer names, if you want to change this.
source_model.summary()
layer_dict = {layer.name:layer for layer in source_model.layers}
print(layer_dict)

test_generator2 = test_datagen.flow_from_directory(
            '../Data/inaturalist_12K/test',
            target_size=(img_width,img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle = True, seed=1234)
            
#Test loss and accuracy on the shuffled test dataset            
history = source_model.evaluate(test_generator2)


#Confusion Matrix
pred_labels = source_model.predict(test_generator)
pred_labels_num = np.argmax(pred_labels, axis = 1)
cm = metrics.confusion_matrix(test_generator.classes, np.argmax(pred_labels, axis = 1))
#metrics.ConfusionMatrixDisplay(cm, display_labels = [0,1,2,3,4,5,6,7,8,9]).plot() 
#With label names:
metrics.ConfusionMatrixDisplay(cm, display_labels = test_generator.class_indices).plot() 
plt.show()



#Sample image predictions
ROWS = 3 
COLUMNS = 10  
ix = 1 
for i in range(ROWS): 
    for j in range(COLUMNS): 
        # specify subplot and turn of axis 
        idx = np.random.choice(len(test_generator[4*j][0])) 
        img = test_generator[4*j][0][idx] 
        ax = plt.subplot(ROWS, COLUMNS, ix) 
        ax.set_xticks([]) 
        ax.set_yticks([]) 
        # plot filter channel in grayscale 
        plt.imshow(img) 
        plt.xlabel(
                    "True: " + class_label_names_dict[str(np.argmax(test_generator[4*j][1][idx]))] +"\n" + "Pred: " + 
                    class_label_names_dict[str(np.argmax(source_model.predict(img.reshape(1,128,128,3))))]
                   )     
        ix += 1 
plt.show()
