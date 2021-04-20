import os
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

import wandb
from wandb.keras import WandbCallback

#To let the gpu memory utilisiation grow as per requirement
physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass

#Default image size:
IMG_SIZE = (128, 128)
DATAPATH = "../Data/inaturalist_12K/test"
MODELPATH = "../TrainedModel/Best_Model"

#Functions to process images
def load_image(path, preprocess=True):
    """Load and rescale image."""
    img = image.load_img(path, target_size=IMG_SIZE)
    if preprocess:
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255
        print
    return img

def class_names(DATAPATH):
   
    class_name=[]
    
    for dir1 in np.sort(os.listdir(DATAPATH)):
        class_name.append(dir1)
    return class_name
  
  
def load_sample_image_from_all_classes(DATAPATH, IMG_HEIGHT, IMG_WIDTH): 
             
    img_data_array=[] 
    class_name=[] 

    for dir1 in np.sort(os.listdir(DATAPATH)): 
        if dir1[0] != ".":
            img_file = random.choice( np.sort(os.listdir(os.path.join(DATAPATH, dir1))))
            image_path= os.path.join(DATAPATH, dir1,  img_file) 
            img1= cv2.imread( image_path, cv2.COLOR_BGR2RGB) 
            img1= cv2.resize(img1, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA) 
            img1=np.array(img1) 
            img1 = img.astype('float32')
            #img1 /= 255
            img_data_array.append(img1) 
            class_name.append(dir1) 
    return img_data_array, class_name 
         
 

def deprocess_image(img):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    img = img.copy()
    img -= img.mean()
    img /= (img.std() + K.epsilon())
    img *= 0.25

    # clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB array
    img *= 255
    if K.image_data_format() == 'channels_first':
        img = img.transpose((1, 2, 0))
    img = np.clip(img, 0, 255).astype('uint8')
    return img
 
#Custom gradient function for guided backpropagation. Consists of guided relu function and its gradient.        
@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad

# Guided backpropagation on multiple images:
#------------------------------------------#

def guided_backpropagation(MODELPATH, num_sample_images = 10):

    # loading the data and the model
    model = tf.keras.models.load_model(MODELPATH)


    gb_model = Model(
        inputs = [model.inputs],
        outputs = [model.get_layer("conv2d_4").output]
    )
        
    for layer in model.layers:
        if hasattr(layer, 'activation') and layer.activation==tf.keras.activations.relu:
            layer.activation = guidedRelu

    # plotting the images
    fig, ax = plt.subplots(2, num_sample_images, figsize=(2*4, num_sample_images*4))
    sample_imgs = tf.convert_to_tensor(test_generator[0][0][:num_sample_images], dtype=tf.float32)
    sample_img_labels = np.array([np.argmax(test_generator[0][1][i]) for i in range(num_sample_images)])
    
    for i in range(num_sample_images):

        with tf.GradientTape() as tape:
            input_img = tf.expand_dims(sample_imgs[i], 0)
            tape.watch(input_img)
            output = gb_model(input_img)[0]
        
        gradients = tape.gradient(output,input_img)[0]

        ax[0][i].set_title("Sample Image")
        ax[0][i].imshow(sample_imgs[i])
        ax[0][i].set_xlabel(class_label_names_dict[str(sample_img_labels[i])])
        ax[1][i].set_title("Guided backprop")
        ax[1][i].imshow(deprocess_image(np.array(gradients)))
        #ax[1][i].imshow(gradients)
    plt.show()

    return model, gb_model



#Execution:
#----------#

class_names = class_names("../Data/inaturalist_12K/test/")
target_dict = {k: v for v, k in enumerate(np.unique(class_names))} 
class_label_names_dict = {str(k): v for k, v in enumerate(np.unique(class_names))} 


#Test data generator alternative for loading images:
#--------------------------------------------------#
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
            '../Data/inaturalist_12K/test',
            target_size=IMG_SIZE,
            batch_size=32,
            class_mode='categorical',
            shuffle = True,
            seed = 1234)

batch = np.random.choice(int(2000/32))
img_index = np.random.choice(32)    
img = test_generator[batch][0][img_index]
img = np.expand_dims(img, axis = 0)
img_true_label = test_generator[batch][1][img_index]

#plot the test image:
plt.figure()
plt.xlabel(class_label_names_dict[str(np.argmax(img_true_label))])
plt.imshow(img[0])
plt.show()


#Guided backprop testing on a sample image:
model = tf.keras.models.load_model(MODELPATH)

gb_model = Model(
    inputs = [model.inputs],
    outputs = [model.get_layer("conv2d_4").output]
)

#List of layers 
layer_list = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]

#Replace the relu activation layer with the guided activation layer
for layer in layer_list:
  if layer.activation == tf.keras.activations.relu:
    layer.activation = guidedRelu

#Observing the gradient flow for a chosen sample image:
with tf.GradientTape() as tape:
  inputs = tf.cast(img, tf.float32)
  tape.watch(inputs)
  outputs = gb_model(inputs)


grads = tape.gradient(outputs,inputs)[0]

plt.imshow(deprocess_image(np.array(grads))) 


#Just to visualise the backpropagated gradients on multiple sample images:
guided_backpropagation(MODELPATH)

