import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

import os

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


def load_image(path, preprocess=True):
    """Load and rescale image."""
    img = image.load_img(path, target_size=IMG_SIZE)
    if preprocess:
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img/255
        print
    return img
 
def deprocess_image(img):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    img = img.copy()
    img = img -  img.mean()
    img /= (img.std() + K.epsilon())
    img = img* 0.25

    # clip to [0, 1]
    img = img + 0.5
    img = np.clip(img, 0, 1)

    # convert to RGB array
    img =img* 255
    if K.image_data_format() == 'channels_first':
        img = img.transpose((1, 2, 0))
    img = np.clip(img, 0, 255).astype('uint8')
    return img
        
@tf.custom_gradient
def guidedRelu(x):
  def grad(dy):
    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
  return tf.nn.relu(x), grad




def class_names(DATAPATH):
   
    class_name=[]
    
    for dir1 in np.sort(os.listdir(DATAPATH)):
        class_name.append(dir1)
    return class_name
    
class_names = class_names("./Data/inaturalist_12K/test/")
target_dict = {k: v for v, k in enumerate(np.unique(class_names))} 
class_label_names_dict = {str(k): v for k, v in enumerate(np.unique(class_names))} 

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
            './Data/inaturalist_12K/test',
            target_size=IMG_SIZE,
            batch_size=32,
            class_mode='categorical',
            shuffle = True)

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

model = keras.models.load_model("./TrainedModel/Best_Model")

gb_model = Model(
    inputs = [model.inputs],
    outputs = [model.get_layer("conv2d_4").output]
)


layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer,'activation')]

for layer in layer_dict:
  if layer.activation == tf.keras.activations.relu:
    layer.activation = guidedRelu

#preprocessed_input = load_image()
with tf.GradientTape() as tape:
  inputs = tf.cast(img, tf.float32)
  tape.watch(inputs)
  outputs = gb_model(inputs)

grads = tape.gradient(outputs,inputs)[0]

plt.imshow(deprocess_image(np.array(grads))) 
