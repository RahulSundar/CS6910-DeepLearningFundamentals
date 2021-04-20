import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow import keras

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


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
            '../Data/inaturalist_12K/test',
            target_size=(img_width,img_height),
            batch_size=32,
            class_mode='categorical',
            shuffle = False)
# Our target layer: we will visualize the filters from this layer.

source_model = keras.models.load_model("../TrainedModel/Best_Model") #Load the best trained model

# See `model.summary()` for list of layer names, if you want to change this.
source_model.summary()
layer_dict = {layer.name:layer for layer in source_model.layers}
print(layer_dict)
#Task: to  build a simple terminal user interface


#1st Convolutional layer:
layer_name = "conv2d" 
activation_layer_name = "activation"

layer = source_model.get_layer(name=layer_name)
activation_layer = source_model.get_layer(name=activation_layer_name)

feature_extractor = keras.Model(inputs=source_model.inputs, outputs=layer.output)
feature_extractor_activation = keras.Model(inputs=source_model.inputs, outputs=activation_layer.output)


filters, biases = layer.get_weights()


# normalize filter values to 0-1 so we can visualize them
filter_min, filter_max = filters.min(), filters.max()
filters = (filters - filter_min) / (filter_max - filter_min)


# plot first few filters
n_filters, ix = 32, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = plt.subplot( 3, n_filters, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
plt.show()

#To get filters of all conv layers uncomment the following:
'''
i = 0 
for layer in model.layers: 
    if 'conv' not in layer.name: 
        continue 
    i += 1 
    globals()["filters"+ str(i)],globals()["biases"+str(i)] = layer.get_weights()
'''    

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



#Extract the feature maps and feature activation maps
feature_maps = feature_extractor(img) 
feature_maps_activation = feature_extractor_activation(img) 

#32 filters in the first layer:
ROWS = 4
COLUMNS = 8 
ix = 1
for _ in range(ROWS):
	for _ in range(COLUMNS):
		# specify subplot and turn of axis
		ax = plt.subplot(ROWS, COLUMNS, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
plt.show() 

ROWS = 4
COLUMNS = 8 
ix = 1
for _ in range(ROWS):
	for _ in range(COLUMNS):
		# specify subplot and turn of axis
		ax = plt.subplot(ROWS, COLUMNS, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		plt.imshow(feature_maps_activation[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
plt.show()
