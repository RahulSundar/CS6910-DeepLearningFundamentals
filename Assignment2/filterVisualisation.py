import numpy as np
import tensorflow as tf
from tensorflow import keras

import wandb
from wandb.keras import WandbCallback

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
#Invalid device or cannot modify virtual devices once initialized.
    pass


# build the VGG16 network
source_model = keras.models.load_model("./TrainedModel/Best_trained_Model/Best_trained_Model") #Load the best trained model

# get the symbolic outputs of each "key" layer (we gave them unique names).
source_model.summary()

# The dimensions of our input image
img_width,img_height = 128, 128


test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
            './Data/inaturalist_12K/test',
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle = True,
            seed = 123)
# Our target layer: we will visualize the filters from this layer.
# See `model.summary()` for list of layer names, if you want to change this.
layer_dict = {layer.name:layer for layer in model.layers}
print(layer_dict)
#Task: to  build a simple terminal user interface
#
layer_name = "conv2d" 

model = Sequential()
for layer in source_model.layers[:-2]: # go through until last layer
    model.add(layer)
#model.add(Dense(3, activation='softmax'))
model.summary()

layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)


layer = model.get_layer(name=layer_name)
filters, biases = layer.get_weights()
# normalize filter values to 0-1 so we can visualize them
f_min, f_max = filters.min(), filters.max()
filters = (filters - f_min) / (f_max - f_min)
# plot first few filters
n_filters, ix = 6, 1
for i in range(n_filters):
	# get the filter
	f = filters[:, :, :, i]
	# plot each channel separately
	for j in range(3):
		# specify subplot and turn of axis
		ax = pyplot.subplot(n_filters, 3, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(f[:, :, j], cmap='gray')
		ix += 1
# show the figure
pyplot.show()


i = 0 
for layer in model.layers: 
    if 'conv' not in layer.name: 
        continue 
    i += 1 
    globals()["filters"+ str(i)],globals()["biases"+str(i)] = layer.get_weights()
    
    
model2 = keras.Model(inputs=model.inputs, outputs=layer.output)
feature_maps = model2.predict(img) 


square = 8
ix = 1
for _ in range(square):
	for _ in range(square):
		# specify subplot and turn of axis
		ax = pyplot.subplot(square, square, ix)
		ax.set_xticks([])
		ax.set_yticks([])
		# plot filter channel in grayscale
		pyplot.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
		ix += 1
# show the figure
pyplot.show() 

