import tensorflow as tf
import numpy as  np

from tensorflow.keras import datasets, layers, models
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt

#Setting memory growth inorder to avoid OOM or resource exhausted error. This prevents complete usage of the GPU memory. 
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass



(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

K.clear_session()
model = models.Sequential()

#First convolutional layer:
model.add(layers.Conv2D(filters = 64, kernel_size = 3,padding = 'same', activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D(pool_size = 2))
#Second Convolutional layer:
model.add(layers.Conv2D(filters = 32, kernel_size =3,padding = 'same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size = 2))
model.add(tf.keras.layers.Dropout(0.3))
#Third convolutional layer:
model.add(layers.Conv2D(filters = 16, kernel_size =3,padding = 'same', activation='relu'))
model.add(layers.MaxPooling2D(pool_size = 2))
model.add(tf.keras.layers.Dropout(0.3))
# Densely connected layer:
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

#Print the model summary with the total number of parameters
model.summary()



model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


history = model.fit(train_images, train_labels, batch_size = 32,epochs=10,  validation_data=(test_images, test_labels))
