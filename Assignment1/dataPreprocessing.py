import numpy as np
import matplotlib.pyplot as plt
from  keras.datasets import fashion_mnist


#Load the data in predefined train and test split ratios:

(trainIn, trainOut), (testIn, testOut) = fashion_mnist.load_data()

#visualisation of the classes:
img_classes_fig = plt.figure("The image classes")
grid_specification = img_classes_fig.add_gridspec(2,5)

num_classes = int(np.max(trainOut) + 1)
ctr = 0
for i in range(num_classes):
    for j in range(trainIn.shape[0]):
        if trainOut[j] == i and i < 5:
            globals()["ax"+str(i+1)]=img_classes_fig.add_subplot(grid_specification[0,i]) 
            plt.imshow(trainIn[j])
            plt.label(str(i))
            break
        elif trainOut[j] == i and i >= 5:
            globals()["ax"+str(5 + i + 1)]=img_classes_fig.add_subplot(grid_specification[1,i-5]) 
            plt.imshow(trainIn[j])
            plt.label(str(i))
            break            

 
