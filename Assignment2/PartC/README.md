# Contents:
Trained weights of the fine tuned YOLOV5s model. 
The pretrianed models are published by ultralytics in their [repository](https://github.com/ultralytics/yolov5).

Data sets were downloaded from the publicly available datasets of Roboflow.

# Instructions:

In order to fine tune the model on your custom dataset, do the following:
1. Dowload and convert the dataset into the YOLOV5 format. (1. jpg images, 2. txt labels, 3 data.yaml file describing the class names and number of classes, location of training and test data.)
2. Download the weights of the required pretained model.
3. Modify the config.yaml of the chosen pretrained model to adapt to the number of classes in your custom dataset. For example, see the config file for yolov5s that was used for mask detection. 2 classes because 1. No mask, 2. Mask

```
# parameters
nc: 2  # number of classes
depth_multiple: 0.33  # model depth multiple
width_multiple: 0.50  # layer channel multiple

# anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5 backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, C3, [1024, False]],  # 9
  ]

# YOLOv5 head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13

   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3/8-small)

   [-1, 1, Conv, [256, 3, 2]],
   [[-1, 14], 1, Concat, [1]],  # cat head P4
   [-1, 3, C3, [512, False]],  # 20 (P4/16-medium)

   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 10], 1, Concat, [1]],  # cat head P5
   [-1, 3, C3, [1024, False]],  # 23 (P5/32-large)

   [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
  ]
```
4. To train the model on this data set you could either choose to fine tune the model further following the instructions in the README file in the official repository of [YOLOV5](https://github.com/ultralytics/yolov5). Number of epochs, type of learning rate scheduling, etc can be chosen as your arguments to the command line while submitting the training script. 
5. Once trained, use the detect script as given in the official repository to visualise the results for a test case. 


