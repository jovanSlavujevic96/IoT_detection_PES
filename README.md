# IoT_detection_PES

This project is intended for master studies of FTN - Faculty of Technical Sciences. The Subject is PES - Projecting of Electonic Systems. 

In this project has been implemented human face detection on frame via opencv's cascade classifier, afterwards face recognition with tensorflow's detection model trained with tensorflow object detection API and alcohol detection with sensor MQ-2. Detection informations have been published to FTN's Cloud server made in cooperation with Wolkabout. Project is written in Python programming language, developed on Ubuntu OS environment on PC, afterwards on Raspberry Pi.

### 1.download and extract this library to $HOME directory : 
### https://github.com/tensorflow/models/releases/tag/v1.12.0

### 2.if you have nvidia gpu and want to use tensorflow-gpu follow this script to install prerequisites:
### https://www.tensorflow.org/install/gpu#ubuntu_1804_cuda_101 
```
P.S. instead of 10.1 install cuda 10.0 (tensorflow 1.15 supports this version of cuda)
on .bashrc set environment variables:
PATH=$PATH:/usr/local/cuda-10.0/bin
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
```
### 3.follow steps of installation of tensorflow with python3 and virtualenv:
FOR THIS PROJECT:$ pip install tensorflow-gpu==1.15.0 
## https://www.tensorflow.org/install/pip

### 4.steps to initialise and connect models/research libraries:
### https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

### 5.go to: $HOME/models-1.12.0/research/object_detection/utils/visualization_utils.py
### change definition of function: visualize_boxes_and_labels_on_image_array (line 543), go to line 675 and instead of:
```
return image
```
### paste this lines:
```
   try:  
     return (image, box_to_display_str_map[box])
   except NameError:
     return (image, '/')
```
