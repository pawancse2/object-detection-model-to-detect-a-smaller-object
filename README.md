# object-detection-model-to-detect-a-smaller-object
Development of object detection model to detect a smaller object/s in a given image.

Introduction 

Keras implementation of RetinaNet object detection as described in Focal Loss for Dense Object Detection by Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollár(https://github.com/fizyr/keras-retinanet)

RetinaNet has been formed by making two improvements over existing single stage object detection models (like YOLO and SSD):
Feature Pyramid Networks for Object Detection
Focal Loss for Dense Object Detection

Feature Pyramid Network
Pyramid networks have been used conventionally to identify objects at different scales. A Feature Pyramid Network (FPN) makes use of the inherent multi-scale pyramidal hierarchy of deep CNNs to create feature pyramids.

The one-stage RetinaNet network architecture uses a Feature Pyramid Network (FPN) backbone on top of a feedforward ResNet architecture (a) to generate a rich, multi-scale convolutional feature pyramid (b). To this backbone RetinaNet attaches two subnetworks, one for classifying anchor boxes (c) and one for regressing from anchor boxes to ground-truth object boxes (d). The network design is intentionally simple, which enables this work to focus on a novel focal loss function that eliminates the accuracy gap between our one-stage detector and state-of-the-art two-stage detectors like Faster R-CNN with FPN while running at faster speeds.
Focal Loss
Focal Loss is an improvement on cross-entropy loss that helps to reduce the relative loss for well-classified examples and putting more focus on hard, misclassified examples.
The focal loss enables training highly accurate dense object detectors in the presence of vast numbers of easy background examples.

![Image of Yaktocat]("C:\\Users\\Pawan\\Downloads\\dataset_test_rgb\\rgb\\test\\24068.png")






