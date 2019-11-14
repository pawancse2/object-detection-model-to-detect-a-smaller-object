# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 00:25:16 2019

@author: Pawan
"""




from __future__ import absolute_import, division, print_function, unicode_literals
import os

import tensorflow as tf

import cProfile
import numpy as np
import xml.etree.ElementTree as ET
import os

import keras
import math

import sys
import cv2
from os import listdir, walk
from os.path import join
from keras_retinanet.bin.train import create_generators,create_models,create_callbacks
from keras_retinanet.models import backbone,load_model,convert_model
from keras_retinanet.utils.config import read_config_file,parse_anchor_parameters
from keras_retinanet.utils.visualization import draw_boxes
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from keras_retinanet.utils.gpu import setup_gpu

print("eagerly value is@@@@@@@",tf.executing_eagerly())
print("tensorflow version is",tf.__version__)

with open('config.ini','w') as f:
    f.write('[anchor_parameters]\nsizes   = 16 32 64 128 256 512\nstrides = 8 16 32 64 128\nratios  = 0.001 0.1 0.442 0.476 0.5 1.0 2.102 2.26 3 4\nscales  =0.1 0.2 0.3 0.4 0.498 0.506 0.625 0.639 1 1.2 1.6 1.8\n')

 #############################################################################################   
#Ratios: [0.442, 1.0, 2.261] 0.442 1.0 2.261
#Scales: [0.4, 0.498, 0.625] 0.4 0.498 0.625
#b = backbone('resnet101')



b = backbone('resnet50')

class args:
    batch_size =2
    #64
    config = read_config_file('config1111.ini')
    #config=True
    random_transform = True # Image augmentation
    annotations = "C:\\Users\\Pawan\\Documents\\ML\\annotations_train_modified2.csv"
    val_annotations = "C:\\Users\\Pawan\\Documents\\ML\\annotations_test_modified2.csv"
    no_resize=False
    classes = "C:\\Users\\Pawan\\Documents\\ML\\classes_train_modified2.csv"
    image_min_side = 672
    image_max_side = 672
    dataset_type = 'csv'
    tensorboard_dir = 'C:\\Users\\Pawan\\Documents\\Tensorboard'
    evaluation = True
    snapshots = True
    snapshot_path = "C:\\Users\\Pawan\\Documents\\ML\\snapshots12"
    backbone = 'resnet50'
    #epochs = 70
    epochs = 70
    steps = 10755//(batch_size)
    weighted_average = True
    gpu=0  
    resize=True
    
    
train_gen,valid_gen = create_generators(args,b.preprocess_image)

###################################################################
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# Define our sequence of augmentation steps that will be applied to every image.
seq = iaa.Sequential(
    [
        #
        # Execute 1 to 9 of the following (less important) augmenters per
        # image. Don't execute all of them, as that would often be way too
        # strong.
        #
        iaa.SomeOf((1, 9),
            [

                        # Blur each image with varying strength using
                        # gaussian blur (sigma between 0 and .5),
                        # average/uniform blur (kernel size 1x1)
                        # median blur (kernel size 1x1).
                        iaa.OneOf([
                            iaa.GaussianBlur((0,0.5)),
                            iaa.AverageBlur(k=(1)),
                            iaa.MedianBlur(k=(1)),
                        ]),

                        # Sharpen each image, overlay the result with the original
                        # image using an alpha between 0 (no sharpening) and 1
                        # (full sharpening effect).
                        iaa.Sharpen(alpha=(0, 0.25), lightness=(0.75, 1.5)),

                        # Add gaussian noise to some images.
                        # In 50% of these cases, the noise is randomly sampled per
                        # channel and pixel.
                        # In the other 50% of all cases it is sampled once per
                        # pixel (i.e. brightness change).
                        iaa.AdditiveGaussianNoise(
                            loc=0, scale=(0.0, 0.01*255), per_channel=0.5
                        ),

                        # Either drop randomly 1 to 10% of all pixels (i.e. set
                        # them to black) or drop them on an image with 2-5% percent
                        # of the original size, leading to large dropped
                        # rectangles.
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5),
                            iaa.CoarseDropout(
                                (0.03, 0.15), size_percent=(0.02, 0.05),
                                per_channel=0.2
                            ),
                        ]),

                        # Add a value of -5 to 5 to each pixel.
                        iaa.Add((-5, 5), per_channel=0.5),

                        # Change brightness of images (85-115% of original value).
                        iaa.Multiply((0.85, 1.15), per_channel=0.5),

                        # Improve or worsen the contrast of images.
                        iaa.ContrastNormalization((0.75, 1.25), per_channel=0.5),

                        # Convert each image to grayscale and then overlay the
                        # result with the original with random alpha. I.e. remove
                        # colors with varying strengths.
                        iaa.Grayscale(alpha=(0.0, 0.25)),

                        # In some images distort local areas with varying strength.
                        sometimes(iaa.PiecewiseAffine(scale=(0.001, 0.01)))
                    ],
            # do all of the above augmentations in random order
            random_order=True
        )
    ],
    # do all of the above augmentations in random order
    random_order=True
)

#######################################################################################

def augment_train_gen(train_gen,visualize=False):
    '''
    Creates a generator using another generator with applied image augmentation.
    Args
        train_gen  : keras-retinanet generator object.
        visualize  : Boolean; False will convert bounding boxes to their anchor box targets for the model.
    '''
    imgs = []
    boxes = []
    targets = []
    size = train_gen.size()
    idx = 0
    while True:
        while len(imgs) < args.batch_size:
            image       = train_gen.load_image(idx % size)
            annotations = train_gen.load_annotations(idx % size)
            image,annotations = train_gen.random_transform_group_entry(image,annotations)
            imgs.append(image)            
            boxes.append(annotations['bboxes'])
            targets.append(annotations)
            idx += 1
        if visualize:
            imgs = seq.augment_images(imgs)
            imgs = np.array(imgs)
            boxes = np.array(boxes)
            yield imgs,boxes
        else:
            imgs = seq.augment_images(imgs)
            imgs,targets = train_gen.preprocess_group(imgs,targets)
            imgs = train_gen.compute_inputs(imgs)
            targets = train_gen.compute_targets(imgs,targets)
            imgs = np.array(imgs)
            yield imgs,targets
        imgs = []
        boxes = []
        targets = []     
##################################################################  

skip_batches = 5

'''
i = 0

for imgs,boxes in augment_train_gen(train_gen,visualize=True):
    if i > skip_batches:
        fig=plt.figure(figsize=(24,96))
        columns = 2
        rows = 8
        for i in range(1, columns*rows + 1):
            draw_boxes(imgs[i], boxes[i], (0, 255, 0), thickness=1)
            fig.add_subplot(rows, columns, i)
            plt.imshow(cv2.cvtColor(imgs[i],cv2.COLOR_BGR2RGB))
        plt.show()
        
    else:
        i += 1      
sys.exit()
'''
#######################################################################
model, training_model, prediction_model = create_models(
            backbone_retinanet=b.retinanet,
            num_classes=train_gen.num_classes(),
            weights=None,
            multi_gpu=True,
            freeze_backbone=True,
            lr=1e-3,
            config=args.config
        )
print("in create models@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
############################################################################# 

callbacks = create_callbacks(
    model,
    training_model,
    prediction_model,
    valid_gen,
    args,
)
print("callbacks@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#############################################################################
#training_model.
#training_model.load_weights('C:\\Users\\Pawan\\Documents\\resnet50_coco_best_v2.1.0.h5',skip_mismatch=True,by_name=True)
training_model.load_weights('C:\\Users\\Pawan\\Documents\\ML\\snapshots11\\resnet50_csv_01.h5',skip_mismatch=True,by_name=True)
print("training model*************************************************************")

############################################################################################

training_model.fit_generator(generator=augment_train_gen(train_gen),
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,)
print("augumenttrain@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
#############################################################################################       