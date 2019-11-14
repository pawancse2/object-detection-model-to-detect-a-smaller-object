import os
from os import listdir, walk
from os.path import join

import numpy as np
import keras
import math
import tensorflow as tf

from keras_retinanet.utils.visualization import draw_boxes
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from keras_retinanet.utils.gpu import setup_gpu
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from tqdm import tqdm
from keras_retinanet.bin.train import create_generators,create_models,create_callbacks
from keras_retinanet.models import backbone,load_model,convert_model
from keras_retinanet.utils.config import read_config_file,parse_anchor_parameters
from keras_retinanet.utils.visualization import draw_boxes
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import matplotlib.pyplot as plt
from keras_retinanet.utils.gpu import setup_gpu
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
b = backbone('resnet50')

class args:
    batch_size =4
    config = read_config_file('config.ini')
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
    epochs = 100
    steps = 10755//(batch_size)
    gpu=0  
    resize=True
    
train_gen,valid_gen = create_generators(args,b.preprocess_image)

model, training_model, prediction_model = create_models(
            backbone_retinanet=b.retinanet,
            num_classes=train_gen.num_classes(),
            weights=None,
            multi_gpu=True,
            freeze_backbone=True,
            lr=1e-9,
            config=args.config
        )
 
training_model.load_weights("C:\\Users\\Pawan\\Documents\\ML\\snapshots12\\resnet50_csv_07.h5")

infer_model = convert_model(training_model,anchor_params=parse_anchor_parameters(read_config_file('C:\\Users\\Pawan\\Documents\\config.ini')))


def test_gen(image_ids, bs = 2, size=672,test = True):
    imgs = []
    scale = None
    idx = 0
    if test:
        path = 'C:\\Users\\Pawan\\Downloads\\dataset_test_rgb\\rgb\\test\\'
    else:
        path = 'C:\\Users\\Pawan\\Downloads\\dataset_test_rgb\\rgb\\test\\'
    
    while idx < len(image_ids):
        if len(imgs) < bs:
            imgs.append(resize_image(preprocess_image(read_image_bgr(path + image_ids[idx] + '.png')),min_side=size,max_side=size)[0])            
            if scale is None:
                scale = resize_image(preprocess_image(read_image_bgr(path + image_ids[idx] + '.png')),min_side=size,max_side=size)[1]
            idx += 1
        else:
            yield np.array(imgs),scale
            imgs = []
            
            
    if len(imgs) > 0:
        yield np.array(imgs),scale


print("###########################################################################################")       
_,_,image_ids = next(walk('C:\\Users\\Pawan\\Downloads\\dataset_test_rgb\\rgb\\test\\'))
image_ids = [i[:-4] for i in image_ids]
image_ids = sorted(image_ids)
#print("image_ids",image_ids)

iter_num = 0
test_bs = 2


for imgs,scale in tqdm(test_gen(image_ids,bs=test_bs),total=math.ceil(len(image_ids)/test_bs)):
    boxes, scores, labels = infer_model.predict_on_batch(imgs)
    boxes /= scale
    for img_num in range(len(imgs)):
        with open('C:\\Users\\Pawan\\Music\\Paku5\\' + image_ids[(iter_num*test_bs) + img_num] + '.txt', 'w') as f:
            for box, score, label in zip(boxes[img_num], scores[img_num], labels[img_num]):
                # scores are sorted so we can break
                if score < 0:
                    break
                f.write(f'{label + 1} {score} {int((box[1]))} {int((box[0]))} {int((box[3]))} {int((box[2]))} \n')
    iter_num += 1        

  