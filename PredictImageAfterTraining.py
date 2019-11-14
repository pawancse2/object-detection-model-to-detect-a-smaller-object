from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
import cv2
from os import walk
import matplotlib.pyplot as plt

_,_,image_ids = next(walk('C:\\Users\\Pawan\\Downloads\\dataset_test_rgb\\rgb\\test\\'))
image_ids = [i[:-4] for i in image_ids]
image_ids = sorted(image_ids)
image_ids=["28164"]
idx = 0
image_id = 1
score_thres = 0.1

for id in image_ids:
    # load image
    #idx += 1
    #if idx == image_id:
        image = read_image_bgr('C:\\Users\\Pawan\\Downloads\\dataset_test_rgb\\rgb\\test\\' + id + '.png')

        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # process image
        boxes = [list(map(int,(line.split()[3],line.split()[2],line.split()[5],line.split()[4]))) for line in open('C:\\Users\\Pawan\\Music\\Paku5\\' + id + '.txt','r').readlines()]
        scores = [float(line.split()[1]) for line in open('C:\\Users\\Pawan\\Music\\Paku5\\' + id + '.txt','r').readlines()]
        labels = [int(line.split()[0]) - 1 for line in open('C:\\Users\\Pawan\\Music\\Paku5\\' + id + '.txt','r').readlines()]
        print("scores is :::::",scores)
        for box, score, label in zip(boxes, scores, labels):
            if score < score_thres:
                break
            color = label_color(label)
            draw_box(draw, box, color=color,thickness=1)
            caption = "{:.3f}".format(score)
            draw_caption(draw, box, caption)

        plt.figure(figsize=(15, 15))
        plt.axis('off')
        plt.imshow(draw)
        plt.show()
        break