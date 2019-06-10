#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from imgaug import augmenters as iaa
from keras import backend as K
import pandas as pd
import numpy as np


# In[ ]:


seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Sometimes(0.5,
        iaa.Crop(percent=(0, 0.1))
    ),
    iaa.Sometimes(0.5,
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5)
    ),
    iaa.Sometimes(0.5,
        iaa.Multiply((0.8, 1.2), per_channel=0.2)
    ),
    iaa.Sometimes(0.5,
        iaa.Affine(
            rotate=(-20, 20),
            shear=(-.2, .2),
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            mode = "edge")
    ),
], random_order=True)


# In[ ]:


submit = pd.read_csv("sample_submission_godl.csv")


# In[ ]:


import cv2
def image_resize(image, side = None, inter = cv2.INTER_AREA):
    dim = None
    h, w = image.shape[:2]
    if h < w:
        r = side/ float(h)
        dim = (int(w * r), side)
    else:
        r = side / float(w)
        dim = (side ,int(h * r))
    return cv2.resize(image, dim, interpolation = inter)

def randomCrop(img, shape=224):
    x = np.random.randint(0, img.shape[1] - shape)
    y = np.random.randint(0, img.shape[0] - shape)
    img = img[y:y+shape, x:x+shape]
    return img
def center(img,crop):
    y,x,_ = img.shape
    startx = x//2-(crop//2)
    starty = y//2-(crop//2)    
    return img[starty:starty+crop,startx:startx+crop]

def top_left(img,crop):
    return img[0:crop,0:crop]

def top_right(img,crop):
    y,x,_ = img.shape
    startx = x-crop
    return img[0:crop,startx:x]

def bottom_right(img,crop):
    y,x,_ = img.shape
    startx = x-crop
    starty = y-crop
    return img[starty:y,startx:x]

def bottom_left(img,crop):
    y,x,_ = img.shape
    starty = y-crop
    return img[starty:y,0:crop]


# In[ ]:


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# In[ ]:


files_Xception = ["GODlXception.val_loss= 0.159.val-f1=0.965.h5",
        "GODlXception.val_loss= 0.174.val-f1=0.961.h5",
        "GODlXception.val_loss= 0.185.val-f1=0.959.h5",
        "GODlXception.val_loss= 0.173.val-f1=0.963.h5",
        "GODlXception.val_loss= 0.134.val-f1=0.967.h5"]


# In[ ]:


import math
import numpy as np
from keras.models import load_model
from keras.applications.xception import preprocess_input
from cv2 import imread
sample_x_all = np.zeros((len(submit), 5))

for file in files_Xception:
    model = load_model(file, custom_objects={'f1':f1})
    count = 0
    sample_x = np.zeros((len(submit), 5))
    step = 25 / len(submit)
    print("\nSTARTED",file.split("/")[-1])
    for j, i in enumerate(submit.image):
        count+=1
        img_or = imread("images/"+i)
        img_or = cv2.cvtColor(img_or, cv2.COLOR_BGR2RGB)
        img = image_resize(img_or, 256)

        center_ = center(img, 224)
        top_left_ = top_left(img, 224)
        top_right_ = top_right(img, 224)
        bottom_right_ = bottom_right(img, 224)
        bottom_left_ = bottom_left(img, 224)

        sample_x[j]+= model.predict(preprocess_input(center_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(top_left_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(top_right_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(bottom_right_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(bottom_left_)[None,...]).squeeze()

        for _ in range(10):
            img = seq.augment_image(img_or)
            img_scale1 = image_resize(img, 256)
            img_scale2 = image_resize(img, 278)

            img_scale1 = randomCrop(img_scale1)
            img_scale2 = randomCrop(img_scale2)

            sample_x[j]+= model.predict(preprocess_input(img_scale1)[None,...]).squeeze()
            sample_x[j]+= model.predict(preprocess_input(img_scale2)[None,...]).squeeze()

        print('\r' + f'Progress: '
        f"[{'=' * int((count) * step) + ' ' * (24 - int((count) * step))}]"
        f"({math.ceil((count) * 100 /len(submit))} %)",
        end='')
    sample_x_all += sample_x
    np.save(file.split("/")[-1][:-3]+".npy",sample_x)


# In[ ]:


files_ResNet101 = [ "GODlResNet101.val_loss= 0.166.val-f1=0.966.h5",
                    "GODlResNet101.val_loss= 0.203.val-f1=0.961.h5",
                    "GODlResNet101.val_loss= 0.183.val-f1=0.961.h5",
                    "GODlResNet101.val_loss= 0.192.val-f1=0.958.h5",
                    "GODlResNet101.val_loss= 0.143.val-f1=0.968.h5"]


# In[ ]:


import math
import numpy as np
from keras.models import load_model
from keras.applications.resnet import preprocess_input
from cv2 import imread
sample_x_all = np.zeros((len(submit), 5))

for file in files:
    model = load_model(file, custom_objects={'f1':f1})
    count = 0
    sample_x = np.zeros((len(submit), 5))
    step = 25 / len(submit)
    print("\nSTARTED",file.split("/")[-1])
    for j, i in enumerate(submit.image):
        count+=1
        img_or = imread("images/"+i)
        img_or = cv2.cvtColor(img_or, cv2.COLOR_BGR2RGB)
        img = image_resize(img_or, 256)

        center_ = center(img, 224)
        top_left_ = top_left(img, 224)
        top_right_ = top_right(img, 224)
        bottom_right_ = bottom_right(img, 224)
        bottom_left_ = bottom_left(img, 224)

        sample_x[j]+= model.predict(preprocess_input(center_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(top_left_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(top_right_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(bottom_right_)[None,...]).squeeze()
        sample_x[j]+= model.predict(preprocess_input(bottom_left_)[None,...]).squeeze()

        for _ in range(10):
            img = seq.augment_image(img_or)
            img_scale1 = image_resize(img, 256)
            img_scale2 = image_resize(img, 278)

            img_scale1 = randomCrop(img_scale1)
            img_scale2 = randomCrop(img_scale2)

            sample_x[j]+= model.predict(preprocess_input(img_scale1)[None,...]).squeeze()
            sample_x[j]+= model.predict(preprocess_input(img_scale2)[None,...]).squeeze()

        print('\r' + f'Progress: '
        f"[{'=' * int((count) * step) + ' ' * (24 - int((count) * step))}]"
        f"({math.ceil((count) * 100 /len(submit))} %)",
        end='')
    sample_x_all += sample_x
    np.save(file.split("/")[-1][:-3]+".npy",sample_x)


# In[ ]:


pred_ = np.argmax(sample_x_all, axis = -1)
submit.category = pred_+1
submit.to_csv("submitFinalSubmit.csv", index = False)

