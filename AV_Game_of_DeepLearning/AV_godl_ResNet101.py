#!/usr/bin/env python
# coding: utf-8

# ## Installing libs

# In[ ]:


get_ipython().system("unzip 'train.zip'")


# In[ ]:


get_ipython().system('pip install -U git+https://github.com/keras-team/keras git+https://github.com/keras-team/keras-applications')


# ## Importing Libs

# In[ ]:


from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
from keras.utils import to_categorical
from sklearn.utils import class_weight
from keras.applications.resnet import ResNet101, preprocess_input
from keras.layers import *
from keras.models import Model
from keras.utils import Sequence
from cv2 import imread
from keras import regularizers
from keras import backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import keras
import cv2


# ## Helping Functions

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
  
def augment_brightness(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1
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


# ## Loading Data

# In[ ]:


train = pd.read_csv("train.csv")
train.head()


# In[ ]:


train.category.value_counts()


# In[ ]:


y = to_categorical(train.category-1)
X_train, X_test, y_train, y_test = train_test_split(train.image.values.tolist(), y, test_size=0.2, random_state=42)


# ## Perform oversampling

# In[ ]:


class_examples = [1702, 941, 732, 665, 961]
max_sample = 1702
for i in range(1,5):
    samples_generate = max_sample-class_examples[i]
    random_generate = np.random.choice(class_examples[i],samples_generate)
    for j in random_generate:
        X_train = np.append(X_train,np.array([X_train[j]]), axis=0)
        y_train =  np.append(y_train,np.array([y_train[j]]), axis=0)


# ## Defining Model

# In[ ]:


base_model = ResNet101(input_shape=(224,224,3),weights='imagenet', include_top=False)

x = base_model.output
x = MaxPool2D()(x)
x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.1))(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.1))(x)
x = Dropout(0.5)(x)
pre = Dense(5, activation='softmax')(x)

model = Model(inputs=[base_model.input],outputs=[pre])
model.summary()


# ## Data Generator

# In[ ]:


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X_id,y, batch_size=256, train = True):
        'Initialization'
        self.batch_size = batch_size
        self.X_id = X_id
        self.y = y
        self.train = train

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_id) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X_id_batch = self.X_id[index*self.batch_size:(index+1)*self.batch_size]
        y_batch = self.y[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X = []
        y = []
        for x_p,y_p in zip(X_id_batch,y_batch):
            img_or = imread("images/"+x_p)
            img_or = cv2.cvtColor(img_or, cv2.COLOR_BGR2RGB)

            p = np.random.rand()

            if self.train:
                img_or = seq.augment_image(img_or)

            if p> 0.8:
                img_or = cv2.cvtColor(img_or, cv2.COLOR_RGB2GRAY)
                img_or = cv2.cvtColor(img_or, cv2.COLOR_GRAY2RGB)

            scale = np.random.randint(256, 300)
            img = image_resize(img_or, scale)

            if self.train:
                center_ = center(img, 224)
                top_left_ = top_left(img, 224)
                top_right_ = top_right(img, 224)
                bottom_right_ = bottom_right(img, 224)
                bottom_left_ = bottom_left(img, 224)

                X.append(preprocess_input(center_))
                X.append(preprocess_input(top_left_))
                X.append(preprocess_input(top_right_))
                X.append(preprocess_input(bottom_right_))
                X.append(preprocess_input(bottom_left_))
                y.append(y_p)
                y.append(y_p)
                y.append(y_p)
                y.append(y_p)
                y.append(y_p)
            else:
                img_ = randomCrop(img, 224)
                X.append(preprocess_input(img_))
                y.append(y_p)
        return np.array(X), np.array(y)


# ## Training model

# ### Training with default lr

# In[ ]:


model.compile("sgd", ["categorical_crossentropy"], [f1,"acc"])


# In[ ]:


training_generator = DataGenerator(X_train, y_train, batch_size=32, train=False)
validation_generator = DataGenerator(X_test, y_test, batch_size=16, train=False)


# In[ ]:


model.fit_generator(training_generator,
                    epochs=15,
                    validation_data=validation_generator,
                    class_weight=class_weights,
                    callbacks=[keras.callbacks.ModelCheckpoint(
                      'GODlResNet101.val_loss={val_loss: .3f}.val-f1={val_f1:.3f}.h5',
                      monitor='val_f1',
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max',
                      period=1
      )])


# In[ ]:


training_generator = DataGenerator(X_train, y_train, batch_size=10, train=True)
validation_generator = DataGenerator(X_test, y_test, batch_size=16, train=False)


# In[ ]:


model.fit_generator(training_generator,
                    epochs=30,
                    initial_epoch=15,
                    validation_data=validation_generator,
                    class_weight=class_weights,
                    callbacks=[keras.callbacks.ModelCheckpoint(
                      'GODlResNet101.val_loss={val_loss: .3f}.val-f1={val_f1:.3f}.h5',
                      monitor='val_f1',
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max',
                      period=1
      )])


# ### Training with lr = 0.001

# In[ ]:


sgd = keras.optimizers.SGD(lr=0.001)
model.compile(sgd, ["categorical_crossentropy"], [f1,"acc"])


# In[ ]:


model.fit_generator(training_generator,
                    epochs=45,
                    initial_epoch=30,
                    validation_data=validation_generator,
                    class_weight=class_weights,
                    callbacks=[keras.callbacks.ModelCheckpoint(
                      'GODlResNet101.val_loss={val_loss: .3f}.val-f1={val_f1:.3f}.h5',
                      monitor='val_f1',
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max',
                      period=1
      )])


# In[ ]:


training_generator = DataGenerator(X_train, y_train, batch_size=32, train=False)
validation_generator = DataGenerator(X_test, y_test, batch_size=16, train=False)


# In[ ]:


model.fit_generator(training_generator,
                    epochs=60,
                    initial_epoch=45,
                    validation_data=validation_generator,
                    class_weight=class_weights,
                    callbacks=[keras.callbacks.ModelCheckpoint(
                      'GODlResNet101.val_loss={val_loss: .3f}.val-f1={val_f1:.3f}.h5',
                      monitor='val_f1',
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max',
                      period=1
      )])


# ### Training with lr = 0.001

# In[ ]:


sgd = keras.optimizers.SGD(lr=0.0001)
model.compile(sgd, ["categorical_crossentropy"], [f1,"acc"])


# In[ ]:


model.fit_generator(training_generator,
                    epochs=60,
                    initial_epoch=45,
                    validation_data=validation_generator,
                    class_weight=class_weights,
                    callbacks=[keras.callbacks.ModelCheckpoint(
                      'GODlResNet101.val_loss={val_loss: .3f}.val-f1={val_f1:.3f}.h5',
                      monitor='val_f1',
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max',
                      period=1
      )])


# In[ ]:


training_generator = DataGenerator(X_train, y_train, batch_size=10, train=True)
validation_generator = DataGenerator(X_test, y_test, batch_size=16, train=False)


# In[ ]:


model.fit_generator(training_generator,
                    epochs=75,
                    initial_epoch=60,
                    validation_data=validation_generator,
                    class_weight=class_weights,
                    callbacks=[keras.callbacks.ModelCheckpoint(
                      'GODlResNet101.val_loss={val_loss: .3f}.val-f1={val_f1:.3f}.h5',
                      monitor='val_f1',
                      save_best_only=True,
                      save_weights_only=False,
                      mode='max',
                      period=1
      )])

