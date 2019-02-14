'''
Training was done on TPU
'''
import os
import tensorflow as tf
import pandas as pd
from skimage import io
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from skimage.transform import resize
from sklearn.model_selection import train_test_split

train = pd.read_csv('training.csv')
X = train.image_name.values
y = train.drop("image_name",axis =1).values
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.25)

def iou(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, x2, y1, y2)
    box2 -- second box, list object with coordinates (x1, x2, y1, y2)
    """

    # Calculate the (x1, x2, y1, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    xi1 = K.maximum(box1[:, 0], box2[:, 0])
    xi2 = K.minimum(box1[:, 1], box2[:, 1])
    yi1 = K.maximum(box1[:, 2], box2[:, 2])
    yi2 = K.minimum(box1[:, 3], box2[:, 3])

    inter_area = K.maximum(xi2-xi1, 0) * K.maximum(yi2-yi1, 0)

    # Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    box1_area = (box1[:, 3] - box1[:, 2]) * (box1[:, 1] - box1[:, 0])
    box2_area = (box2[:, 3] - box2[:, 2]) * (box2[:, 1] - box2[:, 0])
    union_area = (box1_area + box2_area - inter_area)
    iou = inter_area / union_area

    return K.mean(iou)
def iou_loss(box1, box2):
    """
    Implement iou loss between box1 and box2

    Arguments:
    box1 -- first box, list object with coordinates (x1, x2, y1, y2)
    box2 -- second box, list object with coordinates (x1, x2, y1, y2)
    """

    return -iou(box1, box2)

def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x
def transition(x, nb_filter, concat_axis=-1,
               dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (1, 1),
               kernel_initializer="he_uniform",
               padding="same",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x
def denseblock(x, concat_axis, nb_layers, nb_filter, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    list_feat = [x]

    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)
        nb_filter += growth_rate

    return x, nb_filter
def denseblock_altern(x, concat_axis, nb_layers, nb_filter, growth_rate,
                      dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each conv_factory
       is fed to subsequent ones. (Alternative of a above)
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    * The main difference between this implementation and the implementation
    above is that the one above
    """

    for i in range(nb_layers):
        merge_tensor = conv_factory(x, concat_axis, growth_rate,
                                    dropout_rate, weight_decay)
        x = Concatenate(axis=concat_axis)([merge_tensor, x])
        nb_filter += growth_rate

    return x, nb_filter
def DenseNet(nb_classes, img_dim, depth, nb_dense_block, growth_rate,
             nb_filter, dropout_rate=None, weight_decay=1E-4):
    """ Build the DenseNet model
    :param nb_classes: int -- number of classes
    :param img_dim: tuple -- (channels, rows, columns)
    :param depth: int -- how many layers
    :param nb_dense_block: int -- number of dense blocks to add to end
    :param growth_rate: int -- number of filters to add
    :param nb_filter: int -- number of filters
    :param dropout_rate: float -- dropout rate
    :param weight_decay: float -- weight decay
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """

    if K.image_dim_ordering() == "th":
        concat_axis = 1
    elif K.image_dim_ordering() == "tf":
        concat_axis = -1

    model_input = Input(shape=img_dim)

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               name="initial_conv2D",
               use_bias=False,
               kernel_regularizer=l2(weight_decay))(model_input)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = denseblock(x, concat_axis, nb_layers,
                                  nb_filter, growth_rate,
                                  dropout_rate=dropout_rate,
                                  weight_decay=weight_decay)
        # add transition
        x = transition(x, nb_filter, dropout_rate=dropout_rate,
                       weight_decay=weight_decay)

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, concat_axis, nb_layers,
                              nb_filter, growth_rate,
                              dropout_rate=dropout_rate,
                              weight_decay=weight_decay)

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D(data_format=K.image_data_format())(x)
    x = Dense(nb_classes,
              activation='relu',
              kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=[model_input], outputs=[x], name="DenseNet")

    return densenet

model = DenseNet(nb_classes=4,
                  img_dim=(120,160,3),
                  depth=91,
                  nb_dense_block=5,
                  growth_rate=4,
                  nb_filter =32,
                  dropout_rate=.01)
model.summary()

tpu_model = tf.contrib.tpu.keras_to_tpu_model(
    model,
    strategy=tf.contrib.tpu.TPUDistributionStrategy(
        tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
    )
)

tpu_model.compile(tf.train.AdamOptimizer(learning_rate=.01, ), "mse", [iou])

class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, X_path,y, batch_size=10):
        'Initialization'
        self.batch_size = batch_size
        self.X_path = X_path
        self.y = y

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.X_path) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        X_path_batch = self.X_path[index*self.batch_size:(index+1)*self.batch_size]
        y_batch = self.y[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X = []
        y = []
        for x_p,y_p in zip(X_path_batch,y_batch):
            imgX = io.imread('images/'+x_p)
            imgX = resize(imgX,(120,160))
            y.append(y_p/4)
            X.append(imgX)
        X = np.array(X)
        y = np.array(y)
        return np.array(X), y

training_generator = DataGenerator(X_tr,y_tr,batch_size=64)
validation_generator = DataGenerator(X_val,y_val,batch_size=64)

his = tpu_model.fit_generator(training_generator,
                    steps_per_epoch=len(X_tr)/64,
                    epochs=150,
                    validation_data=validation_generator,
                    callbacks=[keras.callbacks.ModelCheckpoint(
	            'Model_DenseNet.val-iou={val_iou:.2f}.val-mse={val_loss:.2f}.h5',
	            monitor='val_iou',
	            verbose=0,
	            save_best_only=True,
	            save_weights_only=False,
	            mode='auto',
	            period=1
      )])


# Get training and test loss histories
training_loss = his.history['loss']
val_loss = his.history['val_loss']
training_iou = his.history['iou']
val_iou = his.history['val_iou']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
plt.plot(epoch_count, training_loss, 'r-')
plt.plot(epoch_count, val_loss, 'b-')
plt.legend(['Training Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(epoch_count, training_iou, 'r-')
plt.plot(epoch_count, val_iou, 'b-')
plt.legend(['Training IoU', 'Validation IoU'])
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.show()
