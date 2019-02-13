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
from tensorflow.keras.layers import Input, BatchNormalization, Activation
from tensorflow.keras.layers import Dense, Dropout, Add, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
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

def main_block(x, filters, n, strides, dropout):
	# Normal part
	x_res = Conv2D(filters, (3,3), strides=strides, padding="same")(x)# , kernel_regularizer=l2(5e-4)
	x_res = BatchNormalization()(x_res)
	x_res = Activation('relu')(x_res)
	x_res = Conv2D(filters, (3,3), padding="same")(x_res)
	# Alternative branch
	x = Conv2D(filters, (1,1), strides=strides)(x)
	# Merge Branches
	x = Add()([x_res, x])

	for i in range(n-1):
		# Residual conection
		x_res = BatchNormalization()(x)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)
		# Apply dropout if given
		if dropout: x_res = Dropout(dropout)(x)
		# Second part
		x_res = BatchNormalization()(x_res)
		x_res = Activation('relu')(x_res)
		x_res = Conv2D(filters, (3,3), padding="same")(x_res)
		# Merge branches
		x = Add()([x, x_res])

	# Inter block part
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	return x
def build_model(input_dims, output_dim, n, k, act= "relu", dropout=None):
	""" Builds the model. Params:
			- n: number of layers. WRNs are of the form WRN-N-K
				 It must satisfy that (N-4)%6 = 0
			- k: Widening factor. WRNs are of the form WRN-N-K
				 It must satisfy that K%2 = 0
			- input_dims: input dimensions for the model
			- output_dim: output dimensions for the model
			- dropout: dropout rate - default=0 (not recomended >0.3)
			- act: activation function - default=relu. Build your custom
				   one with keras.backend (ex: swish, e-swish)
	"""
	# Ensure n & k are correct
	assert (n-4)%6 == 0
	assert k%2 == 0
	n = (n-4)//6
	# This returns a tensor input to the model
	inputs = Input(shape=(input_dims))

	# Head of the model
	x = Conv2D(16, (3,3), padding="same")(inputs)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# 3 Blocks (normal-residual)
	x = main_block(x, 16*k, n, (1,1), dropout) # 0
	x = main_block(x, 32*k, n, (2,2), dropout) # 1
	x = main_block(x, 64*k, n, (2,2), dropout) # 2

	# Final part of the model
	x = AveragePooling2D((8,8))(x)
	x = Flatten()(x)
    x = Dense(16, "relu")(x)
	outputs = Dense(output_dim, "relu")(x)

	model = Model(inputs=inputs, outputs=outputs)
	return model

model = build_model((120,160,3), 4, 10, 4)
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
                  'Model_wrn.val-iou={val_iou:.2f}.val-mse={val_loss:.2f}.h5',
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
