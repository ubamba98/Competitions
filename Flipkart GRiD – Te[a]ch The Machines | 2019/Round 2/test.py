from skimage import io
import numpy as np
import pandas as pd
from keras import backend as K
import math
from skimage.transform import resize
from tensorflow.keras.models import load_model

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

test = pd.read_csv('test.csv')

tpu_model = load_model("Model_DenseNet.val-iou=0.90.val-mse=22.97.h5",custom_objects={'iou':iou,'iou_loss':iou_loss}) #Best DenseNet Model
y_pred = []
for i, image in enumerate(test.image_name):
  step = 25 / len(test)
  imgX = io.imread('images/'+image)
  imgX = resize(imgX,(120, 160))
  y_p = tpu_model.predict(imgX[None,...])*4
  y_pred.append(y_p)
  print('\r' + f'Progress: '
                f"[{'=' * int((i+1) * step) + ' ' * (24 - int((i+1) * step))}]"
                f"({math.ceil((i+1) * 100 /len(test))} %)",
                end='')
y_pred = np.array(y_pred).squeeze()
test.x1 = np.clip(y_pred[:,0],0,640)
test.x2 = np.clip(y_pred[:,1],0,640)
test.y1 = np.clip(y_pred[:,2],0,480)
test.y2 = np.clip(y_pred[:,3],0,480)
print(test.head())
test.to_csv("densenet_submit.csv",index = False)

tpu_model = load_model("Model_wrn.val-iou=0.91.val-mse=22.03.h5",custom_objects={'iou':iou,'iou_loss':iou_loss}) #Best wrn model
y_pred = []
for i, image in enumerate(test.image_name):
  step = 25 / len(test)
  imgX = io.imread('images/'+image)
  imgX = resize(imgX,(120, 160))
  y_p = tpu_model.predict(imgX[None,...])*4
  y_pred.append(y_p)
  print('\r' + f'Progress: '
                f"[{'=' * int((i+1) * step) + ' ' * (24 - int((i+1) * step))}]"
                f"({math.ceil((i+1) * 100 /len(test))} %)",
                end='')
y_pred = np.array(y_pred).squeeze()
test.x1 = np.clip(y_pred[:,0],0,640)
test.x2 = np.clip(y_pred[:,1],0,640)
test.y1 = np.clip(y_pred[:,2],0,480)
test.y2 = np.clip(y_pred[:,3],0,480)
print(test.head())
test.to_csv("wrn_submit.csv",index = False)

#Ensembling
x = pd.read_csv("wrn_submit.csv")
y = pd.read_csv("densenet_submit.csv")
test.x1 = np.clip((x.x1*0.91+y.x1*.9)/1.81,0,640)
test.x2 = np.clip((x.x2*0.91+y.x2*.9)/1.81,0,640)
test.y1 = np.clip((x.y1*0.91+y.y1*.9)/1.81,0,480)
test.y2 = np.clip((x.y2*0.91+y.y2*.9)/1.81,0,480)
print(test.head())
test.to_csv("submit.csv",index = False)
