import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from constants import get_config

config = get_config()
BOX              = config['BOX']
CLASS            = config['CLASS']
NO_OBJECT_SCALE  = config['NO_OBJECT_SCALE']
OBJECT_SCALE     = config['OBJECT_SCALE']
COORD_SCALE      = config['COORD_SCALE']
CLASS_SCALE      = config['CLASS_SCALE']
BATCH_SIZE       = config['BATCH_SIZE']

def subsetting(data):
    confidence = tf.slice(data, [0, 0, 0, 0], [-1, -1, -1, BOX])
    x = tf.slice(data, [0, 0, 0, BOX], [-1, -1, -1, BOX])
    y = tf.slice(data, [0, 0, 0, BOX*2], [-1, -1, -1, BOX])
    w = tf.slice(data, [0, 0, 0, BOX*3], [-1, -1, -1, BOX])
    h = tf.slice(data, [0, 0, 0, BOX*4], [-1, -1, -1, BOX])
    classes = tf.slice(data, [0, 0, 0, BOX*5], [-1, -1, -1, CLASS])
    
    #classes = []
	#for i in range(classes):
    #    ctf = tf.slice(data, [0, 0, BOX*(5+i)], [-1, -1, CLASS])
    #    classes.append(ctf)
    return confidence, x, y, w, h, classes

def yolo_xy_loss(y_true, y_pred, t):
    error = K.square(y_true - y_pred)
    object_true = COORD_SCALE*(error)
    object_false = K.zeros_like(y_true, dtype='float32')
    loss = tf.where(t, object_true, object_false)
    return K.sum(loss)

def yolo_wh_loss(y_true, y_pred, t):
    error = K.square(K.sqrt(y_true)-K.sqrt(y_pred))
    object_true = COORD_SCALE*(error)
    object_false = K.zeros_like(y_true, dtype='float32')
    loss = tf.where(t, object_true, object_false)
    return K.sum(loss)

def yolo_confidence_loss(y_true, y_pred, t):
    error = K.square(y_true-y_pred)
    object_true = OBJECT_SCALE*(error)
    object_false = NO_OBJECT_SCALE*(error)
    object_default = K.zeros_like(y_true)
    loss1 = tf.where(t, object_true, object_default)
    loss2 = tf.where(K.less(tf.to_float(t),0.5), object_false, object_default)
    return K.sum(loss1) + K.sum(loss2)

def yolo_class_loss(y_true, y_pred, t):
    error = K.square(y_true - y_pred)
    object_true = CLASS_SCALE*(error)
    object_false = K.zeros_like(y_true)
    loss = tf.where(t, object_true, object_false)
    return K.sum(loss)

#Solutions are made in sections all confidence, all x's, all y's and etc in sections
def yolo_loss(y_true, y_pred):
    truth_confidence, truth_x, truth_y, truth_w, truth_h, truth_classes = subsetting(y_true)
    pred_confidence, pred_x, pred_y, pred_w, pred_h, pred_classes = subsetting(y_pred)

    #"truth" has a value of 1 if grid has an object else 0
    truth = K.greater(truth_confidence, 0.5)
    truth_for_class = K.greater(truth_classes, 0.5)
    loss_x = yolo_xy_loss(truth_x, pred_x, truth)
    loss_y = yolo_xy_loss(truth_y, pred_y, truth)
    loss_w = yolo_wh_loss(truth_w, pred_w, truth)
    loss_h = yolo_wh_loss(truth_h, pred_h, truth)
    loss_confidence = yolo_confidence_loss(truth_confidence, pred_confidence, truth)
    loss_class = yolo_class_loss(truth_classes, pred_classes, truth_for_class)

    loss = loss_x + loss_y + loss_w + loss_h + loss_confidence + loss_class

    return loss

def yolo_loss2(y_true, y_pred):
    loss = 0
    for i in range (BATCH_SIZE):
        loss += yololoss(y_true[i], y_pred[i])
    return loss


if __name__ == '__main__':
    """ FOR TESTING """
    #print(config)
