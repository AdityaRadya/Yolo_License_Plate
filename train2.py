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

gridcells = 7**2
lamda_confid_obj = 48
lamda_confid_noobj = 1
lamda_xy = 10
lamda_wh = 15
reguralar_wh = 0.1
lamda_class = 20
classes = 2

def subsetting(data):
    confidence = tf.slice(data, [0, 0, 0], [-1, -1, BOX])
    x = tf.slice(data, [0, 0, BOX], [-1, -1, BOX])
    y = tf.slice(data, [0, 0, BOX*2], [-1, -1, BOX])
    w = tf.slice(data, [0, 0, BOX*3], [-1, -1, BOX])
    h = tf.slice(data, [0, 0, BOX*4], [-1, -1, BOX])
    classes = tf.slice(data, [0, 0, BOX*5], [-1, -1, CLASS])

    #classes = []
    #for i in range(classes):
    #    ctf = tf.slice(data, [0, 0, BOX*(5+i)], [-1, -1, CLASS])
    #    classes.append(ctf)
    return confidence, x, y, w, h, classes

def yoloconfidloss(y_true, y_pred, t):
	lo = K.square(y_true-y_pred)
	value_if_true = lamda_confid_obj*(lo)
	value_if_false = lamda_confid_noobj*(lo)
	loss1 = tf.select(t, value_if_true, value_if_false)
	loss = K.mean(loss1) #,axis=0)
	#
	return loss

# shape is (gridcells*2,)
def yoloxyloss(y_true, y_pred, t):
    lo = K.square(y_true-y_pred)
    value_if_true = lamda_xy*(lo)
    value_if_false = K.zeros_like(y_true)
    loss1 = tf.select(t, value_if_true, value_if_false)
    return K.mean(loss1)

# shape is (gridcells*2,)
def yolowhloss(y_true, y_pred, t):
        #lo = K.square(K.sqrt(y_true)-K.sqrt(y_pred))
	# let w,h not too small or large
    lo = K.square(y_true-y_pred)+reguralar_wh*K.square(0.5-y_pred)
    value_if_true = lamda_wh*(lo)
    value_if_false = K.zeros_like(y_true)
    loss1 = tf.select(t, value_if_true, value_if_false)
    return K.mean(loss1)

# shape is (gridcells*classes,)
def yoloclassloss(y_true, y_pred, t):
    lo = K.square(y_true-y_pred)
    value_if_true = lamda_class*(lo)
    value_if_false = K.zeros_like(y_true)
    loss1 = tf.select(t, value_if_true, value_if_false)
    return K.mean(loss1)

# shape is (gridcells*(5+classes), )
def yololoss(y_true, y_pred):
    truth_confid_tf = tf.slice(y_true, [0,0], [-1,gridcells])
    truth_x_tf = tf.slice(y_true, [0,gridcells], [-1,gridcells])
    truth_y_tf = tf.slice(y_true, [0,gridcells*2], [-1,gridcells])
    truth_w_tf = tf.slice(y_true, [0,gridcells*3], [-1,gridcells])
    truth_h_tf = tf.slice(y_true, [0,gridcells*4], [-1,gridcells])  
    truth_classes_tf = []
    for i in range(classes):
        ctf = tf.slice(y_true, [0,gridcells*(5+i)], [-1,gridcells])
        truth_classes_tf.append(ctf)        

    pred_confid_tf = tf.slice(y_pred, [0,0], [-1,gridcells])
    pred_x_tf = tf.slice(y_pred, [0,gridcells], [-1,gridcells])
    pred_y_tf = tf.slice(y_pred, [0,gridcells*2], [-1,gridcells])
    pred_w_tf = tf.slice(y_pred, [0,gridcells*3], [-1,gridcells])
    pred_h_tf = tf.slice(y_pred, [0,gridcells*4], [-1,gridcells])

    pred_classes_tf = []
    for i in range(classes):
        ctf = tf.slice(y_pred, [0,gridcells*(5+i)], [-1,gridcells])
        pred_classes_tf.append(ctf)

    t = K.greater(truth_confid_tf, 0.5) 

    confidloss = yoloconfidloss(truth_confid_tf, pred_confid_tf, t)
    xloss = yoloxyloss(truth_x_tf, pred_x_tf, t)
    yloss = yoloxyloss(truth_y_tf, pred_y_tf, t)
    wloss = yolowhloss(truth_w_tf, pred_w_tf, t)
    hloss = yolowhloss(truth_h_tf, pred_h_tf, t)

    classesloss =0
    for i in range(classes):
        closs = yoloclassloss(truth_classes_tf[i], pred_classes_tf[i], t)
        classesloss += closs

    loss = confidloss+xloss+yloss+wloss+hloss+classesloss
	#loss = wloss+hloss
	#
	#return loss,confidloss,xloss,yloss,wloss,hloss,classesloss
    return loss


if __name__ == '__main__':
    """ FOR TESTING """
    #print(config)
