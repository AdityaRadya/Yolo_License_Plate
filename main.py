import tensorflow as tf
import time
import os
import random as rn
import numpy as np
import pickle

#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.optimizers import SGD

from net import *
from train import yolo_loss
from constants import get_config

IMAGE_DIR = 'annotations'
ANNOTATION_DIR = 'images'

config = get_config()
BATCH_SIZE  = config['BATCH_SIZE']
EPOCHS      = 3

#LOAD MODEL
def load_data():
    X = pickle.load(open("input_x.pickle", "rb"))
    Y = pickle.load(open("output_y.pickle", "rb"))
    return X, Y

#TRAIN
def train():
    return

#SAVE MODEL AND WEIGHTS

#TEST MODEL
def test():
    return

def predict():
    return

#GUI MODEL

def main():
    input_data, output_data = load_data()
    input_data = np.array(input_data)
    output_data = np.array(output_data)
    model = model0()
    model.fit(input_data, output_data, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.3)
    print(model)

main()