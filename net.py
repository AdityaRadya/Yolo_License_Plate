import tensorflow as tf
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import SGD

import train
from constants import get_config
from train import yolo_loss

config = get_config()
IMAGE_H = config['IMAGE_H']
IMAGE_W = config['IMAGE_W']
CHANNEL = config['CHANNEL']
BOX = config['BOX']
CLASS = config['CLASS']
LEARNING_RATE = config['LEARNING_RATE']
BATCH_SIZE = config['BATCH_SIZE']

image_size = (IMAGE_H, IMAGE_W, CHANNEL)

def model0():
    model = Sequential()
    #Layer 0
    model.add(Conv2D(128, (3, 3), input_shape=image_size, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 1
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 2
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 3
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 4
    model.add(Conv2D((BOX*5 + CLASS), (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss=yolo_loss, optimizer=sgd, metrics=['accuracy'])

    return model

def model1():
    model = Sequential()
    #Layer 0
    model.add(Conv2D(128, (3, 3), input_shape=image_size, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 1
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 2
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 3
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 4
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    #Layer 5
    model.add(Conv2D((BOX*5 + CLASS), (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss=yolo_loss, optimizer=sgd, metrics=['accuracy'])

    return model

def model2():
    model = Sequential()
    #Layer 0
    model.add(Conv2D(64, (3, 3), input_shape=image_size, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 1
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 2
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 3
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 4
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    #Layer 5
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    #Layer 6
    model.add(Conv2D((BOX*5 + CLASS), (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss=yolo_loss, optimizer=sgd, metrics=['accuracy'])

    return model

def model3():
    model = Sequential()
    #Layer 0
    model.add(Conv2D(32, (3, 3), input_shape=image_size, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 1
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 2
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 3
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 4
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    #Layer 5
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    #Layer 6
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    #Layer 7
    model.add(Conv2D((BOX*5 + CLASS), (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss=yolo_loss, optimizer=sgd, metrics=['accuracy'])

    return model

def model_ori():
    model = Sequential()
    #Layer 0
    model.add(Conv2D(16, (3, 3), input_shape=image_size, padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 1
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 2
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 3
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Layer 4
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    #Layer 5
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
    #Layer 6
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    #Layer 7
    model.add(Conv2D(1024, (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())
    #Layer 8
    model.add(Conv2D((BOX*5 + CLASS), (3, 3), padding='same'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(BatchNormalization())

    sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss=yolo_loss, optimizer=sgd, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    ''' FOR TESTING '''
    model = model0()
    print(model.summary())
    model = model1()
    print(model.summary())
    model = model2()
    print(model.summary())
    model = model3()
    print(model.summary())
    model = model_ori()
    print(model.summary())