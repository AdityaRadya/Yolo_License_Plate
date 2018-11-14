import tensorflow as tf
import time
import os
import random as rn
import numpy as np
import pickle

from tensorflow.keras.callbacks import TensorBoard

#from tensorflow.keras.models import Model, Sequential
#from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
#from tensorflow.keras.layers import Conv2D, MaxPooling2D
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.layers import LeakyReLU
#from tensorflow.keras.optimizers import SGD

from net import *
from train import yolo_loss
from constants import get_config
from image_generator import Image_Generator

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options = gpu_options))

IMAGE_DIR = '/home/radya/Thesis/images'
ANNOTATION_DIR = '/home/radya/Thesis/annotations'

config = get_config()
BATCH_SIZE  = config['BATCH_SIZE']
EPOCHS      = 10

VALIDATION_SIZE = 0.2 #Percentage of orginal data

#LOAD MODEL
def load_data(image_file, annotation_file):
    input_x = []
    output_y = []
    for img in os.listdir(image_file):
        img_path = os.path.join(image_file, img)
        xml_path = os.path.join(annotation_file, img.replace('jpg','xml'))
        input_x.append(img_path)
        output_y.append(xml_path)
    return input_x, output_y

#GUI MODEL

def main():
    #PREPARE DATA
    input_data, output_data = load_data(IMAGE_DIR, ANNOTATION_DIR)
    input_data = np.array(input_data)
    output_data = np.array(output_data)

    num_training_samples = int(len(input_data)*(1-VALIDATION_SIZE))
    num_validation_samples = int(len(input_data) - num_training_samples)

    test_input_data = input_data[:num_training_samples]
    test_output_data = output_data[:num_training_samples]
    validation_input_data = input_data[num_training_samples:]
    validation_output_data = output_data[num_training_samples:]
    
    #LOAD THE MODEL
    model = model0()
    tensorboard = TensorBoard(log_dir="logs/{}".format("model0"))
    sgd = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
    model.compile(loss=yolo_loss, optimizer=sgd, metrics=['accuracy'])

    #PREPARE BATCH AND TRAINING
    train_batch_generator = Image_Generator(test_input_data, test_output_data, BATCH_SIZE)
    validation_batch_generator = Image_Generator(validation_input_data, validation_output_data, BATCH_SIZE)

    model.fit_generator(generator=train_batch_generator,
                                          steps_per_epoch=(num_training_samples // BATCH_SIZE),
                                          epochs=EPOCHS,
                                          verbose=1,
                                          validation_data=validation_batch_generator,
                                          validation_steps=(num_validation_samples // BATCH_SIZE),
                                          use_multiprocessing=True,
                                          workers=8,
                                          max_queue_size=32,
                                          callbacks=[tensorboard])

    #model.fit(
    #            input_data, output_data, 
    #            batch_size=BATCH_SIZE, 
    #            epochs=EPOCHS, 
    #            validation_split=0.3, 
    #            callbacks=[tensorboard]
    #            )

    
    print(model)

main()