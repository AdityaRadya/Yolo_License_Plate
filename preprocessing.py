import xml.etree.ElementTree
import cv2
import numpy as np
from constants import get_config
import pickle
import os
import sys
import random as rn
import imgaug as iaa

config = get_config()
IMAGE_H, IMAGE_W, CHANNEL = config['IMAGE_H'], config['IMAGE_W'], config['CHANNEL']
CLASS = config['CLASS']
BOX = config['BOX']
LABELS = config['LABELS']
GRID_H, GRID_W = config['GRID_H'], config['GRID_W']
IMAGE_NUM = 1000

def get_center(xmin, ymin, w, h):
    center_w = int(w/2)
    center_h = int(h/2)
    return xmin + center_w, ymin + center_h

def parsing_xml(xml_path):
    root = xml.etree.ElementTree.parse(xml_path).getroot()
    data = []
    for object in root.iter('object'):
        try:
            if object.find('truncated').text == 1:
                pass
        except AttributeError:
            pass
        name = object.find('name').text
        bbox = object.find('bndbox')
        xmin = bbox.find('xmin').text
        ymin = bbox.find('ymin').text
        xmax = bbox.find('xmax').text
        ymax = bbox.find('ymax').text
        temp_object = [
                        name, 
                        int(float(xmin)), 
                        int(float(ymin)), 
                        int(float(xmax)), 
                        int(float(ymax))
                        ]
        data.append(temp_object)
    return data
    
def resize(img, ann):
    h, w, _ = img.shape

    h_scale = IMAGE_H / h
    w_scale = IMAGE_W / w
    
    resize_img = cv2.resize(img, (IMAGE_H, IMAGE_W))
    img = np.array(img)

    for obj in ann:
        obj[1] = int(np.round(obj[1] * w_scale))
        obj[2] = int(np.round(obj[2] * h_scale))
        obj[3] = int(np.round(obj[3] * w_scale))
        obj[4] = int(np.round(obj[4] * h_scale))

    resize_img = np.array(resize_img)
    return resize_img, ann

def get_height(ymin, ymax):
    return ymax - ymin

def get_width(xmin, xmax):
    return xmax - xmin

def output_tensor(ann):
    output_depth = BOX * 5 + CLASS
    output = np.zeros((GRID_H, GRID_W, output_depth))

    for obj in ann:
        h = get_height(obj[2], obj[4])
        w = get_width(obj[1], obj[3])
        grid_x = int((w/IMAGE_W)*GRID_W)
        grid_y = int((h/IMAGE_H)*GRID_H)
        xmin_norm = obj[1] / IMAGE_W
        ymin_norm = obj[2] / IMAGE_H
        h_norm = h / IMAGE_H
        w_norm = w / IMAGE_W
        for boxes in range(BOX):
            #confidence
            """if boxes == 0 and output[grid_y][grid_x][boxes] == 1:
                print("Grid already asign to an object")"""
            output[grid_y][grid_x][boxes] = 1
            #xmin
            output[grid_y][grid_x][BOX+boxes] = xmin_norm
            #ymin
            output[grid_y][grid_x][BOX*2+boxes] = ymin_norm
            #width
            output[grid_y][grid_x][BOX*3+boxes] = w_norm
            #height
            output[grid_y][grid_x][BOX*4+boxes] = h_norm
        #class
        class_index = LABELS.index(obj[0])
        output[grid_y][grid_x][BOX*5+class_index] = 1

    return output

#LOAD and PREPROCESS DATASET
def preprocess_img(img, xml_path):
    ann = parsing_xml(xml_path)
    img, ann = resize(img, ann)
    out_tf = output_tensor(ann)
    return img, out_tf

def load_dataset(image_paths, annotations_paths):
    features_arr = []
    labels_arr = []
    for index in range(len(image_paths)) :
        try:
            img_array = cv2.imread(image_paths[index]) / 255
            xml_path = annotations_paths[index]
            features, labels = preprocess_img(img_array, xml_path)
            features_arr.append(features)
            labels_arr.append(labels)

        except FileNotFoundError:
            print("error")
            pass
    #size_arr = np.arange(len(labels_arr))
    #randomize = np.random.permutation(len(labels_arr))
    #features_arr = features_arr[randomize]
    #labels_arr = labels_arr[randomize]
    return features_arr, labels_arr

      
#if __name__ == '__main__':
    #IMAGE_DIR = '/home/radya/Thesis/images'
    #ANNOTATION_DIR = '/home/radya/Thesis/annotations'
#
    #input_x, output_y = load_dataset(IMAGE_DIR, ANNOTATION_DIR)
    ##print(len(train_data[0]))
    ##nput_x = []
    ##utput_y = []
#
    ##or features, output in train_data:
    ##   input_x.append(features)
    ##   output_y.append(output)
#
    #pickle_out = open("input_x.pickle","wb")
    #pickle.dump(input_x, pickle_out)
    #pickle_out.close()
#
    #pickle_out = open("output_y.pickle","wb")
    #pickle.dump(output_y, pickle_out)
    #pickle_out.close()