#LABELS = ['License PLate','Car', 'Motorcycles', 'Person']
LABELS  =   ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 
            'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 
            'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

IMAGE_H, IMAGE_W, CHANNEL = 416, 416, 3
GRID_H,  GRID_W  = 26 , 26
BOX              = 2
CLASS            = len(LABELS)
OBJ_THRESHOLD    = 0.3#0.5
NMS_THRESHOLD    = 0.3#0.45
BATCH_SIZE       = 16
EPOCHS           = 150
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0
BATCH_SIZE       = 16
LEARNING_RATE    = 0.001


generator_config = {
    'IMAGE_H'           : IMAGE_H, 
    'IMAGE_W'           : IMAGE_W,
    'CHANNEL'           : CHANNEL,
    'GRID_H'            : GRID_H,  
    'GRID_W'            : GRID_W,
    'BOX'               : BOX,
    'LABELS'            : LABELS,
    'CLASS'             : CLASS,
    'OBJ_THRESHOLD'     : OBJ_THRESHOLD,
    'NMS_THRESHOLD'     : NMS_THRESHOLD,
    'BATCH_SIZE'        : BATCH_SIZE,
    'EPOCHS'            : EPOCHS,
    'NO_OBJECT_SCALE'   : NO_OBJECT_SCALE,
    'OBJECT_SCALE'      : OBJECT_SCALE,
    'COORD_SCALE'       : COORD_SCALE,
    'CLASS_SCALE'       : CLASS_SCALE,
    'LEARNING_RATE'     : LEARNING_RATE
}

def get_config():
    return generator_config
