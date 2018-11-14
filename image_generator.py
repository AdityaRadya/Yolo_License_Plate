import numpy as np
import os

from tensorflow.keras.utils import Sequence
from preprocessing import load_dataset

class Image_Generator(Sequence):

    def __init__(self, image_directory, labels_directory, batch_size):
        self.image_filenames, self.labels = image_directory, labels_directory
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

        i_batch_x, o_batch_y = load_dataset(batch_x, batch_y)

        return np.array(i_batch_x), np.array(o_batch_y)

