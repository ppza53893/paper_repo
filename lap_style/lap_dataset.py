import glob
import math
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import Sequence

AUTOTUNE = tf.data.experimental.AUTOTUNE


class TFDataLoader:
    def __init__(self, image_path, style_path, batch_size, name, shuffle=True, limits = None):
        self.image_path = image_path
        self.style_path = style_path
        self.batch_size = batch_size
        self.name = name
        self.shuffle = shuffle

        self.image_list = [p for p in glob.iglob(self.image_path + '/**', recursive=True) if os.path.isfile(p)]
        if limits and isinstance(limits, int):
            self.image_list = self.image_list[:limits]

    @tf.function
    def preprocess(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (128,128))
        img /= 255.
        return img

    @tf.function
    def generate_input(self, img_path):
        img = self.preprocess(img_path)
        style_img = self.preprocess(self.style_path)
        return tf.stack([img, style_img])

    def gen_ds(self):
        self.ds = tf.data.Dataset.from_tensor_slices(self.image_list)
        self.ds = self.ds.map(self.generate_input, num_parallel_calls=AUTOTUNE)
        if self.shuffle:
            self.ds = self.ds.shuffle(1024)
        self.ds = self.ds.cache(filename='cache_{}'.format(self.name)).batch(self.batch_size)
        size = self.ds.cardinality().numpy()
        print(f'{self.name} dataset cardinality: {size}')
        self.ds = self.ds.prefetch(AUTOTUNE)
        return self.ds


class SequenceDataLoader(Sequence):
    def __init__(self, image_path, style_path, batch_size, *args, **kwargs):
        self.image_path = image_path
        self.style_path = style_path
        self.batch_size = batch_size

        # load
        self.image_list = [p for p in glob.iglob(self.image_path + '/**', recursive=True) if os.path.isfile(p)]
        if len(self.image_list) > 30000:
            self.image_list = self.image_list[:30000]
        self.on_epoch_end()

        self.style_image = self.image_load_resize(style_path)
        temp = self.style_image.copy()
        for _ in range(batch_size-1):
            self.style_image = np.r_['0', self.style_image, temp]

    def __len__(self):
        return math.ceil(len(self.image_list)/self.batch_size)
    
    def image_load_resize(self, imgpath):
        image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (128,128), interpolation=cv2.INTER_LANCZOS4)
        image = image /255.
        image = image[np.newaxis]
        return image

    def __getitem__(self, idx):
        x = self.image_list[self.batch_size*idx:self.batch_size*(idx+1)]

        ret_x = None
        for path in x:
            image = self.image_load_resize(path)
            if ret_x is None:
                ret_x = image.copy()
            else:
                ret_x = np.r_['0', ret_x, image]
        return tf.stack([ret_x, self.style_image])

    def on_epoch_end(self):
        random.shuffle(self.image_list)
    
    def gen_ds(self):
        return self
