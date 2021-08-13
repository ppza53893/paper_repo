import glob
import math
import os
import random

import cv2
import numpy as np
import tensorflow as tf
import tqdm
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

    def __len__(self):
        return None

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
        self.ds = self.ds.batch(self.batch_size) #.cache(filename='cache_{}'.format(self.name)).
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


class TFRecordLoader():
    def __init__(
        self,
        image_path = None,
        style_path = None,
        batch_size = 5,
        name = 'ds',
        shuffle=True):

        self.image_path = image_path
        self.style_path = style_path
        self.batch_size = batch_size
        self.name = name
        self.shuffle = shuffle

        if self.image_path is not None:
            self.image_list = [p for p in glob.iglob(self.image_path + '/**', recursive=True) if os.path.isfile(p)]
            if shuffle:
                random.shuffle(self.image_list)
        
        self.built_ds = False
        self.steps_per_epoch = None

    def __len__(self):
        if not self.built_ds:
            return None
        return self.steps_per_epoch

    def preprocess(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (128,128))
        img /= 255.
        return img

    def serialize(self, content, style):
        content_img = self.preprocess(content)
        content_img = tf.io.serialize_tensor(content_img)

        style_img = self.preprocess(style)
        style_img = tf.io.serialize_tensor(style_img)

        ci = tf.train.BytesList(value=[content_img.numpy()])
        si = tf.train.BytesList(value=[style_img.numpy()])

        features = tf.train.Features(
            feature={
                'content': tf.train.Feature(bytes_list = ci),
                'style': tf.train.Feature(bytes_list = si)
                }
        )
        proto = tf.train.Example(features=features)
        return proto.SerializeToString()

    def serialize_helper(self, images):
        string = tf.py_function(self.serialize, [images[0], images[1]], tf.string)
        return string

    def deserialize(self, proto):
        parsed = tf.io.parse_example(proto, {
            'content': tf.io.FixedLenFeature([], tf.string),
            'style': tf.io.FixedLenFeature([], tf.string)
        })
        ci = tf.io.parse_tensor(parsed['content'], out_type=tf.float32)
        si = tf.io.parse_tensor(parsed['style'], out_type=tf.float32)
        return tf.stack([ci, si])

    def write_to_tfrecord(self, folder, exist_ok=False, overwrite=True):
        if overwrite and os.path.exists(folder):
            print(f'{folder} already exists.')
            return
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=exist_ok)
        if self.image_path is None or self.style_path is None:
            raise ValueError('path is not defined.')
        fp = list(zip(self.image_list, [self.style_path]*len(self.image_list)))
        write_ds = tf.data.Dataset.from_tensor_slices(fp)
        size = write_ds.batch(self.batch_size).cardinality()
        for i in tqdm.tqdm(range(size)):
            output = os.path.join(folder, f'{self.name}_ds_{i}.tfrecord')
            shard_ds = write_ds.shard(num_shards=size, index=i).map(self.serialize_helper)
            writer =  tf.data.experimental.TFRecordWriter(output)
            writer.write(shard_ds)

    def load_tfrecord(self, folder, max_size=None):
        file_list = [p for p in glob.glob(folder + '/*') if os.path.isfile(p)]
        random.shuffle(file_list)
        if max_size is not None and isinstance(max_size, int) and max_size < len(file_list):
            file_list = file_list[:max_size]
        self.built_ds = True
        self.steps_per_epoch = len(file_list)
        print('{} -> total files: {}'.format(folder, self.steps_per_epoch))
        read_ds = tf.data.TFRecordDataset(file_list)
        read_ds = read_ds.map(self.deserialize)
        read_ds = read_ds.batch(self.batch_size)
        if self.shuffle:
            read_ds = read_ds.shuffle(2048)
        read_ds = read_ds.prefetch(AUTOTUNE)
        return self, read_ds
