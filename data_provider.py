import tensorflow as tf
import numpy as np

from pathlib import Path
from scipy.io import loadmat


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    # RGB -> BGR
    image = tf.reverse(image, [False, False, True])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image


def rescale_image(image, stride_width=64):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height = tf.to_float(tf.shape(image)[0])
    width = tf.to_float(tf.shape(image)[1])

    # Taken from 'szross'
    scale_up = 625. / tf.minimum(height, width)
    scale_cap = 961. / tf.maximum(height, width)
    scale_up = tf.minimum(scale_up, scale_cap)
    new_height = stride_width * tf.round(
        (height * scale_up) / stride_width) + 1
    new_width = stride_width * tf.round((width * scale_up) / stride_width) + 1
    new_height = tf.to_int32(new_height)
    new_width = tf.to_int32(new_width)

    images = tf.expand_dims(image, 0)
    images = tf.image.resize_images(images, new_height, new_width)
    return images[0, :, :, :]


class Dataset(object):
    def __init__(self, name, root, batch_size=1):
        self.name = name
        self.root = Path(root)
        self.batch_size = batch_size

    def preprocess(self, image):

    def get_keys(self):
        return tf.constant([x.stem for x in (self.root / 'images').glob('*')],
                           tf.string)

    def preprocess(self, image):
        return caffe_preprocess(image)

    def get_images(self, index, shape=None):
        return get_png_images(index, shape)

    def get_png_images(self, index, shape=None):
        path = tf.reduce_join([str(self.root / 'images'), '/', index, '.png'],
                              0)
        image = tf.image.decode_png(tf.read_file(path), channels=3)
        return rescale_image(image)

    def get_jpg_images(self, index, shape=None):
        path = tf.reduce_join([str(self.root / 'images'), '/', index, '.jpg'],
                              0)
        image = tf.image.decode_jpg(tf.read_file(path), channels=3)
        return rescale_image(image)

    def get_normals(self, index, shape=None):
        def wrapper(index, shape):
            path = self.root / 'normals' / (index + '.mat')

            if path.exists():
                normals = loadmat(str(path))['norms'].astype(np.float32)
                mask = np.ones(list(shape[:2]) + [1]).astype(np.float32)
                return normals, mask

            normals = np.zeros(shape).astype(np.float32)
            mask = np.zeros(list(shape[:2]) + [1]).astype(np.float32)
            return normals, mask

        normals, mask = tf.py_func(wrapper, [index, shape],
                                   [tf.float32, tf.float32])
        normals.set_shape([None, None, 3])
        mask.set_shape([None, None, 1])
        return rescale_image(normals), rescale_image(mask)

    def get_segmentation(self, index, shape=None):
        return 0, 0

    def get(self, *names):
        producer = tf.train.string_input_producer(self.get_keys(),
                                                  shuffle=True)
        key = producer.dequeue()
        images = self.get_images(key)
        images = self.preprocess(images)
        image_shape = tf.shape(images)
        tensors = [images]

        for name in names:
            fun = getattr(self, 'get_' + name.split('/')[0])
            use_mask = (
                len(name.split('/')) > 1) and name.split('/')[1] == 'mask'

            label, mask = fun(key, shape=image_shape)
            tensors.append(label)

            if use_mask:
                tensors.append(mask)

        return tf.train.batch(tensors,
                              self.batch_size,
                              capacity=1000,
                              dynamic_pad=True)


class ICT3DFE(Dataset):
    def __init__(self, batch_size=1):
        self.name = 'ICT3DFE'
        self.batch_size = batch_size
        self.root = Path('data/ict3drfe/')
