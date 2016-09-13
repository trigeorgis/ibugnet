import tensorflow as tf
import numpy as np
import menpo.io as mio

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

    def get_keys(self, path='images'):
        path = self.root / path
        keys = [x.stem for x in path.glob('*')]
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def preprocess(self, image):
        return caffe_preprocess(image)

    def get_images(self, index, shape=None, subdir='images'):
        return self._get_images(index, shape=None, subdir='images')

    def _get_images(self, index, shape=None, subdir='images', channels=3, extension='png'):
        path = tf.reduce_join([str(self.root / subdir), '/', index, '.', extension],
                              0)

        if extension == 'png':
            image = tf.image.decode_png(tf.read_file(path), channels=channels)
        elif extension == 'jpg':
            image = tf.image.decode_jpeg(tf.read_file(path), channels=channels)
        else:
            raise RuntimeError()

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
        res = tf.zeros(shape)
        return res, res

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
                              capacity=100,
                              dynamic_pad=True)


class EarWPUTEDB(Dataset):
    def __init__(self, batch_size=1):
        self.name = 'EarWPUTEDB'
        self.batch_size = batch_size
        self.root = Path('/data/tmp/')
        self.dataset = mio.import_pickle(str(self.root / 'LDA-WPUTEDB-Data-dsift.pkl'))
        self.num_classes = 500

    def get_keys(self, path='images'):
        path = self.root / path
        keys = map(str, np.arange(len(self.dataset)))
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def get_images(self, key, shape=None):
        def wrapper(index):
            return self.dataset[int(index)][1].astype(np.float32)

        image = tf.py_func(wrapper, [key],
                                   [tf.float32])[0]

        image.set_shape([358272,])
        return image

    def get_labels(self, key, shape=None):
        def wrapper(index):
            return self.dataset[int(index)][0].astype(np.int32)

        label = tf.py_func(wrapper, [key],
                                   [tf.int32])[0]

        label = tf.one_hot(label, self.num_classes, dtype=tf.int32)
        label.set_shape([500,])
        return label, None

    def get(self, *names):
        producer = tf.train.string_input_producer(self.get_keys(),
                                                  shuffle=True)
        key = producer.dequeue()
        images = self.get_images(key)

        image_shape = tf.shape(images)
        tensors = [images]

        for name in names:
            fun = getattr(self, 'get_' + name.split('/')[0])
            use_mask = (
                len(name.split('/')) > 1) and name.split('/')[1] == 'mask'

            label, mask = fun(key, shape=image_shape)
            tensors.append(label)

        return tf.train.shuffle_batch(tensors,
                              self.batch_size,
                              capacity=2000, min_after_dequeue=200)

class ICT3DFE(Dataset):
    def __init__(self, batch_size=1):
        self.name = 'ICT3DFE'
        self.batch_size = batch_size
        self.root = Path('data/ict3drfe/')

class FDDB(Dataset):
    def __init__(self, batch_size=1):
        self.name = 'FDDB'
        self.batch_size = batch_size
        self.root = Path('data/fddb/')

    def get_images(self, index, shape=None, subdir='images'):
        return self._get_images(index, shape, subdir=subdir, extension='jpg')

    def get_segmentation(self, index, shape=None):
        segmentation = self._get_images(
            index, shape, subdir='semantic_segmentation', channels=1, extension='png')

        return segmentation, tf.ones_like(segmentation)

class AFLW(Dataset):
    def __init__(self, batch_size=1):
        self.name = 'AFLW'
        self.batch_size = batch_size
        self.root = Path('data/aflw/')

    def get_images(self, index, shape=None, subdir='images'):
        return self._get_images(index, shape, subdir=subdir, extension='jpg')

    def get_segmentation(self, index, shape=None):
        segmentation = self._get_images(
            index, shape, subdir='semantic_segmentation', channels=1, extension='png')

        return segmentation, tf.ones_like(segmentation)


class AFLWSingle(Dataset):
    def __init__(self, batch_size=1):
        from menpo.transform import UniformScale, Translation, Homogeneous, scale_about_centre, Rotation
        import menpo3d.io as m3dio

        self.name = 'AFLW'
        self.batch_size = batch_size
        self.root = Path('/data/datasets/aflw_ibug')
        template = m3dio.import_mesh('/vol/construct3dmm/regression/src/template.obj')
        template = Translation(-template.centre()).apply(template)
        self.template = scale_about_centre(template, 1./1000.).apply(template)
        pca_path = '/homes/gt108/Projects/ibugface/pose_settings/pca_params.pkl'
        self.eigenvectors, self.eigenvalues, self.h_mean, self.h_max = mio.import_pickle(pca_path)
        self.image_extension = '.jpg'
        self.lms_extension = '.ljson'

    def get_keys(self, path='images'):

        path = self.root / path
        lms_files = path.glob('*' + self.lms_extension)
        keys = [] # ['face_55135', 'face_49348']

        # Get only files with 68 landmarks
        for p in lms_files:
            try:
                lms = mio.import_landmark_file(p)

                if lms.n_landmarks == 68:
                    keys.append(lms.path.stem)
            except:
                pass

        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def get_data(self, index, shape=(256, 256)):
        from utils_3d import retrieve_camera_matrix, quaternion_from_matrix, crop_face
        def wrapper(index):
            path = self.root / (index + self.image_extension)

            im = mio.import_image(path, normalize=False)
            im = crop_face(im)

            view_t, c_t, proj_t = retrieve_camera_matrix(im, self.template, group=None)
            h = view_t.h_matrix
            vector = self.eigenvectors.dot(h.ravel() - self.h_mean)
            # vector /= self.h_max.max()
            vector = vector[:3]
            if np.any(np.isnan(vector)):
                vector = np.array([0, 0, 0])

            px = im.pixels_with_channels_at_back()

            if len(px.shape) == 2:
                px = px[..., None]

            if px.shape[2] == 1:
                px = np.dstack([px, px, px])

            if px.shape[2] == 4:
                px = px[..., :3]


            return px.astype(np.float32), vector.astype(np.float32)

        images, parameters = tf.py_func(wrapper, [index],
                                   [tf.float32, tf.float32])

        images.set_shape([shape[0], shape[1], 3])
        parameters.set_shape([3])

        return images, parameters

    def get_data_from_scratch(self, index, shape=(256, 256)):
        from utils_3d import retrieve_camera_matrix, quaternion_from_matrix, crop_face
        def wrapper(index):
            path = self.root / (index + self.image_extension)

            im = mio.import_image(path, normalize=False)
            im = crop_face(im)

            view_t, c_t, proj_t = retrieve_camera_matrix(im, self.template, group=None)

            h = view_t.h_matrix.copy()
            h[0, 3] = h[1, 3] = h[2, 3] = 0
            h[3, 3] = 1

            q = quaternion_from_matrix(h)
            q /= np.sqrt(np.sum(q**2))
            translation =  view_t.h_matrix[:3, 3].ravel()

            if np.any(np.isnan(q)):
                q = np.zeros((4, ))
                translation = np.zeros((3, ))
            px = im.pixels_with_channels_at_back()

            if len(px.shape) == 2:
                px = px[..., None]

            if px.shape[2] == 1:
                px = np.dstack([px, px, px])

            px = px[..., :3]
            return [x.astype(np.float32) for x in (px, q, translation)]

        images, quaternion, translation = tf.py_func(wrapper, [index],
                                   [tf.float32, tf.float32, tf.float32])

        images.set_shape([shape[0], shape[1], 3])
        quaternion.set_shape([4])
        translation.set_shape([3])

        return images, quaternion, translation

    def get_quaternions(self):
        keys = self.get_keys(path='.')
        producer = tf.train.string_input_producer(keys,
                                                  shuffle=True, capacity=1000)
        key = producer.dequeue()
        image, q, t = self.get_data_from_scratch(key)
        image = self.preprocess(image)

        return tf.train.batch([image, q, t],
                              self.batch_size,
                              num_threads=4,
                              capacity=200,
                              dynamic_pad=False)

    def get(self):
        keys = self.get_keys(path='.')
        producer = tf.train.string_input_producer(keys,
                                                  shuffle=True, capacity=1000)
        key = producer.dequeue()
        image, parameters = self.get_data(key)
        image = self.preprocess(image)

        return tf.train.batch([image, parameters],
                              self.batch_size,
                              capacity=100,
                              dynamic_pad=False)


class FDDBSingle(AFLWSingle):
    def __init__(self, batch_size=1):
        from menpo.transform import Translation, scale_about_centre
        import menpo3d.io as m3dio

        self.name = 'FDDB'
        self.batch_size = batch_size
        self.root = Path('/vol/atlas/databases/fddb_ibug')
        template = m3dio.import_mesh('/vol/construct3dmm/regression/src/template.obj')
        template = Translation(-template.centre()).apply(template)
        self.template = scale_about_centre(template, 1./1000.).apply(template)
        pca_path = '/homes/gt108/Projects/ibugface/pose_settings/pca_params.pkl'
        self.eigenvectors, self.eigenvalues, self.h_mean, self.h_max = mio.import_pickle(pca_path)
        self.image_extension = '.jpg'
        self.lms_extension = '.ljson'

class LFPWSingle(AFLWSingle):
    def __init__(self, batch_size=1):
        from menpo.transform import Translation, scale_about_centre
        import menpo3d.io as m3dio

        self.name = 'LFPW'
        self.batch_size = batch_size
        self.root = Path('/vol/atlas/databases/lfpw/trainset')
        template = m3dio.import_mesh('/vol/construct3dmm/regression/src/template.obj')
        template = Translation(-template.centre()).apply(template)
        self.template = scale_about_centre(template, 1./1000.).apply(template)
        self.image_extension = '.png'
        self.lms_extension = '.pts'

class JamesRenders(Dataset):
    def __init__(self, batch_size=32):
        self.name = 'JamesRenders'
        self.batch_size = batch_size
        self.root = Path('/vol/construct3dmm/regression/basic/renderings')

    def preprocess_params(self, params):
        return params / 100000.

    def get_data(self, index, shape=(256, 256)):
        def wrapper(index):
            path = self.root / (index + '.pkl')

            p = mio.import_pickle(path)

            px = p['img'].resize(shape).pixels_with_channels_at_back()
            return px.astype(np.float32), p['weights'].astype(np.float32)

        images, parameters = tf.py_func(wrapper, [index],
                                   [tf.float32, tf.float32])

        images.set_shape([shape[0], shape[1], 3])
        parameters.set_shape([200])

        return images, parameters

    def get(self):
        keys = self.get_keys(path='.')
        producer = tf.train.string_input_producer(keys,
                                                  shuffle=True)
        key = producer.dequeue()
        images, parameters = self.get_data(key)
        images = self.preprocess(images)
        parameters = self.preprocess_params(parameters)

        return tf.train.batch([images, parameters],
                              self.batch_size,
                              capacity=80,
                              dynamic_pad=False)
