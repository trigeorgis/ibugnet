import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo

from pathlib import Path
from scipy.io import loadmat
from utils_3d import crop_face
from menpo.image import Image

def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    # RGB -> BGR
    image = tf.reverse(image, [False, False, True])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image


def rescale_image(image, stride_width=64, method=0):
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
    image = tf.image.resize_images(image, (new_height, new_width), method=method)
    return image


class Dataset(object):
    def __init__(self, name, root, batch_size=1):
        self.name = name
        self.root = Path(root)
        self.batch_size = batch_size

    def num_samples(self):
        return len(self._keys)

    def get_keys(self, path='images'):
        path = self.root / path
        keys = [str(x.stem) for x in path.glob('*')]
        self._keys = keys

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

        return image

    def get_normals(self, index, shape=None):
        def wrapper(index, shape):
            path = self.root / 'normals' / "{}.mat".format(index.decode("utf-8"))

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
        return normals, mask

    def get_segmentation(self, index, shape=None):
        res = tf.zeros(shape)
        return res, res

    def get(self, *names):
        producer = tf.train.string_input_producer(self.get_keys(),
                                                  shuffle=True)
        key = producer.dequeue()
        images = self.get_images(key)
        image_shape = tf.shape(images)
        images = rescale_image(images)
        images = self.preprocess(images)
        
        tensors = [images]

        for name in names:
            fun = getattr(self, 'get_' + name.split('/')[0])
            use_mask = (
                len(name.split('/')) > 1) and name.split('/')[1] == 'mask'

            label, mask = fun(key, shape=image_shape)
            tensors.append(rescale_image(label, method=1))

            if use_mask:
                tensors.append(rescale_image(mask, method=1))

        return tf.train.batch(tensors,
                              self.batch_size,
                              capacity=100,
                              dynamic_pad=True)

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
        self.lms_root = self.root / 'landmarks'
        self.lms_extension = 'ljson'
        self.image_extension = 'jpg'

    def get_images(self, index, shape=None, subdir='images'):
        return self._get_images(index, shape, subdir=subdir, extension='jpg')


    def get_keypoints(self, index, shape):
        def wrapper(index, shape):
            index = index.decode("utf-8")

            prefix = index.split('_')[0]

            landmark_indices = list(map(int, index.split('_')[1:]))
            if len(landmark_indices) > 1:
                min_index, max_index = landmark_indices
                landmark_indices = range(min_index, max_index+1)

            kpts = np.zeros(shape[:2], dtype=int)
            im = Image(kpts)

            mask = np.ones(list(shape[:2]) + [1]).astype(np.float32)

            for lms_index in landmark_indices:
                filename = (prefix + '_' + str(lms_index) + '.' + self.lms_extension)
                path = self.lms_root / filename
                if not path.exists():
                    continue
                lms = mio.import_landmark_file(path.as_posix()).lms
                
                if lms.points.shape[0] != 68:
                    min_indices, max_indices = lms.bounds()

                    mask[min_indices[0]:max_indices[0], min_indices[1]:max_indices[1]] = 0
                    continue

                for i in range(68):
                    lms_mask = im.as_masked().copy()
                    patches = np.ones((1, 1, 1, 4, 4), dtype=np.bool)

                    pc = lms.points[i][None, :]
                    lms_mask.mask.pixels[...] = False
                    lms_mask = lms_mask.mask.set_patches(patches, menpo.shape.PointCloud(pc))
                    kpts[lms_mask.mask] = i + 1


            return kpts.astype(np.int32), mask.astype(np.int32)


        kpts, mask = tf.py_func(wrapper, [index, shape],
                                   [tf.int32, tf.int32])

        kpts = tf.expand_dims(tf.reshape(kpts, shape[:2]), 2)
        mask = tf.expand_dims(tf.reshape(mask, shape[:2]), 2)

        return kpts, mask

    def get_segmentation(self, index, shape=None):
        segmentation = self._get_images(
            index, shape, subdir='semantic_segmentation', channels=1, extension='png')

        return segmentation, tf.ones_like(segmentation)
    

def get_pixels(im, channels=3):
    """Returns the pixels off an `Image`.
    
    Args:
      im: A menpo `Image`.
    Returns:
      A `np.array` of dimensions [height, width, channels]
    """
    
    assert channels in [3]

    pixels = im.pixels_with_channels_at_back()

    if len(pixels.shape) == 2:
        pixels = pixels[..., None]

    # If the image is grayscale, make it RGB.
        pixels = np.dstack([pixels, pixels, pixels])

    # If we have an RGBA image return only the RGB channels.
    if pixels.shape[2] == 4:
        pixels = pixels[..., :3]
    
    return pixels

class AFLWSingle(Dataset):
    def __init__(self, batch_size=1):
        self.name = 'AFLW'
        self.batch_size = batch_size
        self.root = Path('/vol/atlas/databases/aflw_ibug')
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

        self._keys = keys
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def get_keypoints(self, index, shape=(256, 256)):
        from utils_3d import crop_face
        def wrapper(index):
            path = self.root / (index.decode("utf-8") + self.image_extension)
            im = mio.import_image(path, normalize=False)

            im = crop_face(im)
            kpts = np.zeros(im.shape, dtype=int)

            for i in range(68):
                mask = im.as_masked().copy()
                patches = np.ones((1, 1, 1, 5, 5), dtype=np.bool)

                pc = mask.landmarks[None].lms.points[i][None, :]
                mask.mask.pixels[...] = False
                mask = mask.mask.set_patches(patches, menpo.shape.PointCloud(pc))
                kpts[mask.mask] = i + 1

            pixels = get_pixels(im)

            return pixels.astype(np.float32), kpts.astype(np.int32)

        images, kpts = tf.py_func(wrapper, [index],
                                   [tf.float32, tf.int32])

        images.set_shape([shape[0], shape[1], 3])
        kpts.set_shape([shape[0], shape[1]])

        return images, kpts

    def get(self):
        keys = self.get_keys(path='.')
        producer = tf.train.string_input_producer(keys,
                                                  shuffle=True, capacity=1000)
        key = producer.dequeue()
        image, kpts = self.get_keypoints(key)
        image = self.preprocess(image)

        return tf.train.batch([image, kpts],
                              self.batch_size,
                              capacity=1000,
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

class Deep3DV1(Dataset):
    def __init__(self, batch_size=32):
        self.name = 'JamesRenders'
        self.batch_size = batch_size
        self.root = Path('/data/datasets/renders/v1')
        self.tfrecord_names = ['train_v2.tfrecords']
        self.model = mio.import_pickle('/vol/construct3dmm/experiments/models/nicp/mein3d/full_unmasked_good_200.pkl')['model']
        self.settings = mio.import_pickle('/vol/construct3dmm/experiments/nicptexture/settings.pkl', encoding='latin1')

    def get(self):
        paths = [str(self.root / x) for x in self.tfrecord_names]
        
        filename_queue = tf.train.string_input_producer(paths)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'uv_height': tf.FixedLenFeature([], tf.int64),
                'uv_width': tf.FixedLenFeature([], tf.int64),
                'parameters': tf.FixedLenFeature([], tf.string),
                'transform': tf.FixedLenFeature([], tf.string),
                'name': tf.FixedLenFeature([], tf.string),
                'uv': tf.FixedLenFeature([], tf.string),
                'rendering': tf.FixedLenFeature([], tf.string),
            })

        image = tf.image.decode_jpeg(features['rendering'])
        uv = tf.image.decode_jpeg(features['uv'])
        image = tf.to_float(image)
        image.set_shape((256, 256, 3))
        uv.set_shape((2007, 3164, 3))

        parameters = tf.decode_raw(features['parameters'], tf.float32)
        transform = tf.decode_raw(features['transform'], tf.float32)
        parameters.set_shape((200))

#         parameters = tf.expand_dims(parameters, 0)

#         n_vertices = self.model._mean.shape[0] // 3
#         h = tf.matmul(parameters, self.model.components.astype(np.float32)) +self. model._mean.astype(np.float32)
#         h = tf.reshape(h, (n_vertices, 3))
#         h = tf.concat(1, [h, tf.ones((n_vertices, 1))])
#         h = tf.matmul(h, tf.reshape(transform, (4, 4)), transpose_b=True)
#         h = (h / tf.expand_dims(h[:, 3], 1))[:, :3]

        parameters /= 10000.
        return tf.train.shuffle_batch([image, uv, parameters ],
                              self.batch_size,
                              capacity=1000,
                              num_threads=3,
                              min_after_dequeue=200)
        
    
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
