import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
import scipy

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


def _rescale_image(image, stride_width=64, method=0):
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
    image = tf.image.resize_images(
        image, (new_height, new_width), method=method)
    return image


class Dataset(object):
    def __init__(self, name, root, batch_size=1):
        self.name = name
        self.root = Path(root)
        self.batch_size = batch_size
        self.image_extension = 'png'
        self.images_root = 'images'

    def rescale_image(self, image, method=0, image_shape=None):
        return _rescale_image(image, method=0)

    def num_samples(self):
        return len(self._keys)

    def get_keys(self):
        path = self.root / self.images_root
        keys = [str(x.stem) for x in path.glob('*')]
        self._keys = keys

        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def preprocess(self, image):
        return caffe_preprocess(image)

    def get_images(self, index, shape=None):
        return self._get_images(
            index,
            shape=None,
            subdir=self.images_root,
            extension=self.image_extension)

    def _get_images(self,
                    index,
                    shape=None,
                    subdir='images',
                    channels=3,
                    extension='png'):
        path = tf.reduce_join(
            [str(self.root / subdir), '/', index, '.', extension], 0)

        if extension == 'png':
            image = tf.image.decode_png(tf.read_file(path), channels=channels)
        elif extension == 'jpg':
            image = tf.image.decode_jpeg(tf.read_file(path), channels=channels)
        else:
            raise RuntimeError()

        return tf.to_float(image)

    def get_normals(self, index, shape=None):
        def wrapper(index, shape):
            path = self.root / 'normals' / "{}.mat".format(
                index.decode("utf-8"))

            if path.exists():
                mat = loadmat(str(path))
                normals = mat['norms'].astype(np.float32)
                mask = mat['cert'][..., None].astype(np.float32)
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

    def get(self, *names, preprocess_inputs=True, create_batches=True):
        producer = tf.train.string_input_producer(
            self.get_keys(), shuffle=True)
        key = producer.dequeue()
        images = self.get_images(key)
        image_shape = tf.shape(images)
        images = self.rescale_image(images, image_shape=image_shape)

        if preprocess_inputs:
            images = self.preprocess(images)

        tensors = [images]

        for name in names:
            fun = getattr(self, 'get_' + name.split('/')[0])
            use_mask = (
                len(name.split('/')) > 1) and name.split('/')[1] == 'mask'

            label, mask = fun(key, shape=image_shape)
            tensors.append(self.rescale_image(label, method=name.split('/')[0], image_shape=image_shape))

            if use_mask:
                tensors.append(self.rescale_image(mask, method=name, image_shape=image_shape))

        if not create_batches:
            return tensors

        return tf.train.batch(
            tensors, self.batch_size, capacity=100, dynamic_pad=True)


class ICT3DFE(Dataset):
    def __init__(self, batch_size=1):
        super().__init__(
            name='ICT3DFE', root=Path('data/ict3drfe/'), batch_size=batch_size)
        self.image_extension = 'png'

    def get_normals(self, index, shape=None):
        normals, mask = super().get_normals(index, shape)
        return normals * np.array([1, 1, 1]), mask


class Photoface(Dataset):
    def __init__(self, batch_size=1):
        super().__init__(
            'photoface', Path('data/photoface/'), batch_size=batch_size)
        self.image_extension = 'png'
        self.images_root = 'albedo'

    def get_normals(self, index, shape=None):
        normals, mask = super().get_normals(index, shape)
        return normals * np.array([1, -1, 1]), mask


class BaselNormals(Dataset):
    def __init__(self, batch_size=1):
        super().__init__(
            '3ddfa_basel',
            Path('/data/datasets/3ddfa_basel_normals'),
            batch_size=batch_size)
        self.image_extension = 'jpg'


class MeinNormals(Dataset):
    def __init__(self, batch_size=1):
        super().__init__(
            'mein3dnormals',
            Path('/data/datasets/renders/emotion_model_normals'),
            batch_size=batch_size)
        self.image_extension = 'jpg'


class HumanPose(Dataset):
    def __init__(self, batch_size=1, root='/data/yz4009/', valid_check=False, n_lms=16):
        super().__init__(
            'human_pose',
            Path(root),
            batch_size=batch_size)
        self.image_extension = 'jpg'
        self.images_root = '.'
        self.valid_check = valid_check
        self.n_lms = n_lms

    def get_keys(self):
        if (self.root / 'keys.pkl').exists():
            keys = mio.import_pickle(self.root / 'keys.pkl')
            self._keys = keys
            return tf.constant(keys)

        path = self.root

        def check_valid(x):
            return all([(path / '{}+svs_dark+{:02d}.pkl'.format(x, i)).exists()
                        for i in [0, 1, 2, 4]])

        if self.valid_check:
            keys = [str(x.stem) for x in path.glob('*.jpg') if check_valid(x.stem)]
        else:
            keys = [str(x.stem) for x in path.glob('*.jpg')]
        self._keys = keys

        mio.export_pickle(keys, self.root / 'keys.pkl')
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def rescale_image(self, image, method=None, image_shape=None):
        if not image_shape is None:
            h, w = tf.to_float(image_shape[0]), tf.to_float(image_shape[1])
            scale = tf.reduce_max([h,w]) / 256.0
            # scale = 1
            nh, nw = tf.to_int32(h/scale), tf.to_int32(w/scale)

            if not method is None:
                if method.split('/')[0] == 'landmarks':
                    return image / scale
                elif method.split('/')[0] == 'pose':
                    return tf.image.resize_bilinear(image, [nh, nw])

            return tf.image.resize_bilinear(image[None, ...], [nh, nw])[0]
        else:
            return image


    def get_pose(self, index, shape):
        def wrapper(index):
            index = index.decode("utf-8")
            result = []

            for i in [0, 1, 2, 4]:
                svs = mio.import_pickle(
                    self.root / '{}+svs_dark+{:02d}.pkl'.format(index, i))
                svs = svs.pixels_with_channels_at_back()[:,:,[0,1,2,3,8]]
                result.append(svs)
            return np.array(result).astype(np.float32)

        svs, = tf.py_func(wrapper, [index], [tf.float32])
        svs.set_shape([4, None, None, 5])
        return svs, tf.ones_like(svs)

    def get_heatmap(self, index, shape, sigma=7):
        def wrapper(index, shape):
            index = index.decode("utf-8")
            lms = mio.import_landmark_file(
                self.root / '{}.ljson'.format(index))
            marked_index = label_index(lms, 'marked')

            img_hm = Image.init_blank(shape[:2], n_channels=lms.n_landmarks)
            xs, ys = np.meshgrid(np.arange(0, shape[1]), np.arange(0, shape[0]))
            
            for c,(y, x) in enumerate(np.round(lms.lms.points)):
                if c in marked_index:
                    try:
                        gaussian = (1 / (sigma * np.sqrt(2 * np.pi)))
                        gaussian *= np.exp(-0.5 * (pow(ys - y, 2) + pow(xs - x, 2)) * pow(1 / sigma, 2))
                        gaussian *= 17
                        img_hm.pixels[c] = gaussian
                    except Exception as e:
                        print(e)

            return img_hm.pixels_with_channels_at_back().astype(np.float32)

        hm, = tf.py_func(wrapper, [index, shape], [tf.float32])
        hm.set_shape([None, None, self.n_lms])
        return hm, tf.ones_like(hm)

    def get_landmarks(self, index, shape):
        def wrapper(index):
            index = index.decode("utf-8")
            lms = mio.import_landmark_file(
                self.root / '{}.ljson'.format(index))
            marked_index = label_index(lms, 'marked')
            mask = np.zeros(lms.lms.points.shape).astype(np.int32)

            mask[[marked_index],:] = 1
            return lms.lms.points.astype(np.float32), mask

        lms, mask = tf.py_func(wrapper, [index], [tf.float32, tf.int32])
        lms.set_shape([self.n_lms,2])
        mask.set_shape([self.n_lms,2])
        return lms, mask

    def get_keypoints(self, index, shape):
        def wrapper(index, shape):
            index = index.decode("utf-8")

            kpts = np.zeros(shape[:2], dtype=int)
            im = Image(kpts)
            mask = np.ones(list(shape[:2]) + [1]).astype(np.float32)

            path = self.root / '{}.ljson'.format(index)
            lms = mio.import_landmark_file(path)
            marked_index = label_index(lms, 'marked')

            for i in range(lms.lms.n_points):
                if i in marked_index:
                    lms_mask = im.as_masked().copy()
                    patches = np.ones((1, 1, 1, 13, 13), dtype=np.bool)

                    pc = lms.lms.points[i][None, :]
                    lms_mask.mask.pixels[...] = False
                    lms_mask = lms_mask.mask.set_patches(
                        patches, menpo.shape.PointCloud(pc))
                    kpts[lms_mask.mask] = i + 1

            return kpts.astype(np.int32), mask.astype(np.int32)

        kpts, mask = tf.py_func(wrapper, [index, shape], [tf.int32, tf.int32])

        kpts = tf.expand_dims(tf.reshape(kpts, shape[:2]), 2)
        mask = tf.expand_dims(tf.reshape(mask, shape[:2]), 2)

        kpts.set_shape([None, None, 1])

        return kpts, mask

    def get_keypoints_visible(self, index, shape):
        def wrapper(index, shape):
            index = index.decode("utf-8")

            kpts = np.zeros(shape[:2], dtype=int)
            im = Image(kpts)
            mask = np.ones(list(shape[:2]) + [1]).astype(np.float32)

            path = self.root / '{}.ljson'.format(index)
            lms = mio.import_landmark_file(path)
            pts_all = lms.lms.points
            visible_index = label_index(lms, 'visible')

            for i in range(pts_all.shape[0]):
                if i in visible_index:
                    lms_mask = im.as_masked().copy()
                    patches = np.ones((1, 1, 1, 13, 13), dtype=np.bool)

                    pc = pts_all[i][None, :]
                    lms_mask.mask.pixels[...] = False
                    lms_mask = lms_mask.mask.set_patches(
                        patches, menpo.shape.PointCloud(pc))
                    kpts[lms_mask.mask] = i + 1

            return kpts.astype(np.int32), mask.astype(np.int32)

        kpts, mask = tf.py_func(wrapper, [index, shape], [tf.int32, tf.int32])

        kpts = tf.expand_dims(tf.reshape(kpts, shape[:2]), 2)
        mask = tf.expand_dims(tf.reshape(mask, shape[:2]), 2)

        kpts.set_shape([None, None, 1])

        return kpts, mask


def label_index(lms, label):
    pts_v = lms[label].points
    pts_all = lms.lms.points
    return np.array([np.argwhere(np.linalg.norm(pts_all - v, axis=-1) == 0) for v in pts_v]).squeeze()

class Deblurring(Dataset):
    def __init__(self, batch_size=1):
        super().__init__('FDDB', Path('data/fddb/'), batch_size)
        self.image_extension = 'png'

    def get_images(self, index, shape=None, subdir='images'):
        images = self._get_images(index, shape, subdir=subdir, extension='jpg')
        images_size = tf.shape(images)

        images = tf.image.resize_images(images, (50, 50))
        images = tf.image.resize_images(images, images_size[:2])

        return images

    def get_deblurred(self, index, shape=None):
        deblurred = self._get_images(
            index, shape, subdir='images', extension='jpg')
        deblurred = tf.to_float(deblurred) / 255.
        return deblurred, tf.ones_like(deblurred)


class FDDB(Dataset):
    def __init__(self, batch_size=1):
        super().__init__('FDDB', Path('data/fddb/'), batch_size)
        self.image_extension = 'png'

    def get_images(self, index, shape=None, subdir='images'):
        return self._get_images(index, shape, subdir=subdir, extension='jpg')

    def get_segmentation(self, index, shape=None):
        segmentation = self._get_images(
            index,
            shape,
            subdir='semantic_segmentation',
            channels=1,
            extension='png')

        return segmentation, tf.ones_like(segmentation)


class AFLW(Dataset):
    def __init__(self, batch_size=1):
        super().__init__('AFLW', Path('data/aflw/'), batch_size)
        self.lms_root = self.root / 'landmarks'
        self.lms_extension = 'ljson'
        self.image_extension = 'jpg'

    def get_keypoints(self, index, shape):
        def wrapper(index, shape):
            index = index.decode("utf-8")

            prefix = index.split('_')[0]

            landmark_indices = list(map(int, index.split('_')[1:]))
            if len(landmark_indices) > 1:
                min_index, max_index = landmark_indices
                landmark_indices = range(min_index, max_index + 1)

            kpts = np.zeros(shape[:2], dtype=int)
            im = Image(kpts)

            mask = np.ones(list(shape[:2]) + [1]).astype(np.float32)

            for lms_index in landmark_indices:
                filename = (
                    prefix + '_' + str(lms_index) + '.' + self.lms_extension)
                path = self.lms_root / filename
                if not path.exists():
                    continue
                lms = mio.import_landmark_file(path.as_posix()).lms

                if lms.points.shape[0] != 68:
                    min_indices, max_indices = lms.bounds()

                    mask[min_indices[0]:max_indices[0], min_indices[1]:
                         max_indices[1]] = 0
                    continue

                for i in range(68):
                    lms_mask = im.as_masked().copy()
                    patches = np.ones((1, 1, 1, 4, 4), dtype=np.bool)

                    pc = lms.points[i][None, :]
                    lms_mask.mask.pixels[...] = False
                    lms_mask = lms_mask.mask.set_patches(
                        patches, menpo.shape.PointCloud(pc))
                    kpts[lms_mask.mask] = i + 1

            return kpts.astype(np.int32), mask.astype(np.int32)

        kpts, mask = tf.py_func(wrapper, [index, shape], [tf.int32, tf.int32])

        kpts = tf.expand_dims(tf.reshape(kpts, shape[:2]), 2)
        mask = tf.expand_dims(tf.reshape(mask, shape[:2]), 2)

        return kpts, mask

    def get_segmentation(self, index, shape=None):
        segmentation = self._get_images(
            index,
            shape,
            subdir='semantic_segmentation',
            channels=1,
            extension='png')

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
        super().__init__('AFLW', Path('/vol/atlas/databases/aflw_ibug'),
                         batch_size)
        self.image_extension = '.jpg'
        self.lms_extension = '.ljson'

    def get_keys(self, path='images'):

        path = self.root / path
        lms_files = path.glob('*' + self.lms_extension)
        keys = []  # ['face_55135', 'face_49348']

        # Get only files with 68 landmarks
        for p in lms_files:
            try:
                lms = mio.import_landmark_file(p)

                if lms.n_landmarks == 68 and not np.isnan(lms.lms.points).any(
                ):
                    keys.append(lms.path.stem)
            except:
                pass

        self._keys = keys
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def get_landmarks(self, index, shape=(256, 256)):
        from utils_3d import crop_face

        def wrapper(index):
            path = self.root / (index.decode("utf-8") + self.image_extension)
            im = mio.import_image(path, normalize=False)

            im = crop_face(im)

            pixels = get_pixels(im)
            landmarks = im.landmarks[None].lms.points.astype(np.float32)

            return pixels.astype(np.float32), landmarks

        images, kpts = tf.py_func(wrapper, [index], [tf.float32, tf.float32])

        images.set_shape([shape[0], shape[1], 3])
        kpts.set_shape([68, 2])

        return images, kpts

    def get_keypoints(self, index, shape=(256, 256)):
        from utils_3d import crop_face

        def wrapper(index):
            path = self.root / (index.decode("utf-8") + self.image_extension)
            im = mio.import_image(path, normalize=False)

            im = crop_face(im)
            kpts = np.zeros(im.shape, dtype=int)


            for i in range(68):
                mask = im.as_masked().copy()
                patches = np.ones((1, 1, 1, 4, 4), dtype=np.bool)

                pc = mask.landmarks[None].lms.points[i][None, :]
                mask.mask.pixels[...] = False
                mask = mask.mask.set_patches(patches,
                                             menpo.shape.PointCloud(pc))
                kpts[mask.mask] = i + 1

            pixels = get_pixels(im)

            return pixels.astype(np.float32), kpts.astype(np.int32)

        images, kpts = tf.py_func(wrapper, [index], [tf.float32, tf.int32])

        images.set_shape([shape[0], shape[1], 3])
        kpts.set_shape([shape[0], shape[1]])
        mask = tf.ones_like(kpts)

        return images, kpts, mask

    def get(self, name):
        keys = self.get_keys(path='.')
        producer = tf.train.string_input_producer(
            keys, shuffle=True, capacity=1000)
        key = producer.dequeue()

        if name == 'landmarks':
            mask = tf.ones((self.batch_size, 68, 2))
            image, kpts = self.get_landmarks(key)
        else:
            image, kpts, mask = self.get_keypoints(key)
        image = self.preprocess(image)

        return tf.train.batch(
            [image, kpts, mask], self.batch_size, capacity=1000, dynamic_pad=False)


class FDDBSingle(AFLWSingle):
    def __init__(self, batch_size=1):
        super().__init__(batch_size)
        from menpo.transform import Translation, scale_about_centre
        import menpo3d.io as m3dio

        self.name = 'FDDB'
        self.root = Path('/vol/atlas/databases/fddb_ibug')
        template = m3dio.import_mesh(
            '/vol/construct3dmm/regression/src/template.obj')
        template = Translation(-template.centre()).apply(template)
        self.template = scale_about_centre(template, 1. /
                                           1000.).apply(template)
        pca_path = '/homes/gt108/Projects/ibugface/pose_settings/pca_params.pkl'
        self.eigenvectors, self.eigenvalues, self.h_mean, self.h_max = mio.import_pickle(
            pca_path)
        self.image_extension = '.jpg'
        self.lms_extension = '.ljson'


class LFPWSingle(AFLWSingle):
    def __init__(self, batch_size=1):
        super().__init__(batch_size)

        from menpo.transform import Translation, scale_about_centre
        import menpo3d.io as m3dio

        self.name = 'LFPW'
        self.batch_size = batch_size
        self.root = Path('/vol/atlas/databases/lfpw/trainset')
        template = m3dio.import_mesh(
            '/vol/construct3dmm/regression/src/template.obj')
        template = Translation(-template.centre()).apply(template)
        self.template = scale_about_centre(template, 1. /
                                           1000.).apply(template)
        self.image_extension = '.png'
        self.lms_extension = '.pts'


class Deep3DV1(Dataset):
    def __init__(self, batch_size=32):
        self.name = 'JamesRenders'
        self.batch_size = batch_size
        self.root = Path('/data/datasets/renders/v1')
        self.tfrecord_names = ['train_v2.tfrecords']
        self.model = mio.import_pickle(
            '/vol/construct3dmm/experiments/models/nicp/mein3d/full_unmasked_good_200.pkl')[
                'model']
        self.settings = mio.import_pickle(
            '/vol/construct3dmm/experiments/nicptexture/settings.pkl',
            encoding='latin1')

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
        return tf.train.shuffle_batch(
            [image, uv, parameters],
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
        producer = tf.train.string_input_producer(keys, shuffle=True)
        key = producer.dequeue()
        images, parameters = self.get_data(key)
        images = self.preprocess(images)
        parameters = self.preprocess_params(parameters)

        return tf.train.batch(
            [images, parameters],
            self.batch_size,
            capacity=80,
            dynamic_pad=False)
