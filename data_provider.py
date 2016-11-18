import tensorflow as tf
import numpy as np
import menpo.io as mio
import menpo
import scipy
import utils

from pathlib import Path
from scipy.io import loadmat
from utils_3d import crop_face
from menpo.image import Image
from menpo.shape import PointCloud
from menpo.transform import Translation


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


def augment_img(img, augmentation):
    flip, rotate, rescale = np.array(augmentation).squeeze()
    rimg = img.rescale(rescale)
    rimg = rimg.rotate_ccw_about_centre(rotate)
    crimg = rimg.warp_to_shape(
        img.shape,
        Translation(-np.array(img.shape) / 2 + np.array(rimg.shape) / 2)
    )
    if flip > 0.5:
        crimg = crimg.mirror()

    img = crimg

    return img

def rotate_points_tensor(points, image, angle):

    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # center coordinates since rotation center is supposed to be in the image center
    points_centered = points - image_center

    rot_matrix = tf.dynamic_stitch([[0], [1], [2], [3]], [tf.cos(angle), -tf.sin(angle), tf.sin(angle), tf.cos(angle)])
    rot_matrix = tf.reshape(rot_matrix, shape=[2, 2])

    points_centered_rot = tf.matmul(rot_matrix, tf.transpose(points_centered))

    return tf.transpose(points_centered_rot) + image_center


def rotate_image_tensor(image, angle):
    s = tf.shape(image)
    image_center = tf.to_float(s[:2]) / 2.

    # Coordinates of new image
    xs, ys = tf.meshgrid(tf.range(0.,tf.to_float(s[1])), tf.range(0., tf.to_float(s[0])))
    coords_new = tf.reshape(tf.pack([ys,xs], 2), [-1, 2])

    # center coordinates since rotation center is supposed to be in the image center
    coords_new_centered = tf.to_float(coords_new) - image_center

    # Perform backward transformation of the image coordinates
    rot_mat_inv = tf.pack([tf.cos(angle), tf.sin(angle), -tf.sin(angle), tf.cos(angle)])
    rot_mat_inv = tf.reshape(rot_mat_inv, shape=[2, 2])
    coord_old_centered = tf.matmul(rot_mat_inv, tf.transpose(coords_new_centered))
    coord_old = tf.to_int32(tf.round(tf.transpose(coord_old_centered) + image_center))


    # Find nearest neighbor in old image
    coord_old_y, coord_old_x = tf.unpack(coord_old, axis=1)


    # Clip values to stay inside image coordinates
    outside_y = tf.logical_or(tf.greater(coord_old_y, s[0]-1), tf.less(coord_old_y, 0))
    outside_x = tf.logical_or(tf.greater(coord_old_x, s[1]-1), tf.less(coord_old_x, 0))
    outside_ind = tf.logical_or(outside_y, outside_x)



    inside_mask = tf.logical_not(outside_ind)
    inside_mask = tf.tile(tf.reshape(inside_mask, s[:2])[...,None], tf.pack([1,1,s[2]]))

    coord_old_y = tf.maximum(tf.minimum(coord_old_y, s[0]-1), 0)
    coord_old_x = tf.maximum(tf.minimum(coord_old_x, s[1]-1), 0)

    coord_old =  tf.pack([coord_old_y,coord_old_x], axis=1)


    def sample_fn(coord, image=image):
        y,x = tf.unpack(coord)
        return image[y,x,:]

    rot_image = tf.map_fn(sample_fn,coord_old, dtype=tf.float32)
    rot_image = tf.reshape(rot_image, s)


    return tf.select(inside_mask, rot_image, tf.zeros_like(image))


class ProtobuffProvider(object):
    def __init__(self, filename='mpii_train.tfrecords', root=None, batch_size=1, rescale=None, augmentation=False):
        self.filename = filename
        self.root = Path(root)
        self.batch_size = batch_size
        self.image_extension = 'jpg'
        self.rescale = rescale
        self.augmentation = augmentation


    def get(self, *keys, preprocess_inputs=False):
        images, *names = self._get_data_protobuff(self.root / self.filename, *keys)
        tensors = [images]

        for name in names:
            tensors.append(name)

        return tf.train.shuffle_batch(
            tensors, self.batch_size, 1000, 200, 4)

    def augmentation_type(self):
        return tf.pack([tf.random_uniform([1]),
                        tf.random_uniform([1]) * 60. - 30. * np.pi / 180.,
                        tf.random_uniform([1]) * 0.5 + 0.75])

    # Data from protobuff
    def _get_data_protobuff(self, filename, *keys):
        filename = str(filename)
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                # images
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                # svs
                'n_svs': tf.FixedLenFeature([], tf.int64),
                'n_svs_ch': tf.FixedLenFeature([], tf.int64),
                # 'svs_0': tf.FixedLenFeature([], tf.string),
                'svs_1': tf.FixedLenFeature([], tf.string),
                # 'svs_2': tf.FixedLenFeature([], tf.string),
                'svs_3': tf.FixedLenFeature([], tf.string),
                # landmarks
                'n_landmarks': tf.FixedLenFeature([], tf.int64),
                'gt': tf.FixedLenFeature([], tf.string),
                'visible': tf.FixedLenFeature([], tf.string),
                'marked': tf.FixedLenFeature([], tf.string),
                'scale': tf.FixedLenFeature([], tf.float32),
                # original infomations
                'original_scale': tf.FixedLenFeature([], tf.float32),
                'original_centre': tf.FixedLenFeature([], tf.string),
                'original_lms': tf.FixedLenFeature([], tf.string),
                # inverse transform to original landmarks
                'restore_translation': tf.FixedLenFeature([], tf.string),
                'restore_scale': tf.FixedLenFeature([], tf.float32)
            }

        )

        # image
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        #
        image = tf.reshape(image, (image_height, image_width, 3))
        image = tf.to_float(image)
        #
        #
        # svs
        n_svs = 2#tf.to_int32(features['n_svs'])
        n_svs_ch = 7
        # svs_0 = tf.image.decode_jpeg(features['svs_0'])
        svs_1 = tf.image.decode_jpeg(features['svs_1'])
        # svs_2 = tf.image.decode_jpeg(features['svs_2'])
        svs_3 = tf.image.decode_jpeg(features['svs_3'])

        pose = tf.reshape(tf.pack([svs_3, svs_1]),(n_svs,n_svs_ch,image_height,image_width))
        pose = tf.transpose(pose, perm=[2, 3, 0, 1])
        pose = tf.reshape(pose, (image_height,image_width,n_svs*n_svs_ch))
        pose = tf.to_float(pose) / 255.
        # pose = tf.Print(pose, [tf.shape(pose)], 'svs range: ', summarize=5)
        #
        # landmarks
        def lms_to_heatmap(lms, h, w, n_landmarks, marked_index):
            xs, ys = tf.meshgrid(tf.range(0.,tf.to_float(w)), tf.range(0., tf.to_float(h)))
            sigma = 5.
            gaussian = (1. / (sigma * np.sqrt(2. * np.pi)))



            def gaussian_fn(lms):
                y, x, idx = tf.unpack(lms)
                idx = tf.to_int32(idx)
                def run_true():
                    return tf.exp(-0.5 * (tf.pow(ys - y, 2) + tf.pow(xs - x, 2)) *
                           tf.pow(1. / sigma, 2.)) * gaussian * 17.

                def run_false():
                    return tf.zeros((h,w))

                return tf.cond(tf.reduce_any(tf.equal(marked_index,idx)), run_true, run_false)


            img_hm = tf.pack(tf.map_fn(gaussian_fn, tf.concat(1, [lms, tf.to_float(tf.range(0,16))[..., None]])))


            return img_hm

        n_landmarks = 16
        gt_lms = tf.decode_raw(features['gt'], tf.float32)
        visible = tf.to_int32(tf.decode_raw(features['visible'], tf.int64))
        marked = tf.to_int32(tf.decode_raw(features['marked'], tf.int64))
        scale = features['scale']

        gt_lms = tf.reshape(gt_lms, (n_landmarks, 2))
        gt_heatmap = lms_to_heatmap(gt_lms, image_height, image_width, n_landmarks, marked)
        gt_heatmap = tf.transpose(gt_heatmap, perm=[1,2,0])

        # augmentation
        if self.augmentation:
            do_flip, do_rotate, do_scale = tf.unpack(self.augmentation_type())

            # rescale
            image_height = tf.to_int32(tf.to_float(image_height) * do_scale[0])
            image_width = tf.to_int32(tf.to_float(image_width) * do_scale[0])

            image = tf.image.resize_images(image, tf.pack([image_height, image_width]))
            pose = tf.image.resize_images(pose, tf.pack([image_height, image_width]))
            gt_heatmap = tf.image.resize_images(gt_heatmap, tf.pack([image_height, image_width]))
            gt_lms *= do_scale


            # rotate
            # image = rotate_image_tensor(image, do_rotate)
            # pose = rotate_image_tensor(pose, do_rotate)
            # gt_heatmap = rotate_image_tensor(gt_heatmap, do_rotate)
            # gt_lms = rotate_points_tensor(gt_lms, image, do_rotate)


            # flip
            def flip_fn(image=image, pose=pose, gt_heatmap=gt_heatmap, gt_lms=gt_lms):
                image = tf.image.flip_left_right(image)
                pose = tf.image.flip_left_right(pose)
                gt_heatmap = tf.image.flip_left_right(gt_heatmap)


                pose = tf.reshape(pose, (image_height,image_width,n_svs,n_svs_ch))
                flip_pose_list = []
                for idx in [1,0,2,3,5,4,6]:
                    flip_pose_list.append(pose[:,:,:,idx])
                pose = tf.pack(flip_pose_list, axis=3)
                pose = tf.reshape(pose, (image_height,image_width,n_svs*n_svs_ch))

                flip_hm_list = []
                flip_lms_list = []
                for idx in [5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10]:
                    flip_hm_list.append(gt_heatmap[:,:,idx])
                    flip_lms_list.append(gt_lms[idx,:])

                gt_heatmap = tf.pack(flip_hm_list, axis=2)
                gt_lms = tf.pack(flip_lms_list)

                return image, pose, gt_heatmap, gt_lms

            def no_flip(image=image, pose=pose, gt_heatmap=gt_heatmap, gt_lms=gt_lms):
                return image, pose, gt_heatmap, gt_lms

            image, pose, gt_heatmap, gt_lms = tf.cond(do_flip[0] > 0.5, flip_fn, no_flip)

        # crop to 256 * 256
        target_h = tf.to_int32(256)
        target_w = tf.to_int32(256)
        offset_h = tf.to_int32((image_height - target_h) / 2)
        offset_w = tf.to_int32((image_width - target_w) / 2)

        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w)

        pose = tf.image.crop_to_bounding_box(
            pose, offset_h, offset_w, target_h, target_w)
        pose = tf.reshape(pose, (target_h,target_w,n_svs,n_svs_ch))
        pose = tf.transpose(pose, perm=[2, 0, 1, 3])

        gt_heatmap = tf.image.crop_to_bounding_box(
            gt_heatmap, offset_h, offset_w, target_h, target_w)

        gt_lms -= tf.to_float(tf.pack([offset_h, offset_w]))

        image.set_shape([None, None, 3])
        pose.set_shape([2, None, None, 7])
        gt_heatmap.set_shape([None, None, 16])
        gt_lms.set_shape([16, 2])
        # return image, []
        return image, pose, gt_heatmap, gt_lms, scale


class DeconposePoseProvider(ProtobuffProvider):
    def __init__(self, filename='mpii_train.tfrecords', root=None,
                 batch_size=1, rescale=None, augmentation=False):

        super().__init__(
            filename,
            root,
            batch_size=batch_size,
            rescale=rescale,
            augmentation=augmentation)

    # Data from protobuff
    def _get_data_protobuff(self, filename, *keys):
        filename = str(filename)
        filename_queue = tf.train.string_input_producer([filename],
                                                        num_epochs=None)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                # images
                'image': tf.FixedLenFeature([], tf.string),
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                # svs
                'n_svs': tf.FixedLenFeature([], tf.int64),
                'n_svs_ch': tf.FixedLenFeature([], tf.int64),
                'svs': tf.FixedLenFeature([], tf.string),
                # landmarks
                'n_landmarks': tf.FixedLenFeature([], tf.int64),
                'gt': tf.FixedLenFeature([], tf.string),
                'visible': tf.FixedLenFeature([], tf.string),
                'marked': tf.FixedLenFeature([], tf.string),
                'scale': tf.FixedLenFeature([], tf.float32),
                # original infomations
                'original_scale': tf.FixedLenFeature([], tf.float32),
                'original_centre': tf.FixedLenFeature([], tf.string),
                'original_lms': tf.FixedLenFeature([], tf.string),
                # inverse transform to original landmarks
                'restore_translation': tf.FixedLenFeature([], tf.string),
                'restore_scale': tf.FixedLenFeature([], tf.float32)
            }

        )

        # image
        image = tf.image.decode_jpeg(features['image'], channels=3)
        image_height = tf.to_int32(features['height'])
        image_width = tf.to_int32(features['width'])
        #
        image = tf.reshape(image, (image_height, image_width, 3))
        image = tf.to_float(image)
        #
        #
        # svs
        n_svs = tf.to_int32(features['n_svs'])
        n_svs_ch = tf.to_int32(features['n_svs_ch'])
        svs = tf.image.decode_jpeg(features['svs'])

        pose = tf.reshape(svs,(n_svs_ch,image_height, n_svs, image_width))
        pose = tf.transpose(pose, perm=[1, 3, 2, 0])
        pose = tf.reshape(pose, (image_height,image_width,n_svs*n_svs_ch))
        pose = tf.to_float(pose) / 255.
        #
        # landmarks
        def lms_to_heatmap(lms, h, w, n_landmarks, marked_index):
            xs, ys = tf.meshgrid(tf.range(0.,tf.to_float(w)), tf.range(0., tf.to_float(h)))
            sigma = 5.
            gaussian = (1. / (sigma * np.sqrt(2. * np.pi)))



            def gaussian_fn(lms):
                y, x, idx = tf.unpack(lms)
                idx = tf.to_int32(idx)
                def run_true():
                    return tf.exp(-0.5 * (tf.pow(ys - y, 2) + tf.pow(xs - x, 2)) *
                           tf.pow(1. / sigma, 2.)) * gaussian * 17.

                def run_false():
                    return tf.zeros((h,w))

                return tf.cond(tf.reduce_any(tf.equal(marked_index,idx)), run_true, run_false)


            img_hm = tf.pack(tf.map_fn(gaussian_fn, tf.concat(1, [lms, tf.to_float(tf.range(0,16))[..., None]])))


            return img_hm

        n_landmarks = 16
        gt_lms = tf.decode_raw(features['gt'], tf.float32)
        visible = tf.to_int32(tf.decode_raw(features['visible'], tf.int64))
        marked = tf.to_int32(tf.decode_raw(features['marked'], tf.int64))
        scale = features['scale']

        gt_lms = tf.reshape(gt_lms, (n_landmarks, 2))
        gt_heatmap = lms_to_heatmap(gt_lms, image_height, image_width, n_landmarks, marked)
        gt_heatmap = tf.transpose(gt_heatmap, perm=[1,2,0])

        # augmentation
        if self.augmentation:
            do_flip, do_rotate, do_scale = tf.unpack(self.augmentation_type())

            # rescale
            image_height = tf.to_int32(tf.to_float(image_height) * do_scale[0])
            image_width = tf.to_int32(tf.to_float(image_width) * do_scale[0])

            image = tf.image.resize_images(image, tf.pack([image_height, image_width]))
            pose = tf.image.resize_images(pose, tf.pack([image_height, image_width]))
            gt_heatmap = tf.image.resize_images(gt_heatmap, tf.pack([image_height, image_width]))
            gt_lms *= do_scale


            # flip
            def flip_fn(image=image, pose=pose, gt_heatmap=gt_heatmap, gt_lms=gt_lms):
                image = tf.image.flip_left_right(image)
                pose = tf.image.flip_left_right(pose)
                gt_heatmap = tf.image.flip_left_right(gt_heatmap)


                pose = tf.reshape(pose, (image_height,image_width,n_svs,n_svs_ch))
                flip_pose_list = []
                for idx in [5,4,3,2,1,0,6,7,8,14,13,12,11,10,9]:
                    flip_pose_list.append(pose[:,:,:,idx])
                pose = tf.pack(flip_pose_list, axis=3)
                pose = tf.reshape(pose, (image_height,image_width,n_svs*n_svs_ch))

                flip_hm_list = []
                flip_lms_list = []
                for idx in [5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10]:
                    flip_hm_list.append(gt_heatmap[:,:,idx])
                    flip_lms_list.append(gt_lms[idx,:])

                gt_heatmap = tf.pack(flip_hm_list, axis=2)
                gt_lms = tf.pack(flip_lms_list)

                return image, pose, gt_heatmap, gt_lms

            def no_flip(image=image, pose=pose, gt_heatmap=gt_heatmap, gt_lms=gt_lms):
                return image, pose, gt_heatmap, gt_lms

            image, pose, gt_heatmap, gt_lms = tf.cond(do_flip[0] > 0.5, flip_fn, no_flip)

        # crop to 256 * 256
        target_h = tf.to_int32(256)
        target_w = tf.to_int32(256)
        offset_h = tf.to_int32((image_height - target_h) / 2)
        offset_w = tf.to_int32((image_width - target_w) / 2)

        image = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w)

        pose = tf.image.crop_to_bounding_box(
            pose, offset_h, offset_w, target_h, target_w)
        pose = tf.reshape(pose, (target_h,target_w,n_svs,n_svs_ch))
        pose = tf.transpose(pose, perm=[2, 0, 1, 3])

        gt_heatmap = tf.image.crop_to_bounding_box(
            gt_heatmap, offset_h, offset_w, target_h, target_w)

        gt_lms -= tf.to_float(tf.pack([offset_h, offset_w]))

        image.set_shape([None, None, 3])
        pose.set_shape([4, None, None, 15])
        gt_heatmap.set_shape([None, None, 16])
        gt_lms.set_shape([16, 2])

        # return image, []
        return image, pose, gt_heatmap, gt_lms, scale


# Data from files
class Dataset(object):
    def __init__(self, name, root, batch_size=1, rescale=None, augmentation=False):
        self.name = name
        self.root = Path(root)
        self.batch_size = batch_size
        self.image_extension = 'png'
        self.images_root = 'images'
        self.rescale = rescale
        self.augmentation = augmentation

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

    def augmentation_type(self):
        return tf.pack([tf.random_uniform([1]),
                        tf.random_uniform([1]) * 60 - 30,
                        tf.random_uniform([1]) * 0.5 + 0.75])

    def get(self, *names, preprocess_inputs=True, create_batches=True):
        producer = tf.train.string_input_producer(
            self.get_keys(), shuffle=True)
        key = producer.dequeue()

        augmentation_type = self.augmentation_type()
        images = self.get_images(key, augmentation_type=augmentation_type)
        image_shape = tf.shape(images)
        images = self.rescale_image(images, image_shape=image_shape)

        if preprocess_inputs:
            images = self.preprocess(images)

        tensors = [images]

        for name in names:
            fun = getattr(self, 'get_' + name.split('/')[0])
            use_mask = (
                len(name.split('/')) > 1) and name.split('/')[1] == 'mask'

            label, mask = fun(key, shape=image_shape, augmentation_type=augmentation_type)
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
    def __init__(self, batch_size=1, root='/data/yz4009/',
                 valid_check=False, n_lms=16, rescale=None,
                 svs_index=[0,1,2,3], augmentation=False):
        super().__init__(
            'human_pose',
            Path(root),
            batch_size=batch_size,
            rescale=rescale,
            augmentation=augmentation)
        self.image_extension = 'jpg'
        self.images_root = '.'
        self.valid_check = valid_check
        self.n_lms = n_lms
        self.svs_index = svs_index

    def get_keys(self):
        # if (self.root / 'keys.pkl').exists():
        #     keys = mio.import_pickle(self.root / 'keys.pkl')
        #     self._keys = keys
        #     return tf.constant(keys)

        path = self.root

        def check_valid(x):
            return all([(path / '{}+svs_dark+{:02d}.pkl'.format(x, i)).exists()
                        for i in self.svs_index])

        if self.valid_check:
            keys = [str(x.stem) for x in path.glob('*.jpg') if check_valid(x.stem)]
        else:
            keys = [str(x.stem) for x in path.glob('*.jpg')]
        self._keys = keys

        # mio.export_pickle(keys, self.root / 'keys.pkl')
        print('Found {} files.'.format(len(keys)))

        if len(keys) == 0:
            raise RuntimeError('No images found in {}'.format(path))
        return tf.constant(keys, tf.string)

    def rescale_image(self, image, method=None, image_shape=None):
        if not image_shape is None and self.rescale:
            h, w = tf.to_float(image_shape[0]), tf.to_float(image_shape[1])
            scale = tf.reduce_max([h,w]) / self.rescale
            # scale = 1
            nh, nw = tf.to_int32(h/scale), tf.to_int32(w/scale)

            if not method is None:
                if method.split('/')[0] == 'landmarks':
                    if len(method.split('/')) > 1:
                        return image
                    return image / scale
                elif method.split('/')[0] == 'pose':
                    return tf.image.resize_bilinear(image, [nh, nw])
                elif method.split('/')[0] == 'scale':
                    return image / scale

            return tf.image.resize_bilinear(image[None, ...], [nh, nw])[0]
        else:
            return image

    def get_images(self, index, shape=None, augmentation_type=None):

        def wrapper(index, augmentation_type):

            index = index.decode("utf-8")
            extension='jpg'

            path = Path("".join([str(self.root / self.images_root), '/', index, '.', extension]))

            if self.augmentation:
                # info = mio.import_pickle(path.parent / '{}_info.pkl'.format(path.stem))
                # database_path = Path('/vol/atlas/databases/body/MPIIHumanPose')
                # mpii_img_path = database_path / 'images' / (path.stem.split('_')[0] + '.jpg')
                # orgimg = utils.import_image(mpii_img_path)
                # corgimg = orgimg.warp_to_shape(
                #     orgimg.shape,
                #     Translation(-np.array(orgimg.shape)/2+info['original_centre']))
                # acorgimg = augment_img(corgimg, augmentation_type)
                # ncimg, *_ = utils.crop_image(
                #     acorgimg,
                #     np.array(acorgimg.shape)/2,
                #     info['original_scale'], (340,340))
                # img = ncimg
                img = utils.import_image(path)
                img = augment_img(img, augmentation_type)
            else:
                img = utils.import_image(path)

            return np.array(img.pixels_with_channels_at_back()).astype(np.float32)

        image, = tf.py_func(wrapper, [index, augmentation_type], [tf.float32])
        image.set_shape([None, None, 3])
        return image

    def get_scale(self, index, shape, augmentation_type=None):
        def wrapper(index, augmentation_type):
            index = index.decode("utf-8")
            info = mio.import_pickle(
                self.root / '{}_info.pkl'.format(index))

            scale = info['scale']

            if self.augmentation:
                scale *= np.array(augmentation_type).squeeze()[2]

            return np.array([scale]).astype(np.float32)

        scale, = tf.py_func(wrapper, [index, augmentation_type], [tf.float32])
        scale.set_shape([1])
        return scale, None


    def get_pose(self, index, shape, augmentation_type=None):
        def wrapper(index, augmentation_type):
            index = index.decode("utf-8")
            result = []

            for i in self.svs_index:
                try:
                    svs = mio.import_pickle(
                        self.root / '{}+svs_dark+{:02d}.pkl'.format(index, i), encoding='latin1')
                except:
                    print (self.root / '{}+svs_dark+{:02d}.pkl'.format(index, i))


                    import glob
                    list(map(glob.os.remove, glob.glob(self.root / '{}+svs_dark+*.pkl'.format(index))))

                    raise Exception(self.root / '{}+svs_dark+{:02d}.pkl'.format(index, i))

                if self.augmentation:
                    svs = augment_img(Image(svs.pixels.astype(np.float32)), augmentation_type)

                    if np.array(augmentation_type).squeeze()[0] > 0.5:
                        svs.pixels = svs.pixels[[1,0,2,3,5,4,6],:,:]

                svs = svs.pixels_with_channels_at_back()
                result.append(svs)
            return np.array(result).astype(np.float32)

        svs, = tf.py_func(wrapper, [index, augmentation_type], [tf.float32])
        svs.set_shape([len(self.svs_index), None, None, 7])
        return svs, tf.ones_like(svs)

    def get_heatmap(self, index, shape, augmentation_type=None, sigma=5):
        def wrapper(index, shape, augmentation_type):
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

            if self.augmentation:
                img_hm = augment_img(img_hm, augmentation_type)
                # if flipped flip also channels
                if np.array(augmentation_type).squeeze()[0] > 0.5:
                    img_hm.pixels = img_hm.pixels[[5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10],:,:]

            return img_hm.pixels_with_channels_at_back().astype(np.float32)

        hm, = tf.py_func(wrapper, [index, shape, augmentation_type], [tf.float32])
        hm.set_shape([None, None, self.n_lms])
        return hm, tf.ones_like(hm)

    def get_landmarks(self, index, shape, augmentation_type=None):
        def wrapper(index, shape, augmentation_type):
            index = index.decode("utf-8")
            lms = mio.import_landmark_file(
                self.root / '{}.ljson'.format(index))
            marked_index = label_index(lms, 'marked')
            mask = np.zeros(lms.lms.points.shape).astype(np.int32)
            mask[[marked_index], :] = 1

            lms = lms.lms

            if self.augmentation:
                img = Image.init_blank(shape[:2])
                img.landmarks['PTS'] = lms
                img = augment_img(img, augmentation_type)
                lms = img.landmarks['PTS'].lms
                if np.array(augmentation_type).squeeze()[0] > 0.5:
                    lms = PointCloud(lms.points[
                        [5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10]])


            return lms.points.astype(np.float32), mask

        lms, mask = tf.py_func(wrapper, [index, shape, augmentation_type], [tf.float32, tf.int32])
        lms.set_shape([self.n_lms,2])
        mask.set_shape([self.n_lms,2])
        return lms, mask

    def get_keypoints(self, index, shape, augmentation_type=None):
        def wrapper(index, shape, augmentation_type):
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

                    label = i
                    if self.augmentation:
                        if np.array(augmentation_type).squeeze()[0] > 0.5:
                            label = [5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10][i]

                    kpts[lms_mask.mask] = label + 1

            if self.augmentation:
                img = Image(kpts)
                img = augment_img(img, augmentation_type)
                kpts = img.pixels_with_channels_at_back()

            return kpts.astype(np.int32), mask.astype(np.int32)

        kpts, mask = tf.py_func(wrapper, [index, shape, augmentation_type], [tf.int32, tf.int32])

        kpts = tf.expand_dims(tf.reshape(kpts, shape[:2]), 2)
        mask = tf.expand_dims(tf.reshape(mask, shape[:2]), 2)

        kpts.set_shape([None, None, 1])

        return kpts, mask

    def get_keypoints_visible(self, index, shape, augmentation_type=None):
        def wrapper(index, shape, augmentation_type):
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

                    label = i

                    if self.augmentation:
                        if np.array(augmentation_type).squeeze()[0] > 0.5:
                            label = [5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10][i]

                    kpts[lms_mask.mask] = label + 1

            if self.augmentation:
                img = Image(kpts.astype(np.float32))
                img = augment_img(img, augmentation_type)
                kpts = img.pixels_with_channels_at_back()

            return kpts.astype(np.int32), mask.astype(np.int32)

        kpts, mask = tf.py_func(wrapper, [index, shape, augmentation_type], [tf.int32, tf.int32])

        kpts = tf.expand_dims(tf.reshape(kpts, shape[:2]), 2)
        mask = tf.expand_dims(tf.reshape(mask, shape[:2]), 2)

        kpts.set_shape([None, None, 1])

        return kpts, mask

def label_index(lms, label):
    pts_v = lms[label].points
    pts_all = lms.lms.points
    return np.unique(np.concatenate(
        [np.argwhere(np.linalg.norm(pts_all - v, axis=-1) == 0) for v in pts_v]
        ).squeeze())

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
