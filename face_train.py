import tensorflow as tf
import numpy as np
import face_model
import losses

from tensorflow.python.platform import tf_logging as logging
from scipy.io import loadmat
from pathlib import Path

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          """Initial learning rate.""")
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          """Epochs after which learning rate decays.""")
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          """Learning rate decay factor.""")
tf.app.flags.DEFINE_integer('batch_size', 32, """The batch size to use.""")
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            """How many preprocess threads to use.""")
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           """If specified, restore this pretrained model """
                           """before beginning any training.""")
tf.app.flags.DEFINE_integer('max_steps', 10000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('train_device', '/gpu:0', """Device to train with.""")
tf.app.flags.DEFINE_string('dataset_path', '', 'Dataset directory')
# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    
    # RGB -> BGR
    image = tf.reverse(image, [False, False, True])
    # Subtract VGG training mean across all channels
    image = image - VGG_MEAN.reshape([1, 1, 3])
    return image

def rescale_images(images, stride_width=64):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height = tf.to_float(tf.shape(images)[1])
    width = tf.to_float(tf.shape(images)[2])

    # Taken from 'szross'
    scale_up = 625. / tf.minimum(height, width)
    scale_cap = 961. / tf.maximum(height, width)
    scale_up  = tf.minimum(scale_up, scale_cap)
    new_height = stride_width * tf.round((height * scale_up) / stride_width) + 1
    new_width = stride_width * tf.round((width * scale_up) / stride_width) + 1
    new_height = tf.to_int32(new_height)
    new_width = tf.to_int32(new_width)

    images = tf.image.resize_images(images, new_height, new_width)
    return images

def load_mat_normals(path):
    return loadmat(path)['norms'].astype(np.float32)

def ict_dataset():
    root_path = Path('/vol/atlas/homes/iasonas/frcnn/data/ict3drfe/')
    indexes = [x.stem for x in (root_path / 'images').glob('*.png')]
    images_paths = tf.pack(map(str, [root_path / 'images' / (index + '.png') for index in indexes]))
    normals_paths = tf.pack(map(str, [root_path / 'normals' / (index + '.mat') for index in indexes]))
    
        
    image_path, normals_path = tf.train.slice_input_producer(
        [images_paths, normals_paths], capacity=1000)

    image = tf.image.decode_png(tf.read_file(image_path), 3)
    normals, = tf.py_func(load_mat_normals, [normals_path], [tf.float32])
    normals.set_shape(image.get_shape())
    image = tf.to_float(image)

    items_to_handlers = {
      'image': image,
      'normals': normals,
    }
    
    image = caffe_preprocess(image)
    images = rescale_images(tf.expand_dims(image, 0))
    normals = rescale_images(tf.expand_dims(normals, 0))

    return images, normals

def train():
    g = tf.Graph()
    
    with g.as_default():
        # Decay the learning rate exponentially based on the number of steps.
        lr = 0.001
        # Initialise session.
        sess = tf.Session()
        
        # Load dataset.
        images, normals = ict_dataset()
        
        # Initialize thread coordinator.
        coord = tf.train.Coordinator()
        tf.train.start_queue_runners(sess, coord)

        # Define model graph.
        with tf.variable_scope('net'):
            net, pyramid = face_model.multiscale_net(images)

        saver = tf.train.Saver()

        if FLAGS.pretrained_model_checkpoint_path:
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)

        sess.run(tf.initialize_all_variables())

        for prediction in [net] + pyramid:
            loss = losses.smooth_l1(prediction, normals)
            slim.losses.add_loss(loss)
        
        total_loss = slim.losses.get_total_loss()
        
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
        
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        logging.set_verbosity(1)
        slim.learning.train(train_op, FLAGS.train_dir, saver=saver)
        
if __name__ == '__main__':
    train()