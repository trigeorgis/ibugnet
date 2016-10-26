import tensorflow as tf
import numpy as np
import hourglass_model
import losses
import data_provider
import utils
import matplotlib.pyplot as plt

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0001,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('batch_size', 1, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_iterations', 4, '''The number of iterations to unfold the pose machine.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many preprocess threads to use.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train_svs',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_string(
    'pretrained_resnet_checkpoint_path', '',
    '''If specified, restore this pretrained resnet '''
    '''before beginning any training.'''
    '''This restores only the weights of the resnet model''')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0',
                           '''Device to train with.''')

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999

keypoint_colours = np.array([plt.cm.spectral(x) for x in np.linspace(0, 1, 14)])[
    ..., :3].astype(np.float32)

def generate_heatmap(logits):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, 14].
    """
    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(prediction, (-1, 14)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap


def train(output_classes=14):
    g = tf.Graph()

    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(batch_size=FLAGS.batch_size)
        images, keypoints, mask = provider.get('keypoints/mask')

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):

                # lms regression net
                prediction = hourglass_model.test_network(images, 1, output_channels=output_classes)


        # losses l1 smooth
        # ones = tf.ones_like(ground_truth)
        # weights = tf.select(ground_truth < .1, ones, ones * 10000) * mask
        # loss = losses.smooth_l1(prediction, ground_truth, weights)
        # tf.scalar_summary('losses/heatmap_smooth_l1', loss)
        #
        # total_loss = slim.losses.get_total_loss()
        # tf.scalar_summary('losses/total loss', total_loss)
        #
        # tf.image_summary('image', images)
        # tf.image_summary('gt_heatmap', tf.reduce_sum(ground_truth,-1)[..., None])
        # tf.image_summary('gt_heatmap_mask', tf.reduce_sum(mask,-1)[..., None])
        # tf.image_summary('pred_heatmap', tf.reduce_sum(prediction,-1)[..., None])

        # losses softmax_cross_entropy
        keypoints = tf.to_int32(keypoints)

        is_background = tf.equal(keypoints, 0)
        ones = tf.to_float(tf.ones_like(is_background))
        zeros = tf.to_float(tf.zeros_like(is_background))

        tf.image_summary('gt', tf.select(is_background, zeros, ones) * tf.to_float(mask) * 255)
        tf.image_summary('predictions', generate_heatmap(prediction))
        tf.image_summary('images', images)

        weights = tf.select(is_background, ones * 0.001, ones) * tf.to_float(mask)
        weights = tf.reshape(weights, (-1,))
        weights.set_shape([None,])

        # Add a cosine loss to every scale and the combined output.
        prediction = tf.reshape(prediction, (-1, output_classes))
        keypoints = tf.reshape(keypoints, (-1,))
        keypoints = slim.layers.one_hot_encoding(keypoints, num_classes=output_classes)
        slim.losses.softmax_cross_entropy(prediction, keypoints, weight=weights)
        total_loss = slim.losses.get_total_loss()

        tf.scalar_summary('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_resnet_checkpoint_path:
            # init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)
            pass

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                {'/'.join(var.op.name.split('/')[1:]):var for var in variables_to_restore if 'logits' not in var.name and not 'adam' in var.name.lower()},
                ignore_missing_vars=True)


        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)

if __name__ == '__main__':
    train()
