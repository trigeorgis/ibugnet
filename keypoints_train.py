import tensorflow as tf
import numpy as np
import resnet_model
import losses
import data_provider
import utils

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0005,
                          '''Initial learning rate.''')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 5.0,
                          '''Epochs after which learning rate decays.''')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.97,
                          '''Learning rate decay factor.''')
tf.app.flags.DEFINE_integer('batch_size', 32, '''The batch size to use.''')
tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            '''How many preprocess threads to use.''')
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train_kpt',
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


def restore_resnet(sess, path):
    def name_in_checkpoint(var):
        name = '/'.join(var.name.split('/')[2:])
        name = name.split(':')[0]
        if 'Adam' in name:
            return None
        return name

    variables_to_restore = slim.get_variables_to_restore(
        include=["net/multiscale/resnet_v1_50"])
    variables_to_restore = {name_in_checkpoint(var): var
                            for var in variables_to_restore
                            if name_in_checkpoint(var) is not None}

    return slim.assign_from_checkpoint_fn(
        path, variables_to_restore, ignore_missing_vars=True)


def train():
    g = tf.Graph()
    num_classes = 69
    batch_size = 1

    with g.as_default():
        # Load dataset.
        provider = data_provider.AFLW(batch_size=batch_size)
        images, keypoints, mask = provider.get('keypoints/mask')
        
        keypoints_transformed = dpm(keypoints)
        
        keypoints = tf.to_int32(keypoints)
        is_background = tf.equal(keypoints, 0)
        ones = tf.to_float(tf.ones_like(is_background))
        zeros = tf.to_float(tf.zeros_like(is_background))
        keypoints = tf.reshape(keypoints, (-1,))
        keypoints = slim.layers.one_hot_encoding(keypoints, num_classes=num_classes)

        weights = tf.select(is_background, ones, ones) * tf.to_float(mask)

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                prediction, pyramid, _ = resnet_model.multiscale_kpts_net(images, scales=(1, 2, 4))

        background_is_very_confident = tf.nn.softmax(prediction)[..., 0] > .99
        prediction_is_actually_background = tf.equal(background_is_very_confident, is_background)

        weights = tf.select(prediction_is_actually_background, zeros, weights)
        weights = tf.reshape(weights, (-1,))
        weights.set_shape([None,])
        heatmaps = utils.generate_heatmap(prediction)
        tf.image_summary('predictions', heatmaps)
        tf.image_summary('images', images)

        # Add a cosine loss to every scale and the combined output.
        for net in [prediction] + pyramid:
            net = tf.reshape(net, (-1, num_classes))
            # slim.losses.softmax_cross_entropy(net, keypoints, weight=weights )
            
            slim.losses.mean_squared_error(net, keypoints_transformed - dpm)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:


        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess,
                                     FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                    FLAGS.pretrained_model_checkpoint_path, variables_to_restore)

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
