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


def restore_resnet(sess, path):
    def name_in_checkpoint(var):
        if 'resnet_v1_50/conv1/weights' in var.name:
            return None
        name = '/'.join(var.name.split('/')[2:])
        name = name.split(':')[0]
        if 'Adam' in name:
            return None
        return name

    variables_to_restore = slim.get_variables_to_restore(
        include=["net/multiscale/resnet_v1_50"])
    variables_to_restore = {name_in_checkpoint(var): var
                            for var in variables_to_restore if name_in_checkpoint(var) is not None}

    return slim.assign_from_checkpoint_fn(path, variables_to_restore, ignore_missing_vars=True)


def train():
    g = tf.Graph()
    
    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose()
        images, ground_truth = provider.get('pose')

        
        # TODO: Current code assumes batch_size=1. 
        # The states Tensor must be of dimensions
        # (batch_size, num_states, height, width, num_parts)

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):

                prediction, states = resnet_model.svs_regression_net(images)
                
                # States currently is
                # (num_states, batch_size, height, width, num_parts)

        
        # Add a cosine loss to every scale and the combined output.
        for i, state in enumerate(states):
            gt = ground_truth[:, i, :, :, :]

            # TODO: Move 100 to a flag.
            # Reweighting the loss by 100. If we do not do this 
            # The loss becomes extremely small.
            ones = tf.ones_like(gt)
            weights = tf.select(gt < .1, ones, ones * 100)
            loss = losses.smooth_l1(state, gt, weights)
            tf.scalar_summary('losses/iteration_{}'.format(i), loss)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        for i in range(len(states)):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None])
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(ground_truth[:, i, :, :, :], -1)[..., None])

            for j in range(7):
                state = states[i][..., j][..., None]
                gt = ground_truth[:, i, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(2, (state, gt)))

        tf.image_summary('image', images)
            
        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                    FLAGS.pretrained_model_checkpoint_path, variables_to_restore, ignore_missing_vars=True)


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
