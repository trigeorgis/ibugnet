import tensorflow as tf
import numpy as np
import face_model
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
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '',
                           '''If specified, restore this pretrained model '''
                           '''before beginning any training.''')
tf.app.flags.DEFINE_integer('max_steps', 100000,
                            '''Number of batches to run.''')
tf.app.flags.DEFINE_string('train_device', '/gpu:0',
                           '''Device to train with.''')
tf.app.flags.DEFINE_string('dataset_path', '', 'Dataset directory')

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


def train():
    g = tf.Graph()
    with g.as_default():
        # Load dataset.
        provider = data_provider.ICT3DFE()
        images, normals = provider.get('normals')

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=True):
                prediction, pyramid = face_model.multiscale_nrm_net(images)

        # Add a smoothed l1 loss to every scale and the combined output.
        for net in [prediction] + pyramid:
            loss = losses.smooth_l1(net, normals)
            slim.losses.add_loss(loss)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver([v for v in tf.trainable_variables() if 'seg' not in v.name and 'nrm' not in v.name])

        if FLAGS.pretrained_model_checkpoint_path:
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)

        train_op = slim.learning.create_train_op(
            total_loss, optimizer, summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            # train_step_fn=train_step_fn,
                            save_summaries_secs=60,
                            save_interval_secs=600)


if __name__ == '__main__':
    train()
