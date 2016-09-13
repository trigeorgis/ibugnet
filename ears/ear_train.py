import tensorflow as tf
import numpy as np
import data_provider

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.001,
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


def network(inputs, scale=1, output_classes=500):


    net = slim.layers.fully_connected(inputs,
                      1024,
                      scope='fc1')

    net = slim.layers.fully_connected(net,
                      1024,
                      scope='fc2')

    net = slim.layers.fully_connected(net,
                      output_classes,
                      scope='pred',
                      activation_fn=None)

    return net

def train():
    g = tf.Graph()
    with g.as_default():
        # Load dataset.
        provider = data_provider.EarWPUTEDB(batch_size=32)
        images, labels = provider.get('labels')

        # Define model graph.
        with tf.variable_scope('net'):
            prediction = network(images)

        # Add a smoothed l1 loss to every scale and the combined output.
        slim.losses.softmax_cross_entropy(prediction, labels)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:

        if FLAGS.pretrained_model_checkpoint_path:
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)

        train_op = slim.learning.create_train_op(
            total_loss, optimizer, summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            save_interval_secs=600)


if __name__ == '__main__':
    train()
