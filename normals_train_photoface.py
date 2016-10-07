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
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train',
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
                            for var in variables_to_restore if name_in_checkpoint(var) is not None}

    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, path)

def reconstruct_image(albedo, normals, light):
    return albedo * (tf.reduce_sum(normals * light[None, None, None, :], 3))

def train():
    g = tf.Graph()
    with g.as_default():
        # Load datasets.
        provider = data_provider.Photoface()
        albedo, normals, mask = provider.get('normals/mask', preprocess_inputs=False)
        albedo = albedo[..., 0]
        light = tf.random_uniform((3,)) * tf.to_float((tf.shape(albedo)[1], tf.shape(albedo)[2], 0))
        images = reconstruct_image(albedo, normals, light)
        images = tf.concat(3, [images[..., None]]*3)
        preprocessed = provider.preprocess(images[0])[None, ...]

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                scales = [1, 2, 4]
                prediction, pyramid = resnet_model.multiscale_nrm_net(preprocessed, scales=scales)

        tf.image_summary('images', images)
        tf.image_summary('albedo', albedo[..., None])
        tf.image_summary('reconstruction', reconstruct_image(albedo, prediction, light))
        # Add a cosine loss to every scale and the combined output.
        for net, level_name in zip([prediction] + pyramid, ['pred'] + scales):
            loss = losses.cosine_loss(net, normals, mask)
            tf.scalar_summary('losses/loss at {}'.format(level_name), loss)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:

        if FLAGS.pretrained_resnet_checkpoint_path:
            restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            variables_to_restore = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables_to_restore)
            saver.restore(sess, FLAGS.pretrained_model_checkpoint_path)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            save_interval_secs=600)


if __name__ == '__main__':
    train()
