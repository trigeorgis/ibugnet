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
tf.app.flags.DEFINE_string('train_dir', 'ckpt/train_normals',
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


class DatasetMixer():
    
    def __init__(self, names, densities=None, batch_size=1):
        self.providers = []
        self.batch_size = batch_size
        
        if densities is None:
            densities = [1] * len(names)

        for name, bs in zip(names, densities):
            provider = getattr(data_provider, name)(batch_size=bs)
            self.providers.append(provider)
            
    def get(self, *names):
        queue = None
        enqueue_ops = []
        for p in self.providers:
            tensors = p.get(*names, create_batches=True)
            
            shapes = [x.get_shape() for x in tensors]


            if queue is None:
                dtypes = [x.dtype for x in tensors]
                queue = tf.FIFOQueue(
                    capacity=1000,
                    dtypes=dtypes, name='fifoqueue')

            enqueue_ops.append(queue.enqueue_many(tensors))

        qr = tf.train.QueueRunner(queue, enqueue_ops)
        tf.train.add_queue_runner(qr)      

        tensors = queue.dequeue()
        
        for t, s in zip(tensors, shapes):
            t.set_shape(s[1:])

        return tf.train.batch(
            tensors,
            self.batch_size,
            num_threads=2,
            enqueue_many=False,
            dynamic_pad=True,
            capacity=200)


    
def train():
    g = tf.Graph()
    with g.as_default():
        # Load datasets.
        provider = DatasetMixer(('BaselNormals', 'MeinNormals'))
        images, normals, mask = provider.get('normals/mask')
        
        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                scales = [1, 2, 4]
                prediction, pyramid = resnet_model.multiscale_nrm_net(images, scales=scales)

        # Add a cosine loss to every scale and the combined output.
        for net, level_name in zip([prediction] + pyramid, ['pred'] + scales):
            loss = losses.cosine_loss(net, normals, mask)
            tf.scalar_summary('losses/loss at {}'.format(level_name), loss)

        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    config = tf.ConfigProto(inter_op_parallelism_threads=2)
    with tf.Session(graph=g, config=config) as sess:

        init_fn = None
        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                    FLAGS.pretrained_model_checkpoint_path,
                    variables_to_restore, ignore_missing_vars=True)

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
