import tensorflow as tf
import numpy as np
import resnet_model
import hourglass_model
import losses
import data_provider
import utils

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path
import matplotlib.pyplot as plt

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('initial_learning_rate', 0.0005,
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
tf.app.flags.DEFINE_string('eval_dir', 'ckpt/eval_svs',
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


def ced_accuracy(t, dists):
    return tf.reduce_sum(tf.to_int32(dists <= t), 1) / tf.shape(dists)[1]


def pckh(preds, gts):
    t_range = np.arange(0,0.51,0.01)
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2), reduction_indices=-1)) / 4.5
    pckh = [ced_accuracy(t, dists) for t in t_range]
    return pckh[-1]

keypoint_colours = np.array(
    [plt.cm.spectral(x) for x in np.linspace(0, 1, 13)]
)[..., :3].astype(np.float32)

def generate_heatmap(logits):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, 13].
    """
    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(prediction, (-1, 13)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap


def test(output_svs=5, output_lms=13, num_iterations=4):
    g = tf.Graph()

    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(batch_size=FLAGS.batch_size)
        images, gt_landmarks = provider.get('landmarks')

        tf.image_summary('images', images)

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):

                # svs regression net
                prediction, states = resnet_model.svs_regression_net_light(images, output_classes=output_svs, num_iterations=num_iterations)
                # States currently is
                # (num_states, batch_size, height, width, num_parts)
                net = tf.concat(3, [images, prediction], name='concat-bridge')
                lms_heatmap_prediction = hourglass_model.network(net, 1, output_channels=output_lms)


        hs = tf.argmax(tf.reduce_max(lms_heatmap_prediction, 2), 1)
        ws = tf.argmax(tf.reduce_max(lms_heatmap_prediction, 1), 1)
        predictions = tf.transpose(tf.to_float(tf.pack([hs, ws])), perm=[1, 2, 0])

    with tf.Session(graph=g) as sess:

        accuracy = pckh(predictions, gt_landmarks)
        accuracy = tf.Print(accuracy, [tf.shape(accuracy), accuracy], summarize=5)
        # These are streaming metrics which compute the "running" metric,
        # e.g running accuracy
        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
            "streaming_pckh": slim.metrics.streaming_mean(tf.reduce_mean(accuracy)),
        })

        # Define the streaming summaries to write:
        summary_ops = []
        for metric_name, metric_value in metrics_to_values.items():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        summary_ops.append(tf.scalar_summary('accuracy_pckh', tf.reduce_mean(accuracy)))

        summary_ops.append(
            tf.image_summary('prediction',
                             generate_heatmap(lms_heatmap_prediction)))

        global_step = slim.get_or_create_global_step()
        # Evaluate every 30 seconds
        logging.set_verbosity(1)
        num_examples = provider.num_samples()
        num_batches = np.ceil(num_examples / FLAGS.batch_size)
        slim.evaluation.evaluation_loop(
            '',
            FLAGS.train_dir,
            FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(metrics_to_updates.values()),
            summary_op=tf.merge_summary(summary_ops),
            eval_interval_secs=30)


if __name__ == '__main__':
    test()
