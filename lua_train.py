import tensorflow as tf
import numpy as np
import resnet_model
import hourglass_model
import losses
import data_provider
import utils
import matplotlib.pyplot as plt
import pickle

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

from flags import FLAGS

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


keypoint_colours = np.array([plt.cm.spectral(x) for x in np.linspace(0, 1, FLAGS.n_landmarks + 1)])[
    ..., :3].astype(np.float32)

def generate_heatmap(logits):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, FLAGS.n_landmarks + 1].
    """
    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(prediction, (-1, FLAGS.n_landmarks + 1)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap


def generate_landmarks(keypoints):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    zeros = tf.to_float(tf.zeros_like(is_background))

    return tf.select(is_background, zeros, ones) * 255


def restore_resnet(sess, path):
    def name_in_checkpoint(var):
        # Uncomment for non lightweight model
        # if 'resnet_v1_50/conv1/weights' in var.name:
        #     return None
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


def build_graph(inputs, tree):
    net = inputs

    if tree['name'] == 'nn.Sequential':
        with tf.variable_scope('nn.Sequential'):
            for tr in tree['children']:
                net = build_graph(net, tr)
    elif tree['name'] == 'nn.ConcatTable':
        net_table = []
        with tf.variable_scope('nn.ConcatTable'):
            for tr in tree['children']:
                net_table.append(build_graph(net, tr))
        net = net_table
    elif tree['name'] == 'nn.JoinTable':
        net = tf.concat(3, net)
    elif tree['name'] == 'nn.CAddTable':
        net = tf.add_n(net)
    elif tree['name'] == 'nn.SpatialConvolution':
        net = slim.conv2d(net,
                          int(tree['nOutputPlane']),
                          (int(tree['kH']),int(tree['kW'])),
                          (int(tree['dH']),int(tree['dW'])),
                          activation_fn=None
                         )
    elif tree['name'] == 'nn.SpatialFullConvolution':
        net = tf.image.resize_bilinear(net, tf.shape(net)[1:3] * int(tree['kH']), name="up_sample")
    elif tree['name'] == 'nn.SpatialBatchNormalization':
        net = slim.batch_norm(net)
    elif tree['name'] == 'nn.ReLU':
        net = slim.nn.relu(net)
    elif tree['name'] == 'nn.Sigmoid':
        net = slim.nn.sigmoid(net)
    elif tree['name'] == 'nn.SpatialMaxPooling':
        net = slim.max_pool2d(
            tf.pad(
                net, [
                    [0,0],
                    [int(tree['padH']),int(tree['padH'])],
                    [int(tree['padW']),int(tree['padW'])],
                    [0,0]
                ]),
            (int(tree['kH']),int(tree['kW'])),
            (int(tree['dH']),int(tree['dW']))
        )
    elif tree['name'] == 'nn.Identity':
        pass
    else:
        raise Exception(tree['name'])

    return net


def train(data, output_svs=7, output_lms=FLAGS.n_landmarks):
    g = tf.Graph()
    logging.set_verbosity(10)


    def keypts_encoding(keypoints):
        keypoints = tf.to_int32(keypoints)
        keypoints = tf.reshape(keypoints, (-1,))
        keypoints = slim.layers.one_hot_encoding(keypoints, num_classes=output_lms+1)
        return keypoints

    def get_weight(keypoints, mask, ng_w=0.01, ps_w=1.0):
        is_background = tf.equal(keypoints, 0)
        ones = tf.to_float(tf.ones_like(is_background))
        weights = tf.select(is_background, ones * ng_w, ones*ps_w) * tf.to_float(mask)

        return weights

    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            valid_check=False,
            rescale = 256)

        images, heatmap, heatmap_mask = provider.get(
            'heatmap/mask')

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):

                lms_prediction = build_graph(images, data)

        # landmark-regression losses
        weight_hm = get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(lms_prediction, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)


        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)


        # image summaries
        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_prediction, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('gt/all', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))


        # learning rate decay
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate,
            global_step,
            FLAGS.learning_rate_decay_step / FLAGS.batch_size,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                {'/'.join(var.op.name.split('/')[2:]):var for var in variables_to_restore},
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
    while True:
        try:
            with open('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/out-graph.pkl', 'br') as f:
                data = pickle.load(f, encoding='latin1')
            train(data)
        except Exception as e:
            print(e)
