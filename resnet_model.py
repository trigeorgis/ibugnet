import tensorflow as tf
import re
from tensorflow.contrib.slim import nets
slim = tf.contrib.slim


def resnet_arg_scope(weight_decay=0.0001,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    """Defines the default ResNet arg scope.
  Args:
    is_training: Whether or not we are training the parameters in the batch
      normalization layers of the model.
    weight_decay: The weight decay to use for regularizing the model.
    batch_norm_decay: The moving average decay when estimating layer activation
      statistics in batch normalization.
    batch_norm_epsilon: Small constant to prevent division by zero when
      normalizing activations by their variance in batch normalization.
    batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
      activations in the batch normalization layer.
  Returns:
    An `arg_scope` to use for the resnet models.
  """
    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
    }

    with slim.arg_scope(
        [slim.conv2d],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc


def add_batchnorm_layers((i, net)):
    name = "{}_skip_{}".format(net.name.split('/')[-2], i)
    net = slim.batch_norm(net,
                          center=False,
                          scale=False,
                          scope="BN_pre_{}".format(name),
                          epsilon=1,
                          outputs_collections='outputs')

    net = tf.mul(net, 10, name="{}_scl".format(name))

    net = slim.conv2d(net, 3, (1, 1), activation_fn=None)
    return net


def network(inputs, scale, output_classes=3):
    out_shape = tf.shape(inputs)[1:3]

    if scale > 1:
        inputs = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
        inputs = slim.layers.avg_pool2d(inputs, (3, 3), (scale, scale),
                                        padding='VALID')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        net, endpoints = nets.resnet_v1.resnet_v1_50(inputs, global_pool=False)
    scope_name = [c[:-5] for c in endpoints.keys() if '50/conv1' in c][0]

    skip_connections = []

    # print(endpoints.keys())
    # skip_connections = [
    #     endpoints[scope_name + x] for x in [
    #         'conv1', 'block1'
    #     ]
    # ]

    skip_connections = map(add_batchnorm_layers, enumerate(skip_connections))
    skip_connections.append(net)

    skip_connections = [
        tf.image.resize_bilinear(x, out_shape,
                                 name="up_nrm_{}_0".format(i))
        for i, x in enumerate(skip_connections, start=1)
    ]

    net = tf.concat(3, skip_connections, name='concat-nrm__00')

    return slim.conv2d(net,
                       output_classes, (1, 1),
                       scope='upscore-fuse-nrm__00',
                       activation_fn=None)


def multiscale_seg_net(inputs, scales=(1, 2, 4)):
    pyramid = []

    for scale in scales:
        reuse_variables = scale != scales[0]
        with tf.variable_scope('scale', reuse=reuse_variables):
            pyramid.append(network(inputs, scale, output_classes=2))

    net = tf.concat(3, pyramid, name='concat-mr-seg')
    net = slim.conv2d(net,
                      2, (1, 1),
                      scope='upscore-fuse-mr-seg',
                      activation_fn=None)

    return net, pyramid


def multiscale_nrm_net(inputs, scales=(1, 2, 4)):
    pyramid = []

    for scale in scales:
        reuse_variables = scale != scales[0]
        with tf.variable_scope('multiscale', reuse=reuse_variables):
            pyramid.append(network(inputs, scale, output_classes=3))

    net = tf.concat(3, pyramid, name='concat-mr-nrm')
    net = slim.conv2d(net,
                      3, (1, 1),
                      scope='upscore-fuse-mr-nrm',
                      activation_fn=None)

    def normalize(x, scale=0):
        with tf.variable_scope('normupscore-fuse-mr-nrm_{}'.format(scale)):
            normalization_matrix = tf.sqrt(1e-12 + tf.reduce_sum(
                tf.square(x), [3]))
            x /= tf.expand_dims(normalization_matrix, 3)
        return x

    return normalize(net), [normalize(x, i) for x, i in zip(pyramid, scales)]
