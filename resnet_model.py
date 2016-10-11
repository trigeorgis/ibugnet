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


def add_batchnorm_layers(i, net):
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


def network(inputs, scale, output_classes=3, return_endpoints=False, state=None):
    out_shape = tf.shape(inputs)[1:3]

    if scale > 1:
        inputs = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
        inputs = slim.layers.avg_pool2d(inputs, (3, 3), (scale, scale),
                                        padding='VALID')

    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        bottleneck, endpoints = nets.resnet_v1.resnet_v1_50(inputs, global_pool=False)

    scope_name = '/'.join(bottleneck.name.split('/')[:3])

    skip_connections = [
        endpoints[scope_name + '/' + x] for x in [
            'conv1', 'block2/unit_4/bottleneck_v1/conv1', 'block3/unit_6/bottleneck_v1/conv1'
        ]
    ]

    if state is not None:
        skip_connections.append(state)

    net = slim.layers.conv2d(bottleneck, 1024, (3, 3), rate=12, scope='upsample_conv')

    skip_connections.append(net)
    skip_connections = [add_batchnorm_layers(i, x) for i, x in enumerate(skip_connections)]
    skip_connections.append(inputs)

    skip_connections = [
        tf.image.resize_bilinear(x, out_shape,
                                 name="up_nrm_{}_0".format(i))
        for i, x in enumerate(skip_connections, start=1)
    ]

    net = tf.concat(3, skip_connections, name='concat-nrm__00')

    result = slim.conv2d(net,
            output_classes, (1, 1),
            scope='upscore-fuse-nrm__00',
            activation_fn=None)
    
    if return_endpoints:
        return result, bottleneck
    else:
        return result


def multiscale_kpts_net(inputs, scales=(1, 2, 4), num_keypoints=69, return_endpoints=False):
    pyramid = []
    endpoints = []

    for scale in scales:
        reuse_variables = scale != scales[0]
        with tf.variable_scope('multiscale', reuse=reuse_variables):
            net = network(
                    inputs,
                    scale,
                    output_classes=num_keypoints,
                    return_endpoints=return_endpoints)
            if return_endpoints:
                pyramid.append(net[0])
                endpoints.append(net[1])
            else:
                pyramid.append(net)

    net = tf.concat(3, pyramid, name='concat-mr-kpts')
    net = slim.conv2d(net,
                      num_keypoints, (1, 1),
                      scope='upscore-fuse-mr-kpts',
                      activation_fn=None)


    return net, pyramid, endpoints


def svs_regression_net(inputs):
    states = []
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]
    output_classes = 7
    
    hidden = tf.Variable(tf.zeros((1, 256, 256, output_classes)))
    hidden = tf.image.resize_bilinear(hidden, (height, width, output_classes))

    for i in range(4):
        with tf.variable_scope('multiscale', reuse=i > 0):
            hidden = network(inputs, 1, output_classes=output_classes, state=hidden)
            states.append(hidden)

    return hidden, states


def multiscale_deblurring_net(inputs, scales=(1, 2, 4)):
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
