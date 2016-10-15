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


def add_batchnorm_layers(i, net, output_channels=3):
    name = "{}_skip_{}".format(net.name.split('/')[-2], i)
    net = slim.batch_norm(
        net,
        center=False,
        scale=False,
        scope="BN_pre_{}".format(name),
        epsilon=1,
        outputs_collections='outputs')

    net = tf.mul(net, 10, name="{}_scl".format(name))
    net = slim.conv2d(net, output_channels, (1, 1), activation_fn=None)

    return net


def network(inputs):
    """Defines a lightweight resnet based model for dense estimation tasks.
    
    Args:
      inputs: A `Tensor` with dimensions [num_batches, height, width, depth].
    Returns:
      A `Tensor` of dimensions [num_batches, height, width, output_channels]."""


    with slim.arg_scope(nets.resnet_v1.resnet_arg_scope()):
        bottleneck, endpoints = nets.resnet_v1.resnet_v1_50(
            inputs, global_pool=False)

    return bottleneck, endpoints


def upsample_net(bottleneck, endpoints, output_shape, state=None,
                output_channels=3,
                internal_channels=3,):
    scope_name = '/'.join(bottleneck.name.split('/')[:3])

    skip_connections = [
        endpoints[scope_name + '/' + x]
        for x in [
            'conv1', 'block2/unit_4/bottleneck_v1/conv1',
            'block3/unit_6/bottleneck_v1/conv1'
        ]
    ]

    net = slim.layers.conv2d(
        bottleneck, 1024, (3, 3), rate=12, scope='upsample_conv')

    skip_connections.append(net)
    skip_connections = [
        add_batchnorm_layers(
            i, x, output_channels=internal_channels)
        for i, x in enumerate(skip_connections)
    ]

    if state is not None:
        skip_connections.append(state)

    skip_connections = [
        tf.image.resize_bilinear(
            x, output_shape, name="up_nrm_{}_0".format(i))
        for i, x in enumerate(
            skip_connections, start=1)
    ]

    return tf.concat(3, skip_connections, name='concat-nrm__00')

def svs_regression_net_light(inputs, output_classes=7, output_n_points=16, num_iterations=3):
    states = []
    results = []

    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]

    hidden = tf.zeros(
        (batch_size, height, width, output_classes), name='hidden')

    
    
    for i in range(num_iterations):
        with tf.variable_scope('multiscale', reuse=i > 0):
            
            if i == 0:
                bottleneck, endpoints = network(inputs)
                output_shape = (height, width)
            
            presigmoid_hidden = upsample_net(
                bottleneck, endpoints, output_shape,
                output_channels=output_classes,
                state=hidden * 10,
                internal_channels=output_classes)

            hidden = slim.conv2d(
                presigmoid_hidden,
                output_classes, (1, 1),
                scope='upscore_hidden',
                activation_fn=tf.sigmoid)

            result = slim.conv2d(
                presigmoid_hidden,
                output_n_points, (1, 1),
                scope='final_result',
                activation_fn=None)

            states.append(hidden)
            results.append(result)

    return states, results


def svs_regression_net(inputs, output_classes=7):
    states = []
    batch_size = tf.shape(inputs)[0]
    height = tf.shape(inputs)[1]
    width = tf.shape(inputs)[2]

    hidden = tf.zeros(
        (batch_size, height, width, output_classes), name='hidden')

    for i in range(4):
        with tf.variable_scope('multiscale', reuse=i > 0):
            hidden = (hidden - .5) * 256
            hidden = network(
                tf.concat(3, (inputs, hidden)),
                1,
                output_channels=output_classes)
            hidden = slim.conv2d(
                hidden, output_classes, (1, 1), activation_fn=tf.sigmoid)
            states.append(hidden)

    return hidden, states