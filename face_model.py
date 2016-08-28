import tensorflow as tf
import re
slim = tf.contrib.slim


def repeat_conv(net,
                output_filters,
                num_repeats,
                kernel_size=(3, 3),
                rate=1,
                scope='conv',
                padding=1):
    for i in range(1, num_repeats + 1):
        net = tf.pad(net, ((0, 0), (padding, padding), (padding, padding),
                           (0, 0)))
        net = slim.layers.conv2d(net,
                                 output_filters,
                                 kernel_size,
                                 rate=rate,
                                 scope='{}_{}'.format(scope, i))
    return net


def add_batchnorm_layers(net):
    name = net.name.split('/')[-2]
    net = slim.batch_norm(net,
                          center=False,
                          scale=False,
                          scope="BN_pre_{}".format(name),
                          epsilon=1,
                          outputs_collections='outputs')

    net = tf.mul(net, 10, name="{}_scl".format(name))

    num, = re.findall(r"(\d+)_", name)
    net = slim.conv2d(net,
                      3, (1, 1),
                      scope='score_nrm_{}_0_hole0'.format(num),
                      activation_fn=None)
    return net


def network(inputs, scale=1, output_classes=3):
    out_shape = tf.shape(inputs)[1:3]

    if scale > 1:
        inputs = tf.pad(inputs, ((0, 0), (1, 1), (1, 1), (0, 0)))
        inputs = slim.layers.avg_pool2d(inputs, (3, 3), (scale, scale),
                                        padding='VALID')

    with slim.arg_scope([slim.layers.max_pool2d, slim.layers.avg_pool2d],
                        padding='SAME',
                        outputs_collections='outputs'):
        with slim.arg_scope(
            [slim.layers.conv2d],
                weights_regularizer=slim.regularizers.l2_regularizer(0.0005),
                padding='VALID',
                outputs_collections='outputs'):
            skip_connections = []

            net = repeat_conv(inputs, 64, 2, scope='conv1')
            skip_connections.append(net)

            net = slim.layers.max_pool2d(net, 3, 2, scope='pool1')
            net = repeat_conv(net, 128, 2, scope='conv2')
            skip_connections.append(net)

            net = slim.layers.max_pool2d(net, [3, 3], [2, 2], scope='pool2')
            net = repeat_conv(net, 256, 3, scope='conv3')
            skip_connections.append(net)

            net = slim.layers.max_pool2d(net, [3, 3], [2, 2], scope='pool3')
            net = repeat_conv(net, 512, 3, scope='conv4')
            skip_connections.append(net)

            net = slim.layers.max_pool2d(net, [3, 3], [1, 1], scope='pool4')
            net = repeat_conv(net, 512, 3, rate=2, padding=2, scope='conv5')
            skip_connections.append(net)

            net = slim.layers.max_pool2d(net, [3, 3], [1, 1], scope='pool5')
            net = tf.pad(net, ((0, 0), (1, 1), (1, 1), (0, 0)))
            net = slim.layers.avg_pool2d(net,
                                         3,
                                         1,
                                         padding='VALID',
                                         scope='cue5')

            # fc6
            net = tf.pad(net, ((0, 0), (12, 12), (12, 12), (0, 0)))
            net = slim.layers.conv2d(net, 1024, (3, 3), rate=12, scope='fc6')
            net = slim.layers.dropout(net, 0.5, scope='drop6')

            # fc7
            net = slim.layers.conv2d(net, 1024, (1, 1), scope='fc7')
            net = slim.layers.dropout(net, 0.5, scope='drop7')
            net = slim.conv2d(
                net,
                output_classes, (1, 1),
                scope='score_nrm_{}_0_hole0'.format(1 + len(skip_connections)),
                activation_fn=None)

            skip_connections = map(add_batchnorm_layers, skip_connections)

            skip_connections.append(net)
            skip_connections = [tf.image.resize_bilinear(
                x, out_shape, name="up_nrm_{}_0".format(i))
                                for i, x in enumerate(skip_connections,
                                                      start=1)]

            net = tf.concat(3, skip_connections, name='concat-nrm__00')

            net = slim.conv2d(net,
                              output_classes, (1, 1),
                              scope='upscore-fuse-nrm__00',
                              activation_fn=None)

    return net


def multiscale_seg_net(inputs, scales=(1, 2, 4)):
    pyramid = []

    for scale in scales:
        reuse_variables = scale != scales[0]
        with tf.variable_scope('RS', reuse=reuse_variables):
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
        with tf.variable_scope('RS', reuse=reuse_variables):
            pyramid.append(network(inputs, scale))

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
