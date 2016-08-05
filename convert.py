import tensorflow as tf
import numpy as np

def assign_caffe_weights(net_caffe, sess):
    caffe_layers = {}

    for i, layer in enumerate(net_caffe.layers):
        layer_name = net_caffe._layer_names[i]
        caffe_layers[layer_name] = layer

    def caffe_bn(layer_name):
        layer = caffe_layers[layer_name]
        return layer.blobs[0].data, layer.blobs[1].data, layer.blobs[2].data

    def caffe_weights(layer_name):
        layer = caffe_layers[layer_name]
        return layer.blobs[0].data

    def caffe_bias(layer_name):
        layer = caffe_layers[layer_name]
        return layer.blobs[1].data


    def caffe2tf_filter(name):
      f = caffe_weights(name)
      return f.transpose((2, 3, 1, 0))

    update_ops = []

    with sess.graph.as_default(), tf.variable_scope("net", reuse=True):
        with sess.graph.as_default(), tf.variable_scope("RS", reuse=True):
            for key in caffe_layers:
                if 'RS' in key or 'upscore-fuse-mr-nrm' in key:
                    continue

                layer = caffe_layers[key]

                if layer.type == 'Convolution':
                    weights = caffe2tf_filter(key)
                    bias = caffe_bias(key)

                    var = tf.get_variable('{}/weights'.format(key))

                    update_ops.append(tf.assign(var, weights))

                    var = tf.get_variable('{}/biases'.format(key))
                    update_ops.append(tf.assign(var, bias))
                elif layer.type == 'BatchNorm':
                    mean, variance, scale = caffe_bn(key)
                    var = tf.get_variable('{}/moving_mean'.format(key))
                    update_ops.append(tf.assign(var, mean / scale))

                    var = tf.get_variable('{}/moving_variance'.format(key))
                    update_ops.append(tf.assign(var, variance / scale))
                    # print('not imple')
                else:
                    continue
        key = 'upscore-fuse-mr-nrm'
        weights = caffe2tf_filter(key)
        bias = caffe_bias(key)

        var = tf.get_variable('{}/weights'.format(key))

        update_ops.append(tf.assign(var, weights))

        var = tf.get_variable('{}/biases'.format(key))
        update_ops.append(tf.assign(var, bias))
    _ = sess.run(update_ops)