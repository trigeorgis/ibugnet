from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables

slim = tf.contrib.slim


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])
    
    # RGB -> BGR
    pixels = image.pixels[[2, 1, 0]]
    # Subtract VGG training mean across all channels
    pixels = pixels - VGG_MEAN.reshape([3, 1, 1])
    pixels = pixels.astype(np.float32, copy=False)
    return pixels


def rescale_image(image, stride_width=64):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height, width = image.shape

    # Taken from 'szross'
    scale_up = 625. / min(height, width)
    scale_cap = 961. / max(height, width)
    scale_up  = min(scale_up, scale_cap)
    new_height = stride_width * round((height * scale_up) / stride_width) + 1
    new_width = stride_width * round((width * scale_up) / stride_width) + 1
    image, tr = image.resize([new_height, new_width], return_transform=True)
    image.inverse_tr = tr
    return image


def frankotchellappa(dzdx, dzdy):
    from numpy.fft import ifftshift, fft2, ifft2
    rows, cols = dzdx.shape
    # The following sets up matrices specifying frequencies in the x and y
    # directions corresponding to the Fourier transforms of the gradient
    # data.  They range from -0.5 cycles/pixel to + 0.5 cycles/pixel.
    # The scaling of this is irrelevant as long as it represents a full
    # circle domain. This is functionally equivalent to any constant * pi
    pi_over_2 = np.pi / 2.0
    row_grid = np.linspace(-pi_over_2, pi_over_2, rows)
    col_grid = np.linspace(-pi_over_2, pi_over_2, cols)
    wy, wx = np.meshgrid(row_grid, col_grid, indexing='ij')

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = ifftshift(wx)
    wy = ifftshift(wy)

    # Fourier transforms of gradients
    DZDX = fft2(dzdx)
    DZDY = fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    # weighting the Fourier coefficients by their frequencies in x and y and
    # then dividing by the squared frequency
    denom = (wx ** 2 + wy ** 2)
    Z = (-1j * wx * DZDX - 1j * wy * DZDY) / denom
    Z = np.nan_to_num(Z)
    return np.real(ifft2(Z))


def create_train_op(
    total_loss,
    optimizer,
    global_step=None,
    update_ops=None,
    variables_to_train=None,
    clip_gradient_norm=0,
    iter_step=1,
    summarize_gradients=False,
    gate_gradients=tf_optimizer.Optimizer.GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    gradient_multipliers=None):
    """Creates an `Operation` that evaluates the gradients and returns the loss.
    Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `None`, then slim.variables.global_step() is used.
    update_ops: an optional list of updates to execute. Note that the update_ops
      that are used are the union of those update_ops passed to the function and
      the value of slim.ops.GetUpdateOps(). Therefore, if `update_ops` is None,
      then the value of slim.ops.GetUpdateOps() is still used.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    clip_gradient_norm: If greater than 0 then the gradients would be clipped
      by it.
    iter_step: accumulate gradients across `iter_step` batches.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    gradient_multipliers: A dictionary of either `Variables` or `Variable` op
      names to the coefficient by which the associated gradient should be
      scaled.
    Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
    """
    if global_step is None:
        global_step = variables.get_or_create_global_step()

    # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
    global_update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    if update_ops is None:
        update_ops = global_update_ops
    else:
        update_ops = set(update_ops)

    if not global_update_ops.issubset(update_ops):
        logging.warning('update_ops in create_train_op does not contain all the '
                        ' update_ops in GraphKeys.UPDATE_OPS')

    # Make sure update_ops are computed before total_loss.
    if update_ops:
        with ops.control_dependencies(update_ops):
            barrier = control_flow_ops.no_op(name='update_barrier')
    total_loss = control_flow_ops.with_dependencies([barrier], total_loss)

    if variables_to_train is None:
        # Default to tf.trainable_variables()
        variables_to_train = tf_variables.trainable_variables()
    else:
        # Make sure that variables_to_train are in tf.trainable_variables()
        for v in variables_to_train:
            assert v in tf_variables.trainable_variables()

    assert variables_to_train


    # Create the gradients. Note that apply_gradients adds the gradient
    # computation to the current graph.
    single_grads = optimizer.compute_gradients(
      total_loss, variables_to_train, gate_gradients=gate_gradients,
      aggregation_method=aggregation_method,
      colocate_gradients_with_ops=colocate_gradients_with_ops)

    accum_grads = [tf.Variable(tf.zeros_like(g), trainable=False) for (g, _) in single_grads]                                        
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_grads]

    accum_ops = [a.assign_add(g) for a, (g, _) in zip(accum_grads, single_grads)]
    grads = [(a / iter_step, v) for a, (_, v) in zip(accum_grads, single_grads)]
    
    def train_step_fn(sess, train_op, global_step, train_step_kwargs):
        sess.run(zero_ops)

        for i in range(iter_step):
            sess.run(accum_ops)

        return slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

    # Scale gradients.
    if gradient_multipliers:
        with ops.name_scope('multiply_grads'):
            grads = multiply_gradients(grads, gradient_multipliers)

    # Clip gradients.
    if clip_gradient_norm > 0:
        with ops.name_scope('clip_grads'):
            grads = clip_gradient_norms(grads, clip_gradient_norm)

    # Summarize gradients.
    if summarize_gradients:
        with ops.name_scope('summarize_grads'):
            slim.learning.add_gradients_summaries(grads)

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

    with ops.name_scope('train_op'):
        # Make sure total_loss is valid.
        total_loss = array_ops.check_numerics(total_loss,
                                              'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    return control_flow_ops.with_dependencies([grad_updates], total_loss), train_step_fn