import tensorflow as tf


def smooth_l1(pred, ground_truth):
    '''Calculates the smoothed l1 loss.

    The prediction and ground truth should be l1 normalised.

    Args:
      pred: A tf Tensor of dimensions [num_images, height, width, 3].
      ground_truth: A tf Tensor of dimensions [num_images, height, width, 3].
    Returns:
      A scalar with the mean loss.
    '''

    absolute_residual = tf.abs(pred - ground_truth)

    loss = tf.select(
        tf.less(absolute_residual, 1), 0.5 * tf.square(absolute_residual),
        absolute_residual - .5)

    return tf.reduce_mean(loss, name='smooth_l1')
