import tensorflow as tf


def smooth_l1(pred, ground_truth):
    """Defines a robust L1 loss.

    This is a robust L1 loss that is less sensitive to outliers
    than the traditional L2 loss. This was defined in Ross
    Girshick's Fast R-CNN, ICCV 2015 paper.

    Args:
      pred: A `Tensor` of dimensions [num_images, height, width, 3].
      ground_truth: A `Tensor` of dimensions [num_images, height, width, 3].
    Returns:
      A scalar with the mean loss.
    """
    residual = tf.abs(pred - ground_truth)

    loss = tf.select(tf.less(residual, 1),
                     0.5 * tf.square(residual),
                     residual - .5)

    return tf.reduce_mean(loss, name='smooth_l1')


def quaternion_loss(pred, ground_truth):
    '''Computes the half cosine distance between two quaternions.
    
    Assumes that the quaternions are l2 normalised.

    Args:
      pred: A `Tensor` of dimensions [num_images, 4]
      ground_truth: A `Tensor` of dimensions [num_images, 4].
    Returns:
      A scalar with the mean cosine loss.
    '''
    loss = 1 - tf.abs(tf.reduce_sum(pred * ground_truth, 1))

    return tf.reduce_mean(loss)


def cosine_loss(pred, ground_truth, dim=3):
    '''Computes the cosine distance between two images.
    
    Assumes that the input images are l2 normalised per pixel.

    Args:
      pred: A `Tensor` of dimensions [num_images, height, width, 3]
      ground_truth: A `Tensor` of dimensions [num_images, height, width, 3].
    Returns:
      A scalar with the mean angular error (cosine loss).
    '''
    loss = 1 - tf.reduce_sum(pred * ground_truth, dim)

    return tf.reduce_mean(loss, name='cosine_loss')