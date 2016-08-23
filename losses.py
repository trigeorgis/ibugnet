import tensorflow as tf

def smooth_l1(pred, ground_truth):
    """Defines a robust L1 loss.

    This is a robust L1 loss that is less sensitive to outliers
    than the traditional L2 loss. This was defined in Ross
    Girshick's Fast R-CNN, ICCV 2015 paper.
    
    Args:
      pred: a Tensor with the predictions.
      ground_truth: a Tensor with the ground truth labels.
    Returns:
      a scalar defining the smoothed l1 loss.
    """
    residual = tf.abs(pred - ground_truth)
    loss = tf.select(tf.less(residual, 1),
                     0.5 * tf.square(residual),
                     residual - .5)
    return tf.reduce_mean(loss, name='smooth_l1')
