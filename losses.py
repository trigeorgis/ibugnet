import tensorflow as tf

def smooth_l1(pred, ground_truth):
    residual = tf.abs(pred - ground_truth)
    loss = tf.select(tf.less(residual, 1),
                     0.5 * tf.square(residual),
                     residual - .5)
    return tf.reduce_mean(loss, name='smooth_l1')