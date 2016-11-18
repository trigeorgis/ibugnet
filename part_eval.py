import tensorflow as tf
import numpy as np
import resnet_model
import hourglass_model
import losses
import data_provider
import utils

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path
import matplotlib.pyplot as plt

slim = tf.contrib.slim

from flags import FLAGS


def ced_accuracy(t, dists):
    # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
    pts_r  = tf.transpose(tf.gather(tf.transpose(dists), [8,12,11,10,2,1,0]))
    pts_l  = tf.transpose(tf.gather(tf.transpose(dists), [9,13,14,15,3,4,5]))
    part_pckh = (tf.to_int32(pts_r <= t) + tf.to_int32(pts_l <= t)) / 2

    return tf.concat(1, [part_pckh, tf.reduce_sum(tf.to_int32(dists <= t), 1)[...,None] / tf.shape(dists)[1]])


def pckh(preds, gts, scales):
    t_range = np.arange(0,0.51,0.01)
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2), reduction_indices=-1)) / scales
    # pckh = [ced_accuracy(t, dists) for t in t_range]
    # return pckh[-1]
    return ced_accuracy(0.5, dists)


keypoint_colours = np.array([plt.cm.spectral(x) for x in np.linspace(0, 1, FLAGS.n_landmarks + 1)])[
    ..., :3].astype(np.float32)

def generate_heatmap(logits):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, FLAGS.n_landmarks + 1].
    """
    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(prediction, (-1, FLAGS.n_landmarks + 1)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap


def generate_landmarks(keypoints):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    zeros = tf.to_float(tf.zeros_like(is_background))

    return tf.select(is_background, zeros, ones) * 255

def test_hg(output_svs=7, output_lms=16, num_iterations=4):
    g = tf.Graph()

    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms)
        images, heatmap, heatmap_mask, gt_landmarks, scales = provider.get(
            'heatmap/mask', 'landmarks', 'scale')


        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                lms_prediction = hourglass_model.network(
                    images, 1, output_channels=output_lms)


        hs = tf.argmax(tf.reduce_max(lms_prediction, 2), 1)
        ws = tf.argmax(tf.reduce_max(lms_prediction, 1), 1)
        predictions = tf.transpose(tf.to_float(tf.pack([hs, ws])), perm=[1, 2, 0])

    with tf.Session(graph=g) as sess:

        accuracy = pckh(predictions, gt_landmarks, scales)
        # accuracy = tf.Print(accuracy, [tf.shape(accuracy), accuracy], summarize=5)
        # These are streaming metrics which compute the "running" metric,
        # e.g running accuracy
        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({
            # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
            "accuracy/pckh_All": slim.metrics.streaming_mean(accuracy[:,-1]),
            "accuracy/pckh_Head": slim.metrics.streaming_mean(accuracy[:,0]),
            "accuracy/pckh_Shoulder": slim.metrics.streaming_mean(accuracy[:,1]),
            "accuracy/pckh_Elbow": slim.metrics.streaming_mean(accuracy[:,2]),
            "accuracy/pckh_Wrist": slim.metrics.streaming_mean(accuracy[:,3]),
            "accuracy/pckh_Hip": slim.metrics.streaming_mean(accuracy[:,4]),
            "accuracy/pckh_Knee": slim.metrics.streaming_mean(accuracy[:,5]),
            "accuracy/pckh_Ankle": slim.metrics.streaming_mean(accuracy[:,6])
        })

        # Define the streaming summaries to write:
        summary_ops = []
        for metric_name, metric_value in metrics_to_values.items():
            op = tf.scalar_summary(metric_name, metric_value)
            op = tf.Print(op, [metric_value], metric_name)
            summary_ops.append(op)

        summary_ops.append(tf.scalar_summary('accuracy/running_pckh', tf.reduce_mean(accuracy[:,-1])))

        summary_ops.append(tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_prediction, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4)))
        summary_ops.append(tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4)))
        summary_ops.append(tf.image_summary('gt/all', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4)))


        global_step = slim.get_or_create_global_step()
        # Evaluate every 30 seconds
        logging.set_verbosity(1)
        num_examples = provider.num_samples()
        num_batches = np.ceil(num_examples / FLAGS.batch_size)
        num_batches = 500
        slim.evaluation.evaluation_loop(
            '',
            FLAGS.train_dir,
            FLAGS.eval_dir,
            num_evals=num_batches,
            eval_op=list(metrics_to_updates.values()),
            summary_op=tf.merge_summary(summary_ops),
            eval_interval_secs=30)



if __name__ == '__main__':
    while True:
        try:
            if FLAGS.train_mode == 0:
                test_hg()
            elif FLAGS.train_mode == 1:
                test_pd()
            elif FLAGS.train_mode == 2:
                test_svs()
        except Exception as e:
            print(e)
