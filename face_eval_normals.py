import data_provider
import tensorflow as tf
import resnet_model
import math

slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('checkpoint_dir', 'ckpt/train_normals',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_string('log_dir', 'ckpt/eval_ict',
                           '''Directory where to write event logs '''
                           '''and checkpoint.''')
tf.app.flags.DEFINE_integer('eval_interval_secs', 300, '''Run the evaluation every many secs.''')

def main():
    # Load the data
    provider = data_provider.ICT3DFE()
    images, normals, mask = provider.get('normals/mask')

    # Define the network
    with tf.variable_scope('net'):
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                            is_training=False):
            predictions, _ = resnet_model.multiscale_nrm_net(images, scales=(1, 2, 4))

    tf.image_summary('images', images)
    tf.image_summary('normals', normals)
    tf.image_summary('predictions', predictions)
    
    # Choose the metrics to compute:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'cosine': tf.contrib.metrics.streaming_mean_cosine_distance(
            predictions, normals, 3, weights=mask)
    })

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in names_to_values.items():
        op = tf.scalar_summary(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    num_examples = provider.num_samples()
    batch_size = 1
    num_batches = math.ceil(num_examples / batch_size)

    # Setup the global step.
    slim.get_or_create_global_step()

    slim.evaluation.evaluation_loop('',
                                    FLAGS.checkpoint_dir,
                                    FLAGS.log_dir,
                                    num_evals=num_batches,
                                    eval_op=list(names_to_updates.values()),
                                    summary_op=tf.merge_summary(summary_ops),
                                    eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    main()
