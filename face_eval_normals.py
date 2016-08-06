import tensorflow as tf
slim = tf.contrib.slim


def main():
    # Load the data
    images, labels = load_ict_data(...)

    # Define the network
    predictions = MyModel(images)

    # Choose the metrics to compute:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'cosine': tf.contrib.metrics.streaming_mean_cosine_distance(
            predictions, labels, 2)
    })

    # Create the summary ops such that they also print out to std output:
    summary_ops = []
    for metric_name, metric_value in metrics_to_values.iteritems():
        op = tf.scalar_summary(metric_name, metric_value)
        op = tf.Print(op, [metric_value], metric_name)
        summary_ops.append(op)

    num_examples = 10000
    batch_size = 32
    num_batches = math.ceil(num_examples / float(batch_size))

    # Setup the global step.
    slim.get_or_create_global_step()

    output_dir = 'ckpt/eval'
    eval_interval_secs = 600  # Run the evaluation every 10 mins.
    slim.evaluation.evaluation_loop('local',
                                    checkpoint_dir,
                                    log_dir,
                                    num_evals=num_batches,
                                    eval_op=names_to_updates.values(),
                                    summary_op=tf.merge_summary(summary_ops),
                                    eval_interval_secs=eval_interval_secs)


if __name__ == '__main__':
    main()
