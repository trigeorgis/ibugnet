import tensorflow as tf
import numpy as np
import resnet_model
import hourglass_model
import losses
import data_provider
import utils
import matplotlib.pyplot as plt

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

from flags import FLAGS

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999


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


def restore_resnet(sess, path):
    def name_in_checkpoint(var):
        # Uncomment for non lightweight model
        # if 'resnet_v1_50/conv1/weights' in var.name:
        #     return None
        name = '/'.join(var.name.split('/')[2:])
        name = name.split(':')[0]
        if 'Adam' in name:
            return None
        return name

    variables_to_restore = slim.get_variables_to_restore(
        include=["net/multiscale/resnet_v1_50"])
    variables_to_restore = {name_in_checkpoint(var): var
                            for var in variables_to_restore if name_in_checkpoint(var) is not None}

    return slim.assign_from_checkpoint_fn(path, variables_to_restore, ignore_missing_vars=True)


def train(output_svs=7, output_lms=FLAGS.n_landmarks):
    g = tf.Graph()
    logging.set_verbosity(10)
    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            valid_check=True)
        images, ground_truth, ground_truth_mask= provider.get(
            'pose/mask')


        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):

                # svs regression net
                prediction, states = resnet_model.svs_regression_net_light(images, output_classes=output_svs, num_iterations=FLAGS.num_iterations)
                # States currently is
                # (num_states, batch_size, height, width, num_parts)

        # Add a cosine loss to every scale and the combined output.
        for i, state in enumerate(states):
            gt = ground_truth[:, i, :, :, :]

            # TODO: Move 100 to a flag.
            # Reweighting the loss by 100. If we do not do this
            # The loss becomes extremely small.
            ones = tf.ones_like(gt)

            weights = tf.select(gt < .1, ones, ones * 100)

            # The non-visible parts have a substracted value of a 100.
            weights = tf.select(gt < 0, tf.zeros_like(gt), weights)

            loss = losses.smooth_l1(state, gt, weights)
            tf.scalar_summary('losses/iteration_{}'.format(i), loss)

        # total loses
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)

        # image summaries
        for i in range(len(states)):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None], max_images=min(FLAGS.batch_size,4))
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(ground_truth[:, i, :, :, :], -1)[..., None], max_images=min(FLAGS.batch_size,4))

            for j in range(output_svs):
                state = states[i][..., j][..., None]
                gt = ground_truth[:, i, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(2, (state, gt)), max_images=min(FLAGS.batch_size,4))

        tf.image_summary('image', images, max_images=min(FLAGS.batch_size,4))

        # learning rate decay
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate,
            global_step,
            FLAGS.learning_rate_decay_step / FLAGS.batch_size,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                {'/'.join(var.op.name.split('/')[2:]):var for var in variables_to_restore},
                ignore_missing_vars=True)


        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)


def train_regress(output_svs=7, output_lms=FLAGS.n_landmarks):
    g = tf.Graph()
    logging.set_verbosity(10)
    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms)
        images, heatmap, heatmap_mask = provider.get('heatmap/mask')


        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                lms_prediction = hourglass_model.network(
                    images, 1, output_channels=output_lms)

        def get_weight(keypoints, mask, ng_w=0.01, ps_w=1.0):
            is_background = tf.equal(keypoints, 0)
            ones = tf.to_float(tf.ones_like(is_background))
            weights = tf.select(is_background, ones * ng_w, ones*ps_w) * tf.to_float(mask)

            return weights
        # landmark-regression losses


        weight_hm = get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(lms_prediction, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

        # total losses
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)


        # image sumarries
        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_prediction, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))
        tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all ', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        # learning rate decay
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate,
            global_step,
            FLAGS.learning_rate_decay_step / FLAGS.batch_size,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                {'/'.join(var.op.name.split('/')[2:]):var for var in variables_to_restore if not var.op.name.split('/')[1] == 'landmarks'},
                ignore_missing_vars=True)


        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)


def train_part_detect(output_lms=FLAGS.n_landmarks):
    g = tf.Graph()
    logging.set_verbosity(10)
    with g.as_default():
        # Load dataset.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms)
        images, keypoints_visible, keypoints_visible_mask = provider.get(
            'keypoints_visible/mask')

        def keypts_encoding(keypoints):
            keypoints = tf.to_int32(keypoints)
            keypoints = tf.reshape(keypoints, (-1,))
            keypoints = slim.layers.one_hot_encoding(keypoints, num_classes=output_lms+1)
            return keypoints

        def get_weight(keypoints, mask, ng_w=0.01, ps_w=1.0):
            is_background = tf.equal(keypoints, 0)
            ones = tf.to_float(tf.ones_like(is_background))
            weights = tf.select(is_background, ones * ng_w, ones*ps_w) * tf.to_float(mask)

            return weights

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                # Part-detector net that is trained only on the visible points.
                part_prediction, pyramid, _ = resnet_model.multiscale_kpts_net(
                    images, scales=(1, 2), num_keypoints=output_lms+1)



        tf.image_summary('predictions/part-detection', generate_heatmap(part_prediction), max_images=min(FLAGS.batch_size,4))
        tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/visiable', generate_landmarks(keypoints_visible), max_images=min(FLAGS.batch_size,4))

        # Add a cosine loss to every scale and the combined output.
        # part-detection losses
        kps_visiable, weight_visible = keypts_encoding(keypoints_visible), get_weight(keypoints_visible, keypoints_visible_mask)
        weight_visible = tf.reshape(weight_visible, (-1,))
        weight_visible.set_shape([None,])

        for net,name in zip([part_prediction] + pyramid, ['final', '1', '2']):
            net = tf.reshape(net, (-1, output_lms+1))
            loss = slim.losses.softmax_cross_entropy(net, kps_visiable, weight=weight_visible)
            tf.scalar_summary('losses/part_detection_scale/{}'.format(name), loss)

        # total losses
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)


        # learning rate decay
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate,
            global_step,
            FLAGS.learning_rate_decay_step / FLAGS.batch_size,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                    FLAGS.pretrained_model_checkpoint_path,
                    {'/'.join(var.op.name.split('/')[2:]):var for var in variables_to_restore if not var.op.name.split('/')[1] == 'landmarks'},
                    ignore_missing_vars=True)

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess,
                                     FLAGS.pretrained_resnet_checkpoint_path)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)


def train_lmr(output_svs=7, output_lms=FLAGS.n_landmarks):
    g = tf.Graph()
    logging.set_verbosity(10)
    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            valid_check=True)
        images, ground_truth, ground_truth_mask, heatmap, heatmap_mask = provider.get(
            'pose/mask','heatmap/mask')


        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                prediction, states, lms_prediction = resnet_model.svs_landmark_regression_net(
                    images, output_svs=output_svs, output_lms=output_lms,
                    num_iterations=FLAGS.num_iterations)

        def get_weight(keypoints, mask, ng_w=0.01, ps_w=1.0):
            is_background = tf.equal(keypoints, 0)
            ones = tf.to_float(tf.ones_like(is_background))
            weights = tf.select(is_background, ones * ng_w, ones*ps_w) * tf.to_float(mask)

            return weights

        # Add a cosine loss to every scale and the combined output.
        for i, state in enumerate(states):
            gt = ground_truth[:, i, :, :, :]
            gt_mask = ground_truth_mask[:, i, :, :, :]

            # TODO: Move 100 to a flag.
            # Reweighting the loss by 100. If we do not do this
            # The loss becomes extremely small.
            ones = tf.ones_like(gt)

            weights = tf.select(gt < .1, ones, ones * 100)

            # The non-visible parts have a substracted value of a 100.
            weights = tf.select(gt < 0, tf.zeros_like(gt), weights) * gt_mask

            loss = losses.smooth_l1(state, gt, weights)
            tf.scalar_summary('losses/iteration_{}'.format(i), loss)

        # landmark-regression losses


        weight_hm = get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(lms_prediction, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

        # total losses
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)


        # image sumarries
        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_prediction, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('mask/heatmap', tf.reduce_sum(heatmap_mask, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))


        tf.image_summary('gt/all ', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))


        for i in range(len(states)):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None], max_images=min(FLAGS.batch_size,4))
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(ground_truth[:, i, :, :, :], -1)[..., None], max_images=min(FLAGS.batch_size,4))

            for j in range(output_svs):
                state = states[i][..., j][..., None]
                gt = ground_truth[:, i, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(2, (state, gt)), max_images=min(FLAGS.batch_size,4))

        # learning rate decay
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate,
            global_step,
            FLAGS.learning_rate_decay_step / FLAGS.batch_size,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                {'/'.join(var.op.name.split('/')[2:]):var for var in variables_to_restore if not var.op.name.split('/')[1] == 'landmarks'},
                ignore_missing_vars=True)


        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)


def train_svs_hg(output_svs=7, output_lms=FLAGS.n_landmarks):
    g = tf.Graph()
    logging.set_verbosity(10)
    with g.as_default():
        # Load datasets.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            valid_check=True)
        images, ground_truth, ground_truth_mask, heatmap, heatmap_mask = provider.get(
            'pose/mask','heatmap/mask')


        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                prediction, states, lms_prediction = resnet_model.svs_hourglass_net(
                    images, output_svs=output_svs, output_lms=output_lms,
                    num_iterations=FLAGS.num_iterations)

        def get_weight(keypoints, mask, ng_w=0.01, ps_w=1.0):
            is_background = tf.equal(keypoints, 0)
            ones = tf.to_float(tf.ones_like(is_background))
            weights = tf.select(is_background, ones * ng_w, ones*ps_w) * tf.to_float(mask)

            return weights

        # Add a cosine loss to every scale and the combined output.
        for i, state in enumerate(states):
            gt = ground_truth[:, i, :, :, :]
            gt_mask = ground_truth_mask[:, i, :, :, :]

            # TODO: Move 100 to a flag.
            # Reweighting the loss by 100. If we do not do this
            # The loss becomes extremely small.
            ones = tf.ones_like(gt)

            weights = tf.select(gt < .1, ones, ones * 100)

            # The non-visible parts have a substracted value of a 100.
            weights = tf.select(gt < 0, tf.zeros_like(gt), weights) * gt_mask

            loss = losses.smooth_l1(state, gt, weights)
            tf.scalar_summary('losses/iteration_{}'.format(i), loss)

        # landmark-regression losses


        weight_hm = get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(lms_prediction, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

        # total losses
        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)


        # image sumarries
        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_prediction, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('mask/heatmap', tf.reduce_sum(heatmap_mask, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))


        tf.image_summary('gt/all ', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))


        for i in range(len(states)):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None], max_images=min(FLAGS.batch_size,4))
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(ground_truth[:, i, :, :, :], -1)[..., None], max_images=min(FLAGS.batch_size,4))

            for j in range(output_svs):
                state = states[i][..., j][..., None]
                gt = ground_truth[:, i, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(2, (state, gt)), max_images=min(FLAGS.batch_size,4))

        # learning rate decay
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate,
            global_step,
            FLAGS.learning_rate_decay_step / FLAGS.batch_size,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(FLAGS.initial_learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess, FLAGS.pretrained_resnet_checkpoint_path)

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                {'/'.join(var.op.name.split('/')[2:]):var for var in variables_to_restore if not var.op.name.split('/')[1] == 'landmarks'},
                ignore_missing_vars=True)


        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)




def train_bl(output_lms=FLAGS.n_landmarks):
    g = tf.Graph()
    logging.set_verbosity(10)
    with g.as_default():
        # Load dataset.
        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms)
        images, keypoints_visible, keypoints_visible_mask, heatmap, heatmap_mask = provider.get(
            'keypoints_visible/mask','heatmap/mask')

        def keypts_encoding(keypoints):
            keypoints = tf.to_int32(keypoints)
            keypoints = tf.reshape(keypoints, (-1,))
            keypoints = slim.layers.one_hot_encoding(keypoints, num_classes=output_lms+1)
            return keypoints

        def get_weight(keypoints, mask, ng_w=0.01, ps_w=1.0):
            is_background = tf.equal(keypoints, 0)
            ones = tf.to_float(tf.ones_like(is_background))
            weights = tf.select(is_background, ones * ng_w, ones*ps_w) * tf.to_float(mask)

            return weights

        # Define model graph.
        with tf.variable_scope('net'):
            with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                is_training=True):
                # Part-detector net that is trained only on the visible points.
                part_prediction, pyramid, _ = resnet_model.multiscale_kpts_net(images, scales=(1, 2), num_keypoints=output_lms+1)
                net = tf.concat(3, [part_prediction, images])
                # Regressor net that is trained on the whole points.
                lms_prediction = hourglass_model.network(net, 1, output_channels=output_lms)



        tf.image_summary('predictions/part-detection', generate_heatmap(part_prediction), max_images=min(FLAGS.batch_size,4))
        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_prediction, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('mask/visible', keypoints_visible_mask, max_images=min(FLAGS.batch_size,4))
        tf.image_summary('mask/heatmap', tf.reduce_sum(heatmap_mask, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))


        tf.image_summary('gt/visiable', generate_landmarks(keypoints_visible), max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))


        # Add a cosine loss to every scale and the combined output.
        # part-detection losses
        kps_visiable, weight_visible = keypts_encoding(keypoints_visible), get_weight(keypoints_visible, keypoints_visible_mask)
        weight_visible = tf.reshape(weight_visible, (-1,))
        weight_visible.set_shape([None,])

        for net,name in zip([part_prediction] + pyramid, ['final', '1', '2']):
            net = tf.reshape(net, (-1, output_lms+1))
            loss = slim.losses.softmax_cross_entropy(net, kps_visiable, weight=weight_visible)
            tf.scalar_summary('losses/part_detection_scale/{}'.format(name), loss)

        # landmark-regression losses
        weight_hm = get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(lms_prediction, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)


        total_loss = slim.losses.get_total_loss()
        tf.scalar_summary('losses/total loss', total_loss)


        # learning rate decay
        global_step = slim.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate,
            global_step,
            FLAGS.learning_rate_decay_step / FLAGS.batch_size,
            FLAGS.learning_rate_decay_factor,
            staircase=True)

        tf.scalar_summary('learning rate', learning_rate)

        optimizer = tf.train.AdamOptimizer(learning_rate)

    with tf.Session(graph=g) as sess:
        init_fn = None

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model...')
            variables_to_restore = slim.get_model_variables()
            init_fn = slim.assign_from_checkpoint_fn(
                    FLAGS.pretrained_model_checkpoint_path,
                    {'/'.join(var.op.name.split('/')[2:]):var for var in variables_to_restore if not var.op.name.split('/')[1] == 'landmarks'},
                    ignore_missing_vars=True)

        if FLAGS.pretrained_resnet_checkpoint_path:
            init_fn = restore_resnet(sess,
                                     FLAGS.pretrained_resnet_checkpoint_path)

        train_op = slim.learning.create_train_op(total_loss,
                                                 optimizer,
                                                 summarize_gradients=True)

        logging.set_verbosity(1)
        slim.learning.train(train_op,
                            FLAGS.train_dir,
                            save_summaries_secs=60,
                            init_fn=init_fn,
                            save_interval_secs=600)


if __name__ == '__main__':
    while True:
        try:
            if FLAGS.train_mode == 0:
                train()
            elif FLAGS.train_mode == 1:
                train_lmr()
            elif FLAGS.train_mode == 2:
                train_bl()
            elif FLAGS.train_mode == 3:
                train_regress()
            elif FLAGS.train_mode == 4:
                train_part_detect()
            elif FLAGS.train_mode == 5:
                train_svs_hg()
        except Exception as e:
            print(e)
