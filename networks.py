import tensorflow as tf
import numpy as np
import resnet_model
import hourglass_model
import losses
import data_provider
import utils
import matplotlib.pyplot as plt
import pickle

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

from flags import FLAGS

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999




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

# general framework
class DeepNetwork(object):
    """docstring for DeepNetwork"""
    def __init__(self, data_provider, *data_keys, output_lms=FLAGS.n_landmarks):
        super(DeepNetwork, self).__init__()
        self.output_lms = output_lms
        self.data_provider = data_provider
        self.data_keys = data_keys

    def _build_network(self, inputs):
        pass


    def _build_losses(self, predictions, states, images, datas):
        pass


    def _build_summaries(self, predictions, states, images, datas):
        pass


    def _build_restore_fn(self, sess):
        init_fn = None

        if FLAGS.pretrained_model_checkpoint_path:
            print('Loading whole model ...')
            variables_to_restore = slim.get_model_variables()
            init_fn =  slim.assign_from_checkpoint_fn(
                FLAGS.pretrained_model_checkpoint_path,
                variables_to_restore,
                ignore_missing_vars=True)
        return init_fn


    def train(self):
        g = tf.Graph()
        logging.set_verbosity(10)


        with g.as_default():
            # Load datasets.

            images, *datas = self.data_provider.get(*self.data_keys, preprocess_inputs=False)
            images /= 255.

            # Define model graph.
            with tf.variable_scope('net'):
                with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                    is_training=True):

                    predictions, states = self._build_network(images)

            # custom losses
            self._build_losses(predictions, states, images, datas)

            # total losses
            total_loss = slim.losses.get_total_loss()
            tf.scalar_summary('losses/total loss', total_loss)


            # image summaries
            self._build_summaries(predictions, states, images, datas)
            tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4))

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
            init_fn = self._build_restore_fn(sess)
            train_op = slim.learning.create_train_op(total_loss,
                                                     optimizer,
                                                     summarize_gradients=True)
            logging.set_verbosity(1)
            slim.learning.train(train_op,
                                FLAGS.train_dir,
                                save_summaries_secs=60,
                                init_fn=init_fn,
                                save_interval_secs=600)

    def _eval_matrix(self, lms_predictions, states, images, datas):
        *_, gt_landmarks, scales = datas

        hs = tf.argmax(tf.reduce_max(lms_predictions, 2), 1)
        ws = tf.argmax(tf.reduce_max(lms_predictions, 1), 1)
        predictions = tf.transpose(tf.to_float(tf.pack([hs, ws])), perm=[1, 2, 0])

        return utils.pckh(predictions, gt_landmarks, scales)

    def _eval_summary_ops(self, accuracy, lms_predictions, states, images, datas):

        # These are streaming metrics which compute the "running" metric,
        # e.g running accuracy
        metrics_to_values, metrics_to_updates = slim.metrics.aggregate_metric_map({

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
        summary_ops.append(tf.image_summary('predictions/landmark-regression', tf.reduce_sum(lms_predictions, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4)))
        summary_ops.append(tf.image_summary('images', images, max_images=min(FLAGS.batch_size,4)))

        return summary_ops, metrics_to_updates

    def eval(self):

        g = tf.Graph()

        with g.as_default():
            # Load datasets.
            images, *datas = self.data_provider.get(*self.data_keys, preprocess_inputs=False)
            images /= 255.

            # Define model graph.
            with tf.variable_scope('net'):
                with slim.arg_scope([slim.batch_norm, slim.layers.dropout],
                                    is_training=False):

                    lms_predictions, states = self._build_network(images)


        with tf.Session(graph=g) as sess:

            accuracy = self._eval_matrix(lms_predictions, states, images, datas)
            # accuracy = tf.Print(accuracy, [tf.shape(accuracy), accuracy], summarize=5)
            # These are streaming metrics which compute the "running" metric,
            # e.g running accuracy
            summary_ops, metrics_to_updates = self._eval_summary_ops(
                accuracy, lms_predictions, states, images, datas)

            global_step = slim.get_or_create_global_step()
            # Evaluate every 30 seconds
            logging.set_verbosity(1)

            # num_examples = provider.num_samples()
            # num_batches = np.ceil(num_examples / FLAGS.batch_size)
            # num_batches = 500

            slim.evaluation.evaluation_loop(
                '',
                FLAGS.train_dir,
                FLAGS.eval_dir,
                num_evals=FLAGS.eval_size,
                eval_op=list(metrics_to_updates.values()),
                summary_op=tf.merge_summary(summary_ops),
                eval_interval_secs=30)


# Hourglass Part
class DNHourglass(DeepNetwork):
    """docstring for DNHourglass"""
    def __init__(self, output_lms=16):

        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            valid_check=False)

        super(DNHourglass, self).__init__(provider, 'heatmap/mask', 'landmarks', 'scale', output_lms=output_lms)

    def _build_network(self, inputs):

        prediction = hourglass_model.network(
            inputs, 1, output_channels=self.output_lms)

        return prediction, None

    def _build_losses(self, predictions, states, images, datas):
        heatmap, heatmap_mask, *_ = datas

        # landmark-regression losses
        weight_hm = utils.get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(predictions, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        heatmap,heatmap_mask, *_ = datas

        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all ', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

# Resnet SVS Part
class DNSVSRes(DeepNetwork):
    """docstring for DNSVSPart"""
    def __init__(self, output_svs=7, output_lms=16):

        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            valid_check=True)

        super(DNSVSRes, self).__init__(provider, 'pose/mask', 'landmarks', 'scale', output_lms=output_lms)

        self.output_svs=output_svs

    def _build_network(self, inputs):

        prediction, states = resnet_model.svs_regression_net_light(
            inputs, output_classes=self.output_svs, num_iterations=FLAGS.num_iterations)

        return prediction, states

    def _build_losses(self, predictions, states, images, datas):
        ground_truth, ground_truth_mask, *_ = datas

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

    def _build_summaries(self, predictions, states, images, datas):
        ground_truth, ground_truth_mask, *_ = datas

        for i in range(len(states)):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None], max_images=min(FLAGS.batch_size,4))
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(ground_truth[:, i, :, :, :], -1)[..., None], max_images=min(FLAGS.batch_size,4))

            for j in range(self.output_svs):
                state = states[i][..., j][..., None]
                gt = ground_truth[:, i, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(2, (state, gt)), max_images=min(FLAGS.batch_size,4))

# Multiscale Part-Detect Part
class DNPartDetect(DeepNetwork):
    """docstring for DNSVSPart"""
    def __init__(self, output_lms=16):

        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            valid_check=False)

        super(DNPartDetect, self).__init__(provider, 'keypoints_visible/mask', 'landmarks', 'scale', output_lms=output_lms)

    def _build_network(self, inputs):

        part_prediction, pyramid, _ = resnet_model.multiscale_kpts_net(
            inputs, scales=(1, 2), num_keypoints=self.output_lms+1)

        return part_prediction, pyramid

    def _build_losses(self, predictions, states, images, datas):
        keypoints_visible, keypoints_visible_mask, *_ = datas

        # part-detection losses
        kps_visiable, weight_visible = utils.keypts_encoding(keypoints_visible, self.output_lms), utils.get_weight(keypoints_visible, keypoints_visible_mask)
        weight_visible = tf.reshape(weight_visible, (-1,))
        weight_visible.set_shape([None,])

        for net, name in zip([predictions] + states, ['final', '1', '2']):
            net = tf.reshape(net, (-1, self.output_lms+1))
            loss = slim.losses.softmax_cross_entropy(net, kps_visiable, weight=weight_visible)
            tf.scalar_summary('losses/part_detection_scale/{}'.format(name), loss)

    def _build_summaries(self, predictions, states, images, datas):
        keypoints_visible, keypoints_visible_mask, *_ = datas

        tf.image_summary('predictions/part-detection', utils.generate_heatmap(predictions,self.output_lms), max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/visiable', utils.generate_landmarks(keypoints_visible), max_images=min(FLAGS.batch_size,4))


# Multiscale Part-Detect + Hourglass
class DNPartDetectHourglass(DeepNetwork):
    """docstring for DNSVSPart"""
    def __init__(self, output_lms=16):

        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            valid_check=False)

        super(DNPartDetectHourglass, self).__init__(provider, 'keypoints_visible/mask','heatmap/mask', 'landmarks', 'scale', output_lms=output_lms)

    def _build_network(self, inputs):

        # Part-detector net that is trained only on the visible points.
        part_prediction, pyramid, _ = resnet_model.multiscale_kpts_net(inputs, scales=(1, 2), num_keypoints=self.output_lms+1)
        net = tf.concat(3, [part_prediction, inputs])
        # Regressor net that is trained on the whole points.
        lms_prediction = hourglass_model.network(net, 1, output_channels=self.output_lms)

        return lms_prediction, [part_prediction] + pyramid

    def _build_losses(self, predictions, states, images, datas):
        keypoints_visible, keypoints_visible_mask, heatmap, heatmap_mask, *_ = datas

        # part-detection losses
        kps_visiable, weight_visible = utils.keypts_encoding(keypoints_visible, self.output_lms), utils.get_weight(keypoints_visible, keypoints_visible_mask)
        weight_visible = tf.reshape(weight_visible, (-1,))
        weight_visible.set_shape([None,])

        for net,name in zip(states, ['final', '1', '2']):
            net = tf.reshape(net, (-1, self.output_lms+1))
            loss = slim.losses.softmax_cross_entropy(net, kps_visiable, weight=weight_visible)
            tf.scalar_summary('losses/part_detection_scale/{}'.format(name), loss)

        # landmark-regression losses
        weight_hm = utils.get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(predictions, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        keypoints_visible, keypoints_visible_mask, heatmap, heatmap_mask, *_ = datas

        tf.image_summary('predictions/part-detection', utils.generate_heatmap(states[0],self.output_lms), max_images=min(FLAGS.batch_size,4))
        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        tf.image_summary('gt/visiable', utils.generate_landmarks(keypoints_visible), max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))





# Multiscale SVS + Landmark Regression Framework

class DNQuickSVSLMS(DeepNetwork):
    """docstring for DNSVSHourglass"""
    def __init__(self, output_svs=7, output_lms=16, provider=None):


        if not provider:
            provider = data_provider.ProtobuffProvider(
                batch_size=FLAGS.batch_size,
                root=FLAGS.dataset_dir,
                rescale=FLAGS.rescale,
                augmentation=FLAGS.eval_dir=='',
                )

        keys = ['heatmap/mask', 'landmarks', 'scale']
        if FLAGS.eval_dir=='':
            keys = ['pose/mask'] + keys

        super(DNQuickSVSLMS, self).__init__(provider, *keys, output_lms=output_lms)

        self.output_svs = output_svs

    def _build_network(self, inputs):
        pass

    def _build_losses(self, predictions, states, images, datas):
        pose, gt_heatmap, *_ = datas
        # Add a cosine loss to every scale and the combined output.
        for i, state in enumerate(states):
            gt = pose[:, i, :, :, :]

            ones = tf.ones_like(gt)
            weights = tf.select(gt < .1, ones, ones * 100)

            loss = losses.smooth_l1(state, gt, weights)
            tf.scalar_summary('losses/iteration_{}'.format(i), loss)

        # landmark-regression losses
        weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(predictions, gt_heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        pose, gt_heatmap, *_ = datas

        batch_summariy = tf.concat(1, [
            tf.reduce_sum(images, -1)[...,None],
            tf.reduce_sum(predictions, -1)[...,None],
            tf.reduce_sum(gt_heatmap, -1)[...,None]
        ])

        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None], max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all ', tf.reduce_sum(gt_heatmap, -1)[...,None], max_images=min(FLAGS.batch_size,4))

        for i in range(len(states)):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None], max_images=min(FLAGS.batch_size,4))
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(pose[:, i, :, :, :], -1)[..., None], max_images=min(FLAGS.batch_size,4))

            batch_summariy = tf.concat(1, [
                batch_summariy,
                tf.reduce_sum(states[i], -1)[..., None],
                tf.reduce_sum(pose[:, i, :, :, :], -1)[..., None]
            ])

            for j in range(self.output_svs):
                state = states[i][..., j][..., None]
                gt = pose[:, i, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(1, (state, gt)), max_images=min(FLAGS.batch_size,4))


        tf.image_summary('batch', batch_summariy, max_images=min(FLAGS.batch_size,4))



class DNSVSLMS(DeepNetwork):
    """docstring for DNSVSHourglass"""
    def __init__(self, output_svs=7, output_lms=16, svs_index=[0,1,2,3]):

        provider = data_provider.HumanPose(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            n_lms=output_lms,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            valid_check=FLAGS.eval_dir=='',
            svs_index=svs_index)

        keys = ['heatmap/mask', 'landmarks', 'scale']
        if FLAGS.eval_dir=='':
            keys = ['pose/mask'] + keys

        super(DNSVSLMS, self).__init__(provider, *keys, output_lms=output_lms)

        self.output_svs = output_svs

    def _build_network(self, inputs):
        pass

    def _build_losses(self, predictions, states, images, datas):
        ground_truth,ground_truth_mask,heatmap,heatmap_mask, *_ = datas
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
        weight_hm = utils.get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(predictions, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        ground_truth,ground_truth_mask,heatmap,heatmap_mask, *_ = datas

        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all ', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))

        for i in range(len(states)):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None], max_images=min(FLAGS.batch_size,4))
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(ground_truth[:, i, :, :, :], -1)[..., None], max_images=min(FLAGS.batch_size,4))

            for j in range(self.output_svs):
                state = states[i][..., j][..., None]
                gt = ground_truth[:, i, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(2, (state, gt)), max_images=min(FLAGS.batch_size,4))



# SVS Hourglass + Hourglass
class DNSVSHourglass(DNSVSLMS):
    """docstring for DNSVSHourglass"""
    def __init__(self, output_svs=7, output_lms=16):

        super(DNSVSHourglass, self).__init__(output_svs=output_svs, output_lms=output_lms)


    def _build_network(self, inputs):

        prediction, states, lms_prediction = resnet_model.svs_hourglass_net(
            inputs, output_svs=self.output_svs, output_lms=self.output_lms,
            num_iterations=FLAGS.num_iterations)

        return lms_prediction, states


# SVS Resnet + Hourglass
class DNSVSScaleHourglass(DNSVSLMS):
    """docstring for DNSVSHourglass"""
    def __init__(self, output_svs=7, output_lms=16):

        super(DNSVSScaleHourglass, self).__init__(output_svs=output_svs, output_lms=output_lms)


    def _build_network(self, inputs):

        prediction, states, lms_prediction = resnet_model.svs_landmark_regression_net(
            inputs, output_svs=self.output_svs, output_lms=self.output_lms,
            num_iterations=FLAGS.num_iterations)

        return lms_prediction, states


# Torch Net
class DNTorch(DNHourglass):
    """docstring for DNHourglass"""
    def __init__(self, path, output_lms=16):

        super(DNTorch, self).__init__(output_lms=output_lms)

        self.network_path = path

    def _build_network(self, inputs):

        with open(self.network_path, 'br') as f:
            data = pickle.load(f, encoding='latin1')

        prediction = utils.build_graph_old(inputs, data)

        return prediction, None

    def _build_losses(self, predictions, states, images, datas):
        heatmap, heatmap_mask, *_ = datas

        # landmark-regression losses
        weight_hm = utils.get_weight(heatmap, heatmap_mask, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(predictions, heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        heatmap,heatmap_mask, *_ = datas

        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all ', tf.reduce_sum(heatmap * tf.to_float(heatmap_mask), -1)[...,None] * 255.0, max_images=min(FLAGS.batch_size,4))


# SVS Hourglass + Hourglass
class DNSVSResTorch(DNSVSLMS):
    """docstring for DNSVSHourglass"""
    def __init__(self, path, output_svs=7, output_lms=16, svs_index=[0,1,2,3]):

        super(DNSVSResTorch, self).__init__(
            output_svs=output_svs,
            output_lms=output_lms,
            svs_index=svs_index[::len(svs_index)//FLAGS.num_iterations])

        self.network_path = path


    def _build_network(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        states = []
        hidden = tf.zeros(
            (batch_size, height, width, self.output_svs), name='hidden')

        with open(self.network_path, 'br') as f:
            data = pickle.load(f, encoding='latin1')

        subnet_svs = data['children'][0]
        for i in range(3):
            subnet_svs['children'][-1]['children'].pop()

        subnet_regression = data['children'][2]


        for i in range(FLAGS.num_iterations):
            with tf.variable_scope('multiscale', reuse=i > 0):
                hidden = utils.build_graph(tf.concat(3, (inputs, hidden)), subnet_svs)
                hidden = slim.conv2d_transpose(
                    hidden,
                    tf.shape(hidden)[-1],
                    16,
                    16,
                    activation_fn=None,
                    padding='VALID'
                )
                hidden = slim.conv2d(
                    hidden,
                    self.output_svs,
                    1,
                    activation_fn=None,
                )
                states.append(hidden)

        lms_prediction = utils.build_graph(tf.concat(3, (inputs, hidden)),  subnet_regression)

        return lms_prediction, states


class DNSVSHGTorch(DNSVSLMS):
    """docstring for DNSVSHourglass"""
    def __init__(self, path, output_svs=7, output_lms=16, svs_index=[0,1,2,3]):

        super(DNSVSHGTorch, self).__init__(
            output_svs=output_svs,
            output_lms=output_lms,
            svs_index=svs_index[::len(svs_index)//FLAGS.num_iterations])

        self.network_path = path


    def _build_network(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        states = []
        hidden = tf.zeros(
            (batch_size, height, width, self.output_svs), name='hidden')

        with open(self.network_path, 'br') as f:
            data = pickle.load(f, encoding='latin1')

        subnet_regression = data['children'][2].copy()
        subnet_regression['children'][-2]['children'].pop()
        subnet_regression['children'].pop()


        for i in range(FLAGS.num_iterations):
            with tf.variable_scope('multiscale', reuse=i > 0):
                hidden = tf.concat(3, (inputs, hidden))
                hidden = slim.conv2d(
                    hidden,
                    19,
                    1,
                    activation_fn=None
                )

                hidden = utils.build_graph(hidden, subnet_regression)

                hidden = slim.conv2d(
                    hidden,
                    self.output_svs,
                    1,
                    activation_fn=None
                )
                hidden = slim.conv2d_transpose(
                    hidden,
                    self.output_svs,
                    4,
                    4,
                    activation_fn=None,
                    padding='VALID'
                )
                states.append(hidden)

        net = tf.concat(3, (inputs, hidden))
        net = slim.conv2d(
            net,
            19,
            1,
            activation_fn=None
        )

        with open(self.network_path, 'br') as f:
            data2 = pickle.load(f, encoding='latin1')
        prediction = utils.build_graph(net,  data2['children'][2])

        return prediction, states


class DNQuickSVSHGTorch(DNQuickSVSLMS):
    """docstring for DNSVSHourglass"""
    def __init__(self, path, output_svs=7, output_lms=16):

        super(DNQuickSVSHGTorch, self).__init__(
            output_svs=output_svs,
            output_lms=output_lms)

        self.network_path = path


    def _build_network(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        states = []
        hidden = tf.zeros(
            (batch_size, height, width, self.output_svs), name='hidden')

        with open(self.network_path, 'br') as f:
            data = pickle.load(f, encoding='latin1')

        subnet_regression = data['children'][2].copy()
        subnet_regression['children'][-2]['children'].pop()
        subnet_regression['children'].pop()


        for i in range(FLAGS.num_iterations):
            with tf.variable_scope('multiscale', reuse=i > 0):
                hidden = tf.concat(3, (inputs, hidden))
                hidden = slim.conv2d(
                    hidden,
                    19,
                    1,
                    activation_fn=None
                )

                hidden = utils.build_graph(hidden, subnet_regression)

                hidden = slim.conv2d(
                    hidden,
                    self.output_svs,
                    1,
                    activation_fn=None
                )
                hidden = slim.conv2d_transpose(
                    hidden,
                    self.output_svs,
                    4,
                    4,
                    activation_fn=None,
                    padding='VALID'
                )
                states.append(hidden)

        net = tf.concat(3, (inputs, hidden))
        net = slim.conv2d(
            net,
            19,
            1,
            activation_fn=None
        )

        with open(self.network_path, 'br') as f:
            data2 = pickle.load(f, encoding='latin1')
        prediction = utils.build_graph(net,  data2['children'][2])

        return prediction, states


class DNQuickSVSDecompose(DNQuickSVSLMS):
    """docstring for DNSVSHourglass"""
    def __init__(self, path, output_svs=[1,7,15], output_lms=16, svs_index=[3,2,1]):

        provider = data_provider.DeconposePoseProvider(
            batch_size=FLAGS.batch_size,
            root=FLAGS.dataset_dir,
            rescale=FLAGS.rescale,
            augmentation=FLAGS.eval_dir=='',
            )

        super(DNQuickSVSDecompose, self).__init__(
            output_svs=output_svs,
            output_lms=output_lms,
            provider=provider)

        self.network_path = path
        self.svs_index = svs_index


    def _build_network(self, inputs):
        batch_size = tf.shape(inputs)[0]
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]

        states = []
        hidden = tf.zeros(
            (batch_size, height, width, 1), name='hidden')

        with open(self.network_path, 'br') as f:
            data = pickle.load(f, encoding='latin1')

        subnet_regression = data['children'][2].copy()
        subnet_regression['children'][-2]['children'].pop()
        subnet_regression['children'].pop()


        for i, output_svs in enumerate(self.output_svs):
            hidden = tf.concat(3, [inputs] + states)
            hidden = slim.conv2d(
                hidden,
                19,
                1,
                activation_fn=None
            )

            hidden = utils.build_graph(hidden, subnet_regression)

            hidden = slim.conv2d(
                hidden,
                output_svs,
                1,
                activation_fn=None
            )
            hidden = slim.conv2d_transpose(
                hidden,
                output_svs,
                4,
                4,
                activation_fn=None,
                padding='VALID'
            )
            states.append(hidden)

        net = tf.concat(3, [inputs] + states)
        net = slim.conv2d(
            net,
            19,
            1,
            activation_fn=None
        )

        with open(self.network_path, 'br') as f:
            data2 = pickle.load(f, encoding='latin1')
        prediction = utils.build_graph(net,  data2['children'][2])

        return prediction, states

    def _build_losses(self, predictions, states, images, datas):
        pose, gt_heatmap, *_ = datas
        # Add a cosine loss to every scale and the combined output.
        for j, (pose_idx, state) in enumerate(zip(self.svs_index, states)):
            gt = pose[:, pose_idx, :, :, :]

            # decompose
            if j == 0:
                gt = tf.reduce_sum(gt, 3)[..., None]
            elif j == 1:
                gt=tf.transpose(gt, perm=[3,0,1,2])
                parts = []
                for indexes in [[0,1],[2,3],[4,5],[6,7,8],[9,10],[11,12],[13,14]]:

                    part = tf.gather(gt, indexes)
                    part = tf.reduce_sum(part, 0)

                    parts.append(part)

                gt = tf.pack(parts, axis=3)

            # losses
            ones = tf.ones_like(gt)
            gt = tf.select(gt > 1., ones, gt)
            weights = tf.select(gt < .1, ones, ones * 100)

            loss = losses.smooth_l1(state, gt, weights)
            tf.scalar_summary('losses/iteration_{}'.format(j), loss)

        # landmark-regression losses
        weight_hm = utils.get_weight(gt_heatmap, ng_w=0.1, ps_w=1) * 500
        l2norm = slim.losses.mean_squared_error(predictions, gt_heatmap, weight=weight_hm)
        tf.scalar_summary('losses/lms_pred', l2norm)

    def _build_summaries(self, predictions, states, images, datas):
        pose, gt_heatmap, *_ = datas

        batch_summariy = tf.concat(1, [
            tf.reduce_sum(images, -1)[...,None],
            tf.reduce_sum(predictions, -1)[...,None],
            tf.reduce_sum(gt_heatmap, -1)[...,None]
        ])

        tf.image_summary('predictions/landmark-regression', tf.reduce_sum(predictions, -1)[...,None], max_images=min(FLAGS.batch_size,4))
        tf.image_summary('gt/all ', tf.reduce_sum(gt_heatmap, -1)[...,None], max_images=min(FLAGS.batch_size,4))

        for i, pose_idx in enumerate(self.svs_index):
            tf.image_summary('state/it_{}'.format(i), tf.reduce_sum(states[i], -1)[..., None], max_images=min(FLAGS.batch_size,4))
            tf.image_summary('gt/it_{}'.format(i), tf.reduce_sum(pose[:, pose_idx, :, :, :], -1)[..., None], max_images=min(FLAGS.batch_size,4))

            batch_summariy = tf.concat(1, [
                batch_summariy,
                tf.reduce_sum(states[i], -1)[..., None],
                tf.reduce_sum(pose[:, pose_idx, :, :, :], -1)[..., None]
            ])

            for j in range(self.output_svs[i]):
                state = states[i][..., j][..., None]
                gt = pose[:, pose_idx, ..., j][..., None]
                tf.image_summary('state/it_{}/part_{}'.format(i, j),  tf.concat(1, (state, gt)), max_images=min(FLAGS.batch_size,4))


        tf.image_summary('batch', batch_summariy, max_images=min(FLAGS.batch_size,4))

class DNSVSPartTunning(DNSVSHGTorch):
    """docstring for DNSVSPartTunning"""
    def __init__(self, path, output_svs=7, output_lms=16, svs_index=[0,1,2,3]):
        super(DNSVSPartTunning, self).__init__(
            path, output_svs=output_svs, output_lms=output_lms,
            svs_index=svs_index)



    def _build_losses(self, predictions, states, images, datas):
        ground_truth,ground_truth_mask,heatmap,heatmap_mask, *_ = datas
        # Add a cosine loss to every scale and the combined output.
        for i, state in enumerate(states):
            gt = ground_truth[:, i, :, :, :]
            gt_mask = ground_truth_mask[:, i, :, :, :]

            ones = tf.ones_like(gt)

            weights = tf.select(gt < .1, ones, ones * 100)

            # The non-visible parts have a substracted value of a 100.
            weights = tf.select(gt < 0, tf.zeros_like(gt), weights) * gt_mask

            # loss = losses.smooth_l1(state, gt, weights)
            l2norm = slim.losses.mean_squared_error(state, gt, weight=weights)
            tf.scalar_summary('losses/iteration_{}'.format(i), l2norm)
