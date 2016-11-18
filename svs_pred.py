import tensorflow as tf
import numpy as np
import resnet_model
import hourglass_model
import losses
import data_provider
import glob
import cv2
import sys
import utils
import networks
import menpo.io as mio
import scipy.io as sio
from tensorflow.python.platform import tf_logging as logging
from menpo.visualize import print_progress
from pathlib import Path
import matplotlib.pyplot as plt
from menpo.image import Image
from menpo.shape import PointCloud


slim = tf.contrib.slim

FLAGS = tf.app.flags.FLAGS



def pred_mpii(graph_path='/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl',
         ckpt_dir='/homes/yz4009/wd/gitdev/ibugnet/ckpt/train_svs_tunning_hg_quick/',
         store_path='/homes/yz4009/wd/databases/body/mpii-test/'):

    # load model
    sess = tf.Session()
    images = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    gt_landmarks = tf.placeholder(tf.float32, shape=(1,16,2))

    net_model = networks.DNSVSHGTorch(graph_path)

    with tf.variable_scope('net'):
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=False):
            lms_heatmap_prediction, states = net_model._build_network(images)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    model_path = slim.evaluation.tf_saver.get_checkpoint_state(ckpt_dir).model_checkpoint_path
    saver.restore(sess, model_path)
    print(model_path)

    # load data

    database_path = Path('/vol/atlas/databases/body/MPIIHumanPose')

    # annotations = loadmatToDict(str(database_path / 'mpii_human_pose_v1_u12_1.mat'))
    annotations = sio.loadmat(str(database_path / 'mpii_human_pose_v1_u12_1.mat'), squeeze_me=True, struct_as_record=False)
    annotations = annotations['RELEASE']

    anno_list = annotations.annolist[annotations.img_train == 0]
    rectidxs = annotations.single_person[annotations.img_train == 0]
    pred = [{}] * anno_list.shape[0]

    store_path = Path(store_path)

    for imgidx, (anno, ridx) in enumerate(zip(print_progress(anno_list), rectidxs)):

        img_name = anno.image.name
        img = utils.import_image('{}/images/{}'.format(database_path, img_name))


        rects = anno.annorect
        if not type(rects) == np.ndarray:
            rects = np.array([rects])

        if len(rects) == 0:
            rect = lambda:None
            objpos = lambda:None
            objpos.y, objpos.x = np.array(img.shape) / 2
            rect.scale = 1
            rect.objpos = objpos
            rects = np.array([rect])

        if not type(ridx) == np.ndarray:
            ridx = np.array([ridx])

        if ridx.shape[0] == 0:
            ridx = [1]

        pred[imgidx] = {
            'annorect': [{}] * np.max(ridx),
            'frame_sec': anno.frame_sec,
            'vididx': anno.vididx,
            'image': {'name': img_name}
        }
        for j, rid in enumerate(ridx):
            rect = rects[rid-1]
            pimg = img.copy()

            try:
                scale = rect.scale
                if type(scale) == np.ndarray:
                    scale = 1
            except:
                scale = 1

            try:
                centre = np.array([rect.objpos.y,rect.objpos.x])
            except:
                centre = np.array(img.shape) / 2


            # square bounding box
            cimg, trans, c_scale = utils.crop_image(pimg, centre, scale, [384,384])
            offset = 256 / 2
            offset_pt = np.array([offset,offset])
            ccimg, ctrans = cimg.crop(cimg.centre()-offset_pt,cimg.centre()+offset_pt, return_transform=True)


            # predict
            def fn_predict(input_pixels):

                lms_hm_prediction, = sess.run(
                    [lms_heatmap_prediction],
                    feed_dict={images: input_pixels[None, ...]})

                hs = np.argmax(np.max(lms_hm_prediction.squeeze().transpose(2,0,1), 2), 1)
                ws = np.argmax(np.max(lms_hm_prediction.squeeze().transpose(2,0,1), 1), 1)
                pts_predictions = np.stack([hs,ws]).T

                return pts_predictions


            pts_pred_l = fn_predict(ccimg.pixels_with_channels_at_back() / 255)
            pts_predictions = pts_pred_l

            if FLAGS.flip_pred:
                rcimg, t_refl = ccimg.mirror(return_transform=True)
                pts_pred_r = fn_predict(rcimg.pixels_with_channels_at_back() / 255)
                pts_pred_r = t_refl.apply(pts_pred_r[[5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10],:])

                pts_predictions = (pts_pred_l + pts_pred_r) / 2

            # store points


            pred[imgidx]['annorect'][rid-1] = {
                'annopoints': {
                    'point': [{}] * 16
                }
            }

            orig_pts = trans.apply(PointCloud(ctrans.apply(pts_predictions) * c_scale)).points
            cimg.landmarks['LJSON'] = PointCloud(orig_pts)

            for pidx, pts in enumerate(orig_pts):

                pred[imgidx]['annorect'][rid-1]['annopoints']['point'][pidx] = {
                    'x': pts[1],
                    'y': pts[0],
                    'id': pidx
                }

            img_path = store_path / '{}'.format(img_name)

    #             Save files
            mio.export_landmark_file(
                cimg.landmarks['LJSON'],
                '{}/{}-{}.ljson'.format(img_path.parent, rid, img_path.stem),
                overwrite=True)

def mat_mpii(store_path='/homes/yz4009/wd/databases/body/mpii-test/'):
    database_path = Path('/vol/atlas/databases/body/MPIIHumanPose')

    annotations = sio.loadmat(str(database_path / 'mpii_human_pose_v1_u12_1.mat'), squeeze_me=True, struct_as_record=False)
    annotations = annotations['RELEASE']

    anno_list = annotations.annolist[annotations.img_train == 0]
    rectidxs = annotations.single_person[annotations.img_train == 0]
    pred = [{}] * anno_list.shape[0]

    store_path = Path(store_path)

    for imgidx, (anno, ridx) in enumerate(zip(print_progress(anno_list), rectidxs)):

        img_name = anno.image.name

        rects = anno.annorect
        if not type(rects) == np.ndarray:
            rects = np.array([rects])

        if len(rects) == 0:
            rect = lambda:None
            objpos = lambda:None
            objpos.y, objpos.x = [0,0]
            rect.scale = 1
            rect.objpos = objpos
            rects = np.array([rect])

        if not type(ridx) == np.ndarray:
            ridx = np.array([ridx])

        if ridx.shape[0] == 0:
            ridx = [1]

        pred[imgidx] = {
            'annorect': [{}] * np.max(ridx),
            'frame_sec': anno.frame_sec,
            'vididx': anno.vididx,
            'image': {'name': img_name}
        }
        for j, rid in enumerate(ridx):
            rect = rects[rid-1]

            try:
                scale = rect.scale
                if type(scale) == np.ndarray:
                    scale = 1
            except:
                scale = 1

            try:
                centre = np.array([rect.objpos.y,rect.objpos.x])
            except:
                centre = np.array([0,0])


            # store points
            img_path = store_path / '{}'.format(img_name)

            pred[imgidx]['annorect'][rid-1] = {
                'annopoints': {
                    'point': [{}] * 16
                }
            }

            lms = mio.import_landmark_file(
                '{}/{}-{}.ljson'.format(img_path.parent, rid, img_path.stem))

            orig_pts = lms.lms.points

            for pidx, pts in enumerate(orig_pts):

                pred[imgidx]['annorect'][rid-1]['annopoints']['point'][pidx] = {
                    'x': pts[1],
                    'y': pts[0],
                    'id': pidx
                }

    sio.savemat(str(store_path) + '/mat/pred_keypoints_mpii.mat', {'prediction':pred})



def pred_lsp(graph_path='/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl',
             ckpt_dir='/homes/yz4009/wd/gitdev/ibugnet/ckpt/train_svs_tunning_hg_quick/',
             store_path='/homes/yz4009/wd/databases/body/lsp-test-svs/'):

    # load model
    sess = tf.Session()
    images = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    gt_landmarks = tf.placeholder(tf.float32, shape=(1,16,2))

    net_model = networks.DNSVSHGTorch(graph_path)

    with tf.variable_scope('net'):
        with slim.arg_scope([slim.batch_norm, slim.layers.dropout], is_training=False):
            lms_heatmap_prediction, states = net_model._build_network(images)

    variables_to_restore = slim.get_variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    model_path = slim.evaluation.tf_saver.get_checkpoint_state(ckpt_dir).model_checkpoint_path
    saver.restore(sess, model_path)
    print(model_path)

    #load data
    store_path = Path(store_path)
    image_load_path = Path('/vol/atlas/databases/body/lsp_dataset/images')
    annotations = sio.loadmat('/vol/atlas/databases/body/lsp_dataset/joints.mat', squeeze_me=True, struct_as_record=False)

    results = []
    for nimg in print_progress(list(range(1001,2001))):

        image_name = Path('im{:04d}.jpg'.format(nimg))
        load_path = image_load_path / image_name

        img = utils.import_image(load_path)

        anno_points = annotations['joints'][:2,:,nimg-1].T[:,-1::-1]
        joints_lms = np.zeros((16,2))
        visiblepts = [0,1,2,3,4,5,10,11,12,13,14,15,8,9]
        marked_ids = visiblepts
        joints_lms[marked_ids] = anno_points

        img.landmarks['JOINT'] = PointCloud(joints_lms)
        img.landmarks['JOINT']['visible'] = visiblepts
        img.landmarks['JOINT']['marked'] = marked_ids

        head_scale = np.linalg.norm(anno_points[2] - anno_points[13])
        scale = 0.89

        pimg = img
        centre = np.array(img.shape) / 2
        # square bounding box
        cimg, trans, c_scale = utils.crop_image(pimg, centre, scale, [256,256], base=200)
        offset = 256 / 2
        offset_pt = np.array([offset,offset])
        ccimg, ctrans = cimg.crop(cimg.centre()-offset_pt,cimg.centre()+offset_pt, return_transform=True)


        # predict
        def fn_predict(input_pixels):

            lms_hm_prediction, = sess.run(
                [lms_heatmap_prediction],
                feed_dict={images: input_pixels[None, ...]})

            hs = np.argmax(np.max(lms_hm_prediction.squeeze().transpose(2,0,1), 2), 1)
            ws = np.argmax(np.max(lms_hm_prediction.squeeze().transpose(2,0,1), 1), 1)
            pts_predictions = np.stack([hs,ws]).T

            return pts_predictions


        pts_pred_l = fn_predict(ccimg.pixels_with_channels_at_back() / 255)
        pts_predictions = pts_pred_l

        if FLAGS.flip_pred:
            rcimg, t_refl = ccimg.mirror(return_transform=True)
            pts_pred_r = fn_predict(rcimg.pixels_with_channels_at_back() / 255)
            pts_pred_r = t_refl.apply(pts_pred_r[[5,4,3,2,1,0,6,7,8,9,15,14,13,12,11,10],:])

            pts_predictions = (pts_pred_l + pts_pred_r) / 2


        # store points


        orig_pts = trans.apply(PointCloud(pts_predictions * c_scale)).points
        cimg.landmarks['LJSON'] = PointCloud(orig_pts)
        img_path = store_path / '{}'.format(image_name)

        results.append(np.linalg.norm(orig_pts[[0,1,2,3,4,5,10,11,12,13,14,15,8,9]] - anno_points, axis=-1) / head_scale)

        # Save files

        mio.export_landmark_file(
            cimg.landmarks['LJSON'],
            '{}/{}.ljson'.format(img_path.parent, img_path.stem),
            overwrite=True
            )

    print((np.array(results) < 0.2).astype(np.int).astype(np.float32).mean(axis=0))
    print((np.array(results) < 0.2).astype(np.int).astype(np.float32).mean(axis=0).mean())


def mat_lsp(store_path='/homes/yz4009/wd/databases/body/lsp-test-svs/'):
    mat_file = [lms.lms.points[[0,1,2,3,4,5,10,11,12,13,14,15,8,9],-1::-1] for lms in mio.import_landmark_files(store_path)]
    predictions = np.array(mat_file).transpose(2,1,0)
    sio.savemat(str(store_path) + '/mat/pred_keypoints_lsp.mat', {'pred':predictions})


if __name__ == '__main__':
    this = sys.modules[__name__]

    getattr(this, '{}_{}'.format(FLAGS.pred_mode, FLAGS.db_name))()
