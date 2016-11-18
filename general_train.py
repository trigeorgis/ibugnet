import tensorflow as tf
import numpy as np
import utils
import time
import networks
import traceback
import matplotlib.pyplot as plt

from tensorflow.python.platform import tf_logging as logging
from pathlib import Path

slim = tf.contrib.slim

from flags import FLAGS

# The decay to use for the moving average.
MOVING_AVERAGE_DECAY = 0.9999



def get_model(netname):
    if netname == "hg":
        return networks.DNHourglass()
    elif netname == "svs_res":
        return networks.DNSVSRes()
    elif netname == "pd":
        return networks.DNPartDetect()
    elif netname == "torch":
        return networks.DNTorch('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/out-graph.pkl')
    elif netname == "yorgos_rep":
        return networks.DNTorch('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl')
    elif netname == "yorgos_w":
        return networks.DNTorch('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl')
    elif netname == "pd_hg":
        return networks.DNPartDetectHourglass()
    elif netname == "svs_hg_hg":
        return networks.DNSVSHourglass()
    elif netname == "svs_ms_hg":
        return networks.DNSVSScaleHourglass()
    elif netname == "svs_tunning":
        return networks.DNSVSResTorch('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl')
    elif netname == "svs_tunning_hg":
        return networks.DNSVSHGTorch('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl')
    elif netname == "svs_tunning_hg_svs":
        return networks.DNSVSPartTunning('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl')
    elif netname == "svs_tunning_hg_quick":
        return networks.DNQuickSVSHGTorch('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl')
    elif netname == "svs_decompose":
        return networks.DNQuickSVSDecompose('/homes/yz4009/wd/gitdev/human-pose-estimation-master/tfmodels/yorgos_graph_weight.pkl')
    else:
        raise Exception('Unknown model ' + netname)


if __name__ == '__main__':
    while True:
        try:
            if FLAGS.eval_dir == '':
                get_model(FLAGS.train_model).train()
            else:
                get_model(FLAGS.train_model).eval()

        except Exception as e:
            traceback.print_exc()
            time.sleep(10)
