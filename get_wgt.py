import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib

sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

latest_checkpoint = tf.train.latest_checkpoint("C:/Users/KSH/Desktop/feature/train/model/")
meta_graph_location = latest_checkpoint + ".meta"
saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
saver.restore(sess, latest_checkpoint)
slim.get_variables()
variables = sess.run(slim.get_variables())
def relu6(x):
    return np.maximum(np.minimum(x, 6), 0)
def sigm(x):
    return 1/(np.exp(-x)+1)
def l2_normalize(x):
    return x / np.sqrt(np.max(np.sum(x ** 2,), epsilon))


data0 = np.random.normal(size=[4,1152])
data0_normalized = data0 / np.linalg.norm(data0,axis=1).reshape([4,1])


feature = np.matmul(relu6(np.matmul(relu6(np.matmul(data0_normalized,variables[1])),variables[2])),variables[3])
feature = sigm(np.matmul(relu6(feature),variables[4]))
