# Placeholders
#----------------------------------
#
# This function introduces how to 
# use placeholders in TensorFlow

import logging

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start logging
logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')

# Using Placeholders
sess = tf.Session()

x = tf.placeholder(tf.float32, shape=(4, 4))
y = tf.identity(x)

rand_array = np.random.rand(4, 4)

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("/tmp/variable_logs", sess.graph)
logging.debug(sess.run(y, feed_dict={x: rand_array}))
logging.debug("this will write to log /tmp/variable_logs")
