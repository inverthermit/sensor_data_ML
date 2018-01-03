from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
a = tf.constant(np.array([[1.0, 1.0, 0.0, 0.0]]))
b = tf.constant(np.array([[1.0, 0.0, 0.0, 0.0]]))
print(sess.run(b*tf.log(a)))
