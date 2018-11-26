# Matrices and Matrix Operations
#----------------------------------
#
# This function introduces various ways to create
# matrices and how to use them in TensorFlow

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

# start logging
logging.basicConfig(level=logging.DEBUG)
logging.debug('This will get logged')

# Declaring matrices
sess = tf.Session()

# Identity matrix
identity_matrix = tf.diag([1.0,1.0,1.0])
logging.debug(sess.run(identity_matrix))

# 2x3 random norm matrix
A = tf.truncated_normal([2,3])
logging.debug(sess.run(A))

# 2x3 constant matrix
B = tf.fill([2,3], 5.0)
logging.debug(sess.run(B))

# 3x2 random uniform matrix
C = tf.random_uniform([3,2])
logging.debug(sess.run(C))  # Note that we are reinitializing, hence the new random variables

# Create matrix from np array
D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))
logging.debug(sess.run(D))

# Matrix addition/subtraction
logging.debug(sess.run(A+B))
logging.debug(sess.run(B-B))

# Matrix Multiplication
logging.debug(sess.run(tf.matmul(B, identity_matrix)))

# Matrix Transpose
logging.debug(sess.run(tf.transpose(C))) # Again, new random variables

# Matrix Determinant
logging.debug(sess.run(tf.matrix_determinant(D)))

# Matrix Inverse
logging.debug(sess.run(tf.matrix_inverse(D)))

# Cholesky Decomposition
logging.debug(sess.run(tf.cholesky(identity_matrix)))

# Eigenvalues and Eigenvectors
logging.debug(sess.run(tf.self_adjoint_eig(D)))
