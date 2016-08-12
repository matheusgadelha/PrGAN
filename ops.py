import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np


def linear(x, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        w = tf.get_variable(
            name='w',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())
        h = tf.matmul(x, w) + b
        if activation is not None:
            h = activation(h)
        return h


def l2norm(a, b): return tf.sqrt(tf.reduce_sum(tf.pow(a-b, 2), 1))


def show_graph_operations():
    operations = [op.name for op in tf.get_default_graph().get_operations()]
    for o in operations: print o


def load_flatten_imgbatch(img_paths):
    images = []
    for path in img_paths:
        images.append(mpimg.imread(path).flatten())
    return np.array(images)

