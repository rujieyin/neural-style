import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf  # Import TensorFlow after Scipy or Scipy will break
from PIL import Image

import click
import time

from util import *

IMAGE_HEIGHT = 227
IMAGE_WIDTH = 227
COLOR_CHANNELS = 3

VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

class Weights:

    def __init__(self, *args, **kwargs):
        self.weights = {}
        graph = kwargs.get('graph', None)
        shape = kwargs.get('shape', None)
        if graph:
            for key, _ in graph.iteritems():
                # remove first dim for batch size
                weight_shape = graph[key].get_shape()[1:]
                self.weights[key+'_w'] = tf.Variable(tf.zeros(weight_shape), name = key+'_w')
            self.shape = { key: weight.get_shape() for key, weight in self.weights.iteritems() }
        else:
            for key, s in shape.iteritems():
                self.weights[key] = tf.Variable(tf.zeros(s), name = key)
            self.shape = shape

    def add(self, W): #  NOT inplace sum
        Sum = Weights(shape = self.shape)
        for key, _ in Sum.weights.iteritems():
            Sum.weights[key] = tf.add(self.weights[key], W.weights[key])
        return Sum

    def sub(self, W):
        Sub = Weights(shape = self.shape)
        for key, _ in Sub.weights.iteritems():
            Sub.weights[key] = tf.sub(self.weights[key], W.weights[key])
        return Sub

    def sqr_norm(self):
        return sum([tf.reduce_sum(tf.square(w)) for _, w in self.weights.iteritems()])

    def l1_norm(self):
        return sum([tf.reduce_sum(tf.abs(w)) for _, w in self.weights.iteritems()])

    def soft_thresh(self, s):
        W = Weights(shape = self.shape)
        for key, w in self.weights.iteritems():
            W.weights[key] = tf.maximum(tf.abs(w) - s, tf.zeros(w.get_shape()))
            W.weights[key] = tf.mul(tf.sign(w), W.weights[key])
        return W

    def compute_reg(self, graph):

        def _inner_prod(t1, t2):
            # use broadcast of tf.mul, reduce sum in consistant dim
            return tf.reduce_sum(tf.mul(t1,t2), [1, 2, 3])

        # sum of inner products of output coeffs and weights
        return sum([_inner_prod(weight, graph[key[:-2]]) for key, weight in self.weights.iteritems() ])


def build_graph(image):
    graph = load_vgg_model(VGG_MODEL, input_image = image)
    model_var = tf.all_variables()

    def _normalize_graph(graph):
        for key, val in graph.iteritems():
            graph[key] = tf.scalar_mul(.1/tf.reduce_mean(val), val)

    _normalize_graph(graph)

    return (graph, model_var)


def reg_loss(regs, labels):
    return tf.reduce_mean(tf.squared_difference(regs , labels))

def residual_loss(beta, z, u):
    return beta.sub(z).add(u).sqr_norm()

def start_session(model_var):
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_variables(model_var))
    sess.run(tf.initialize_all_variables())
    return sess

def update_z(z, beta, u, s):
    op = []
    for key, val in beta.add(u).soft_thresh(s).weights.iteritems():
        op.append(z.weights[key].assign(val))
    return tf.group(*op)

def check(z, beta, u, s):
    tmp = beta.add(u).soft_thresh(s)
    min_nz = []
    for _, val in tmp.weights.iteritems():
        min_nz.append( tf.reduce_min(tf.abs(val) + tf.scalar_mul(1,tf.to_float(tf.equal(val, 0))) ) )
    return tf.reduce_min(tf.pack(min_nz))

def update_u(u, beta, z):
    op = []
    for key, val in u.add(beta).sub(z).weights.iteritems():
        op.append(u.weights[key].assign(val))
    return tf.group(*op)

def tf_count_nonzero(t):
    elements_equal_to_value = tf.equal(t, 0)
    as_ints = tf.cast(tf.equal(t, 0), tf.int32)
    count = tf.size(t) - tf.reduce_sum(as_ints)
    return count

def binary_reg():
    # Content image to use.
    CONTENT_IMAGE = 'images/inputs/hummingbird-photo_p1-rot.jpg' #'images/inputs/hummingbird-small.jpg'
    content_image = load_image(CONTENT_IMAGE, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
    # Style image to use.
    STYLE_IMAGE = 'images/inputs/Nr2_original_p1-ds.jpg' #'images/inputs/Nr2_orig.jpg'
    style_image = load_image(STYLE_IMAGE, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
    labels = tf.constant([0 , 1], dtype = 'float32')

    graph, model_var = build_graph(tf.concat(0, [content_image, style_image]))

    beta = Weights(graph = graph)
    regs = beta.compute_reg(graph)

    z = Weights(graph = graph)
    u = Weights(graph = graph)

    loss = reg_loss(regs, labels) + residual_loss(beta, z, u)

    sess = start_session(model_var)

    print( "total number of weight variables: %.4e" % sess.run(sum([tf.size(v) for key, v in beta.weights.iteritems()])) )

    # add tensorboard summaries
    for key, val in beta.weights.iteritems():
        tf.histogram_summary("beta-"+key, val, collections = ("beta", ) )
    merged_beta = tf.merge_all_summaries("beta")
    for key, val in z.weights.iteritems():
        tf.histogram_summary("z-"+key, val, collections = ("z", ) )
        tf.scalar_summary("nnz/z-"+key, tf_count_nonzero(val), collections = ("z", ) )
        tf.scalar_summary("l1/z-"+key, tf.reduce_sum(tf.abs(val)), collections = ("z", ) )
    merged_z = tf.merge_all_summaries("z")
    writer = tf.train.SummaryWriter("output/logs/{}".format(time.strftime('%Y-%m-%d_%H%M%S')), sess.graph)

    # create a placeholder so that lr can be updated during iteration
    learning_rate = tf.placeholder(tf.float32, shape=[])
    opt = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)#0.000001)
    opt_op = opt.minimize(loss, var_list=beta.weights.values())

    itr = 0
    lr = 1e-6 # fixed
    s = 1e-6
    loss_bd = 1.0e-5
    z_norm = 1
    max_nitr = 200

    while z_norm > 1e-30 and itr < max_nitr:
        writer.add_summary(sess.run(merged_z), itr)
        loss_val = sess.run(loss)
        loss_bk = 1e10
        print("iteration %d, loss: %.4e" % (itr, loss_val))
        while loss_bk > loss_val + 1e-6:
            print("iteration %d:" % itr)
            sess.run(opt_op, feed_dict={learning_rate: lr})#opt_op.run()
            loss_bk = loss_val
            loss_val = sess.run(loss)
            print("loss: %.4e" % loss_val)
            writer.add_summary(sess.run(merged_beta), itr)
            itr = itr + 1
        print("before updata: %.4e" % sess.run(z.l1_norm()))
        print("before threshold: %.4e" % sess.run(beta.add(u).l1_norm()))
        sess.run(update_z(z, beta, u, s))
        print("after threshold: %.4e" % sess.run(z.l1_norm()))
        sess.run(update_u(u, beta, z))
        z_norm = sess.run(z.sqr_norm())
        writer.add_summary(sess.run(merged_z), itr)
        s = s * 1.2 # faster than equispaced stepsize
        print("update u and z, new threshold: %.2e" % s)


if __name__ == '__main__':
    binary_reg()
