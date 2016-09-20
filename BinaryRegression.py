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
    opt = tf.train.AdamOptimizer(learning_rate=0.0000001)
    opt_op = opt.minimize(loss, var_list=beta.weights.values())

    sess = start_session(model_var)

    print( "total number of weight variables: %.4e" % sess.run(sum([tf.size(v) for key, v in beta.weights.iteritems()])) )

    loss_val = sess.run(loss)
    itr = 0
    while loss_val > 1.0e-2 :
        print("iteration %d:" % itr)
        opt_op.run()
        print("loss: %.4e" % loss_val)
        loss_val = sess.run(loss)
        itr = itr + 1
        # print( "reg_loss: %.4e" % sess.run(reg_loss(regs, labels)))
        # print( "residual_loss: %.4e" % sess.run(residual_loss(beta, z, u)))

if __name__ == '__main__':
    binary_reg()
