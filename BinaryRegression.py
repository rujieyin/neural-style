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
            self.shape = { self.weights[key].get_shape() for key in self.weights.keys() }
        else:
            for key, s in self.shape.iteritems():
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

def build_graph(image):
    graph = load_vgg_model(VGG_MODEL, input_image = image)
    model_var = tf.all_variables()
    # # weights for each layer of output
    # weights = {}
    # for key in graph.keys():
    #     # remove first dim for batch size
    #     weight_shape = graph[key].get_shape()[1:]
    #     weights[key+'_w'] = tf.Variable(tf.random_normal(weight_shape), name = key+'_w')

    def _inner_prod(t1, t2):
        # use broadcast of tf.mul, keep the first batch dimension in sum
        return tf.reduce_sum(tf.mul(t1,t2), [1, 2, 3])

    weights = Weights(graph = graph)
    weights = weights.weights

    # sum of inner products of output coeffs and weights
    regs = sum([_inner_prod(weights[key], graph[key[:-2]]) for key in weights.keys() ])

    return (regs, model_var, weights)


def binary_reg():
    # Content image to use.
    CONTENT_IMAGE = 'images/inputs/hummingbird-photo_p1-rot.jpg' #'images/inputs/hummingbird-small.jpg'
    content_image = load_image(CONTENT_IMAGE, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
    # Style image to use.
    STYLE_IMAGE = 'images/inputs/Nr2_original_p1-ds.jpg' #'images/inputs/Nr2_orig.jpg'
    style_image = load_image(STYLE_IMAGE, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
    label = [0 , 1]

    regs, model_var, weights = build_graph(tf.concat(0, [content_image, style_image]))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_variables(model_var))
    sess.run(tf.initialize_all_variables())

    print( "total number of weight variables: %.4e" % sess.run(sum([tf.size(v) for key, v in weights.iteritems()])) )

    #print("%.4e" % (sess.run(tf.reduce_sum(graph['input']))) )
    print(["%.4e" % reg for reg in sess.run(regs)])

if __name__ == '__main__':
    binary_reg()
