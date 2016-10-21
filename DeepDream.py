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

def fltr_loss(obj, x):
    score = tf.reduce_mean(obj)
    grad = tf.gradients(score, x)[0]
    return loss, grad

def fltr_loss(obj1, obj2, x):
    score = tf.reduce_mean(tf.mul(obj1, obj2))
    grad = tf.gradients(score, x)[0]
    return loss, grad

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


@click.command()
@click.option("--img", "-I", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--fltr", "-f", type=(unicode, int), default=None)
@click.option("--cov", "-c", type=(unicode, int, int), default=None)

def deepdream(img):

    VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'


    sess = tf.InteractiveSession()

    if img:
        image = load_image(img)
        imgname = os.path.basename(img).split(".")[0]
    else:
        image = np.random.uniform(size=(224,224,3)) + 100.0 # gray image with random Noise
        imgname = "random"

    IMAGE_WIDTH = image.shape[2]
    IMAGE_HEIGHT = image.shape[1]
    COLOR_CHANNELS = image.shape[3]
    model = load_vgg_model(VGG_MODEL, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, color_channels=COLOR_CHANNELS)

    if fltr:
        layer = fltr[0]
        score, grad = fltr_loss(model[layer][:,:,:,fltr[1]], model['input'])
        filename = "output/" + imgname + "-" + layer + "-fltr" + str(fltr[1]) + ".png"
    elif cov:
        layer = cov[0]
        score, grad = cov_loss(model[layer][:,:,:,fltr[1]], model[cov[0]][:,:,:,fltr[2]], model['input'])
        filename = "output/" + imgname + "-" + layer + "-cov" + str(fltr[1]) + "-" + str(fltr[2]) + ".png"
    else:
        print("No filter or covariance specified to amplify!")
        return

    sess.run(tf.initialize_all_variables())
    sess.run(model['input'].assign(image))

    for i in range(20):
        grad_val = sess.run(grad)
        grad_val /= grad_val.std() + 1e-8
        image += grad_val * 1.0
        sess.run(model['input'].assign(image))

    if img:
        save_image(filename, image)
    else:
        save_image(filename, visstd(image))

if __name__ == '__main__':
    deepdream()
