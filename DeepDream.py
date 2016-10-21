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
    return score, grad

def cov_loss(obj1, obj2, x):
    score = tf.reduce_mean(tf.mul(obj1, obj2))
    grad = tf.gradients(score, x)[0]
    return score, grad

def covs_loss(covfile, model):
    score_total = tf.Variable(0, name="score")
    grad_total = tf.Variable(tf.zeros_like(model['input']), name="grad")
    for layer, cov_loc in covfile.iteritems():
        for loc1, loc2 in cov_loc:
            score, grad = cov_loss(model[layer][:,:,:, loc1], model[layer][:,:,:,loc2], model['input'])
            score_total = score_total + score
            grad_total = grad_total + grad
    return score_total, grad_total

def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


@click.command()
@click.option("--img", "-I", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--fltr", "-f", type=(unicode, int), default=(None, None))
@click.option("--cov", "-c", type=(unicode, int, int), default=(None, None, None))
@click.option("--covfile", "-cf", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--ascend", "-a", type=bool, default=True)

def deepdream(img, fltr, cov, ascend, covfile):

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

    if covfile:
        covs = load_dictionary(covfile)
        print(" Total number of covariances: {}".format(Ncovs))
        filename = "output/" + imgname + "mixed_cov.png"
        score, grad = covs_loss(covfile, model)
        Ncovs = sum(map(lambda x: len(x[0]), covs.items()))
        grad = tf.truediv(grad , Ncovs)

    else:
        if fltr[0]:
            layer = fltr[0]
            score, grad = fltr_loss(model[layer][:,:,:,fltr[1]], model['input'])
            filename = "output/" + imgname + "-" + layer + "-fltr" + str(fltr[1]) + ".png"
        elif cov[0]:
            layer = cov[0]
            score, grad = cov_loss(model[layer][:,:,:,cov[1]], model[cov[0]][:,:,:,cov[2]], model['input'])
            filename = "output/" + imgname + "-" + layer + "-cov" + str(cov[1]) + "-" + str(cov[2]) + ".png"
        else:
            print("No filter or covariance specified to amplify!")
            return

    if not ascend:
        filename = filename.split(".")[0] + "-descend." + filename.split(".")[-1]

    sess.run(tf.initialize_all_variables())
    sess.run(model['input'].assign(image))

    for i in range(20):
        grad_val, score_val = sess.run([grad, score])
        grad_val /= grad_val.std() + 1e-8
        if ascend:
            image += grad_val * 1.0
        else:
            image -= grad_val * 1.0
        sess.run(model['input'].assign(image))
        print("iter {}: {:.4e}".format(i, score_val))

    if img:
        save_image(filename, image)
    else:
        save_image(filename, visstd(image))

if __name__ == '__main__':
    deepdream()
