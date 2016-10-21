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

def get_covs(covfile, model):
    scores = []
    grads = []
    for layer, cov_loc in covfile.iteritems():
        for loc1, loc2 in zip(cov_loc[0], cov_loc[1]):
            score, grad = cov_loss(model[layer][:,:,:, loc1], model[layer][:,:,:,loc2], model['input'])
            scores.append(score)
            grads.append(grad)
    return tf.pack(scores), grads


def visstd(a, s=0.1):
    '''Normalize the image range for visualization'''
    return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5


@click.command()
@click.option("--img", "-I", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--imgpair", "-Is", nargs=2, type=click.Path(exists=True, dir_okay=False), default=None )
@click.option("--fltr", "-f", type=(unicode, int), default=(None, None))
@click.option("--cov", "-c", type=(unicode, int, int), default=(None, None, None))
@click.option("--covfile", "-cf", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--ascend", 'direction', flag_value='a', default=True)
@click.option('--descend', 'direction', flag_value='d')

def deepdream(img, imgpair, fltr, cov, direction, covfile):

    VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

    sess = tf.InteractiveSession()

    if img:
        image = load_image(img)
        imgname = os.path.basename(img).split(".")[0]
    elif imgpair:
        image = load_image(imgpair[0])
        imgname = os.path.basename(imgpair[0]).split(".")[0]
    else:
        image = np.random.uniform(size=(224,224,3)) + 100.0 # gray image with random Noise
        imgname = "random"

    IMAGE_WIDTH = image.shape[2]
    IMAGE_HEIGHT = image.shape[1]
    COLOR_CHANNELS = image.shape[3]
    model = load_vgg_model(VGG_MODEL, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, color_channels=COLOR_CHANNELS)

    if imgpair:
        image2 = load_image(imgpair[1], image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
        imgname2 = os.path.basename(imgpair[1]).split(".")[0]


    if covfile:
        covs = load_dictionary(covfile)
        filename = "output/" + imgname + "mixed_cov.png"
        scores, grads = get_covs(covs, model)


        Ncovs = sum(map(lambda x: len(x[1][0]), covs.items()))
        print(" Total number of covariances: {}".format(Ncovs))
        # grad = tf.truediv(grad , float(Ncovs) )

        if imgpair:
            sess.run(tf.initialize_all_variables())
            sess.run(model['input'].assign(image))
            scores1 = sess.run(scores)
            sess.run(model['input'].assign(image2))
            scores2 = sess.run(scores)
            signs = np.sign(scores1 - scores2).astype(float)
            score = tf.reduce_sum(tf.mul(scores, signs))
            grad = tf.add_n([ g*s for g,s in zip(grads, signs)]) / Ncovs
        else:
            print("Need image pair input for multiple covariance.")

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

    sess.run(tf.initialize_all_variables())
    sess.run(model['input'].assign(image))


    if direction == 'd':
        filename = filename.split(".")[0] + "-descend." + filename.split(".")[-1]


    for i in range(10):
        grad_val, score_val = sess.run([grad, score])
        grad_val /= grad_val.std() + 1e-8
        if direction == 'a':
            image += grad_val * 1.0
        else:
            image -= grad_val * 1.0
        sess.run(model['input'].assign(image))
        print("iter {}: {:.4e}".format(i, score_val))

    if img or imgpair:
        save_image(filename, image)
        print("save image without scaling.")
    else:
        save_image(filename, visstd(image))
        print("save image with scaling.")

if __name__ == '__main__':
    deepdream()
