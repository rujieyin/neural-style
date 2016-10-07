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


# Noise ratio. Percentage of weight of the noise for intermixing with the
# content image.
NOISE_RATIO = 0.2#0.6

def content_loss_func(sess, model):
    """
    Content loss function as defined in the paper.
    """
    def _content_loss(p, x):
        # N is the number of filters (at layer l).
        N = p.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = p.shape[1] * p.shape[2]
        # Interestingly, the paper uses this form instead:
        #
        #   0.5 * tf.reduce_sum(tf.pow(x - p, 2))
        #
        # But this form is very slow in "painting" and thus could be missing
        # out some constants (from what I see in other source code), so I'll
        # replicate the same normalization constant as used in style loss.
        return (1.0 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))
    return _content_loss(sess.run(model['conv4_2']), model['conv4_2'])

# STYLE_LAYERS = [
#     ('conv1_1', 0.5),
#     ('conv2_1', 1.0),
#     ('conv3_1', 1.5),
#     ('conv4_1', 3.0),
#     ('conv5_1', 4.0),
# ]
STYLE_LAYERS = [
    ('conv1_1', .2),
    ('conv2_1', .2),
    ('conv3_1', .2),
    ('conv4_1', .2),
    ('conv5_1', .2),
]

def style_loss_func(sess, model):
    """
    Style loss function as defined in the paper.
    """
    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x):
        """
        The style loss calculation.
        """
        # N is the number of filters (at layer l).
        N = a.shape[3]
        # M is the height times the width of the feature map (at layer l).
        M = a.shape[1] * a.shape[2]
        # A is the style representation of the original image (at layer l).
        A = _gram_matrix(a, N, M)
        # G is the style representation of the generated image (at layer l).
        G = _gram_matrix(x, N, M)
        result = (1.0 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    E = [_style_loss(sess.run(model[layer_name]), model[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss

def style_loss_func_new(sess, model, filters):

    # def _get_filter_weights(fltr, ndim):
    #     findex = fltr.keys()
    #     findex.sort()
    #     # N = tf.size(findex)
    #     N = sess.run(tf.size(findex))
    #     idx = tf.constant(findex, dtype=tf.int64, shape=[N, 1])
    #     value = tf.constant([fltr[fi] for fi in findex], dtype=tf.float32, shape=[N])
    #     shape = tf.constant(ndim, dtype=tf.int64, shape=[1])
    #     return tf.sparse_tensor_to_dense(tf.SparseTensor(idx, value, shape))

    def _gram_matrix(F, N, M):
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x, fs):
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]
        fw = fs #_get_filter_weights(fs, N)
        A = _gram_matrix(tf.mul(a, fw), N, M)
        G = _gram_matrix(tf.mul(x, fw), N, M)
        result = (1.0 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result

    # total_weights = tf.to_float(tf.add_n([ tf.add_n(dic.values()) for dic in filters.values()]))
    E = [_style_loss(sess.run(model[layer_name]), model[layer_name], filters[layer_name]) for layer_name in filters.keys()]
    return tf.add_n(E)# tf.div(tf.add_n(E), total_weights)


def style_loss_func_bk(sess, model, filters):

    def _get_filter_coeffs(x, findex):
        rank_x = 4
        begin_idx = [0] * (rank_x - 1)
        slice_size = [-1] * (rank_x - 1) + [1]
        return tf.concat(rank_x-1, [tf.slice(x, begin_idx + [i], slice_size) for i in findex])

    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)

    def _style_loss(a, x, fs):
        findex = fs.keys()
        fw = tf.constant(fs.values(), dtype='float32')#fw = [fs[i] for i in findex]
        a_sub = _get_filter_coeffs(a, findex)
        x_sub = _get_filter_coeffs(x, findex)
        # add weights
        a_sub = tf.div(tf.mul(fw, a_sub), tf.reduce_sum(fw))
        x_sub = tf.div(tf.mul(fw, x_sub), tf.reduce_sum(fw))
        sub_shape = tf.shape(a_sub)
        N = sub_shape[3]
        M = sub_shape[1] * sub_shape[2]
        A = _gram_matrix(a_sub, N, M)
        G = _gram_matrix(x_sub, N, M)
        normalization = 4 * tf.square(tf.to_float(N*M)) # otherwise (N*M)**2 may be out of bound
        result =  tf.div( tf.reduce_sum(tf.pow(G - A, 2)) , normalization)
        return result

    total_weights = tf.to_float(tf.add_n([ tf.add_n(dic.values()) for dic in filters.values()]))
    E = [_style_loss(sess.run(model[layer_name]), model[layer_name], filters[layer_name]) for layer_name in filters.keys()]
    return tf.div(tf.add_n(E), total_weights)

def tv_loss_func(image):
    image = tf.squeeze(image) # remove first dimension
    image_size = tf.to_int32(image.get_shape())
    slice_size = image_size + [-1,-1,0]

    def _subimage(begin_idx):
        return tf.slice(image, begin_idx, slice_size)

    diff = tf.squared_difference(_subimage([1,0,0]), _subimage([0,0,0]) ) \
            + tf.squared_difference(_subimage([0,1,0]), _subimage([0,0,0]) )
    return tf.reduce_sum(tf.sqrt(diff + 1e-10))

def generate_noise_image(content_image, image_height, image_width, color_channels, noise_ratio = NOISE_RATIO):
    """
    Returns a noise image intermixed with the content image at a certain ratio.
    """
    noise_image = np.random.uniform(
            -20, 20,
            (1, image_height, image_width, color_channels)).astype('float32')
    # White noise image from the content representation. Take a weighted average
    # of the values
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)
    return input_image


@click.command()
@click.option("--restore", "-r", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--stylew", "-w", type = float, default=1.0)


def train(restore, stylew):
    # Output folder for the images.
    OUTPUT_DIR = 'output/'
    # Style image to use.
    STYLE_IMAGE = 'images/inputs/Nr2_original_p1-ds.jpg' #'images/inputs/Nr2_orig.jpg'
    # Content image to use.
    CONTENT_IMAGE = 'images/inputs/hummingbird-photo_p1-rot.jpg' #'images/inputs/hummingbird-small.jpg'
    # Initial imae to use.
    INITIAL_IMAGE = restore #'output/12000.png'
    # Constant to put more emphasis on content loss.
    BETA = 5
    # Constant to put more emphasis on style loss.
    ALPHA = stylew#1#tf.constant(1.0)
    # Path to the deep learning model. This is more than 500MB so will not be
    # included in the repository, but available to download at the model Zoo:
    # Link: https://github.com/BVLC/caffe/wiki/Model-Zoo
    #
    # Pick the VGG 19-layer model by from the paper "Very Deep Convolutional
    # Networks for Large-Scale Image Recognition".
    VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'

    sess = tf.InteractiveSession()

    content_image = load_image(CONTENT_IMAGE)
    # Image dimensions constants.
    IMAGE_WIDTH = content_image.shape[2]
    IMAGE_HEIGHT = content_image.shape[1]
    COLOR_CHANNELS = content_image.shape[3]
    # resize style image to the same as content image
    style_image = load_image(STYLE_IMAGE, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
    model = load_vgg_model(VGG_MODEL, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, color_channels=COLOR_CHANNELS)
    try:
        input_image = load_image(INITIAL_IMAGE, image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT)
        print('use initial image: ', INITIAL_IMAGE)
    except:
        input_image = generate_noise_image(content_image, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, color_channels=COLOR_CHANNELS)
        # input_image = generate_noise_image(style_image, image_height=IMAGE_HEIGHT, image_width=IMAGE_WIDTH, color_channels=COLOR_CHANNELS)

    # add image summary
    tf.image_summary("style image", tf.to_float(style_image), collections=("input",) )
    tf.image_summary("content image", tf.to_float(content_image), collections=("input",) )
    tf.image_summary("mixed image", tf.to_float(model['input'] + MEAN_VALUES), collections=("output",) )
    input_merged = tf.merge_all_summaries("input")
    output_merged = tf.merge_all_summaries("output")

    sess.run(tf.initialize_all_variables())
    # Construct content_loss using content_image.
    sess.run(model['input'].assign(content_image))
    content_loss = content_loss_func(sess, model)
    # Construct style_loss using style_image.
    sess.run(model['input'].assign(style_image))
    style_loss = style_loss_func_new(sess, model,load_dictionary("test.pkl"))
    # style_loss = style_loss_func(sess, model)
    # total variation loss on reconstruction
    tv_loss = tv_loss_func(model['input'])
    # Instantiate equation 7 of the paper.
    total_loss = BETA * content_loss + ALPHA * style_loss + 1e-3 * tv_loss
    optimizer = tf.train.AdamOptimizer(0.2) #2.0
    train_step = optimizer.minimize(total_loss)

    # add summary on loss
    tf.scalar_summary("loss/tv loss", tv_loss)
    tf.scalar_summary("loss/content loss", content_loss)
    tf.scalar_summary("loss/style loss", style_loss)
    tf.scalar_summary("loss/total loss", total_loss)
    merged = tf.merge_all_summaries("summaries")
    writer = tf.train.SummaryWriter("output/logs/{}".format(time.strftime('%Y-%m-%d_%H%M%S')))#, sess.graph)

    ITERATIONS = 15000  # The art.py uses 5000 iterations, and yields far more appealing results. If you can wait, use 5000.
    sess.run(tf.initialize_all_variables()) # this initialize new variables from optimizer
    sess.run(model['input'].assign(input_image))
    for it in range(ITERATIONS):
        sess.run(train_step)
        if it%100 == 0:
            # Print every 100 iteration.
            mixed_image = sess.run(model['input'])
            tv_loss_val, content_loss_val, style_loss_val, total_loss_val = sess.run([tv_loss, content_loss, style_loss, total_loss])
            print('Iteration %d' % (it))
            # print('sum : %.2f' % sess.run(tf.reduce_sum(mixed_image)))
            # print('tv_loss: %.2f' % tv_loss_val * 1e-3)
            # print('content_loss: %.2f' % content_loss_val * BETA)
            # print('style_loss: %.2f' % style_loss_val * ALPHA)
            print('total_loss: %.4e' % total_loss_val)

            if not os.path.exists(OUTPUT_DIR):
                os.mkdir(OUTPUT_DIR)

            if it == 0:
                writer.add_summary(sess.run(input_merged), it)
            writer.add_summary(sess.run(merged), it)

            if it%1000 == 0:
                writer.add_summary(sess.run(output_merged), it)
                filename = 'output/%d.png' % (it)
                save_image(filename, mixed_image)

    save_image('output/art.jpg', mixed_image)

if __name__ == '__main__':
    train()
