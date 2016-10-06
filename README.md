This repo is an extension of the neural-style repo.

The original documentation: 
# Implementing of a Neural Algorithm of Artistic Style #

This is an implementation of the "[A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)". This uses the selected filtered responses of a pre-trained model (VGG-19) to capture low level to high level features and transfer them to the content image.

# How to run

You will need to install dependencies:

- TensorFlow
- Scipy
- Numpy

You will need to download the [VGG-19 model](http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat).

Then just run art.py.

References:
- [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
- [https://github.com/jcjohnson/neural-style](https://github.com/jcjohnson/neural-style)
- [https://github.com/ckmarkoh/neuralart_tensorflow](https://github.com/ckmarkoh/neuralart_tensorflow)

Add-ons:
# Binary Regression on network coefficients #

different regression models:

- Weights_individual,
  lr : 4e-6

- Weights_covariance,
  lr : 4e-2
