import numpy as np
import tensorflow as tf  # Import TensorFlow after Scipy or Scipy will break

class Weights:

    def __init__(self, *args, **kwargs):
        self.weights = {}
        shape = kwargs.get('shape', None)
        val = kwargs.get('val', None) # initialize with given values
        if shape:
            for key, s in shape.iteritems():
                self.weights[key] = tf.Variable(tf.zeros(s), name = key)
            self.shape = shape
        elif val:
            for key, value in val.iteritems():
                self.weights[key] = tf.Variable(value, name = key)
            self.shape = { key: weight.get_shape() for key, weight in self.weights.iteritems() }

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

# weights on each inidividual filter
class Weights_individual(Weights):

    def __init__(self, *args, **kwargs):
        self.weights = {}
        graph = kwargs.get('graph', None)
        # shape = kwargs.get('shape', None)
        # npzfile = kwargs.get('npzfile', None)
        if graph: # initialize by corresponding graph structure
            for key, _ in graph.iteritems():
                # remove first dim for batch size
                weight_shape = graph[key].get_shape()[1:]
                self.weights[key+'_w'] = tf.Variable(tf.zeros(weight_shape), name = key+'_w')
            self.shape = { key: weight.get_shape() for key, weight in self.weights.iteritems() }
        else:
            super(Weights_individual, self).__init__(args, kwargs)

    def get_X(self, graph, sess):
        X = {}
        for key, value in graph.iteritems():
            X[key] = graph[key].eval()
        return X

    def compute_reg(self, X):

        def _inner_prod(t1, t2):
            # use broadcast of tf.mul, reduce sum in consistant dim
            return tf.reduce_sum(tf.mul(t1,t2), [1, 2, 3])

        # sum of inner products of output coeffs and weights
        return tf.add_n([_inner_prod(weight, X[key[:-2]]) for key, weight in self.weights.iteritems() ])

# weights on covariance of filters in each layer
class Weights_covariance(Weights):

    def __init__(self, *args, **kwargs):
        self.weights = {}
        graph = kwargs.get('graph', None)
        if graph:
            for key, _ in graph.iteritems():
                Nfilter = graph[key].get_shape()[3]
                weight_shape = [Nfilter, Nfilter]#Nfilter.concatenate(Nfilter)
                self.weights[key+'_w'] = tf.Variable(tf.zeros(weight_shape), name = key+'_w')
            self.shape = { key: weight.get_shape() for key, weight in self.weights.iteritems() }
        else:
            super(Weights_individual, self).__init__(args, kwargs)

    def _gram_matrix(self, F):
        F = np.split(F, F.shape[0])
        gram = map(lambda x: np.tensordot(x, x, axes = ( [0,1,2], [0,1,2] )), F)
        return np.stack(gram)
        # Ft = tf.reshape(F, (M, N))
        # return tf.matmul(tf.transpose(Ft), Ft)

    def get_X(self, graph, sess):
        X = {}
        for key, value in graph.iteritems():
            coeffs = graph[key].eval()
            X[key] = self._gram_matrix(coeffs)
        return X

    def compute_reg(self, X):

        def _inner_prod(t1, t2):
            return tf.reduce_sum(tf.mul(t1,t2), [1,2])

        return tf.add_n([_inner_prod(weight, X[key[:-2]]) for key, weight in self.weights.iteritems() ])
