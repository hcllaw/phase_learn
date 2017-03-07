### Main Tensorflow module for phase and fourier neural network ###
import numpy as np
import tensorflow as tf

# Kernel Module (Layer) for implementing Fourier NN or Phase NN for distribution regression (dr) or normal regression and classification etc.
# Use norm = True for Phase Features
# Use rescale = True for scaling of the features
# Use dr = True for distributional Regression
def kernel_module(x, weights, n_hidden, bias = None, dr = False, rescale = True, norm = False):
    x_shape = tf.shape(x)
    x_rank = tf.rank(x)
    if dr:
        tf.assert_equal(x_rank, 3, message = 'Inputs for distribution regression must be of rank 3')
        batch_s = x_shape[0]
        bag_size = x_shape[1]
        dim = x_shape[2]
        layer_0 = tf.reshape(tf.matmul(tf.reshape(x,[batch_s * bag_size, dim]), weights), [batch_s, bag_size, n_hidden])
    else:
        tf.assert_equal(x_rank, 2, message = 'Inputs must be of rank 2')
        layer_0 = tf.matmul(x, weights)
    if bias:
        layer_0 = tf.add(layer_0, bias)
    layer_sin = tf.sin(layer_0)
    layer_cos = tf.cos(layer_0)
    layer_1 = tf.concat(x_rank - 1, [layer_sin, layer_cos])
    if dr:
        layer_1 = tf.reduce_mean(layer_1, 1)
        if norm:
            mean_layer_sin = tf.reduce_mean(layer_sin, 1) 
            mean_layer_cos = tf.reduce_mean(layer_cos, 1)
            amplitude = tf.sqrt( tf.add( tf.square(mean_layer_cos), tf.square(mean_layer_sin) ) ) # Calculate amplitude
            amplitude_layer = tf.concat(1, [amplitude, amplitude]) # amplitude layer
            layer_1 = tf.div(layer_1, amplitude_layer) # Phase layer
        elif not dr and norm:
            raise ValueError('Normalisation for phase features available only for distribution regression')
    if rescale:
            layer_1 = tf.scalar_mul( tf.sqrt(1.0/tf.to_float(n_hidden) ), layer_1 )
    return layer_1
ß
# Neural Network using the fourier or phase module followed by a batch normalisation layer and output layer.
def kernel_nn(x, weights, n_hidden, epsilon, mean_all, var_all, bias = None, train = True, rescale = True, dr = False, norm = False):
    if bias:
        fourier_layer = kernel_module(x, weights['h1'], n_hidden, bias['b1'], dr = dr, rescale = rescale, norm = norm)
    else:
        fourier_layer = kernel_module(x, weights['h1'], n_hidden, bias, dr = dr, rescale = rescale, norm = norm)
    batch_mean, batch_var =  tf.cond( train , lambda: tf.nn.moments(fourier_layer,[0]) , lambda: (mean_all, var_all))
    scale1 = tf.Variable(tf.ones([n_hidden * 2]))
    beta1 = tf.Variable(tf.zeros([n_hidden * 2]))
    layer_bn = tf.nn.batch_normalization(fourier_layer, batch_mean, batch_var, beta1, scale1, epsilon)
    out_layer = tf.matmul(layer_bn, weights['out']) 
    if bias:
        out_layer = out_layer + bias['out']
    return out_layer, batch_mean, batch_var