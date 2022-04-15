from functools import reduce
from typing import Optional

import tensorflow as tf

tfs = tf.sparse
tfkl = tf.keras.layers

class SeparableConvolution(tfkl.Layer):
    """A layer that implements the 4D spatio-angular separable convolution.
    Please see https://ieeexplore.ieee.org/abstract/document/8561240"""

    def __init__(
        self,
        i_channels,
        h_channels,
        o_channels,
        kernel_size,
        activation=None,
        hidden_activation=None,
        initializer=None,
        **kwargs
    ):
        super(SeparableConvolution, self).__init__(**kwargs)
        self.i_channels = i_channels
        self.h_channels = h_channels
        self.o_channels = o_channels
        self.kernel_size = kernel_size
        self.activation = activation if activation is not None else tf.identity
        self.hidden_activation = hidden_activation if hidden_activation is not None else tf.identity
        initializer = initializer if initializer is not None else tf.keras.initializers.GlorotUniform()
        hw_kernel_shape = [kernel_size] * 2 + [i_channels, h_channels]
        self.hw_kernel = tf.Variable(initializer(hw_kernel_shape))
        self.hw_bias = tf.Variable(tf.zeros([h_channels]))
        uv_kernel_shape = [kernel_size] * 2 + [h_channels, o_channels]
        self.uv_kernel = tf.Variable(initializer(uv_kernel_shape))
        self.uv_bias = tf.Variable(tf.zeros([o_channels]))

    @tf.function
    def call(self, x):
        shape = tf.shape(x)
        b = shape[0]; u = shape[1]; v = shape[2]; h = shape[3]; w = shape[4]
        x = tf.reshape(x, [b * u * v, h, w, self.i_channels])
        x = tf.nn.conv2d(x, self.hw_kernel, [1] * 4, 'SAME') + self.hw_bias
        x = tf.reshape(x, [b, u, v, h, w, self.h_channels])
        x = tf.transpose(x, [0, 3, 4, 1, 2, 5])
        x = self.hidden_activation(x)
        x = tf.reshape(x, [b * h * w, u, v, self.h_channels])
        x = tf.nn.conv2d(x, self.uv_kernel, [1] * 4, 'SAME') + self.uv_bias
        x = tf.reshape(x, [b, h, w, u, v, self.o_channels])
        x = tf.transpose(x, [0, 3, 4, 1, 2, 5])
        return self.activation(x)

class SeparableNet(tfkl.Layer):
    """A layer that implements a stack of 4D spatio-angular separable convolutions."""

    def __init__(
        self,
        number_of_convolutions: Optional[int] = 3,
        number_of_filters: Optional[int] = 64,
        kernel_size: Optional[int] = 3,
        spectral_resolution: Optional[int] = 3,
        **kwargs
    ):
        super(SeparableNet, self).__init__(**kwargs)
        self.number_of_convolutions = number_of_convolutions
        self.number_of_filters = number_of_filters
        self.kernel_size = kernel_size
        self.spectral_resolution = spectral_resolution
        i_channels = [spectral_resolution] + [number_of_filters] * (number_of_convolutions - 1)
        o_channels = [number_of_filters] * (number_of_convolutions - 1) + [spectral_resolution]
        activations = [tf.nn.elu] * (number_of_convolutions - 1) + [tf.identity]
        self.convolutions = [
            SeparableConvolution(i_chan, o_chan, o_chan, kernel_size, activation)
            for i_chan, o_chan, activation in zip(i_channels, o_channels, activations)]

    def call(self, inputs):
        x = tf.transpose(inputs, [0, 3, 4, 1, 2, 5])
        x = reduce(lambda x, c: c(x), self.convolutions, x)
        return tf.transpose(x, [0, 3, 4, 1, 2, 5])

class SeparableNets:
    def __new__(cls, count, **kwargs):
        return [SeparableNet(**kwargs) for _ in range(count)]

class ResProx(tfkl.Layer):
    """A layer that implements a flattened 2D convolutions.
    I.e. a light field is first reshaped so that it consists of a 2D image with Ang * Spe channels,
    where Ang is the number of sup-aperture views and Spe is the spectral resolution (=number of color channels)."""

    def __init__(
        self,
        number_of_convolutions: Optional[int] = 3,
        number_of_filters: Optional[int] = 128,
        kernel_size: Optional[int] = 3,
        angular_resolution: Optional[int] = 5,
        spectral_resolution: Optional[int] = 3,
        **kwargs
    ):
        super(ResProx, self).__init__(**kwargs)
        self.number_of_convolutions = number_of_convolutions
        self.number_of_filters = number_of_filters
        self.kernel_size = kernel_size
        self.angular_resolution = angular_resolution
        self.spectral_resolution = spectral_resolution
        self.signal_resolution = self.angular_resolution**2 * self.spectral_resolution

    def build(self, input_shape):
        filterses = [self.number_of_filters if i + 1 < self.number_of_convolutions else self.signal_resolution
                     for i in range(self.number_of_convolutions)]
        activations = ['elu'] * (self.number_of_convolutions - 1) + [None]
        self.convolutions = [tfkl.Conv2D(filters, self.kernel_size, padding='same', activation=activation)
                             for filters, activation in zip(filterses, activations)]

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        cnn_input_shape = tf.concat([input_shape[:-3], [self.signal_resolution]], axis=0)
        x = tf.reshape(inputs, cnn_input_shape)
        x = reduce(lambda x, c: c(x), self.convolutions, x)
        return tf.reshape(x, input_shape)

class ResProxes:
    def __new__(cls, count, **kwargs):
        return [ResProx(**kwargs) for _ in range(count)]

class BlockFromResidual(tfkl.Layer):

    def __init__(self, residual, delta_initial_value=None, **kwargs):
        super(BlockFromResidual, self).__init__(**kwargs)
        self.residual = residual
        self.delta = tf.Variable(delta_initial_value, dtype=tf.float32) if delta_initial_value is not None else 1.0

    def call(self, inputs):
        return inputs + self.delta * self.residual(inputs)

class BlocksFromResiduals:
    def __new__(cls, residuals, delta_initial_value=None, **kwargs):
        return [BlockFromResidual(residual, delta_initial_value, **kwargs) for residual in residuals]

class StackOfResidualBlocks(tfkl.Layer):

    def __init__(
        self,
        number_of_convolutions: Optional[int] = 3,
        number_of_filters: Optional[int] = 128,
        kernel_size: Optional[int] = 3,
        angular_resolution: Optional[int] = 5,
        spectral_resolution: Optional[int] = 3,
        number_of_blocks: Optional[int] = 2,
        delta_initial_value: Optional[float] = 0.1,
        **kwargs
    ):
        super(StackOfResidualBlocks, self).__init__(**kwargs)
        self.blocks = [BlockFromResidual(ResProx(
            number_of_convolutions, number_of_filters, kernel_size, angular_resolution, spectral_resolution), delta_initial_value)
            for _ in range(number_of_blocks)]

    def call(self, inputs):
        return reduce(lambda x, block: block(x), self.blocks, inputs)

class StepHQS(tfkl.Layer):

    def __init__(
        self,
        proximals,
        eta_initial_value,
        delta_initial_value,
        **kwargs
    ):
        super(StepHQS, self).__init__(**kwargs)
        self.proximals = proximals
        self.etas = [tf.Variable(eta_initial_value) for _ in proximals]
        self.deltas = [tf.Variable(delta_initial_value) for _ in self.etas]
        self.delta_0 = tf.Variable(delta_initial_value)

    def call(self, inputs):
        sensing_matrix, acquisition_flat, real_shape = inputs
        x = -self.delta_0 * tfs.sparse_dense_matmul(
            sensing_matrix, acquisition_flat, adjoint_a=True)
        for proximal, eta, delta in zip(self.proximals, self.etas, self.deltas):
            v = tf.reshape(proximal(tf.reshape(x, real_shape)), [-1, 1])
            sensing_diff = tfs.sparse_dense_matmul(sensing_matrix, x) - acquisition_flat
            sensing_grad = tfs.sparse_dense_matmul(sensing_matrix, sensing_diff, adjoint_a=True)
            prox_v_grad = eta * (x - v)
            x -= delta * (sensing_grad + prox_v_grad)
        return x
