import math
from typing import Optional

import tensorflow as tf

tfkr = tf.keras.regularizers

@tf.function
def pairwise_squared_distances(X: tf.Tensor):
    r = tf.reduce_sum(X * X, 1)
    r = tf.reshape(r, [-1, 1])
    return r - 2 * tf.matmul(X, X, transpose_b=True) + tf.transpose(r)

class EntropyRegularizer(tfkr.Regularizer):

    def __init__(
        self,
        weight: Optional[float] = 1.0,
        threshold: Optional[float] = -2.5,
        dim: Optional[int] = 3,
        axis: Optional[int] = 5,
        number_of_samples: Optional[int] = 1024,
        k: Optional[int] = None,
        epsilon: Optional[float] = 1e-12,
        **kwargs
    ):
        super(EntropyRegularizer, self).__init__(**kwargs)
        self.weight = weight
        self.threshold = threshold
        self.dim = dim
        self.axis = axis
        self.number_of_samples = number_of_samples
        if k is None:
            k = int(math.sqrt(number_of_samples))
        self.k = k
        self.epsilon = epsilon

    @property
    def log_of_L2_unit_ball_volume(self):
        return (self.dim / 2) * tf.math.log(math.pi) - tf.math.lgamma(1 + self.dim / 2)

    @property
    def additive_constant(self):
        return tf.math.digamma(float(self.number_of_samples)) - tf.math.digamma(float(self.k)) +\
            self.log_of_L2_unit_ball_volume

    def __call__(self, inputs):
        # Permute axes so that the dimension of interest is positioned last (i.e. shape is [..., dim]).
        x = tf.transpose(inputs, self.get_perm(inputs))
        # Flatten structure except last axis to consider inputs as a list of samples
        # and then take only a few samples because KL is very costly (quadratic).
        samples = tf.reshape(x, [-1, self.dim])[:self.number_of_samples]
        entropy = self.kozachenko_leonenko_estimate(samples)
        return self.weight * tf.maximum(0., self.threshold - entropy)

    def get_perm(self, inputs):
        size = tf.size(tf.shape(inputs))
        return tf.roll(tf.range(size), size - 1 - self.axis, 0)

    def kozachenko_leonenko_estimate(self, samples):
        squared_distances = pairwise_squared_distances(samples)
        negative_k_least_squared_distances, _ = tf.math.top_k(-squared_distances, k=self.k)
        log_k_distances = tf.math.log(-negative_k_least_squared_distances[:, -1] + self.epsilon)
        mean_logs = tf.reduce_mean(log_k_distances) * (self.dim / 2)
        return mean_logs + self.additive_constant

    @classmethod
    def from_config(cls, config):
        return super(EntropyRegularizer, cls).from_config(config)

    def get_config(self):
        return {
            'weight': self.weight,
            'threshold': self.threshold,
            'dim': self.dim,
            'axis': self.axis,
            'number_of_samples': self.number_of_samples,
            'k': self.k,
            'epsilon': self.epsilon}
