from functools import reduce
from typing import Optional
import tensorflow as tf

tfl = tf.linalg
tfk = tf.keras
tfkl = tfk.layers
tfkm = tfk.models

class PixelwiseMatrixGenerator(tfkl.Layer):
    """A layer that produces a possibly random modulation field from a collection of modulation layers by element-wise product."""

    def __init__(
        self,
        modulation_layers,
        normalize: Optional[bool] = False,
        **kwargs
    ):
        super(PixelwiseMatrixGenerator, self).__init__(**kwargs)
        self.modulation_layers = modulation_layers
        self.normalize = normalize

    def call(self, sh):
        """Produces a multi-shot degradation operator in the form of a modulation field.
        Arguments:

        sh: shape information [B, H, W, U, V, C, S] where:
            B is the batch size, H, W are the spatial resolution,
            U, V are the angular resolution, C is the spectral resolution,
            S is the number of acquisition shots.
            
        Returns:
            A flattened multi-shot modulation field with shape [B * H * W, C * U * V, S]"""

        modulations = [layer(sh) for layer in self.modulation_layers]
        modulation = reduce(lambda x, y: x * y, modulations)
        P = tf.reduce_prod(tf.shape(modulation)[:3])
        C = tf.reduce_prod(tf.shape(modulation)[3:-1])
        S = tf.reduce_prod(tf.shape(modulation)[-1])
        modulation = modulation / tf.cast(C, tf.float32) if self.normalize else modulation
        return tf.transpose(
            tf.reshape(modulation, tf.stack([P, C, S])),
            [0, 2, 1])

class PixelwiseClosedFormHQS(tfkl.Layer):
    """A layer that applies the unrolled half-quadratic splitting algorithm for a fixed number of steps.
    In this layer, the minimization of the data-term is performed in closed form at each unrolled iteration,
    exploiting the fact that the degradation operator corresponds to a modulation field."""

    def __init__(
        self,
        proximals,
        mu_initial_value,
        mu_lower_bound: Optional[float] = 1e-2,
        mu_upper_bound: Optional[float] = 1e2,
        **kwargs
    ):
        """
        Arguments:

        proximals: a list of (trainable) layers that correspond to the proximal operators to be applied for each step.
        mu_initial_value: initial value for the weight of the penalty term mu in the HQS algorithm.
        mu_lower_bound: lower bound to constrain mu during training.
        mu_upper_bound: upper bound to constrain mu during training."""

        super(PixelwiseClosedFormHQS, self).__init__(**kwargs)
        self.proximals = proximals
        clip_mu = lambda x, lb=mu_lower_bound, ub=mu_upper_bound: tf.clip_by_value(x, lb, ub)  # noqa: E731
        self.mu_0 = tf.Variable(mu_initial_value, dtype=tf.float32, constraint=clip_mu)
        self.mus = [tf.Variable(mu_initial_value, dtype=tf.float32, constraint=clip_mu) for _ in proximals]

    @tf.function
    def call(self, inputs):
        """
        Reconstructs light field from the degradation operator and measurements.
        Arguments:
        inputs: [A, y, real_shape] where:
        A is the flattened modulation field corresponding to the degradation operator.
        y is the flattened measurement.
        real_shape is the original shape of the light field (i.e. non flattened)"""

        A, y, real_shape = inputs
        Aty = tfl.matvec(A, y, transpose_a=True)
        G = tf.matmul(A, A, transpose_b=True)  # Gram matrix of A.
        S = tf.slice(tf.shape(A), [1], [1])[0]  # Number of measurements.
        x = PixelwiseClosedFormHQS.project(A, G, S, self.mu_0, Aty)
        for proximal, mu in zip(self.proximals, self.mus):
            v = tf.reshape(proximal(tf.reshape(x, real_shape)), tf.shape(x))
            x = PixelwiseClosedFormHQS.project(A, G, S, mu, mu * v + Aty)
        return tf.clip_by_value(x, 0, 1)

    @staticmethod
    @tf.function
    def project(A, G, S, mu, vec):
        return (vec - tfl.matvec(A, tfl.matvec(tfl.inv(G + mu * tf.eye(S)), tfl.matvec(A, vec)), transpose_a=True)) / mu

class PixelwiseSingleStepHQS(tfkl.Layer):
    """A layer that applies the unrolled half-quadratic splitting algorithm for a fixed number of steps.
    In this layer, the minimization of the data-term is performed using an approximation consisting of a single
    step of gradient descent."""

    def __init__(
        self,
        proximals,
        mu_initial_value: float,
        step_size_initial_value: float,
        mu_lower_bound: Optional[float] = 1e-2,
        mu_upper_bound: Optional[float] = 1e2,
        step_size_lower_bound: Optional[float] = 1e-3,
        step_size_upper_bound: Optional[float] = 1e2,
        **kwargs
    ):
        """
        Arguments:

        proximals: a list of (trainable) layers that correspond to the proximal operators to be applied for each step.
        mu_initial_value: initial value for the weight of the penalty term mu in the HQS algorithm.
        step_size_initial_value: the initial value for the step size of the gradient descent approximation.
        mu_lower_bound: lower bound to constrain mu during training.
        mu_upper_bound: upper bound to constrain mu during training.
        step_size_lower_bound: lower bound to constraint the gradient descent step size during training.
        step_size_upper_bound: upper bound to constraint the gradient descent step size during training."""

        super(PixelwiseSingleStepHQS, self).__init__(**kwargs)
        self.proximals = proximals
        clip_mu = lambda x, lb=mu_lower_bound, ub=mu_upper_bound: tf.clip_by_value(x, lb, ub)  # noqa: E731
        clip_step_size = lambda x, lb=step_size_lower_bound, ub=step_size_upper_bound: tf.clip_by_value(x, lb, ub)  # noqa: E731
        self.step_size_0 = tf.Variable(mu_initial_value, dtype=tf.float32, constraint=clip_step_size)
        self.step_sizes = [tf.Variable(step_size_initial_value, dtype=tf.float32, constraint=clip_step_size) for _ in proximals]
        self.mus = [tf.Variable(mu_initial_value, dtype=tf.float32, constraint=clip_mu) for _ in proximals]

    @tf.function
    def call(self, inputs):
        """
        Reconstructs light field from the degradation operator and measurements.
        Arguments:

        inputs: [A, y, real_shape] where:
        A is the flattened modulation field corresponding to the degradation operator.
        y is the flattened measurement.
        real_shape is the original shape of the light field (i.e. non flattened)"""

        A, y, real_shape = inputs
        x = -self.step_size_0 * tfl.matvec(A, y, transpose_a=True)
        for proximal, mu, step_size in zip(self.proximals, self.mus, self.step_sizes):
            v = tf.reshape(proximal(tf.reshape(x, real_shape)), tf.shape(x))
            x = PixelwiseSingleStepHQS.project(A, y, step_size, mu, x, v)
        return tf.clip_by_value(x, 0, 1)

    @staticmethod
    @tf.function
    def project(A, y, step_size, mu, x, v):
        grad = tfl.matvec(A, tfl.matvec(A, x) - y, transpose_a=True) + mu * (x - v)
        return x - step_size * grad

class PixelwiseAcquisitionReconstruction(tfkl.Layer):
    """A layer that modelize the whole acquisition + corruption + reconstruction pipeline."""

    def __init__(
        self,
        generator,
        reconstructor,
        corruptor=None,
        normalize_corruption=True,
        **kwargs
    ):
        """Arguments:
        generator: layer that generates the flattened modulation field,
        reconstructor: layer that reconstructs a light field from the modulation field and the measurements
        corruptor: layer that models the corruption occuring at the sensor.
        """
        super(PixelwiseAcquisitionReconstruction, self).__init__(**kwargs)
        self.generator = generator
        self.reconstructor = reconstructor
        self.corruptor = corruptor
        self.normalize_corruption = normalize_corruption

    def call(self, ground_truth):
        sh = tf.shape(ground_truth)
        P = tf.reduce_prod(sh[:3])
        C = tf.reduce_prod(sh[3:])
        x = tf.reshape(ground_truth, tf.stack([P, C]))
        A = self.generator(sh)
        y = tfl.matvec(A, x)
        if self.corruptor is not None and self.normalize_corruption:
            factor = tf.cast(C, tf.float32)
            y = factor * self.corruptor(y / factor)
        elif self.corruptor is not None and not self.normalize_corruption:
            y = self.corruptor(y)
        rec = self.reconstructor([A, y, sh])
        return tf.reshape(rec, sh)

class PixelwiseModel(tfkm.Model):
    """A functional model to wrap the whole pipeline."""

    def __new__(
        cls,
        acquisition_reconstruction,
        **kwargs
    ):
        inputs = tfk.Input([None] * 4 + [3])
        outputs = acquisition_reconstruction(inputs)
        return tfkm.Model(inputs, outputs, **kwargs)
