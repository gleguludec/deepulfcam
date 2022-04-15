from functools import reduce
import random
from typing import Optional

import numpy as np
import tensorflow as tf

tfkl = tf.keras.layers

class Corruption(tfkl.Layer):
    """A layer to apply element-wise (i.e. sensor-pixel-wise) physically realistic corruption."""

    def __init__(
        self,
        signal_factor: Optional[float] = 1.0,
        readout_noise_stddev: Optional[float] = 40.0,
        full_well_depth: Optional[int] = 20000,
        number_of_ADC_bits: Optional[int] = 14,
        **kwargs
    ):
        super(Corruption, self).__init__(**kwargs)
        self.signal_factor = signal_factor
        self.readout_noise_stddev = readout_noise_stddev
        self.full_well_depth = full_well_depth
        self.number_of_ADC_bits = number_of_ADC_bits

    @property
    def max_ADC(self):
        return 2**self.number_of_ADC_bits - 1

    @property
    def ADC_gain(self):
        return self.max_ADC / self.full_well_depth

    @property
    def epsilon_number_of_electrons(self):
        return self.full_well_depth / 100

    def call(self, sensing):
        sensing *= self.signal_factor
        number_of_electrons = sensing * self.full_well_depth
        # To simulate Poisson noise in a differentiable manner.
        # `epsilon_number_of_electrons` for numerical stability because gradient of sqrt diverges in 0.
        shot_noise_stddev = tf.math.sqrt(tf.maximum(number_of_electrons, self.epsilon_number_of_electrons))
        number_of_electrons = tf.random.normal(
            tf.shape(number_of_electrons),
            mean=number_of_electrons,
            stddev=shot_noise_stddev)
        number_of_electrons = tf.random.normal(
            tf.shape(number_of_electrons),
            mean=number_of_electrons,
            stddev=self.readout_noise_stddev)
        number_of_electrons = tf.clip_by_value(number_of_electrons, 0, self.full_well_depth)
        quantization_noise = tf.random.uniform(tf.shape(number_of_electrons), minval=-0.5, maxval=0.5)
        digital_units = number_of_electrons * self.ADC_gain + quantization_noise
        return digital_units / (self.max_ADC * self.signal_factor)

class SeededModulation(tfkl.Layer):
    """A layer to model a stochastic non-physical learnable coded mask."""

    def __init__(
        self,
        number_of_shots: Optional[int] = 1,
        seed_dimension: Optional[int] = 8,
        number_of_layers: Optional[int] = 4,
        filters_per_layer: Optional[int] = 32,
        spectral_resolution: Optional[int] = 3,
        **kwargs
    ):
        super(SeededModulation, self).__init__(**kwargs)
        self.number_of_shots = number_of_shots
        self.seed_dimension = seed_dimension
        self.spectral_resolution = spectral_resolution
        self.layers = [tfkl.Dense(filters_per_layer, activation='relu') for _ in range(number_of_layers - 1)]
        self.layers.append(tfkl.Dense(spectral_resolution, activation='sigmoid'))

    def call(self, sh):
        """Produces a multi-shot modulation field.

        Arguments:
        sh: shape of the input light field [B, H, W, U, V, C]

        Returns:
        a modulation field of shape [B, H, W, U, V, C, S]"""

        seed_shape = tf.concat([sh[:-1], [self.number_of_shots, self.seed_dimension]], 0)
        out_sh = tf.concat([seed_shape[:-1], [self.spectral_resolution]], 0)
        seed = tf.random.normal(seed_shape)
        x = tf.reshape(seed, [-1, self.seed_dimension])
        x = reduce(lambda x, layer: layer(x), self.layers, x)
        x = tf.reshape(x, out_sh)
        return tf.transpose(x, [0, 1, 2, 3, 4, 6, 5])  # Swap spectral and shot dimensions.

class PaletteModulation(tfkl.Layer):
    """A layer to model a stochastic non-physical learnable coded mask using a
    fixed learnable palette of transmittance for each pixel of the mask."""

    def __init__(
        self,
        number_of_shots: Optional[int] = 1,
        number_of_colors: Optional[int] = 16,
        spectral_resolution: Optional[int] = 3,
        mode: Optional[str] = "learnable",
        **kwargs
    ):
        super(PaletteModulation, self).__init__(**kwargs)
        self.number_of_shots = number_of_shots
        self.number_of_colors = number_of_colors
        self.spectral_resolution = spectral_resolution
        self.mode = mode
        self.color_matrix = PaletteModulation.create_color_matrix(number_of_colors, spectral_resolution, mode)

    @staticmethod
    def create_color_matrix(number_of_colors, spectral_resolution, mode):
        if mode == "learnable":
            return tf.Variable(
                tf.random.uniform([number_of_colors, spectral_resolution]),
                dtype=tf.float32)
        elif mode == "rgbw" and (number_of_colors == 4 and spectral_resolution == 3):
            return tf.concat([tf.eye(3), tf.ones(1, 3)])
        elif mode == "rgbw":
            raise ValueError("(number_of_colors, spectral_resolution) must be (4, 3) when mode is `rgbw`.")
        else:
            raise ValueError(mode)

    def call(self, sh):
        """Produces a multi-shot modulation field.

        Arguments:
        sh: shape of the input light field [B, H, W, U, V, C]

        Returns:
        a modulation field of shape [B, H, W, U, V, C, S]"""

        in_sh = tf.concat([sh[:-1], [self.number_of_shots]], 0)
        out_sh = tf.concat([in_sh, [self.spectral_resolution]], 0)
        indices = tf.random.uniform(in_sh, maxval=self.number_of_colors, dtype=tf.int32)
        indices = tf.reshape(indices, [-1])
        colors = tf.gather(self.color_matrix, indices)
        colors = tf.reshape(colors, out_sh)
        colors = tf.clip_by_value(colors, 0, 1)
        return tf.transpose(colors, [0, 1, 2, 3, 4, 6, 5])  # Swap spectral and shot dimensions.

class UniformModulation(tfkl.Layer):
    """A layer to stochastically produce a modulation field in which all elements are drawn independently from a uniform distribution."""

    def __init__(
        self,
        number_of_shots: Optional[int] = 1,
        pixel_resolution: Optional[int] = 1,
        **kwargs
    ):
        super(UniformModulation, self).__init__(**kwargs)
        self.number_of_shots = number_of_shots
        self.pixel_resolution = pixel_resolution

    def call(self, sh):
        """Produces a multi-shot modulation field.

        Arguments:
        sh: shape of the input light field [B, H, W, U, V, C]

        Returns:
        a modulation field of shape [B, H, W, U, V, C, S]"""

        out_sh = tf.concat([sh, [self.number_of_shots]], 0)
        n = sh[:1]
        hw = sh[1:3]
        ijcs = out_sh[3:]
        lowres_hw = tf.cast(tf.math.ceil(hw / self.pixel_resolution), tf.int32)
        lowres_sh = tf.concat([n, lowres_hw, ijcs], axis=0)
        modulation = tf.random.uniform(lowres_sh)
        for axis in [1, 2]:
            modulation = tf.repeat(modulation, self.pixel_resolution, axis=axis)
        return tf.slice(modulation, tf.zeros_like(out_sh), out_sh)

class CFAModulation(tfkl.Layer):
    """A layer to produce a modulation field corresponding to a (optionally trainable) color filter array."""

    def __init__(
        self,
        number_of_shots: Optional[int] = 1,
        pattern_height: Optional[int] = 2,
        pattern_width: Optional[int] = 2,
        pixel_size: Optional[int] = 1,
        spectral_resolution: Optional[int] = 3,
        mode: Optional[str] = "learnable",
        full_dimension: Optional[str] = False,
        **kwargs
    ):
        super(CFAModulation, self).__init__(**kwargs)
        self.number_of_shots = number_of_shots
        self.pattern_height = pattern_height
        self.pattern_width = pattern_width
        self.pixel_size = pixel_size
        self.spectral_resolution = spectral_resolution
        self.mode = mode
        self.full_dimension = full_dimension
        self.pattern = CFAModulation.create_pattern(pattern_height, pattern_width, spectral_resolution, mode)

    @staticmethod
    def create_pattern(pattern_height, pattern_width, spectral_resolution, mode):
        shape = [pattern_height, pattern_width, spectral_resolution]
        if mode == "learnable":
            return tf.Variable(
                tf.random.uniform(shape),
                constraint=lambda x: tf.clip_by_value(x, 0, 1))
        elif mode == "bayer" and (shape == [2, 2, 3]):
            r, g, b = tf.eye(3)
            return tf.convert_to_tensor([[r, g], [g, b]])
        elif mode == "bayer":
            raise ValueError("(pattern_height, pattern_width, spectral_resolution) must be (2, 2, 3) when mode is `bayer`.")  # noqa: E501
        elif mode == "mono":
            return tf.ones(shape)
        else:
            raise ValueError(mode)

    def call(self, sh):
        """Produces a multi-shot modulation field.

        Arguments:
        sh: shape of the input light field [B, H, W, U, V, C]

        Returns:
        a modulation field of shape [B, H, W, U, V, C, S]"""

        out_sh = tf.concat([sh, [self.number_of_shots]], 0)
        hwc = tf.gather(sh, [1, 2, tf.size(sh) - 1])
        scaled_pattern = tf.repeat(tf.repeat(self.pattern, self.pixel_size, 0), self.pixel_size, 1)
        psh = tf.shape(scaled_pattern)
        mul = tf.cast(tf.math.ceil(hwc / psh), tf.int32)
        cfa = tf.tile(scaled_pattern, mul)
        cfa = tf.slice(cfa, tf.zeros_like(hwc), hwc)
        cfa = cfa[tf.newaxis, :, :, tf.newaxis, tf.newaxis, :, tf.newaxis]
        return cfa * tf.ones(out_sh) if self.full_dimension else cfa

class ColorCodedApertureModulation(tfkl.Layer):
    """A layer to produce a modulation field corresponding to a polychromatic coded aperture (i.e. mask location on the angular plane)."""

    def __init__(
        self,
        number_of_shots: Optional[int] = 1,
        aperture_resolution: Optional[int] = 5,
        spectral_resolution: Optional[int] = 3,
        **kwargs
    ):
        super(ColorCodedApertureModulation, self).__init__(**kwargs)
        self.number_of_shots = number_of_shots
        self.aperture_resolution = aperture_resolution
        self.spectral_resolution = spectral_resolution
        color_coded_aperture_shape = [self.aperture_resolution] * 2 + [self.spectral_resolution, self.number_of_shots]
        self.color_coded_aperture = tf.Variable(tf.random.uniform(color_coded_aperture_shape),
            constraint=lambda x: tf.clip_by_value(x, 0, 1))

    def call(self, sh):
        out_sh = tf.concat([sh, [self.number_of_shots]], 0)
        return tf.ones(out_sh) * self.color_coded_aperture[tf.newaxis, tf.newaxis, tf.newaxis, :, :, :, :]

class MonochromeCodedApertureModulation(tfkl.Layer):
    """A layer to produce a modulation field corresponding to a monochromatic coded aperture (i.e. mask location on the angular plane)."""

    def __init__(
        self,
        number_of_shots: Optional[int] = 1,
        aperture_resolution: Optional[int] = 5,
        mode: Optional[str] = "learned",
        **kwargs
    ):
        super(MonochromeCodedApertureModulation, self).__init__(**kwargs)
        if mode not in {"learned", "uniform"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.mode = mode
        self.number_of_shots = number_of_shots
        self.aperture_resolution = aperture_resolution
        color_coded_aperture_shape = [self.aperture_resolution] * 2 + [self.number_of_shots]
        if mode == "learned":
            self.monochrome_coded_aperture = tf.Variable(tf.random.uniform(color_coded_aperture_shape),
                constraint=lambda x: tf.clip_by_value(x, 0, 1))

    def call(self, sh):
        if self.mode == "learned":
            coded_aperture = self.monochrome_coded_aperture
        else:
            coded_aperture = tf.random.uniform([self.aperture_resolution] * 2 + [self.number_of_shots])
        out_sh = tf.concat([sh, [self.number_of_shots]], 0)
        return tf.ones(out_sh) * coded_aperture[tf.newaxis, tf.newaxis, tf.newaxis, :, :, tf.newaxis, :]

# WARNING: Supports only single shot for now.
class InterpolatedViewCodedMaskModulatingField(tfkl.Layer):
    """A layer to produce the physically accurate modulating field corresponding to a trainable mask located somewhere between the 
    angular (aperture) plane and sensor plane. The layer modelizes the multiplexing by the coded mask of the continuous light field
    obtained by interpolation of the available discrete views of the light field."""

    def __init__(
        self,
        gamma: Optional[float] = 0.5,
        aperture_pixel_size: Optional[float] = 0.25,
        mask_pixel_size: Optional[float] = 0.04,
        sensor_pixel_size: Optional[float] = 1/64,
        aperture_resolution: Optional[int] = 5,
        mask_resolution: Optional[int] = 32,
        sensor_resolution: Optional[int] = 64,
        spectral_resolution: Optional[int] = 3,
        number_of_shots: Optional[int] = 1,
        view_interpolation_mode: Optional[str] = "linear",
        mask_mode: Optional[str] = "learned",
        **kwargs
    ):
        """Arguments:
        gamma: number to caracterize the position of the mask. 0 means located on the sensor plane, 1 means located on the aperture plane.
        aperture_pixel_size: size of an angular pixel, i.e. baseline of the input light field
        mask_pixel_size: size of an individual pixel of the mask
        sensor_pxiel_size: size of a pixel on the sensor
        aperture_resolution: number of sub-aperture for a given dimension in the input light field
        mask_resolution: number of pixels on the coded mask along a given spatial dimension
        sensor_resolution: number of pixels on the sensor along a given spatial dimension
        spectral_resolution: number of chromatic channels
        number_of_shots: number of shots for the acquisition (must be 1)
        view_interpolation_mode: algorithm to use to interpolate the missing views using the available views.
        mask_mode: whether the mask is learned or not."""

        super(InterpolatedViewCodedMaskModulatingField, self).__init__(**kwargs)
        if number_of_shots != 1:
            raise ValueError("Only single-shot is supported for now.")
        if mask_mode not in {'learned', 'uniform'}:
            raise ValueError(f"Mask mode unsupported: {mask_mode}.")
        self.mask_mode = mask_mode
        self.number_of_shots = number_of_shots
        self.gamma = gamma
        self.aperture_pixel_size = aperture_pixel_size
        self.mask_pixel_size = mask_pixel_size
        self.sensor_pixel_size = sensor_pixel_size
        self.aperture_resolution = aperture_resolution
        self.mask_resolution = mask_resolution
        self.sensor_resolution = sensor_resolution
        self.spectral_resolution = spectral_resolution
        self.normalization_factor = (aperture_resolution * sensor_resolution) ** 2

        if mask_mode == "learned":
            self.mask_values = tf.Variable(tf.random.uniform([mask_resolution ** 2, self.spectral_resolution], dtype=tf.float32),
                constraint=lambda x: tf.clip_by_value(x, 0, 1))
        self.form_factors = InterpolatedViewCodedMaskModulatingField.get_form_factors(view_interpolation_mode,
            aperture_resolution, mask_resolution, sensor_resolution, aperture_pixel_size, mask_pixel_size, sensor_pixel_size, gamma)

    @staticmethod
    def get_form_factors(view_interpolation_mode: str, A: int, M: int, S: int, delta_A: float, delta_M: float, delta_S: float, gamma: float):
        if view_interpolation_mode == "nn":
            get_form_factors = InterpolatedViewCodedMaskModulatingField.get_1d_nearest_neighbor_form_factors
        elif view_interpolation_mode == "linear":
            get_form_factors = InterpolatedViewCodedMaskModulatingField.get_1d_linear_form_factors
        one_d_ff = get_form_factors(A, M, S, delta_A, delta_M, delta_S, gamma)
        one_d_ff = tf.constant(one_d_ff)
        two_d_ff = one_d_ff[tf.newaxis, tf.newaxis, tf.newaxis] * one_d_ff[..., tf.newaxis, tf.newaxis, tf.newaxis]
        return tf.sparse.from_dense(tf.reshape(tf.transpose(two_d_ff, [0, 3, 2, 5, 1, 4]), [-1, M**2]))

    @staticmethod
    def get_1d_nearest_neighbor_form_factors(A: int, M: int, S: int, delta_A: float, delta_M: float, delta_S: float, gamma: float):
        convex_polygons_as_constraints = InterpolatedViewCodedMaskModulatingField.get_polygons_as_constraints(A, M, S, delta_A, delta_M, delta_S, gamma)
        return ConvexIntegralHelper.get_areas(convex_polygons_as_constraints).reshape(S, M, A)

    @staticmethod
    def get_1d_linear_form_factors(A: int, M: int, S: int, delta_A: float, delta_M: float, delta_S: float, gamma: float):
        polygons = InterpolatedViewCodedMaskModulatingField.get_polygons_as_constraints(A - 1, M, S, delta_A, delta_M, delta_S, gamma)
        left_interpolation_vectors = InterpolatedViewCodedMaskModulatingField.get_left_interpolation_vectors(A - 1, M, S, delta_A)
        right_interpolation_vectors = InterpolatedViewCodedMaskModulatingField.get_right_interpolation_vectors(A - 1, M, S, delta_A)
        left_factors = ConvexIntegralHelper.get_integrals(polygons, left_interpolation_vectors).reshape(S, M, A - 1)
        right_factors = ConvexIntegralHelper.get_integrals(polygons, right_interpolation_vectors).reshape(S, M, A - 1)
        return np.pad(left_factors, [[0,0],[0,0],[1,0]]) + np.pad(right_factors, [[0,0],[0,0],[0,1]])

    @staticmethod
    def get_left_interpolation_vectors(A, M, S, delta_A):
        affine_vectors = [[[
            [-i_A + A / 2, 0, 1 / delta_A]
            for i_A in range(A)] for _ in range(M)] for _ in range(S)]
        return np.array(affine_vectors, dtype='float32').reshape(-1, 3)

    @staticmethod
    def get_right_interpolation_vectors(A, M, S, delta_A):
        affine_vectors = [[[
            [i_A - A / 2 + 1, 0, -1 / delta_A]
            for i_A in range(A)] for _ in range(M)] for _ in range(S)]
        return np.array(affine_vectors, dtype='float32').reshape(-1, 3)

    @staticmethod
    def get_polygons_as_constraints(A, M, S, delta_A, delta_M, delta_S, gamma):
        polygons = [[[[
                        [-(i_A - A / 2 + 1) * delta_A, 0, 1],
                        [(i_A - A / 2) * delta_A, 0, -1],
                        [-(i_M - M / 2 + 1) * delta_M, 1 - gamma, gamma],
                        [(i_M - M / 2) * delta_M, gamma - 1, -gamma],
                        [-(i_S - S / 2 + 1) * delta_S, 1, 0],
                        [(i_S - S / 2) * delta_S, -1, 0]
                    ]
                    for i_A in range(A)]
            for i_M in range(M)]
        for i_S in range(S)]
        return np.array(polygons, dtype='float32').reshape(-1, 6, 3)

    def call(self, sh):
        b = sh[0]
        if self.mask_mode == "learned":
            mask_values = self.mask_values
        else:
            mask_values = tf.random.uniform([self.mask_resolution ** 2, self.spectral_resolution], dtype=tf.float32)
        modulation_field = tf.sparse.sparse_dense_matmul(self.form_factors, mask_values)
        modulation_field = tf.reshape(modulation_field, sh[1:])
        modulation_field = modulation_field[tf.newaxis, ..., tf.newaxis]
        return self.normalization_factor * tf.repeat(modulation_field, b, 0)

class ConvexIntegralHelper:
    """A helper class to perform various operators on convex sets."""

    @staticmethod
    def get_areas(convex_polygons_as_constraints: np.ndarray):
        factors = map(lambda x: ConvexIntegralHelper._compute_area(ConvexIntegralHelper._to_loop(x)), convex_polygons_as_constraints)
        return np.array(list(factors)).reshape(convex_polygons_as_constraints.shape[:-2])

    @staticmethod
    def get_integrals(convex_polygons_as_constraints: np.ndarray, integral_vectors: np.ndarray):
        factors = map(
            lambda x: ConvexIntegralHelper._compute_integral(ConvexIntegralHelper._to_loop(x[0]), x[1]),
            zip(convex_polygons_as_constraints, integral_vectors))
        return np.array(list(factors), dtype='float32').reshape(convex_polygons_as_constraints.shape[:-2])

    @staticmethod
    def _to_loop(constraints: np.ndarray):
        big_float = 10.0  # Warning: needs to be big enough but not too big in order to ensure numerical stability.
        loop = np.array([[big_float, 0], [0, big_float], [-big_float, 0], [0, -big_float]], dtype='float32')
        loop = reduce(lambda loop, constraint: ConvexIntegralHelper._cut(loop, constraint), constraints, loop)
        return np.array(loop, dtype='float32').reshape(-1, 2)
        
    @staticmethod
    def _cut(loop, constraint: np.ndarray):
        if len(loop) == 0:
            return loop
        f = lambda v: np.dot(v, constraint[1:]) + constraint[0]
        cut_loop = []
        last_v = loop[-1]
        last_phi = f(last_v)
        for v in loop:
            phi = f(v)
            if phi <= 0:
                if last_phi > 0:
                    t = phi / (phi - last_phi)
                    cut_loop.append(t * last_v + (1 - t) * v)
                cut_loop.append(v)
            else:
                if last_phi <= 0:
                    t = last_phi / (last_phi - phi)
                    cut_loop.append(t * v + (1 - t) * last_v)
            last_v, last_phi = v, phi
        return cut_loop

    @staticmethod
    def _compute_area(loop: np.ndarray):
        vectorial_triangles = np.stack([loop, np.roll(loop, -1, axis=0)], -2) - np.mean(loop, axis=0)
        return np.abs(np.sum(np.linalg.det(vectorial_triangles))) / 2

    @staticmethod
    def _compute_integral(loop: np.ndarray, integral_vectors: np.ndarray):
        loop_length = loop.shape[0]
        v0 = np.mean(loop, 0)
        vectorial_triangles = np.stack([loop, np.roll(loop, -1, axis=0)], -2) - v0
        affine_triangles = np.concatenate([np.repeat(v0.reshape(1, 1, 2), loop_length, 0), vectorial_triangles], -2)
        affine_matrices = np.concatenate([np.repeat(np.array([1.,0,0]).reshape(1,3,1), loop_length, 0), affine_triangles], -1)
        determinants = np.linalg.det(vectorial_triangles)
        affine_weighter = np.array([1/2, 1/6, 1/6], dtype='float32')
        integrals = (affine_weighter @ affine_matrices @ integral_vectors) * np.abs(determinants)
        return np.sum(integrals)
