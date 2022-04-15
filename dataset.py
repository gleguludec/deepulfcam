import logging
import os
from glob import glob
from pathlib import Path
from typing import Optional, Callable

import tensorflow as tf
from PIL import Image

tfd = tf.data
tfde = tfd.experimental

class PathsHelper:

    @staticmethod
    def get_dataset_by_name_from_config(
        datasets_config: dict,
        angular_resolution: int,
        logger: Optional[logging.Logger] = None
    ):
        name_to_dataset = {}
        for dataset_name, dataset_config in datasets_config.items():
            file_descriptions = dataset_config['file_descriptions']
            options = dataset_config.get('options', {})
            paths_tensor = PathsHelper.get_paths_tensor_from_description(
                file_descriptions, angular_resolution)
            dataset = DatasetBuilder(paths_tensor, **options).make()
            name_to_dataset[dataset_name] = dataset
            if logger:
                logger.info(f"{paths_tensor.shape[0]} light fields in {dataset_name}.")
        return name_to_dataset

    @staticmethod
    def get_paths_tensor_from_description(file_descriptions, angular_resolution):
        paths_tensors = []
        for description in file_descriptions:
            # A bit of a trick but useful for more flexibility:
            # We dynamically create the formatting function to map (v, u) to file name.
            formatting = description['format'].replace('angular_resolution', f"{angular_resolution}")
            exec(f"format_function = lambda v, u, name: f'{formatting}'")
            paths_tensor = PathsHelper.get_paths_tensor(
                os.path.expandvars(description['root_dir']),
                locals()['format_function'],
                angular_resolution,
                angular_resolution,
                description.get('names_to_ignore', []))
            paths_tensors.append(paths_tensor)
        return tf.concat(paths_tensors, axis=0)

    @staticmethod
    def get_paths_tensor(
            root_path_as_str: str,
            angular_coordinates_to_file_name: Callable,
            angular_height: int,
            angular_width: int,
            names_to_ignore=None):
        names_to_ignore = names_to_ignore or {}
        root = Path(root_path_as_str)
        folder_paths = [Path(s) for s in glob(str(root / "*")) if os.path.isdir(s)]
        file_paths = [[[str(p / angular_coordinates_to_file_name(v, u, p.name))
                        for u in range(angular_height)]
                       for v in range(angular_width)]
                      for p in folder_paths if p.name not in names_to_ignore]
        return tf.convert_to_tensor(file_paths)


class DatasetBuilder:

    DEFAULT_SHUFFLE_BUFFER_SIZE = 2000
    DEFAULT_PATCH_HEIGHT = 100
    DEFAULT_PATCH_WIDTH = 100
    DEFAULT_COLOR_CHANNELS = 3
    DEFAULT_PATCH_STRIDE_RATIO = 3
    DEFAULT_SATURATION_LOWER_FACTOR = 0.75
    DEFAULT_SATURATION_UPPER_FACTOR = 1.5
    DEFAULT_HUE_MAX_DELTA = 0.1

    def __init__(self, paths_tensor: tf.Tensor, **kwargs):
        self.paths_tensor = paths_tensor

        self._shuffle_before_extraction = kwargs.get(
            'shuffle_before_extraction', False)
        self._shuffle_after_extraction = kwargs.get(
            'shuffle_after_extraction', False)
        self._make_patches = kwargs.get(
            'make_patches', 'patch_height' in kwargs or 'patch_width' in kwargs)
        self._repeat = kwargs.get(
            'repeat', False)
        self._augment_data = kwargs.get(
            'augment_data',
            any(k in kwargs for k in ['saturation_lower_factor', 'saturation_lower_factor', 'hue_max_delta']))
        self._prefetch = kwargs.get(
            'prefetch', False)
        self._use_experimental_augmentation = kwargs.get(
            'use_experimental_augmentation', False)
        self._batch_size = kwargs.get(
            'batch_size', None)
        self._shuffle_buffer_size = kwargs.get(
            'shuffle_buffer_size', DatasetBuilder.DEFAULT_SHUFFLE_BUFFER_SIZE)
        self._patch_height = kwargs.get(
            'patch_height', DatasetBuilder.DEFAULT_PATCH_HEIGHT)
        self._patch_width = kwargs.get(
            'patch_width', DatasetBuilder.DEFAULT_PATCH_WIDTH)
        self._patch_stride_ratio = kwargs.get(
            'patch_stride_ratio', DatasetBuilder.DEFAULT_PATCH_STRIDE_RATIO)
        self._color_channels = kwargs.get(
            'color_channels', DatasetBuilder.DEFAULT_COLOR_CHANNELS)
        self._saturation_lower_factor = kwargs.get(
            'saturation_lower_factor', DatasetBuilder.DEFAULT_SATURATION_LOWER_FACTOR)
        self._saturation_upper_factor = kwargs.get(
            'saturation_upper_factor', DatasetBuilder.DEFAULT_SATURATION_UPPER_FACTOR)
        self._hue_max_delta = kwargs.get(
            'hue_max_delta', DatasetBuilder.DEFAULT_HUE_MAX_DELTA)

        self._initialize_fullsize_height_and_width()

    @property
    def height(self) -> int:
        return self._patch_height if self._make_patches else self.fullsize_height

    @property
    def width(self) -> int:
        return self._patch_width if self._make_patches else self.fullsize_width

    @property
    def fullsize_height(self) -> int:
        return self._fullsize_height

    @property
    def fullsize_width(self) -> int:
        return self._fullsize_width

    @property
    def angular_height(self) -> int:
        return self.paths_tensor.shape[1]

    @property
    def angular_width(self) -> int:
        return self.paths_tensor.shape[2]

    @property
    def patch_shape(self) -> list:
        return [self._patch_height, self._patch_width,
                self.angular_height, self.angular_width,
                self._color_channels]

    @property
    def full_shape(self) -> list:
        return [self.fullsize_height, self.fullsize_width,
                self.angular_height, self.angular_width,
                self._color_channels]

    @property
    def patch_strides(self) -> list:
        return [self._patch_height // self._patch_stride_ratio,
                self._patch_width // self._patch_stride_ratio]

    def _initialize_fullsize_height_and_width(self) -> None:
        path_as_string = tf.compat.as_str_any(
            self.paths_tensor[0, 0, 0].numpy())
        with Image.open(path_as_string) as image:
            self._fullsize_width, self._fullsize_height = image.size

    def _get_light_field_tensor(self, paths) -> tf.Tensor:
        images = []
        for v in range(self.angular_height):
            for u in range(self.angular_width):
                images.append(self._get_image(paths[v, u]))
        return tf.concat(images, axis=2)

    def _get_image(self, path) -> tf.Tensor:
        image = tf.io.read_file(path)
        image = tf.io.decode_png(image, channels=self._color_channels)
        return tf.image.convert_image_dtype(image, tf.float32)

    def _extract_patches(self, lf_tensor: tf.Tensor) -> tf.Tensor:
        lf_tensor = tf.expand_dims(lf_tensor, axis=0)
        patches = tf.image.extract_patches(
            images=lf_tensor,
            sizes=[1, self._patch_height, self._patch_width, 1],
            strides=[1] + self.patch_strides + [1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        return tf.reshape(patches, [-1] + self.patch_shape)

    def _augment(self, lf_tensor: tf.Tensor) -> tf.Tensor:
        x = tf.transpose(lf_tensor, perm=[2, 3, 0, 1, 4])
        x = tf.reshape(x, [self.height * self.angular_height * self.angular_width,
                           self.width, self._color_channels])
        x = tf.image.random_saturation(
            x, self._saturation_lower_factor, self._saturation_upper_factor)
        x = tf.image.random_hue(x, self._hue_max_delta)
        x = tf.reshape(x, [
            self.angular_height, self.angular_width,
            self.height, self.width, self._color_channels])
        return tf.transpose(x, perm=[2, 3, 0, 1, 4])

    @tf.function
    def _experimental_augment(self, lf_tensor: tf.Tensor) -> tf.Tensor:
        x = tf.transpose(lf_tensor, [2, 3, 0, 1, 4])
        x = tf.image.adjust_gamma(x, tf.random.uniform([], minval=0.3, maxval=1.0))
        x = tf.image.random_saturation(x, 1.0, 1.5)
        random_perm = tf.concat([tf.reshape(tf.transpose(tf.random.shuffle([[2, 0], [3, 1]])), [-1]), [4]], 0)
        return tf.transpose(x, random_perm)

    def make(self) -> tfd.Dataset:
        ds = tfd.Dataset.from_tensor_slices(self.paths_tensor)
        if self._shuffle_before_extraction:
            ds = ds.shuffle(512)
        ds = ds.map(self._get_light_field_tensor, num_parallel_calls=tfde.AUTOTUNE)
        if self._make_patches:
            ds = ds.map(self._extract_patches, num_parallel_calls=tfde.AUTOTUNE)
            ds = ds.unbatch()
        else:
            ds = ds.map(lambda x: tf.reshape(x, self.full_shape))
        if self._shuffle_after_extraction:
            ds = ds.shuffle(self._shuffle_buffer_size)
        if self._repeat:
            ds = ds.repeat()
        if self._augment_data:
            ds = ds.map(self._augment, num_parallel_calls=tfde.AUTOTUNE)
        if self._use_experimental_augmentation:
            ds = ds.map(self._experimental_augment, num_parallel_calls=tfde.AUTOTUNE)
        if self._batch_size:
            ds = ds.batch(self._batch_size)
        if self._prefetch:
            ds = ds.prefetch(tfde.AUTOTUNE)
        return ds
