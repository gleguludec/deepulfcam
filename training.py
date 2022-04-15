import argparse
import inspect
import logging
import os
import sys
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Optional

import tensorflow as tf
import yaml

import custom
from custom import (functions, physical_layers, reconstruction_layers, regularizers, experimental)
from dataset import PathsHelper

tfk = tf.keras
tfkm = tfk.models
tfkl = tfk.losses
tfkc = tfk.callbacks
tfko = tfk.optimizers
tfkos = tfko.schedules

def get_custom_objects():
    modules = [functions, physical_layers, reconstruction_layers, regularizers, experimental]
    return reduce(
        lambda x, y: dict(x, **y), [{
            name: obj
            for name, obj in inspect.getmembers(
                module, lambda m: inspect.isfunction(m) or inspect.isclass(m))}
            for module in modules])

custom_objects = get_custom_objects()

class YamlConfigLoader:

    def __init__(self, config_path, logger=None):
        self.config_path = config_path
        self.loader = yaml.SafeLoader
        self.logger = logger

    def get_config(self, placeholder_to_value=None):
        self.add_all_yaml_constructors()
        with open(self.config_path, 'r') as file:
            content = file.read()
        for placeholder, value in (placeholder_to_value or {}).items():
            content = content.replace(placeholder, value)
        config = yaml.load(content, Loader=self.loader)
        copy_config_to = os.path.expandvars(config.get('copy_config_to', ''))
        if copy_config_to:
            Path(copy_config_to).parent.mkdir(parents=True, exist_ok=True)
            with open(copy_config_to, 'w') as file:
                file.write(content)
        if self.logger:
            self.logger.info("Configuration:")
            self.logger.info(content)
        return config

    def add_all_yaml_constructors(self):
        modules = [tfko, tfkos,
                   physical_layers, reconstruction_layers, regularizers,
                   experimental, sys.modules[__name__]]
        self.add_constructors_for_modules(modules)
        self.add_keras_callbacks_constructors()
        self.add_pseudo_constructors()

    def add_constructors_for_modules(self, modules: list):
        cls_members = reduce(lambda x, y: x + y, [inspect.getmembers(m, inspect.isclass) for m in modules])
        for name, cls in cls_members:
            ctor = lambda loader, node, cls=cls: cls(**loader.construct_mapping(node, deep=True))  # noqa: E731
            self.loader.add_constructor(f"!{name}", ctor)

    def add_keras_callbacks_constructors(self):
        self.loader.add_constructor("!ModelCheckpoint", self.ModelCheckpoint_ctor)
        self.loader.add_constructor("!Tensorboard", self.Tensorboard_ctor)

    def add_pseudo_constructors(self):
        self.loader.add_constructor("!ExtractedLayer", self.ExtractedLayer_ctor)
        self.loader.add_constructor("!ModelFromFile", self.ModelFromFile_ctor)
        self.loader.add_constructor("!ModelWithWeightLoading", self.ModelWithWeightLoading_ctor)

    def ModelCheckpoint_ctor(self, loader: yaml.Loader, node: yaml.Node):
        values = loader.construct_mapping(node, deep=True)
        values['filepath'] = os.path.expandvars(values['filepath'])
        Path(values['filepath']).parent.mkdir(parents=True, exist_ok=True)
        return tfkc.ModelCheckpoint(**values)

    def Tensorboard_ctor(self, loader: yaml.Loader, node: yaml.Node):
        values = loader.construct_mapping(node, deep=True)
        values['log_dir'] = os.path.expandvars(values['log_dir'])
        return tfkc.TensorBoard(**values)

    def ExtractedLayer_ctor(self, loader: yaml.Loader, node: yaml.Node):
        values = loader.construct_mapping(node, deep=True)
        model_path = os.path.expandvars(values['model_path'])
        layer_name = values['layer_name']
        model = tfkm.load_model(model_path, custom_objects, compile=False)
        return model.get_layer(layer_name)

    def ModelFromFile_ctor(self, loader: yaml.Loader, node: yaml.Node):
        values = loader.construct_mapping(node, deep=True)
        model_path = os.path.expandvars(values['model_path'])
        self.logger is None or self.logger.info(f"Loaded initial model from file: {model_path}")
        return tfkm.load_model(model_path, custom_objects)

    def ModelWithWeightLoading_ctor(self, loader: yaml.Loader, node: yaml.Node):
        values = loader.construct_mapping(node, deep=True)
        model = values['model']
        weights_path = values['weights_path']
        model.load_weights(weights_path)
        return model

class Training:

    def __init__(self, config: dict, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger

    def launch(self):
        model = self.config['model']
        if self.training_config.get('compile', True):
            self.compile(model)
        try:
            model(tf.zeros([1, 64, 64, 5, 5, 3]))  # Feed with dummy tensor to force build of dynamic model.
            model.summary(line_length=144, print_fn=self.logger.info)
        except ValueError:
            logger.warning("Could not display model summary.")
        train_set, dev_set, test_set = [ds.map(lambda t: (t, t)) for ds in self.get_train_dev_test_sets()]
        model.fit(
            train_set,
            validation_data=dev_set,
            callbacks=self.training_config.get('callbacks', []),
            epochs=self.training_config['epochs'])
        model.evaluate(test_set)

    def compile(self, model: tfkm.Model):
        optimizer = self.training_config['optimizer']
        model.compile(optimizer, tfkl.mean_absolute_error, [custom.functions.psnr])
        self.logger is None or self.logger.info("Compiled model.")

    def get_train_dev_test_sets(self):
        name_to_dataset = PathsHelper.get_dataset_by_name_from_config(
            self.config['datasets'], self.config['angular_resolution'], self.logger)
        return [name_to_dataset[name] for name in ['train', 'dev', 'test']]

    @property
    def training_config(self):
        return self.config['training']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    parser.add_argument('--config_name', type=str, default=None)
    parser.add_argument('--silent', nargs='?', default=False, const=True)

    additional_arg_names = [
        'number_of_convolutions',
        'number_of_iterations',
        'number_of_shots',
        'delta_initial_value',
        'mu_initial_value',
        'cfa_pattern_size',
        'signal_factor',
        'mask_position_ratio',
        'mask_mode',
        'coded_aperture_mode'
    ]

    for arg_name in additional_arg_names:
        parser.add_argument(f'--{arg_name}', type=str, default='')

    args = parser.parse_args()
    if args.silent:
        logger = None
    else:
        logging.basicConfig(stream=sys.stdout, format='%(message)s', level=logging.INFO)
        logger = logging.getLogger()

    placeholder_to_value = dict(
        {'{config_name}': args.config_name or datetime.now().strftime('%Y-%m-%d-%H-%M-%S')},
        **{f'{{{arg_name}}}': args.__dict__[arg_name] for arg_name in additional_arg_names}
    )

    config = YamlConfigLoader(args.config_path, logger).get_config(placeholder_to_value)
    Training(config, logger).launch()
