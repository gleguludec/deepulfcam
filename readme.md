# DeepULFCam / Deep Unrolling for Light Field Compressed Acquisition using Coded Masks

This code describes and allows the execution of an algorithm for dense light field reconstruction using a small number of simulated monochromatic projections of the light field. Those simulated projections consist of the composition of:
- the filtering of a light field by a color coded mask placed between the aperture plane and the sensor plane, performing both angular and spectral multiplexing,
- a color filtering array performing color multiplexing,
- a monochromatic sensor.

The composition of these filterings/projects is modeled as linear projection operator.

The light field is then reconstructed by performing the 'unrolling' of an interative reconstruction algorithm (namely the HQS 'half-quadratic splitting' algorithm, a variant of ADMM, an optimization algorithm, applied to the solving a regularized least-squares problem) into a deep convolutional neural network. The algorithm makes use of the structure of the projection operator to efficiently solve the quadratic data-fidelty minimization sub-problem in closed form.

This code is designed to be compatible with Python 3.7 and Tensorflow 2.3.
Other required dependencies are: Pillow, PyYAML.

## How to use

The ```configs``` folder contains yaml file 'templates' for several set-ups. Those files characterize the various experiments by describing the training hyperparameters, schedule, reconstruction algorithm architecture, acquisition model and datasets used for training.

Those configuration templates contain placeholder fields indicated as ```{{placeholder_field}}```. Those fields are to be given as arguments to the training script. As an example:

```python training.py --config_path configs/standard.yml  --config_name my_experiment --number_of_convolutions 4 --number_of_iterations 12 --number_of_shots 1 --delta_initial_value 0.1 --mu_initial_value 1.0```

Not all placeholder fields have to be provided to the training script depending on the configuration file. The different placeholder fields are:

- ```number_of_convolutions```: the number of convolutions in each regularizer block.
- ```number_of_iterations```: the number of unrolled iterations of the optimization algorithm into a network.
- ```number_of_shots'```: the number of measurements used by reconstruction.
- ```delta_initial_value'```: the gating parameter of the residual part of the convolutional regularizers.
- ```mu_initial_value'```: the penalty parameter of the unrolled HQS algorithm.
- ```cfa_pattern_size'```: the size of a pattern on the periodic color filter array.
- ```signal_factor'```: a parameter indicating the amount of noise occurring on the photosensor, the higher, the lower the corruption.
- ```mask_position_ratio'```: a parameter between 0 and 1 to place the coded mask somewhere between the sensor and the reference plane (i.e. the aperture).
- ```mask_mode'```: whether the coded mask is randomly sampled for each training/testing LF sample or is to be learned.
- ```coded_aperture_mode```: whether the coded aperture is randomly sampled for each training/testing LF sample or is to be learned.

## Remarks

The folder ```data``` provides only a minimal amount of light fields. It is divided into three subfolders: ```training```, ```validation``` and ```test```.
Each of these can contain several light fields organized into folders, one folder containing all the sub-aperture views of a given light field in ```.PNG``` format. The naming conventions for the sub-aperture image files can be edited in the template configuration files.

Users willing to train models with a reasonable amount of data might need to download (and possibly process and format) light fields from:
  - the Kalantari dataset: https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/SIGASIA16/
  - the Stanford Lytro LF Archive dataset: http://lightfields.stanford.edu/LF2016.html