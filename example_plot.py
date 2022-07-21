import numpy as np
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import os
import random

from data.visu_density import plot_heatmap_2d
from data.plot_samples import plot_samples_2d
from utils.train_utils import sanity_check, train_density_estimation, nll
from normalizingflows.flow_catalog import Made
from data.data_manager import Dataset

import normalizingflows.flow_catalog as flowcat

tfd = tfp.distributions
tfb = tfp.bijectors

global_step = []
train_losses = []
val_losses = []
min_val_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)  # high value to ensure that first loss < min_loss
min_train_loss = tf.convert_to_tensor(np.inf, dtype=tf.float32)
min_val_epoch = 0
min_train_epoch = 0
delta_stop = 1000  # threshold for early stopping

timesteps = 21

hidden_shape = [10, 10]  # hidden shape for MADE network of MAF
layers = 16  # number of layers of the flow

base_dist = tfd.Normal(loc=0.0, scale=1.0)  # specify base distribution

bijectors = []
for i in range(0, layers):
    bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn = Made(params=2, hidden_units=hidden_shape, activation="relu")))
    bijectors.append(tfb.Permute(permutation=[1, 0]))  # data permutation after layers of MAF
    
bijector = tfb.Chain(bijectors=list(reversed(bijectors)), name='chain_of_maf')

maf = tfd.TransformedDistribution(
    distribution=tfd.Sample(base_dist, sample_shape=[2]),
    bijector=bijector,
)

checkpoint = tf.train.Checkpoint(model=maf)
checkpoint.read('checkpoint' + str(timesteps))

plot_heatmap_2d(maf, -4.0, 4.0, -4.0, 4.0, mesh_count=200, name='density_estimation')  # to save the image, specify a directory as name

# plot samples of the best model
plot_samples_2d(maf.sample(1000), name='sample_example_' + str(timesteps))  # to save the image, specify a directory as name

print("Number of trainable variables:", flowcat.get_trainable_variables(maf))

# plot train and validation loss curve
plt.plot(global_step, train_losses, label="train loss")
plt.plot(global_step, val_losses, label="val loss")
plt.legend()