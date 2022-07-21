import pylab
import matplotlib
matplotlib.use('TkAgg')
import imp
import datetime
from operator import add
from random import randint
import csv
from math import cos
import random
import os

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

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

samples = maf.sample()

checkpoint = tf.train.Checkpoint(model=maf)

# Test reproduction of density
test_set = []
checkpoint.read('checkpoint' + str(0))
plot_samples_2d(maf.sample(1000), name='test_original_example_' + str(0))

for v in maf.trainable_variables:
    test_set = test_set + v.numpy().flatten().tolist()

counter = 0
for v in maf.trainable_variables:
    correct_vals = v.numpy().flatten().tolist()
    arr = test_set[counter:counter+len(correct_vals)]
    counter += len(correct_vals)
    nval = tf.convert_to_tensor(np.reshape(arr, v.shape), v.dtype)
    v.assign(nval)
plot_samples_2d(maf.sample(1000), name='test_example_' + str(0))

training_set_in = []
training_set_out = []

for t in range(37):
    print("Loading checkpoint", t)
    # Read all variables in starting distribution into a flattened array
    checkpoint.read('checkpoint' + str(t))
    flattened_variables_in = []
    for v in maf.trainable_variables:
        flattened_variables_in = flattened_variables_in + v.numpy().flatten().tolist()

    training_set_in = training_set_in + [flattened_variables_in]

    # Read all variables in next timestep distribution into a flattened array
    checkpoint.read('checkpoint' + str(t+1))
    flattened_variables_out = []
    for v in maf.trainable_variables:
        flattened_variables_out = flattened_variables_out + v.numpy().flatten().tolist()

    training_set_out = training_set_out + [flattened_variables_out]

testing_set_in = training_set_in[35:]
testing_set_out = training_set_out[35:]

training_set_in = training_set_in[:35]
training_set_out = training_set_out[:35]

model = tf.keras.Sequential()
model.add(Dense(10000, input_shape=(len(flattened_variables_in),), activation='relu'))
model.add(Dense(10000, activation='relu'))
model.add(Dense(10000, activation='relu'))
model.add(Dense(len(flattened_variables_out)))

#loss =
#'binary_crossentropy' for binary classification.
#'sparse_categorical_crossentropy' for multi-class classification.
#'mse' (mean squared error) for regression.

opt = SGD(learning_rate=0.1, momentum=0.09)
model.compile(optimizer=opt, loss='mse')

#model.compile(optimizer='sgd', loss='mse')

#verbose = 2 for output every epoch
#verbose = 0 for no output
model.fit(training_set_in, training_set_out, epochs=100, batch_size=32, verbose=2)

print("Done fitting.")
    
    
# evaluate the model
loss = model.evaluate(testing_set_in, testing_set_out, verbose=2)

print("Predict a simulation.")
its = 50

checkpoint.read('custom_gaussian_-1.0_0.2_0.05_0.05')
current_set = []
for v in maf.trainable_variables:
    current_set = current_set + v.numpy().flatten().tolist()

current_set = np.array([current_set])
print(current_set.shape)
#predict
for i in range(its):
    # Assign the new values
    counter = 0
    for v in maf.trainable_variables:
        correct_vals = v.numpy().flatten().tolist()
        arr = current_set[0][counter:counter+len(correct_vals)]
        counter += len(correct_vals)
        nval = tf.convert_to_tensor(np.reshape(arr, v.shape), v.dtype)
        v.assign(nval)
    plot_samples_2d(maf.sample(1000), name='sample_example_' + str(i))

    current_set = model.predict(current_set)
    
    #checkpoint.read('checkpoint' + str(i))
    #plot_samples_2d(maf.sample(1000), name='sample_base_example_' + str(i))