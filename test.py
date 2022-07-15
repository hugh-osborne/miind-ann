import pylab
import matplotlib
matplotlib.use('TkAgg')
import imp
import datetime
from operator import add
from random import randint
import csv
from math import cos

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# As a test, let's try to learn the Fitzhugh-Nagumo neuron model

def fn(y, dt):
    param_dict={

    'tau_m':   20e-3,
    'E_r': -65e-3,
    'E_e':  0e-3,
    'tau_s':  5e-3,

    'V_min':-66.0e-3,
    'V_max':  -55.0e-3,
    'V_th': -55.0e-3, #'V_max', # sometimes used in other scripts
    'N_V': 200,
    'w_min':0.0,
    'w_max':  0.8,
    'N_w': 20,

    'I': 0.5, #0.5 to match FN mesh

    }

    v = y[0];
    w = y[1];

    v_prime = v - v**3/3 - w + param_dict['I']
    w_prime = .08*(v + .7 - .8*w)

    return [v_prime, w_prime]

model = tf.keras.Sequential()
model.add(Dense(10, input_shape=(2,), activation='relu'))
model.add(Dense(2))

#loss =
#‘binary_crossentropy‘ for binary classification.
#‘sparse_categorical_crossentropy‘ for multi-class classification.
#‘mse‘ (mean squared error) for regression.

opt = SGD(learning_rate=0.1, momentum=0.09)
model.compile(optimizer=opt, loss='mse')

#model.compile(optimizer='sgd', loss='mse')

# Define equally spaced points in state space (or could be random)
input_pairs = []
for v in np.arange(-2.5, 2.5, 0.1):
    for w in np.arange(-1.0, 1.0, 0.1):
        input_pairs = input_pairs + [[v,w]]
        
output_pairs = []
for pair in input_pairs: 
    output_pairs = output_pairs + [fn(pair, 0.1)]
    
input_pairs = np.array(input_pairs)
output_pairs = np.array(output_pairs)

print(input_pairs.shape, output_pairs.shape)

#verbose = 2 for output every epoch
#verbose = 0 for no output
model.fit(input_pairs, output_pairs, epochs=100, batch_size=32, verbose=2)


print("Done fitting.")
# Define some other points to test
input_pairs = []
for v in np.arange(-2.5, 2.5, 0.15):
    for w in np.arange(-1.0, 1.0, 0.15):
        input_pairs = input_pairs + [[v,w]]
        
output_pairs = []
for pair in input_pairs: 
    output_pairs = output_pairs + [fn(pair, 0.1)]
    
    
# evaluate the model
loss = model.evaluate(input_pairs, output_pairs, verbose=2)

print("Predict the vector field.")

#f, ax = plt.subplots()

#X = np.arange(-2.5, 2.5, 0.15)
#Y = np.arange(-1.0, 1.0, 0.15)

#V = np.zeros([Y.shape[0],0])
#W = np.zeros([Y.shape[0],0])
#for x in X.tolist():
#    v_row = []
#    w_row = []
#    for y in Y.tolist():
#        yhat = model.predict(np.array([[x, y]]))
#        print(np.array(v_row).shape, np.array(yhat).shape)
#        v_row = np.append(v_row, [yhat[-1][0]], axis=0)
#        w_row = np.append(w_row, [yhat[-1][1]], axis=0)

#    v_row = np.reshape(v_row, (Y.shape[0],1))
#    w_row = np.reshape(w_row, (Y.shape[0],1))
#    V = np.append(V, v_row, axis=1)
#    W = np.append(W, w_row, axis=1)

#q = ax.quiver(X, Y, V, W, zorder=1)

#faster?

f, ax = plt.subplots()

X = np.arange(-2.5, 2.5, 0.15)
Y = np.arange(-1.0, 1.0, 0.15)

V = np.zeros([Y.shape[0],0])
W = np.zeros([Y.shape[0],0])

pairs = []

for x in X.tolist():
    for y in Y.tolist():
        pairs = pairs + [[x,y]]
        
yhat = model.predict(np.array(pairs))
count = 0
for x in X.tolist():
    v_row = []
    w_row = []
    for y in Y.tolist():
        v_row = np.append(v_row, [yhat[count][0]], axis=0)
        w_row = np.append(w_row, [yhat[count][1]], axis=0)
        count += 1

    v_row = np.reshape(v_row, (Y.shape[0],1))
    w_row = np.reshape(w_row, (Y.shape[0],1))
    V = np.append(V, v_row, axis=1)
    W = np.append(W, w_row, axis=1)

q = ax.quiver(X, Y, V, W, zorder=1)


#time step

f, ax = plt.subplots()

X = np.arange(-2.5, 2.5, 0.15)
Y = np.arange(-1.0, 1.0, 0.15)

V = np.zeros([Y.shape[0],0])
W = np.zeros([Y.shape[0],0])
for x in X.tolist():
    v_row = []
    w_row = []
    for y in Y.tolist():
        yhat = fn([x,y], 0.1)
        v_row = np.append(v_row, [yhat[0]], axis=0)
        w_row = np.append(w_row, [yhat[1]], axis=0)

    v_row = np.reshape(v_row, (Y.shape[0],1))
    w_row = np.reshape(w_row, (Y.shape[0],1))
    V = np.append(V, v_row, axis=1)
    W = np.append(W, w_row, axis=1)

q = ax.quiver(X, Y, V, W, zorder=1)


plt.show()

print("Now predict a trace.")
its = 1000
trace_vs = [-1.0]
trace_ws = [0.2]

model_vs = [trace_vs[-1]]
model_ws = [trace_ws[-1]]
#predict
for i in range(its):
    yhat = model.predict(np.array([[trace_vs[-1], trace_ws[-1]]]))
    print(yhat)
    trace_vs = trace_vs + [trace_vs[-1] + 0.1*yhat[-1][0]]
    trace_ws = trace_ws + [trace_ws[-1] + 0.1*yhat[-1][1]]
    nt = fn([model_vs[-1],model_ws[-1]], 0.1)
    model_vs = model_vs + [model_vs[-1] + 0.1*nt[0]]
    model_ws = model_ws + [model_ws[-1] + 0.1*nt[1]]
    
plt.figure()
plt.plot(trace_vs,trace_ws)
plt.plot(model_vs, model_ws)
plt.show() 

