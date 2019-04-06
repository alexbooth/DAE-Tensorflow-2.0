from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape


def Conv(n_filters, filter_width):
    return Conv2D(n_filters, filter_width, 
                  strides=2, padding="same", activation="relu")

def Deconv(n_filters, filter_width):
    return Conv2DTranspose(n_filters, filter_width, 
                           strides=2, padding="same", activation="relu")

def Encoder(inputs):
    X = Conv(16, 5)(inputs)
    X = Conv(16, 5)(X)
    X = Conv(16, 3)(X)
    X = Conv(16, 3)(X)
    X = Flatten()(X)
    X = Dense(128, activation="relu")(X)
    X = Dense(64,  activation="relu")(X)
    X = Dense(32,  activation="relu")(X)
    Z = Dense(2,   activation="tanh", name="encoder_output")(X)
    return Z

def Decoder(Z):
    X = Dense(32,  activation="relu", name="decoder_input")(Z)
    X = Dense(64,  activation="relu")(X)
    X = Dense(128, activation="relu")(X)
    X = Dense(64,  activation="relu")(X)
    X = Reshape((2, 2, 16))(X)
    X = Deconv(16, 3)(X)
    X = Deconv(16, 3)(X)
    X = Deconv(16, 5)(X)
    X = Deconv(16, 5)(X)
    X = Conv2D(1, 1)(X)
    return X 

def AutoEncoder():
    X = tf.keras.Input(shape=(32, 32, 1))
    Z = Encoder(X)
    X_pred = Decoder(Z)
    return tf.keras.Model(inputs=X, outputs=X_pred)

