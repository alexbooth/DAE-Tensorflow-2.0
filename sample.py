from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import tensorflow as tf
from data_manager import DataManager

import matplotlib.pyplot as plt

from model import Encoder, Decoder

from absl import app
from absl import flags

flags.DEFINE_integer("sample_size", 10, "samples to test")
flags.DEFINE_string("model", "./trained_model/DAE_model.h5", "todo")
flags.DEFINE_boolean("use_noise", False, "sample noisey images")
FLAGS = flags.FLAGS

def sample(model, manager, n_samples):
    """Passes n random samples through the model and displays X & X_pred"""
    _, X = manager.get_batch(n_samples, use_noise=FLAGS.use_noise)
    X_pred = model.predict(X)
    x_dim, y_dim = X[0].shape[0], X[0].shape[1]
    X_stitched = np.reshape(X.swapaxes(0,1), (x_dim, y_dim*n_samples))
    X_pred_stitched = np.reshape(X_pred.swapaxes(0,1), (x_dim, y_dim*n_samples))
    stitched_img = np.vstack((X_stitched, X_pred_stitched))
    plt.imshow(stitched_img)
    plt.show()

def load_model():
    """Set up and return the model."""
    model_path = os.path.abspath(FLAGS.model)
    model = tf.keras.models.load_model(model_path)

    # holds dimensions of latent vector once we find it
    z_dim = None

    # define encoder
    encoder_in  = tf.keras.Input(shape=(32, 32, 1))
    encoder_out = Encoder(encoder_in)
    encoder = tf.keras.Model(inputs=encoder_in, outputs=encoder_out)
 
    # load encoder weights and get the dimensions of the latent vector
    for i, layer in enumerate(model.layers):
        encoder.layers[i] = layer
        if layer.name == "encoder_output":
            z_dim = (layer.get_weights()[0].shape[-1])
            break

    # define encoder
    decoder_in  = tf.keras.Input(shape=(z_dim,))
    decoder_out = Decoder(decoder_in)
    decoder = tf.keras.Model(inputs=decoder_in, outputs=decoder_out)

    # load decoder weights
    found_decoder_weights = False
    decoder_layer_cnt = 0
    for i, layer in enumerate(model.layers):
        if not found_decoder_weights: 
            found_decoder_weights = ("decoder_input" == layer.name)
        else:
            decoder.layers[decoder_layer_cnt] = layer
            decoder_layer_cnt += 1

    #encoder.compile()
    #decoder.compile()
    encoder.summary()
    decoder.summary()

    return model

def main(argv):
    manager = DataManager()
    model = load_model()
    #sample(model, manager, FLAGS.sample_size)

if __name__ == '__main__':
    app.run(main)
