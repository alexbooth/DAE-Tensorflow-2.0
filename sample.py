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
# TODO delete
flags.DEFINE_integer("run_num", 0, "for frames")
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
        print(layer.name)
        weights = layer.get_weights()
        if len(layer.get_weights()) > 0:
            print(weights[0].shape, weights[1].shape)
        if "decoder_input" == layer.name:
            found_decoder_weights = True
        if found_decoder_weights:
            decoder_layer_cnt += 1
            print("dec:" + decoder.layers[decoder_layer_cnt].name)
            decoder.layers[decoder_layer_cnt].set_weights(weights)

    encoder.summary()
    decoder.summary()

    return encoder, decoder

def sample_decoder(decoder):
    Z = np.mgrid[-1:1:10j, -1:1:10j].reshape(2,-1).T 
    X_pred = decoder.predict(Z)
    out_img = np.zeros((10*32, 10*32))
    for x in range(10):
        for y in range(10):
            x_begin = x*32
            y_begin = y*32
            x_end   = x_begin + 32
            y_end   = y_begin + 32
            out_img[x_begin:x_end, y_begin:y_end] = X_pred[x+y*10].reshape((32,32))
    x_dim, y_dim = X_pred[0].shape[0], X_pred[0].shape[1]
    X_pred_stitched = np.reshape(X_pred.swapaxes(0,1), (x_dim, y_dim*100))
    plt.imsave("./image/frame_" + str(FLAGS.run_num) + ".png", out_img, format="png", vmin=-0.5, vmax=3)
    
        

def main(argv):
    manager = DataManager()
    encoder, decoder = load_model()
    #sample(model, manager, FLAGS.sample_size)
    sample_decoder(decoder)
    sample_decoder(decoder)

if __name__ == '__main__':
    app.run(main)
