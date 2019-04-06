from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import numpy as np
import tensorflow as tf
from datetime import datetime
from data_manager import DataManager

from absl import app
from absl import flags

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MSE

from model import AutoEncoder

flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_integer("batch_size", 32, "batch size")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_string("logdir", "./tmp/log", "log file directory")
flags.DEFINE_boolean("keep_training", False, "continue training same weights")
flags.DEFINE_boolean("keep_best", False, "only save model if it got the best loss")
FLAGS = flags.FLAGS

best_loss = np.inf
model_path = None

def train(model):
    dm = DataManager() 
    n_batches = dm.training_set_size // FLAGS.batch_size
    n_epochs = FLAGS.epochs

    loss = None
    for epoch in range(n_epochs):
        for _ in range(n_batches):
            X, X_noise = dm.get_batch(FLAGS.batch_size, True)
            loss = model.train_on_batch(X, X_noise)
        print("Epoch {} - loss: {}".format(epoch, loss))
        save_model(model, epoch, loss)
    print("Finished training.")

def save_model(model, epoch, loss):
    """Write logs and save the model"""
    train_summary_writer = tf.summary.create_file_writer(summary_path)
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=epoch)

    # save model
    global best_loss
    if not FLAGS.keep_best: 
        model.save(model_path)
    elif loss < best_loss:
        best_loss = loss
        model.save(model_path)

def load_model():
    """Set up and return the model."""
    model = AutoEncoder()
    optimizer = Adam(FLAGS.learning_rate)
    loss = MSE

    # load most recent weights if model_path exists 
    if os.path.isfile(model_path):
        print("Loading model from", model_path)
        model.load_weights(model_path)

    model.compile(optimizer, loss)
    model.summary()
    return model

def setup_paths():
    """Create log and trained_model dirs. """
    global model_path, summary_path
    os.makedirs(FLAGS.logdir, exist_ok=True)
    os.makedirs("./trained_model", exist_ok=True)
    timestamp = str(datetime.now())

    if FLAGS.keep_training and os.listdir(FLAGS.logdir):
        files = filter(os.path.isdir, glob.glob(FLAGS.logdir + "/*"))
        files = sorted(files, key=lambda x: os.path.getmtime(x))
        timestamp = os.path.basename(os.path.normpath(list(reversed(files))[0]))

    model_path = os.path.join("./trained_model/DAE-model-" + timestamp + ".h5")
    summary_path = os.path.join(FLAGS.logdir, timestamp)

def main(argv):
    setup_paths()
    model = load_model()
    train(model)

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
