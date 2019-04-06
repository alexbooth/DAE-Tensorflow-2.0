from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
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
flags.DEFINE_string("checkpoint_dir", "./tmp/checkpoints", "checkpoint directory")
flags.DEFINE_string("log_file", "./tmp/log", "log file directory")
flags.DEFINE_boolean("keep_training", False, "continue training same weights")
FLAGS = flags.FLAGS
timestamp = str(datetime.now())
best_loss = np.inf


def train(model, manager):
    n_batches = manager.training_set_size // FLAGS.batch_size
    n_epochs = FLAGS.epochs

    loss = None
    for epoch in range(n_epochs):
        for _ in range(n_batches):
            X, X_noise = manager.get_batch(FLAGS.batch_size, True)
            loss = model.train_on_batch(X, X_noise)
        print("Epoch {} - loss: {}".format(epoch, loss))
        save_model(model, epoch, loss)
    print("Finished training.")

def save_model(model, epoch, loss):
    # write logs
    # TODO include validation loss
    summary_path = os.path.join(FLAGS.log_file, timestamp)
    train_summary_writer = tf.summary.create_file_writer(summary_path)
    with train_summary_writer.as_default():
        tf.summary.scalar("loss", loss, step=epoch)

    # save checkpoint
    #checkpoint_path = os.path.join(FLAGS.checkpoint_dir, timestamp, "ckpt")
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "ckpt")
    model.save_weights(checkpoint_path, save_format="tf")

    # save model if it has the best loss so far
    global best_loss
    if loss < best_loss:
        best_loss = loss
        model.save("./trained_model/DAE_model.h5")


def load_model():
    """Set up and return the model."""
    model = AutoEncoder()

    if FLAGS.keep_training:
        global timestamp
        # fancy one liner to grab the most recent checkpoint dir name
        #timestamp = list(reversed(next(os.walk(FLAGS.checkpoint_dir))[1]))[0] 
        #checkpoint_path = os.path.join(FLAGS.checkpoint_dir, timestamp, "ckpt")
        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "ckpt")
        if os.path.isfile(checkpoint_path + ".index"):
            model.load_weights(checkpoint_path)

    optimizer = Adam(FLAGS.learning_rate)
    loss = MSE
    model.compile(optimizer, loss)
    model.summary()
    return model

def main(argv):
    dm = DataManager() 
    model = load_model()
    train(model, dm)

if __name__ == '__main__':
    if not tf.test.is_gpu_available():
        print("WARNING: Not training with GPU. Training may be slow.")
    app.run(main)
