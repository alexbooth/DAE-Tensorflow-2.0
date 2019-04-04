from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import tensorflow as tf

# TODO use 1000 data points as validation set
# TODO delete after using to draw (or put in helper somewhere)

def standardize_dataset(x, axis=None):
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.sqrt(((x - mean)**2).mean(axis=axis, keepdims=True))
    return (x - mean) / std

def add_gaussian_noise(X, mean=0, std=1):
    """Returns a copy of X with Gaussian noise."""
    return X.copy() + std * np.random.standard_normal(X.shape) + mean

class DataManager:
    def __init__(self):
        self.X = None
        self.training_set_size = None
        self.load_data()

    def load_data(self):
        """Loads 28x28 MNIST data, pads so each image size is 32x32,
        then standardizes images."""
        with open("./data/mnist/mnist.pkl", "rb") as f:
            data = pickle.load(f)
            data = data.astype(np.float32)
            data = data.reshape(data.shape[0], 28, 28, 1)
            data = np.pad(data, ((0,0), (2,2),(2,2),(0,0)), "constant", constant_values=0)
            data = standardize_dataset(data, axis=(1,2))
            np.random.shuffle(data)
            self.X = data
            self.training_set_size = data.shape[0]

    def get_batch(self, batch_size, use_noise=False):
        """Returns tuple of (X, X_pred). Adds noise to X_pred if specified.
           Otherwise, X == X_pred.

           Arguments:
                batch_size: integer > 0
                use_noise: boolean

           Returns:
                Tuple of two numpy.ndarray of the same shape"""
        indexes = np.random.randint(self.X.shape[0], size=batch_size)
        if use_noise:
            return self.X[indexes,:], add_gaussian_noise(self.X[indexes,:])
        return self.X[indexes,:], self.X[indexes,:]
