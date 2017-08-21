import numpy as np
import time
import json
import warnings

from collections import deque
from collections import OrderedDict
from collections import Iterable
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Layer
from keras import metrics


try:
    import requests
except ImportError:
    requests = None

if K.backend() == 'tensorflow':
    import tensorflow as tf

class EarlyStoppingBound(Callback):
    """Stop training when a monitored quantity has stopped improving.
    # Arguments
        monitor: quantity to be monitored.
        bound: limit for the monitored quantity
        verbose: verbosity mode.
        mode: one of {upper, lower}. If it is an upper or lower bound
        for the monitored quantity
    """

    def __init__(self, bound, monitor, mode, verbose=0):
        super(EarlyStoppingBound, self).__init__()
        self.bound = bound
        self.monitor = monitor
        self.verbose = verbose
        self.stopped_epoch = 0

        if mode == 'upper':
            self.monitor_op = np.less
        elif mode == 'lower':
            self.monitor_op = np.greater

    def on_train_begin(self, logs=None):
        self.wait = 0  # Allow instances to be re-used
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Early stopping requires %s available!' %
                          (self.monitor), RuntimeWarning)

        if not self.monitor_op(current , self.bound):
            self.stopped_epoch = epoch
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch))
            
            
            
            
class CustomVariationalLayer(Layer):
    def __init__(self, img_rows, img_cols, weight=False, 
                 model="gaussian_logsigma", **kwargs):
        self.is_placeholder = True
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.weight = weight
        self.model=model 
        
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def binary_loss(self, x, x_decoded_mean_squash):
        xent_loss = self.img_rows * self.img_cols * \
            metrics.binary_crossentropy(x, x_decoded_mean_squash)
        return xent_loss
    
    def gaussian_loss(self, x, mu, logsigma=None):
        loss = 0.
        quad = K.batch_dot(K.pow(x - mu, 2)
        if logsigma is not None:
            loss += 0.5 * K.sum(logsigma, axis=-1)
            sigma_inv = K.exp(-logsigma)
            loss += K.batch_dot(sigma_inv, quad)
        else:
            loss += K.sum(quad, axis=-1)
        return loss
        
        
        
    def vae_loss(self, x, x_decoded_mean_squash, z_mean, z_log_var, 
                 x_log_var=None, weight=None):
        x = K.flatten(x)
        x_decoded_mean_squash = K.flatten(x_decoded_mean_squash)
        if x_log_var is not None:
            x_log_var = K.flatten(x_log_var)
        if model == "binary":
            xent_loss = self.binary_loss(x, x_decoded_mean_squash)
        elif model.startswith("gaussian"):
            xent_loss = self.gaussian_loss(x, x_decoded_mean_squash, 
                                           x_log_var)
        kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - \
            K.exp(z_log_var), axis=-1)
        loss = kl_loss + xent_loss
        if self.weight:
            loss = weight * loss
        return K.mean(loss)

    def call(self, inputs):
        x, x_decoded_mean_squash, z_mean, z_log_var = inputs[:4]
        inputs = inputs[4:]
        x_log_var = None
        if self.model == "gaussian_logsigma":
            x_log_var = inputs.pop(0)
        weight = None
        if self.weight:
            weight = inputs.pop(0)
        
        loss = self.vae_loss(x, x_decoded_mean_squash, z_mean, 
                             z_log_var, weight)
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x
