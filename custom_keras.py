import numpy as np
import time
import json
import warnings

from collections import deque
from collections import OrderedDict
from collections import Iterable
from keras import backend as K
from keras.callbacks import Callback

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
