"""Placeholder for new gpuarray backend in sandbox. Supports old pickles
which refered to theano.sandbox.gpuarray."""

import warnings
from theano.gpuarray import *

message = "theano.sandbox.gpuarray has been moved to theano.gpuarray." + \
    " Please update your code and pickles."
warnings.warn(message)
