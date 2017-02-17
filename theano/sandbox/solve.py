from __future__ import absolute_import, print_function, division
import warnings
from theano.tensor.slinalg import solve  # noqa

message = ("The module theano.sandbox.solve will soon be deprecated.\n"
           "Please use tensor.slinalg.solve instead.")

warnings.warn(message)
