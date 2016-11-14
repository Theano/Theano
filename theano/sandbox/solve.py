import warnings
from theano.tensor.slinalg import solve  # noqa

message = ("The module theano.sandbox.solve will soon be deprecated.\n"
           "Please use tensor.slinalg.solve instead.")

warnings.warn(message)
