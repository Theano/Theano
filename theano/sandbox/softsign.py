from __future__ import absolute_import, print_function, division
from theano.tensor.nnet.nnet import softsign  # noqa
import sys

print(
    "DEPRECATION WARNING: softsign was moved from theano.sandbox.softsign to "
    "theano.tensor.nnet.nnet ", file=sys.stderr
    )
