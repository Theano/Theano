from __future__ import print_function
import sys
from theano.tensor.nnet.blocksparse import *

print("DEPRECATION: theano.sandbox.blocksparse does not exist anymore,"
      "it has been moved to theano.tensor.nnet.blocksparse.", file=sys.stderr)
