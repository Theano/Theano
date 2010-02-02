import sys
print >> sys.stderr, "DEPRECATION: theano.sandbox.conv no longer provides conv.  They have been moved to theano.tensor.nnet.conv"
from theano.tensor.nnet.conv import *
