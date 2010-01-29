import sys
print >> sys.stderr, "DEPRECATION: theano.sandbox.downsample is deprecated. Use theano.tensor.signal.downsample instead."

from theano.tensor.signal.downsample import *

