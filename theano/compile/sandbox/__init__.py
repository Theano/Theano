import sys
print >> sys.stderr, "DEPRECATION: theano.compile.sandbox no longer provides shared, shared_constructor, and pfunc.  They have been moved to theano.compile."
from theano.compile.sharedvalue import shared, shared_constructor
from theano.compile.pfunc import pfunc
