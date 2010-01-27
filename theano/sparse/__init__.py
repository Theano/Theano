import sys,scipy

enable_sparse=True
if not scipy.__version__.startswith('0.7.'):
    sys.stderr.write("WARNING: scipy version = %s. We request version >=0.7.0 for the sparse code as it has bugs fixed in the sparse matrix code.\n" % scipy.__version__)
    enable_sparse=False

if enable_sparse:
    from basic import *

