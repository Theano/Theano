import sys

enable_sparse=True

try:
    import scipy
    if scipy.__version__ < '0.7':
        sys.stderr.write("WARNING: scipy version = %s. We request version >=0.7.0 for the sparse code as it has bugs fixed in the sparse matrix code.\n" % scipy.__version__)
        enable_sparse=False
except ImportError:
    enable_sparse=False
    sys.stderr.write("WARNING: scipy can't be imported. We disable the sparse matrix code.")
if enable_sparse:
    from basic import *
    import sharedvar
    from sharedvar import sparse_constructor as shared

