from pkg_resources import parse_version as V
import sys

try:
    import scipy
    enable_sparse = V(scipy.__version__) >= V('0.7')

    if not enable_sparse:
        sys.stderr.write("WARNING: scipy version = %s."
                " We request version >=0.7.0 for the sparse code as it has"
                " bugs fixed in the sparse matrix code.\n" % scipy.__version__)
except ImportError:
    enable_sparse = False
    sys.stderr.write("WARNING: scipy can't be imported."
            " We disable the sparse matrix code.")

if enable_sparse:
    from basic import *
    import sharedvar
    from sharedvar import sparse_constructor as shared

