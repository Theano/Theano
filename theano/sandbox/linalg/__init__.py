from __future__ import absolute_import, print_function, division
from theano.tensor.slinalg import (cholesky, solve, eigvalsh)
from theano.tensor.nlinalg import (matrix_inverse,
                                   diag, extract_diag, alloc_diag,
                                   det, eig, eigh,
                                   trace)
from theano.sandbox.linalg.ops import psd, spectral_radius_bound
