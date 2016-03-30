from __future__ import absolute_import, print_function, division
import unittest

import theano
import numpy
import scipy.sparse as sp

from theano import sparse
from theano import gof, tensor, compile

from theano.sparse.tests.test_basic import eval_outputs
from theano.sparse.basic import (
    _is_sparse_variable, _is_dense_variable,
    as_sparse_variable, _is_sparse, _mtypes, _mtype_to_str)
from theano.sparse import SparseType, dense_from_sparse, transpose

from theano.sparse.tests.test_basic import sparse_random_inputs
from theano.tests import unittest_tools as utt
from theano.sparse import verify_grad_sparse

# To maintain compatibility
from theano.sparse.basic import TrueDot, true_dot
