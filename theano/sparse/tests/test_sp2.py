import time
import unittest

from nose.plugins.skip import SkipTest
import numpy as np
try:
    import scipy.sparse as sp
    import scipy.sparse
except ImportError:
    pass  # The variable enable_sparse will be used to disable the test file.

import theano
from theano import tensor
from theano import sparse

if not theano.sparse.enable_sparse:
    raise SkipTest('Optional package sparse disabled')

from theano.sparse.sandbox import sp2 as S2

from theano.tests import unittest_tools as utt
from theano.sparse.basic import verify_grad_sparse


if __name__ == '__main__':
    unittest.main()
