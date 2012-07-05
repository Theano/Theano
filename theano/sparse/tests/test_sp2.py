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


class BinomialTester(utt.InferShapeTester):
    n = tensor.scalar()
    p = tensor.scalar()
    shape = tensor.lvector()
    _n = 5
    _p = .25
    _shape = np.asarray([3, 5], dtype='int64')

    inputs = [n, p, shape]
    _inputs = [_n, _p, _shape]

    def setUp(self):
        super(BinomialTester, self).setUp()
        self.op_class = S2.Binomial

    def test_op(self):
        for sp_format in sparse.sparse_formats:
            for o_type in sparse.float_dtypes:
                f = theano.function(
                    self.inputs,
                    S2.Binomial(sp_format, o_type)(*self.inputs))

                tested = f(*self._inputs)

                assert tested.shape == tuple(self._shape)
                assert tested.format == sp_format
                assert tested.dtype == o_type
                assert np.allclose(np.floor(tested.todense()),
                                   tested.todense())

    def test_infer_shape(self):
        for sp_format in sparse.sparse_formats:
            for o_type in sparse.float_dtypes:
                self._compile_and_check(
                    self.inputs,
                    [S2.Binomial(sp_format, o_type)(*self.inputs)],
                    self._inputs,
                    self.op_class)


if __name__ == '__main__':
    unittest.main()
