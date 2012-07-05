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

# Already in test_basic.py
# Not used here
# def as_sparse_format(data, format):
#     if format == 'csc':
#         return scipy.sparse.csc_matrix(data)
#     elif format == 'csr':
#         return scipy.sparse.csr_matrix(data)
#     else:
#         raise NotImplementedError()


# Already in test_basic.py
# Not used here
# def eval_outputs(outputs):
#     return compile.function([], outputs)()[0]


# Already in test_basic.py
# Not used here
# def random_lil(shape, dtype, nnz):
#     rval = sp.lil_matrix(shape, dtype=dtype)
#     huge = 2 ** 30
#     for k in range(nnz):
#         # set non-zeros in random locations (row x, col y)
#         idx = np.random.random_integers(huge, size=len(shape)) % shape
#         value = np.random.rand()
#         #if dtype *int*, value will always be zeros!
#         if "int" in dtype:
#             value = int(value * 100)
#         rval.__setitem__(
#                 idx,
#                 value)
#     return rval


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


# ##################
# Optimization tests
# ##################
def test_local_mul_s_d():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_d")

    for sp_format in sparse.sparse_formats:
        inputs = [getattr(theano.sparse, sp_format + '_matrix')(),
                  tensor.matrix()]

        f = theano.function(inputs,
                            sparse.mul_s_d(*inputs),
                            mode=mode)

        assert not any(isinstance(node.op, sparse.MulSD) for node
                       in f.maker.env.toposort())


def test_local_mul_s_v():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_mul_s_v")

    for sp_format in ['csr']:  # Not implemented for other format
        inputs = [getattr(theano.sparse, sp_format + '_matrix')(),
                  tensor.vector()]

        f = theano.function(inputs,
                            S2.mul_s_v(*inputs),
                            mode=mode)

        assert not any(isinstance(node.op, S2.MulSV) for node
                       in f.maker.env.toposort())


def test_local_structured_add_s_v():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_structured_add_s_v")

    for sp_format in ['csr']:  # Not implemented for other format
        inputs = [getattr(theano.sparse, sp_format + '_matrix')(),
                  tensor.vector()]

        f = theano.function(inputs,
                            S2.structured_add_s_v(*inputs),
                            mode=mode)

        assert not any(isinstance(node.op, S2.StructuredAddSV) for node
                       in f.maker.env.toposort())


def test_local_sampling_dot_csr():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("specialize", "local_sampling_dot_csr")

    for sp_format in ['csr']:  # Not implemented for other format
        inputs = [tensor.matrix(),
                  tensor.matrix(),
                  getattr(theano.sparse, sp_format + '_matrix')()]

        f = theano.function(inputs,
                            S2.sampling_dot(*inputs),
                            mode=mode)

        assert not any(isinstance(node.op, S2.SamplingDot) for node
                       in f.maker.env.toposort())

if __name__ == '__main__':
    unittest.main()
