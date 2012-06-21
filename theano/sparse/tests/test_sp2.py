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


def as_sparse_format(data, format):
    if format == 'csc':
        return scipy.sparse.csc_matrix(data)
    elif format == 'csr':
        return scipy.sparse.csr_matrix(data)
    else:
        raise NotImplementedError()


def eval_outputs(outputs):
    return compile.function([], outputs)()[0]


def random_lil(shape, dtype, nnz):
    rval = sp.lil_matrix(shape, dtype=dtype)
    huge = 2 ** 30
    for k in range(nnz):
        # set non-zeros in random locations (row x, col y)
        idx = np.random.random_integers(huge, size=len(shape)) % shape
        value = np.random.rand()
        #if dtype *int*, value will always be zeros!
        if "int" in dtype:
            value = int(value * 100)
        rval.__setitem__(
                idx,
                value)
    return rval


class TestCast(utt.InferShapeTester):
    compatible_types = (tensor.int_dtypes +
                        tensor.continuous_dtypes)
    x_csc = [theano.sparse.csc_matrix(dtype=t) for t in compatible_types]
    x_csr = [theano.sparse.csr_matrix(dtype=t) for t in compatible_types]

    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    properties = (data, indices, indptr)

    def setUp(self):
        super(TestCast, self).setUp()
        self.op_class = S2.Cast

    def test_cast(self):
        cast_csc = dict([
            (x, [theano.function([x], S2.Cast(t)(x))
                 for t in self.compatible_types])
            for x in self.x_csc])

        cast_csr = dict([
            (x, [theano.function([x], S2.Cast(t)(x))
                 for t in self.compatible_types])
            for x in self.x_csr])

        for x in self.x_csc:
            for f, t in zip(cast_csc[x], self.compatible_types):
                a = sp.csc_matrix(self.properties, dtype=x.dtype).copy()
                assert f(a).dtype == t

        for x in self.x_csr:
            for f, t in zip(cast_csr[x], self.compatible_types):
                a = sp.csr_matrix(self.properties, dtype=x.dtype)
                assert f(a).dtype == t

    def test_infer_shape(self):
        for x in self.x_csc:
            for t in self.compatible_types:
                a = sp.csc_matrix(self.properties, dtype=x.dtype)
                self._compile_and_check([x],
                                        [S2.Cast(t)(x)],
                                        [a],
                                        self.op_class)

        for x in self.x_csr:
            for t in self.compatible_types:
                a = sp.csr_matrix(self.properties, dtype=x.dtype)
                self._compile_and_check([x],
                                        [S2.Cast(t)(x)],
                                        [a],
                                        self.op_class)

    def test_grad(self):
        for dtype in tensor.float_dtypes:
            for t in tensor.float_dtypes:
                eps = None
                if t == 'float32':
                    eps = 7e-4
                a = sp.csc_matrix(self.properties, dtype=dtype)
                verify_grad_sparse(S2.Cast(t), [a], eps=eps)

        for dtype in tensor.float_dtypes:
            for t in tensor.float_dtypes:
                eps = None
                if t == 'float32':
                    eps = 7e-4
                a = sp.csr_matrix(self.properties, dtype=dtype)
                verify_grad_sparse(S2.Cast(t), [a], eps=eps)


class _HVStackTester(utt.InferShapeTester):
    """Test for both HStack and VStack.

    """
    nb = 3  # Number of sparse matrix to stack
    x = {}
    mat = {}

    for format in sparse.sparse_formats:
        variable = getattr(theano.sparse, format + '_matrix')
        spa = getattr(sp, format + '_matrix')

        x[format] = [variable() for t in range(nb)]
        mat[format] = [spa(np.random.random_integers(5, size=(3, 4)) - 1,
                           dtype=theano.config.floatX)
                       for t in range(nb)]

    def test_op(self):
        for format in sparse.sparse_formats:
            for out_f in sparse.sparse_formats:
                for dtype in sparse.all_dtypes:
                    blocks = self.mat[format]

                    f = theano.function(
                        self.x[format],
                        self.op_class(
                            format=out_f, dtype=dtype)(*self.x[format]),
                        allow_input_downcast=True)

                    tested = f(*blocks)
                    expected = self.expected_f(blocks, format=out_f, dtype=dtype)

                    assert np.allclose(tested.toarray(), expected.toarray())
                    assert tested.format == expected.format
                    assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check(self.x[format],
                                    [self.op_class()(*self.x[format])],
                                    self.mat[format],
                                    self.op_class)

    def test_grad(self):
        for format in sparse.sparse_formats:
            for out_f in sparse.sparse_formats:
                for dtype in sparse.float_dtypes:
                    eps = None
                    if dtype == 'float32':
                        eps = 7e-4

                    verify_grad_sparse(
                        self.op_class(format=out_f, dtype=dtype),
                        self.mat[format],
                        structured=False,
                        eps=eps)


def _hv_switch(op, expected_function):
    """Return the right test class for HStack or VStack.

    :Parameters:
    - `op`: HStack or VStack class.
    - `expected_function`: function from scipy for comparaison.

    """
    class XStackTester(_HVStackTester):
        op_class = op

        def expected_f(self, a, format=None, dtype=None):
            return expected_function(a, format, dtype)

        def setUp(self):
            super(XStackTester, self).setUp()

    return XStackTester

HStackTester = _hv_switch(S2.HStack, sp.hstack)
VStackTester = _hv_switch(S2.VStack, sp.vstack)

class test_structured_add_s_v(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_structured_add_s_v_grad(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.rand(3), dtype=dtype)
                
                theano.sparse.verify_grad_sparse(S2.structured_add_s_v,
                    [spmat, mat], structured=True)
    
    def test_structured_add_s_v(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                x = theano.sparse.SparseType(format, dtype=dtype)()
                y = tensor.vector(dtype=dtype)
                f = theano.function([x, y], S2.structured_add_s_v(x, y))
                
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                spones = spmat.copy()
                spones.data = np.ones_like(spones.data)
                mat = np.asarray(np.random.rand(3), dtype=dtype)
                
                out = f(spmat, mat)
                
                assert np.allclose(out.toarray(), spones.multiply(spmat + mat))


class test_mul_s_v(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_structured_add_s_v_grad(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.rand(3), dtype=dtype)
                
                theano.sparse.verify_grad_sparse(S2.mul_s_v,
                    [spmat, mat], structured=True)
    
    def test_mul_s_v(self):
        sp_types = {'csc': sp.csc_matrix,
            'csr': sp.csr_matrix}
        
        for format in ['csr', 'csc']:
            for dtype in ['float32', 'float64']:
                x = theano.sparse.SparseType(format, dtype=dtype)()
                y = tensor.vector(dtype=dtype)
                f = theano.function([x, y], S2.mul_s_v(x, y))
                
                spmat = sp_types[format](random_lil((4, 3), dtype, 3))
                mat = np.asarray(np.random.rand(3), dtype=dtype)
                
                out = f(spmat, mat)
                
                assert np.allclose(out.toarray(), spmat.toarray() * mat)

if __name__ == '__main__':
    unittest.main()
