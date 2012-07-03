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
                    expected = self.expected_f(blocks,
                                               format=out_f,
                                               dtype=dtype)

                    assert np.allclose(tested.toarray(), expected.toarray())
                    assert tested.format == expected.format
                    assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check(self.x[format],
                                    [self.op_class(theano.config.floatX)
                                     (*self.x[format])],
                                    self.mat[format],
                                    self.op_class)

    def test_grad(self):
        for format in sparse.sparse_formats:
            for out_f in sparse.sparse_formats:
                for dtype in sparse.float_dtypes:
                    verify_grad_sparse(
                        self.op_class(format=out_f, dtype=dtype),
                        self.mat[format],
                        structured=False,
                        eps=7e-4)


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
    return XStackTester

HStackTester = _hv_switch(S2.HStack, sp.hstack)
VStackTester = _hv_switch(S2.VStack, sp.vstack)


class AddSSDataTester(utt.InferShapeTester):
    x = {}
    a = {}

    def setUp(self):
        super(AddSSDataTester, self).setUp()
        self.op_class = S2.AddSSData

        for format in sparse.sparse_formats:
            variable = getattr(theano.sparse, format + '_matrix')

            rand = np.array(np.random.random_integers(3, size=(3, 4)) - 1,
                          dtype=theano.config.floatX)
            constant = as_sparse_format(rand, format)

            self.x[format] = [variable() for t in range(2)]
            self.a[format] = [constant for t in range(2)]

    def test_op(self):
        for format in sparse.sparse_formats:
            f = theano.function(
                self.x[format],
                S2.add_s_s_data(*self.x[format]))

            tested = f(*self.a[format])
            expected = 2 * self.a[format][0]

            assert np.allclose(tested.toarray(), expected.toarray())
            assert tested.format == expected.format
            assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check(self.x[format],
                                    [S2.add_s_s_data(*self.x[format])],
                                    self.a[format],
                                    self.op_class)

    def test_grad(self):
        for format in sparse.sparse_formats:
            verify_grad_sparse(self.op_class(),
                               self.a[format],
                               structured=True)


class PoissonTester(utt.InferShapeTester):
    x = {}
    a = {}

    for format in sparse.sparse_formats:
        variable = getattr(theano.sparse, format + '_matrix')

        rand = np.array(np.random.random_integers(3, size=(3, 4)) - 1,
                        dtype=theano.config.floatX)

        x[format] = variable()
        a[format] = as_sparse_format(rand, format)

    def setUp(self):
        super(PoissonTester, self).setUp()
        self.op_class = S2.Poisson

    def test_op(self):
        for format in sparse.sparse_formats:
            f = theano.function(
                [self.x[format]],
                S2.poisson(self.x[format]))

            tested = f(self.a[format])

            assert tested.format == format
            assert tested.dtype == self.a[format].dtype
            assert np.allclose(np.floor(tested.data), tested.data)
            assert tested.shape == self.a[format].shape

    def test_infer_shape(self):
        for format in sparse.sparse_formats:
            self._compile_and_check([self.x[format]],
                                    [S2.poisson(self.x[format])],
                                    [self.a[format]],
                                    self.op_class)


class MultinomialTester(utt.InferShapeTester):
    p = sparse.csr_matrix()
    _p = sp.csr_matrix(np.asarray([[0.0, 0.5, 0.0, 0.5],
                                   [0.1, 0.2, 0.3, 0.4],
                                   [0.0, 1.0, 0.0, 0.0],
                                   [0.3, 0.3, 0.0, 0.4]]))

    def setUp(self):
        super(MultinomialTester, self).setUp()
        self.op_class = S2.Multinomial

    def test_op(self):
        n = tensor.lscalar()
        f = theano.function([self.p, n], S2.multinomial(n, self.p))

        _n = 5
        tested = f(self._p, _n)
        assert tested.shape == self._p.shape
        assert np.allclose(np.floor(tested.todense()), tested.todense())
        assert tested[2, 1] == _n

        n = tensor.lvector()
        f = theano.function([self.p, n], S2.multinomial(n, self.p))

        _n = np.asarray([1, 2, 3, 4], dtype='int64')
        tested = f(self._p, _n)
        assert tested.shape == self._p.shape
        assert np.allclose(np.floor(tested.todense()), tested.todense())
        assert tested[2, 1] == _n[2]

    def test_infer_shape(self):
        self._compile_and_check([self.p],
                                [S2.multinomial(5, self.p)],
                                [self._p],
                                self.op_class)


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


class _StructuredMonoidUnaryTester(unittest.TestCase):
    def test_op(self):
        for format in sparse.sparse_formats:
            x = getattr(theano.sparse, format + '_matrix')()
            spa = getattr(sp, format + '_matrix')

            a = spa(np.random.random_integers(5, size=(3, 4)) - 1,
                    dtype=theano.config.floatX)

            f = theano.function([x], self.op(x))

            tested = f(a)
            expected = self.expected_op(a.todense())
            expected[a.todense() == 0] = 0

            assert tested.shape == expected.shape
            assert tested.dtype == expected.dtype
            assert np.allclose(tested.todense(), expected)


class StructuredSigmoidTester(_StructuredMonoidUnaryTester):
    def setUp(self):
        super(StructuredSigmoidTester, self).setUp()
        self.op = S2.structured_sigmoid
        self.expected_op = lambda x: 1.0 / (1.0 + np.exp(-x))


class StructuredExpTester(_StructuredMonoidUnaryTester):
    def setUp(self):
        super(StructuredExpTester, self).setUp()
        self.op = S2.structured_exp
        self.expected_op = np.exp


class StructuredLogTester(_StructuredMonoidUnaryTester):
    def setUp(self):
        super(StructuredLogTester, self).setUp()
        self.op = S2.structured_log
        self.expected_op = np.log


class StructuredPowTester(_StructuredMonoidUnaryTester):
    def setUp(self):
        super(StructuredPowTester, self).setUp()
        self.op = lambda x: S2.structured_pow(x, 2)
        self.expected_op = lambda x: np.power(x, 2)


class StructuredMinimumTester(_StructuredMonoidUnaryTester):
    def setUp(self):
        super(StructuredMinimumTester, self).setUp()
        self.op = lambda x: S2.structured_minimum(x, 2)
        self.expected_op = lambda x: np.minimum(x, 2)


class StructuredMaximumTester(_StructuredMonoidUnaryTester):
    def setUp(self):
        super(StructuredMaximumTester, self).setUp()
        self.op = lambda x: S2.structured_maximum(x, 2)
        self.expected_op = lambda x: np.maximum(x, 2)


class StructuredAddTester(_StructuredMonoidUnaryTester):
    def setUp(self):
        super(StructuredAddTester, self).setUp()
        self.op = lambda x: S2.structured_add(x, 2)
        self.expected_op = lambda x: np.add(x, 2)


class MulSVTester(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_mul_s_v_grad(self):
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


class StructuredAddSVTester(unittest.TestCase):
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


class SamplingDotTester(utt.InferShapeTester):
    x = [tensor.matrix() for t in range(2)]
    x.append(sparse.csr_matrix())
    a = [np.array(np.random.random_integers(maximum, size=(3, 3)) - 1,
                      dtype=theano.config.floatX)
         for maximum in [5, 5, 2]]
    a[2] = sp.csr_matrix(a[2])

    def setUp(self):
        super(SamplingDotTester, self).setUp()
        self.op_class = S2.SamplingDot

    def test_op(self):
        f = theano.function(
            self.x,
            S2.sampling_dot(*self.x))

        tested = f(*self.a)
        x, y, p = self.a
        expected = p.multiply(np.dot(x, y.T))

        assert np.allclose(tested.toarray(), expected)
        assert tested.format == 'csr'
        assert tested.dtype == expected.dtype

    def test_infer_shape(self):
        self._compile_and_check(self.x,
                                [S2.sampling_dot(*self.x)],
                                self.a,
                                self.op_class,
                                excluding=['local_sampling_dot_csr'])

    def test_grad(self):
        def _helper(x, y):
            return S2.sampling_dot(x, y, self.a[2])
        verify_grad_sparse(_helper, self.a[:2])


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
