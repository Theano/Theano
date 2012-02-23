import time
import unittest

from nose.plugins.skip import SkipTest
import numpy
try:
    import scipy.sparse as sp
    import scipy.sparse
except ImportError:
    pass  # The variable enable_sparse will be used to disable the test file.

import theano
from theano import compile, config, gof
from theano.sparse import enable_sparse
from theano.gof.python25 import all, any, product

if enable_sparse == False:
    raise SkipTest('Optional package sparse disabled')

from theano.sparse.basic import _is_dense, _is_sparse, _mtypes
from theano.sparse.basic import _is_dense_variable, _is_sparse_variable
from theano.sparse.basic import verify_grad_sparse
from theano.sparse import as_sparse_variable, CSC, CSR, CSM, CSMProperties
from theano.sparse import SparseType, CSMGrad
from theano.sparse import StructuredDot, StructuredDotCSC
from theano.sparse import StructuredDotGradCSC, StructuredDotGradCSR
from theano.sparse import AddSS, AddSD, MulSS, MulSD, Transpose, Neg
from theano.sparse import add, mul, structured_dot, transpose
from theano.sparse import (csc_from_dense, csr_from_dense, dense_from_sparse,
        SparseFromDense)
from theano.sparse import Dot, Usmm, UsmmCscDense, sp_ones_like, GetItemScalar
#from theano.sparse import get_item_2d, get_item_scalar

from theano.tests import unittest_tools as utt
from theano import tensor
from theano.tensor.basic import _allclose


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
        idx = numpy.random.random_integers(huge, size=len(shape)) % shape
        value = numpy.random.rand()
        #if dtype *int*, value will always be zeros!
        if "int" in dtype:
            value = int(value * 100)
        rval.__setitem__(
                idx,
                value)
    return rval


class T_verify_grad_sparse(unittest.TestCase):
    class FailOp(gof.op.Op):
        def __init__(self, structured):
            self.structured = structured

        def __eq__(self, other):
            return (type(self) == type(other)) and \
                self.structured == other.structured

        def __hash__(self):
            return hash(type(self)) ^ hash(self.structured)

        def make_node(self, x):
            x = as_sparse_variable(x)
            return gof.Apply(self, [x], [x.type()])

        def perform(self, node, (x, ), (out, )):
            assert _is_sparse(x)
            out[0] = -x

        def grad(self, (x,), (gz,)):
            assert _is_sparse_variable(x) and _is_sparse_variable(gz)
            if self.structured:
                return sp_ones_like(x) * dense_from_sparse(gz),
            else:
                return gz,

        def infer_shape(self, node, shapes):
            return [shapes[0]]

    def test_grad_fail(self):
        self.assertRaises(verify_grad_sparse.E_grad,
                          verify_grad_sparse,
                          self.FailOp(structured=False),
                          [sp.csr_matrix(random_lil((10, 40),
                                                    config.floatX, 3))])

        self.assertRaises(verify_grad_sparse.E_grad,
                          verify_grad_sparse,
                          self.FailOp(structured=True),
                          [sp.csr_matrix(random_lil((10, 40),
                                                    config.floatX, 3))])


class T_transpose(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_transpose_csc(self):
        sp = scipy.sparse.csc_matrix(scipy.sparse.eye(5, 3))
        a = as_sparse_variable(sp)
        self.assertFalse(a.data is sp)
        self.assertTrue(a.data.shape == (5, 3))
        self.assertTrue(a.type.dtype == 'float64', a.type.dtype)
        self.assertTrue(a.type.format == 'csc', a.type.format)
        ta = transpose(a)
        self.assertTrue(ta.type.dtype == 'float64', ta.type.dtype)
        self.assertTrue(ta.type.format == 'csr', ta.type.format)

        vta = eval_outputs([ta])
        self.assertTrue(vta.shape == (3, 5))

    def test_transpose_csr(self):
        a = as_sparse_variable(scipy.sparse.csr_matrix(scipy.sparse.eye(5, 3)))
        self.assertTrue(a.data.shape == (5, 3))
        self.assertTrue(a.type.dtype == 'float64')
        self.assertTrue(a.type.format == 'csr')
        ta = transpose(a)
        self.assertTrue(ta.type.dtype == 'float64', ta.type.dtype)
        self.assertTrue(ta.type.format == 'csc', ta.type.format)

        vta = eval_outputs([ta])
        self.assertTrue(vta.shape == (3, 5))


class SparseInferShapeTester(utt.InferShapeTester):
    def test_getitem_2d(self):
        raise SkipTest('infer_shape not implemented for GetItem2d yet')

    def test_getitem_scalar(self):
        x = SparseType('csr', dtype=config.floatX)()
        self._compile_and_check([x],
                                [x[2, 2]],
                                [sp.csr_matrix(random_lil((10, 40),
                                               config.floatX, 3))],
                                GetItemScalar)

    def test_csm_grad(self):
        for sparsetype in ('csr', 'csc'):
            x = tensor.vector()
            y = tensor.ivector()
            z = tensor.ivector()
            s = tensor.ivector()
            call = getattr(sp, sparsetype + '_matrix')
            spm = call(random_lil((300, 400), config.floatX, 5))
            out = tensor.grad(dense_from_sparse(
                CSM(sparsetype)(x, y, z, s)
            ).sum(), x)
            self._compile_and_check([x, y, z, s],
                                    [out],
                                    [spm.data, spm.indices, spm.indptr,
                                     spm.shape],
                                    CSMGrad
                                   )

    def test_transpose(self):
        x = SparseType('csr', dtype=config.floatX)()
        self._compile_and_check([x],
                                [x.T],
                                [sp.csr_matrix(random_lil((10, 40),
                                               config.floatX, 3))],
                                Transpose)

    def test_neg(self):
        x = SparseType('csr', dtype=config.floatX)()
        self._compile_and_check([x],
                                [-x],
                                [sp.csr_matrix(random_lil((10, 40),
                                               config.floatX, 3))],
                                Neg)

    def test_add_ss(self):
        x = SparseType('csr', dtype=config.floatX)()
        y = SparseType('csr', dtype=config.floatX)()
        self._compile_and_check([x, y],
                                [x + y],
                                [sp.csr_matrix(random_lil((10, 40),
                                               config.floatX, 3)),
                                 sp.csr_matrix(random_lil((10, 40),
                                               config.floatX, 3))],
                                AddSS)

    def test_add_sd(self):
        x = SparseType('csr', dtype=config.floatX)()
        y = tensor.matrix()
        self._compile_and_check(
                [x, y],
                [x + y],
                [sp.csr_matrix(random_lil((10, 40),
                               config.floatX, 3)),
                 numpy.random.randn(10, 40).astype(config.floatX)],
                AddSD)

    def test_mul_ss(self):
        x = SparseType('csr', dtype=config.floatX)()
        y = SparseType('csr', dtype=config.floatX)()
        self._compile_and_check([x, y],
                                [x * y],
                                [sp.csr_matrix(random_lil((10, 40),
                                               config.floatX, 3)),
                                ] * 2,
                                MulSS)

    def test_mul_sd(self):
        x = SparseType('csr', dtype=config.floatX)()
        y = tensor.matrix()
        self._compile_and_check(
                [x, y],
                [x * y],
                [sp.csr_matrix(random_lil((10, 40),
                               config.floatX, 3)),
                 numpy.random.randn(10, 40).astype(config.floatX)],
                MulSD)

    def test_dot(self):
        x = SparseType('csc', dtype=config.floatX)()
        y = SparseType('csc', dtype=config.floatX)()
        self._compile_and_check(
                [x, y],
                [Dot()(x, y)],
                [sp.csc_matrix(random_lil((4, 5),
                               config.floatX, 3)),
                 sp.csc_matrix(random_lil((5, 3),
                               config.floatX, 3))],
                Dot)

    def test_structured_dot(self):
        x = SparseType('csc', dtype=config.floatX)()
        y = SparseType('csc', dtype=config.floatX)()
        self._compile_and_check(
                [x, y],
                [structured_dot(x, y)],
                [sp.csc_matrix(random_lil((4, 5),
                               config.floatX, 3)),
                 sp.csc_matrix(random_lil((5, 3),
                               config.floatX, 3))],
                StructuredDot)

    def test_csm(self):
        # We also need the grad of CSM to be implemetned.
        raise SkipTest('infer_shape not implemented for CSM')

    def test_structured_dot_grad(self):
        # We also need the grad of CSM to be implemetned.
        raise SkipTest('infer_shape not implemented for the grad'
                       ' of structured_dot')
        for format, op in [('csc', StructuredDotGradCSC),
                           ('csr', StructuredDotGradCSR)]:
            x = SparseType(format, dtype=config.floatX)()
            y = SparseType(format, dtype=config.floatX)()
            grads = tensor.grad(dense_from_sparse(structured_dot(x, y)).sum(),
                                [x, y])
            self._compile_and_check(
                    [x, y],
                    [grads[0]],
                    [as_sparse_format(random_lil((4, 5),
                                   config.floatX, 3), format),
                     as_sparse_format(random_lil((5, 3),
                                   config.floatX, 3), format)],
                    op)
            self._compile_and_check(
                    [x, y],
                    [grads[1]],
                    [as_sparse_format(random_lil((4, 5),
                                   config.floatX, 3), format),
                     as_sparse_format(random_lil((5, 3),
                                   config.floatX, 3), format)],
                    op)

    def test_dense_from_sparse(self):
        x = SparseType('csr', dtype=config.floatX)()
        self._compile_and_check([x],
                                [dense_from_sparse(x)],
                                [sp.csr_matrix(random_lil((10, 40),
                                               config.floatX, 3))],
                                dense_from_sparse.__class__)

    def test_sparse_from_dense(self):
        x = tensor.matrix()
        self._compile_and_check([x],
                                [csc_from_dense(x)],
                                [numpy.random.randn(10, 40).astype(
                    config.floatX)],

                                csc_from_dense.__class__)


class T_AddMul(unittest.TestCase):
    def testAddSS(self):
        self._testSS(add)

    def testAddSD(self):
        self._testSD(add)

    def testAddDS(self):
        self._testDS(add)

    def testMulSS(self):
        self._testSS(mul,
                     numpy.array([[1., 0], [3, 0], [0, 6]]),
                     numpy.array([[1., 0], [3, 0], [0, 6]]))

    def testMulSD(self):
        self._testSD(mul,
                     numpy.array([[1., 0], [3, 0], [0, 6]]),
                     numpy.array([[1., 0], [3, 0], [0, 6]]))

    def testMulDS(self):
        self._testDS(mul,
                     numpy.array([[1., 0], [3, 0], [0, 6]]),
                     numpy.array([[1., 0], [3, 0], [0, 6]]))

    def _testSS(self, op, array1=numpy.array([[1., 0], [3, 0], [0, 6]]),
                array2=numpy.asarray([[0, 2.], [0, 4], [5, 0]])):
        for mtype in _mtypes:
            a = mtype(array1)
            aR = as_sparse_variable(a)
            self.assertFalse(aR.data is a)
            self.assertTrue(_is_sparse(a))
            self.assertTrue(_is_sparse_variable(aR))

            b = mtype(array2)
            bR = as_sparse_variable(b)
            self.assertFalse(bR.data is b)
            self.assertTrue(_is_sparse(b))
            self.assertTrue(_is_sparse_variable(bR))

            apb = op(aR, bR)
            self.assertTrue(_is_sparse_variable(apb))

            self.assertTrue(apb.type.dtype == aR.type.dtype, apb.type.dtype)
            self.assertTrue(apb.type.dtype == bR.type.dtype, apb.type.dtype)
            self.assertTrue(apb.type.format == aR.type.format, apb.type.format)
            self.assertTrue(apb.type.format == bR.type.format, apb.type.format)

            val = eval_outputs([apb])
            self.assertTrue(val.shape == (3, 2))
            if op is add:
                self.assertTrue(numpy.all(val.todense() == (a + b).todense()))
                ans = numpy.array([[1., 2], [3, 4], [5, 6]])
                self.assertTrue(numpy.all(val.todense() == ans))
                verify_grad_sparse(op, [a, b], structured=False)
            elif op is mul:
                self.assertTrue(numpy.all(val.todense()
                                          == (a.multiply(b)).todense()))
                ans = numpy.array([[1, 0], [9, 0], [0, 36]])
                self.assertTrue(numpy.all(val.todense() == ans))

    def _testSD(self, op, array1=numpy.array([[1., 0], [3, 0], [0, 6]]),
                array2=numpy.asarray([[0, 2.], [0, 4], [5, 0]])):
        for mtype in _mtypes:
            a = numpy.array(array1)
            aR = tensor.as_tensor_variable(a)
            self.assertFalse(aR.data is a)  # constants are copied
            self.assertTrue(_is_dense(a))
            self.assertTrue(_is_dense_variable(aR))

            b = mtype(array2)
            bR = as_sparse_variable(b)
            self.assertFalse(bR.data is b)  # constants are copied
            self.assertTrue(_is_sparse(b))
            self.assertTrue(_is_sparse_variable(bR))

            apb = op(aR, bR)

            self.assertTrue(apb.type.dtype == aR.type.dtype, apb.type.dtype)
            self.assertTrue(apb.type.dtype == bR.type.dtype, apb.type.dtype)

            val = eval_outputs([apb])
            self.assertTrue(val.shape == (3, 2))
            if op is add:
                self.assertTrue(_is_dense_variable(apb))
                self.assertTrue(numpy.all(val == (a + b)))
                ans = numpy.array([[1., 2], [3, 4], [5, 6]])
                self.assertTrue(numpy.all(val == ans))
            elif op is mul:
                self.assertTrue(_is_sparse_variable(apb))
                self.assertTrue(numpy.all(val.todense() == (b.multiply(a))))
                self.assertTrue(numpy.all(val.todense() == numpy.array([[1, 0],
[9, 0], [0, 36]])))

    def _testDS(self, op, array1=numpy.array([[1., 0], [3, 0], [0, 6]]),
                array2=numpy.asarray([[0, 2.], [0, 4], [5, 0]])):
        for mtype in _mtypes:
            a = mtype(array1)
            aR = as_sparse_variable(a)
            self.assertFalse(aR.data is a)
            self.assertTrue(_is_sparse(a))
            self.assertTrue(_is_sparse_variable(aR))

            b = numpy.asarray(array2)
            bR = tensor.as_tensor_variable(b)
            self.assertFalse(bR.data is b)
            self.assertTrue(_is_dense(b))
            self.assertTrue(_is_dense_variable(bR))

            apb = op(aR, bR)

            self.assertTrue(apb.type.dtype == aR.type.dtype, apb.type.dtype)
            self.assertTrue(apb.type.dtype == bR.type.dtype, apb.type.dtype)

            val = eval_outputs([apb])
            self.assertTrue(val.shape == (3, 2))
            if op is add:
                self.assertTrue(_is_dense_variable(apb))
                self.assertTrue(numpy.all(val == (a + b)))
                ans = numpy.array([[1., 2], [3, 4], [5, 6]])
                self.assertTrue(numpy.all(val == ans))
            elif op is mul:
                self.assertTrue(_is_sparse_variable(apb))
                ans = numpy.array([[1, 0], [9, 0], [0, 36]])
                self.assertTrue(numpy.all(val.todense() == (a.multiply(b))))
                self.assertTrue(numpy.all(val.todense() == ans))

    def test_upcast(self):
        array1 = numpy.array([[1, 0], [3, 0], [0, 6]], dtype='float32')
        array2 = numpy.array([[1, 0], [3, 0], [0, 6]], dtype='int32')
        array3 = numpy.array([[1, 0], [3, 0], [0, 6]], dtype='int8')

        # AddSS and MulSS
        for mtype in _mtypes:
            a = mtype(array1)
            aR = as_sparse_variable(a)
            b = mtype(array2)
            bR = as_sparse_variable(b)
            c = mtype(array3)
            cR = as_sparse_variable(c)

            # Ops that do not upcast
            self.assertRaises(NotImplementedError, add, aR, bR)
            self.assertRaises(NotImplementedError, add, bR, aR)
            self.assertRaises(NotImplementedError, add, bR, cR)
            self.assertRaises(NotImplementedError, add, cR, bR)
            self.assertRaises(NotImplementedError, add, aR, cR)
            self.assertRaises(NotImplementedError, add, cR, aR)

            self.assertRaises(NotImplementedError, mul, aR, bR)
            self.assertRaises(NotImplementedError, mul, bR, aR)
            self.assertRaises(NotImplementedError, mul, bR, cR)
            self.assertRaises(NotImplementedError, mul, cR, bR)
            self.assertRaises(NotImplementedError, mul, aR, cR)
            self.assertRaises(NotImplementedError, mul, cR, aR)

        # AddSD and MulSD
        for mtype in _mtypes:
            a = mtype(array1)
            a_sv = as_sparse_variable(a)
            a_dv = tensor.as_tensor_variable(array1)
            b = mtype(array2)
            b_sv = as_sparse_variable(b)
            b_dv = tensor.as_tensor_variable(array2)
            c = mtype(array3)
            c_sv = as_sparse_variable(c)
            c_dv = tensor.as_tensor_variable(array3)

            # add does not upcast
            self.assertRaises(NotImplementedError, add, a_sv, b_dv)
            self.assertRaises(NotImplementedError, add, b_sv, a_dv)
            self.assertRaises(NotImplementedError, add, b_sv, c_dv)
            self.assertRaises(NotImplementedError, add, c_sv, b_dv)
            self.assertRaises(NotImplementedError, add, a_sv, c_dv)
            self.assertRaises(NotImplementedError, add, c_sv, a_dv)

            # mul may upcast the dense input if needed
            if (config.cast_policy in ('custom', 'numpy') or
                (config.cast_policy == 'numpy+floatX' and
                 config.floatX == 'float64')):
                # The result should be a float64 (not implemented).
                self.assertRaises(NotImplementedError, mul, a_sv, b_dv)
            elif (config.cast_policy == 'numpy+floatX' and
                  config.floatX == 'float32'):
                # The result should be a float32.
                assert mul(a_sv, b_dv).dtype == 'float32'
            else:
                raise NotImplementedError()
            self.assertRaises(NotImplementedError, mul, b_sv, a_dv)
            assert mul(b_sv, c_dv).dtype == 'int32'
            self.assertRaises(NotImplementedError, mul, c_sv, b_dv)
            assert mul(a_sv, c_dv).dtype == 'float32'
            self.assertRaises(NotImplementedError, mul, c_sv, a_dv)


class T_conversion(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    if 0:
        def test0(self):
            a = tensor.as_tensor_variable(numpy.random.rand(5))
            s = csc_from_dense(a)
            val = eval_outputs([s])
            self.assertTrue(str(val.dtype) == 'float64')
            self.assertTrue(val.format == 'csc')

    if 0:
        def test1(self):
            a = tensor.as_tensor_variable(numpy.random.rand(5))
            s = csr_from_dense(a)
            val = eval_outputs([s])
            self.assertTrue(str(val.dtype) == 'float64')
            self.assertTrue(val.format == 'csr')

    if 1:
        def test2(self):
            #call dense_from_sparse
            for t in _mtypes:
                s = t(scipy.sparse.identity(5))
                d = dense_from_sparse(s)
                # s should be copied into the graph as a constant
                s[0, 0] = 3.0  # changes s, but not the copy
                val = eval_outputs([d])
                return
                self.assertTrue(str(val.dtype) == s.dtype)
                self.assertTrue(numpy.all(val[0] == [1, 0, 0, 0, 0]))

    @staticmethod
    def check_format_ndim(format, ndim):
        x = tensor.tensor(
                dtype=config.floatX,
                broadcastable=([False] * ndim),
                name='x')

        s = SparseFromDense(format)(x)
        s_m = - s
        d = dense_from_sparse(s_m)
        c = d.sum()
        g = tensor.grad(c, x)
        f = theano.function([x], [s, g])
        f(numpy.array(0, dtype=config.floatX, ndmin=ndim))
        f(numpy.array(7, dtype=config.floatX, ndmin=ndim))

    def test_format_ndim(self):
        for format in 'csc', 'csr':
            for ndim in 0, 1, 2:
                self.check_format_ndim(format, ndim)

            self.assertRaises(TypeError, self.check_format_ndim, format, 3)
            self.assertRaises(TypeError, self.check_format_ndim, format, 4)


class test_structureddot(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_structureddot_csc_grad(self):

        #shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = sp.csc_matrix(random_lil((4, 3), 'float32', 3))

        mat = numpy.asarray(numpy.random.randn(3, 2), 'float32')

        verify_grad_sparse(structured_dot, [spmat, mat], structured=True)

        def buildgraph_T(spmat, mat):
            return structured_dot(mat.T, spmat.T)

        verify_grad_sparse(buildgraph_T, [spmat, mat], structured=True)

    def test_structureddot_csr_grad(self):

        #shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = sp.csr_matrix(random_lil((4, 3), 'float64', 3))

        mat = numpy.asarray(numpy.random.randn(3, 2), 'float64')

        verify_grad_sparse(structured_dot, [spmat, mat], structured=True)

        def buildgraph_T(spmat, mat):
            return structured_dot(mat.T, spmat.T)

        verify_grad_sparse(buildgraph_T, [spmat, mat], structured=True)

    def test_upcast(self):

        typenames = ('float32', 'int64', 'int8', 'int32',
                     'int16', 'float64', 'complex64', 'complex128')
        for dense_dtype in typenames:
            for sparse_dtype in typenames:
                correct_dtype = theano.scalar.upcast(sparse_dtype, dense_dtype)
                a = SparseType('csc', dtype=sparse_dtype)()
                b = tensor.matrix(dtype=dense_dtype)
                d = structured_dot(a, b)
                assert d.type.dtype == correct_dtype

                # compile and run a function

                f = theano.function([a, b], d)

                M, N, K, nnz = (4, 3, 5, 3)
                spmat = sp.csc_matrix(random_lil((M, N), sparse_dtype, nnz))
                # the following madness is necessary to workaround
                # an intc vs. int32 bug.
                # The lil makes an intc on my computer when sparse_dtype
                # is int32.
                spmat.dtype = numpy.dtype(sparse_dtype)
                mat = numpy.asarray(numpy.random.randn(N, K) * 9,
                                    dtype=dense_dtype)
                print 'DTYPES', sparse_dtype, dense_dtype
                print 'sym types', a.type, b.type
                print 'dtype strings', spmat.dtype, mat.dtype
                print 'numpy dtype num', mat.dtype.num
                print 'scipy dtype num', spmat.data.dtype.num
                theano_result = f(spmat, mat)
                scipy_result = spmat * mat
                assert theano_result.shape == scipy_result.shape
                assert theano_result.dtype == scipy_result.dtype
                assert _allclose(theano_result, scipy_result)

    def test_opt_unpack(self):
        #
        # Test that a graph involving
        # structured_dot(assembled_csc_matrix) is optimized to be just
        # a structured_dot_csc Op and no assembly of a csc_matrix.
        #
        # The optimization from structured_dot -> structured_dot_csc
        # is currently disabled, So this test is not expected to pass

        return
        #
        kerns = tensor.Tensor(dtype='int64', broadcastable=[False])('kerns')
        spmat = sp.lil_matrix((4, 6), dtype='int64')
        for i in range(5):
            # set non-zeros in random locations (row x, col y)
            x = numpy.floor(numpy.random.rand() * spmat.shape[0])
            y = numpy.floor(numpy.random.rand() * spmat.shape[1])
            spmat[x, y] = numpy.random.rand() * 10
        spmat = sp.csc_matrix(spmat)

        images = tensor.Tensor(dtype='float32',
                               broadcastable=[False, False])(
            'images')

        cscmat = CSC(kerns, spmat.indices[:spmat.size],
                     spmat.indptr, spmat.shape)
        f = theano.function([kerns, images], structured_dot(cscmat, images.T))

        sdcscpresent = False
        for node in f.maker.env.toposort():
            print node.op
            assert not isinstance(node.op, CSM)
            assert not isinstance(node.op, CSMProperties)
            if isinstance(f.maker.env.toposort()[1].op, StructuredDotCSC):
                sdcscpresent = True
        assert sdcscpresent

        kernvals = numpy.array(spmat.data[:spmat.size])
        #print 'kdtype', kernvals.dtype, kernvals.shape,
        #print kernvals.ndim, kernvals.dtype.num
        #print 'type of kernvals = ', kernvals.dtype
        bsize = 3
        imvals = 1.0 * numpy.array(numpy.arange(bsize * spmat.shape[1]).\
                reshape(bsize, spmat.shape[1]), dtype='float32')
        outvals = f(kernvals, imvals)
        print outvals

    def test_dot_sparse_sparse(self):
        #test dot for 2 input sparse matrix
        sparse_dtype = 'float64'
        sp_mat = {'csc': sp.csc_matrix,
                  'csr': sp.csr_matrix}

        for sparse_format_a in ['csc', 'csr']:
            for sparse_format_b in ['csc', 'csr']:
                a = SparseType(sparse_format_a, dtype=sparse_dtype)()
                b = SparseType(sparse_format_b, dtype=sparse_dtype)()
                d = theano.dot(a, b)
                f = theano.function([a, b], theano.Out(d, borrow=True))
                topo = f.maker.env.toposort()
                for M, N, K, nnz in [(4, 3, 2, 3),
                                  (40, 30, 20, 3),
                                  (40, 30, 20, 30),
                                  (400, 3000, 200, 6000),
                                  ]:
                    a_val = sp_mat[sparse_format_a](
                        random_lil((M, N), sparse_dtype, nnz))
                    b_val = sp_mat[sparse_format_b](
                        random_lil((N, K), sparse_dtype, nnz))
                    f(a_val, b_val)

    def test_csc_correct_output_faster_than_scipy(self):
        sparse_dtype = 'float64'
        dense_dtype = 'float64'

        a = SparseType('csc', dtype=sparse_dtype)()
        b = tensor.matrix(dtype=dense_dtype)
        d = theano.dot(a, b)
        f = theano.function([a, b], theano.Out(d, borrow=True))

        for M, N, K, nnz in [(4, 3, 2, 3),
                (40, 30, 20, 3),
                (40, 30, 20, 30),
                (400, 3000, 200, 6000),
                ]:
            spmat = sp.csc_matrix(random_lil((M, N), sparse_dtype, nnz))
            mat = numpy.asarray(numpy.random.randn(N, K), dense_dtype)
            theano_times = []
            scipy_times = []
            for i in xrange(5):
                t0 = time.time()
                theano_result = f(spmat, mat)
                t1 = time.time()
                scipy_result = spmat * mat
                t2 = time.time()

                theano_times.append(t1 - t0)
                scipy_times.append(t2 - t1)

            theano_time = numpy.min(theano_times)
            scipy_time = numpy.min(scipy_times)

            speedup = scipy_time / theano_time
            print scipy_times
            print theano_times
            print ('M=%(M)s N=%(N)s K=%(K)s nnz=%(nnz)s theano_time'
                   '=%(theano_time)s speedup=%(speedup)s') % locals()

            # fail if Theano is slower than scipy by more than a certain amount
            overhead_tol = 0.003  # seconds overall
            overhead_rtol = 1.2  # times as long
            self.assertTrue(numpy.allclose(theano_result, scipy_result))
            if not theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
                self.assertFalse(theano_time > overhead_rtol * scipy_time +
                                 overhead_tol)

    def test_csr_correct_output_faster_than_scipy(self):

        #contrast with test_grad, we put csr in float32, csc in float64

        sparse_dtype = 'float32'
        dense_dtype = 'float32'

        a = SparseType('csr', dtype=sparse_dtype)()
        b = tensor.matrix(dtype=dense_dtype)
        d = theano.dot(a, b)
        f = theano.function([a, b], d)

        for M, N, K, nnz in [(4, 3, 2, 3),
                (40, 30, 20, 3),
                (40, 30, 20, 30),
                (400, 3000, 200, 6000),
                ]:
            spmat = sp.csr_matrix(random_lil((M, N), sparse_dtype, nnz))
            mat = numpy.asarray(numpy.random.randn(N, K), dense_dtype)
            t0 = time.time()
            theano_result = f(spmat, mat)
            t1 = time.time()
            scipy_result = spmat * mat
            t2 = time.time()

            theano_time = t1 - t0
            scipy_time = t2 - t1
            #print theano_result
            #print scipy_result
            print 'theano took', theano_time,
            print 'scipy took', scipy_time
            overhead_tol = 0.002  # seconds
            overhead_rtol = 1.1  # times as long
            self.assertTrue(numpy.allclose(theano_result, scipy_result))
            if not theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
                self.assertFalse(theano_time > overhead_rtol * scipy_time +
                                 overhead_tol)


class DotTests(unittest.TestCase):
    def setUp(self):
        x_size = (10, 1000)
        y_size = (1000, 10000)

        self.x_csr = scipy.sparse.csr_matrix(
            numpy.random.binomial(1, 0.5, x_size), dtype=theano.config.floatX)
        self.x_csc = scipy.sparse.csc_matrix(
            numpy.random.binomial(1, 0.5, x_size), dtype=theano.config.floatX)
        self.y = numpy.asarray(numpy.random.uniform(-1, 1, y_size),
                               dtype=theano.config.floatX)
        self.y_csr = scipy.sparse.csr_matrix(
            numpy.random.binomial(1, 0.5, y_size), dtype=theano.config.floatX)
        self.y_csc = scipy.sparse.csc_matrix(
            numpy.random.binomial(1, 0.5, y_size), dtype=theano.config.floatX)

    def test_csr_dense(self):
        x = theano.sparse.csr_matrix('x')
        y = theano.tensor.matrix('y')

        f_a = theano.function([x, y], theano.sparse.dot(x, y))
        f_b = lambda x, y: x * y

        assert _allclose(f_a(self.x_csr, self.y), f_b(self.x_csr, self.y))

        # Test infer_shape
        f_a = theano.function([x, y], theano.sparse.dot(x, y).shape)
        f_b = lambda x, y: (x * y).shape
        assert numpy.all(f_a(self.x_csr, self.y) == f_b(self.x_csr, self.y))
        topo = f_a.maker.env.toposort()
        if theano.config.mode != 'FAST_COMPILE':
            nb = 0
        else:
            nb = 1
        assert sum([isinstance(node.op, (Dot, Usmm, UsmmCscDense))
                    for node in topo]) == nb

    def test_csc_dense(self):
        x = theano.sparse.csc_matrix('x')
        y = theano.tensor.matrix('y')

        f_a = theano.function([x, y], theano.sparse.dot(x, y))
        f_b = lambda x, y: x * y

        assert _allclose(f_a(self.x_csc, self.y), f_b(self.x_csc, self.y))

        # Test infer_shape
        f_a = theano.function([x, y], theano.sparse.dot(x, y).shape)
        f_b = lambda x, y: (x * y).shape
        assert numpy.all(f_a(self.x_csc, self.y) == f_b(self.x_csc, self.y))
        topo = f_a.maker.env.toposort()
        if theano.config.mode != 'FAST_COMPILE':
            nb = 0
        else:
            nb = 1
        assert sum([isinstance(node.op, (Dot, Usmm, UsmmCscDense))
                    for node in topo]) == nb

    def test_sparse_sparse(self):
        for d1, d2 in [('float32', 'float32'),
                       ('float32', 'float64'),
                       ('float64', 'float32'),
                       ('float64', 'float64'),
                       ('float32', 'int16'),
                       ('float32', 'complex64'),
                       ]:
            for x_f, y_f in [('csc', 'csc'),
                             ('csc', 'csr'),
                             ('csr', 'csc'),
                             ('csr', 'csr'),
                             ]:
                x = theano.sparse.SparseType(format=x_f, dtype=d1)('x')
                y = theano.sparse.SparseType(format=x_f, dtype=d2)('x')

                f_a = theano.function([x, y], theano.sparse.dot(x, y))
                f_b = lambda x, y: x * y

                vx = getattr(self, 'x_' + x_f).astype(d1)
                vy = getattr(self, 'y_' + y_f).astype(d2)
                assert _allclose(f_a(vx, vy), f_b(vx, vy).toarray())

                # Test infer_shape
                f_a = theano.function([x, y], theano.sparse.dot(x, y).shape)
                f_b = lambda x, y: (x * y).shape
                assert numpy.all(f_a(vx, vy) == f_b(vx, vy))
                topo = f_a.maker.env.toposort()
                if theano.config.mode != 'FAST_COMPILE':
                    nb = 0
                else:
                    nb = 1
                assert sum([isinstance(node.op, (Dot, Usmm, UsmmCscDense))
                            for node in topo]) == nb


class UsmmTests(unittest.TestCase):
    """ Test the Usmm and UsmmCscDense class and related optimization """
    def setUp(self):
        x_size = (10, 100)
        y_size = (100, 200)
        z_size = (x_size[0], y_size[1])

        self.rng = numpy.random.RandomState(seed=utt.fetch_seed())
        self.x = numpy.asarray(self.rng.binomial(1, 0.5, x_size),
                               dtype=theano.config.floatX)
        self.y = numpy.asarray(self.rng.uniform(-1, 1, y_size),
                               dtype=theano.config.floatX)
        self.z = numpy.asarray(self.rng.uniform(-1, 1, z_size),
                               dtype=theano.config.floatX)

    def test(self):
        def mat(format, name, dtype):
            if format == 'dense':
                return theano.tensor.matrix(name, dtype=dtype)
            else:
                return theano.sparse.matrix(format, name, dtype=dtype)

        params = product(*([['float32', 'float64', 'int16', 'complex64']] * 4 +
                           [['dense', 'csc', 'csr']] * 2))

        # All test are too slow, so we randomly take 100 of them.
        # The buildbot change the seed, so we will finish by running them all.
        # As of this writing they where all passing.
        #params = self.rng.permutation(list(params))[:500]

        for dtype1, dtype2, dtype3, dtype4, format1, format2 in params:
            if format1 == 'dense' and format2 == 'dense':
                # Usmm won't be used!
                continue
            x = mat(format1, 'x', dtype1)
            y = mat(format2, 'y', dtype2)
            a = theano.tensor.scalar('a', dtype=dtype3)
            z = theano.shared(numpy.asarray(self.z, dtype=dtype4).copy())

            f_b = lambda z, a, x, y: z - a * (x * y)
            x_data = numpy.asarray(self.x, dtype=dtype1)
            if format1 != 'dense':
                x_data = as_sparse_format(x_data, format1)
            y_data = numpy.asarray(self.y, dtype=dtype2)
            if format2 != 'dense':
                y_data = as_sparse_format(y_data, format2)
            a_data = numpy.asarray(1.5, dtype=dtype3)
            z_data = numpy.asarray(self.z, dtype=dtype4)

            f_b_out = f_b(z_data, a_data, x_data, y_data)

            # Can it work inplace?
            inplace = dtype4 == theano.scalar.upcast(dtype1, dtype2, dtype3)

            # To make it easier to check the toposort
            mode = theano.compile.mode.get_default_mode().excluding('fusion')

            if inplace:
                updates = {z: z - a * theano.sparse.dot(x, y)}
                f_a = theano.function([a, x, y], [],
                                      updates=updates,
                                      mode=mode)
                f_a(a_data, x_data, y_data)
                f_a_out = z.get_value(borrow=True)
            else:
                f_a = theano.function([a, x, y],
                                      z - a * theano.sparse.dot(x, y),
                                      mode=mode)
                # In DebugMode there is a strange difference with complex
                # So we raise a little the threshold a little.
                try:
                    orig_atol = theano.tensor.basic.float64_atol
                    orig_rtol = theano.tensor.basic.float64_rtol
                    theano.tensor.basic.float64_atol = 1e-7
                    theano.tensor.basic.float64_rtol = 1e-6
                    f_a_out = f_a(a_data, x_data, y_data)
                finally:
                    theano.tensor.basic.float64_atol = orig_atol
                    theano.tensor.basic.float64_rtol = orig_rtol

            assert _allclose(f_a_out, f_b_out, rtol=1e-5)
            topo = f_a.maker.env.toposort()
            up = theano.scalar.upcast(dtype1, dtype2, dtype3, dtype4)

            fast_compile = theano.config.mode == "FAST_COMPILE"

            if (y.type.dtype == up and format1 == 'csc' and format2 == 'dense'
                and not fast_compile) and up in ('float32', 'float64'):
                # The op UsmmCscDense should be inserted
                assert (sum([isinstance(node.op, tensor.Elemwise) and
                             isinstance(node.op.scalar_op,
                                        theano.scalar.basic.Cast)
                             for node in topo]) == len(topo) - 5)
                new_topo = []
                for node in topo:
                    if not (isinstance(node.op, tensor.Elemwise) and \
                       isinstance(node.op.scalar_op,
                                  theano.scalar.basic.Cast)):
                        new_topo.append(node)
                topo = new_topo
                assert len(topo) == 5, topo
                # Usmm is tested at the same time in debugmode
                # Check if the optimization local_usmm and local_usmm_csx is
                # applied
                assert isinstance(topo[0].op,
                                  theano.sparse.basic.CSMProperties)
                assert isinstance(topo[1].op, theano.tensor.DimShuffle)
                assert isinstance(topo[2].op, theano.tensor.Subtensor)
                assert topo[3].op == theano.tensor.neg
                assert isinstance(topo[4].op, theano.sparse.UsmmCscDense)
                if inplace:
                    assert topo[4].op.inplace
            elif not fast_compile:
                # The op Usmm should be inserted
                assert len(topo) == 3, topo
                assert isinstance(topo[0].op, theano.tensor.DimShuffle)
                assert topo[1].op == theano.tensor.neg
                assert isinstance(topo[2].op, theano.sparse.Usmm)

    def test_infer_shape(self):
        def mat(format, name, dtype):
            if format == 'dense':
                return theano.tensor.matrix(name, dtype=dtype)
            else:
                return theano.sparse.matrix(format, name, dtype=dtype)

        params = [('float32', 'float64', 'int16', 'complex64', 'csc', 'dense'),
                  ('float32', 'float64', 'int16', 'complex64', 'csr', 'dense')]
        for dtype1, dtype2, dtype3, dtype4, format1, format2 in params:
            if format1 == 'dense' and format2 == 'dense':
                # Usmm won't be used!
                continue
            x = mat(format1, 'x', dtype1)
            y = mat(format2, 'y', dtype2)
            a = theano.tensor.scalar('a', dtype=dtype3)
            z = theano.shared(numpy.asarray(self.z, dtype=dtype4).copy())

            f_b = lambda z, a, x, y: z - a * (x * y)
            x_data = numpy.asarray(self.x, dtype=dtype1)
            if format1 != 'dense':
                x_data = as_sparse_format(x_data, format1)
            y_data = numpy.asarray(self.y, dtype=dtype2)
            if format2 != 'dense':
                y_data = as_sparse_format(y_data, format2)
            a_data = numpy.asarray(1.5, dtype=dtype3)
            z_data = numpy.asarray(self.z, dtype=dtype4)

            f_b_out = f_b(z_data, a_data, x_data, y_data)

            # Can it work inplace?
            inplace = dtype4 == theano.scalar.upcast(dtype1, dtype2, dtype3)

            # To make it easier to check the toposort
            mode = theano.compile.mode.get_default_mode().excluding('fusion')

            # test infer_shape of Dot got applied
            f_shape = theano.function([a, x, y],
                                      (z - a * theano.sparse.dot(x, y)).shape,
                                      mode=mode)
            assert all(f_shape(a_data, x_data, y_data) == f_b_out.shape)
            topo = f_shape.maker.env.toposort()
            if theano.config.mode != 'FAST_COMPILE':
                nb = 0
            else:
                nb = 1
            assert sum([isinstance(node.op, (Dot, Usmm, UsmmCscDense))
                        for node in topo]) == nb


class test_zeros_like(unittest.TestCase):
    def test(self):
        x = theano.sparse.csr_matrix()
        f = theano.function([x], theano.sparse.sp_zeros_like(x))
        vx = scipy.sparse.csr_matrix(numpy.asarray(
                numpy.random.binomial(1, 0.5, (100, 100)),
                dtype=theano.config.floatX))

        fx = f(vx)

        assert fx.nnz == 0
        assert fx.shape == vx.shape


def test_shape_i():
    sparse_dtype = 'float32'

    a = SparseType('csr', dtype=sparse_dtype)()
    f = theano.function([a], a.shape[1])
    assert f(sp.csr_matrix(random_lil((100, 10), sparse_dtype, 3))) == 10


def test_shape():
    # Test that getting the shape of a sparse variable
    # does not actually create a dense tensor in the process.
    sparse_dtype = 'float32'

    a = SparseType('csr', dtype=sparse_dtype)()
    f = theano.function([a], a.shape)
    assert numpy.all(f(sp.csr_matrix(random_lil((100, 10), sparse_dtype, 3)))
                     == (100, 10))
    if theano.config.mode != 'FAST_COMPILE':
        topo = f.maker.env.toposort()
        assert len(topo) == 3
        assert isinstance(topo[0].op, tensor.opt.Shape_i)
        assert isinstance(topo[1].op, tensor.opt.Shape_i)
        assert isinstance(topo[2].op, tensor.opt.MakeVector)


def test_may_share_memory():
    a = scipy.sparse.csc_matrix(scipy.sparse.eye(5, 3))
    b = scipy.sparse.csc_matrix(scipy.sparse.eye(4, 3))
    as_ar = lambda a: theano._asarray(a, dtype='int32')
    for a_, b_, rep in [(a, a, True),
                        (b, b, True),
                        (a, b, False),
                        (a, a.data, True),
                        (a, a.indptr, True),
                        (a, a.indices, True),
                        (a, as_ar(a.shape), False),
                        (a.data, a, True),
                        (a.indptr, a, True),
                        (a.indices, a, True),
                        (as_ar(a.shape), a, False),
                        (b, b.data, True),
                        (b, b.indptr, True),
                        (b, b.indices, True),
                        (b, as_ar(b.shape), False),
                        (b.data, b, True),
                        (b.indptr, b, True),
                        (b.indices, b, True),
                        (as_ar(b.shape), b, False),
                        (b.data, a, False),
                        (b.indptr, a, False),
                        (b.indices, a, False),
                        (as_ar(b.shape), a, False),
                        ]:

        assert SparseType.may_share_memory(a_, b_) == rep


def test_sparse_shared_memory():
    # Note : There are no inplace ops on sparse matrix yet. If one is
    # someday implemented, we could test it here.
    a = random_lil((3, 4), 'float32', 3).tocsr()
    m1 = random_lil((4, 4), 'float32', 3).tocsr()
    m2 = random_lil((4, 4), 'float32', 3).tocsr()
    x = SparseType('csr', dtype='float32')()
    y = SparseType('csr', dtype='float32')()

    sdot = theano.sparse.structured_dot
    z = sdot(x * 3, m1) + sdot(y * 2, m2)

    f = theano.function([theano.In(x, mutable=True),
                         theano.In(y, mutable=True)], z, mode='FAST_RUN')

    def f_(x, y, m1=m1, m2=m2):
        return numpy.dot(x * 3, m1) + numpy.dot(y * 2, m2)

    assert SparseType.may_share_memory(a, a)  # This is trivial
    result = f(a, a)
    result_ = f_(a, a)
    assert (result_.todense() == result.todense()).all()


def test_size():
    """
    Ensure the `size` attribute of sparse matrices behaves as in numpy.
    """
    for sparse_type in ('csc_matrix', 'csr_matrix'):
        x = getattr(theano.sparse, sparse_type)()
        y = getattr(scipy.sparse, sparse_type)((5, 7)).astype(config.floatX)
        get_size = theano.function([x], x.size)

        def check():
            assert y.size == get_size(y)
        # We verify that the size is correctly updated as we store more data
        # into the sparse matrix (including zeros).
        check()
        y[0, 0] = 1
        check()
        y[0, 1] = 0
        check()


class Test_getitem(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(utt.fetch_seed())

    def test_GetItem2D(self):
        sparse_formats = ('csc', 'csr')
        for format in sparse_formats:
            x = theano.sparse.matrix(format, name='x')
            a = theano.tensor.iscalar('a')
            b = theano.tensor.iscalar('b')
            c = theano.tensor.iscalar('c')
            d = theano.tensor.iscalar('d')

            # index
            m = 1
            n = 5
            p = 10
            q = 15

            vx = as_sparse_format(self.rng.binomial(1, 0.5, (100, 97)),
                                  format).astype(theano.config.floatX)
            #mode_no_debug = theano.compile.mode.get_default_mode()
            #if isinstance(mode_no_debug, theano.compile.DebugMode):
            #    mode_no_debug = 'FAST_RUN'
            f1 = theano.function([x, a, b, c, d], x[a:b, c:d])
            r1 = f1(vx, m, n, p, q)
            t1 = vx[m:n, p:q]
            assert r1.shape == t1.shape
            assert numpy.all(t1.toarray() == r1.toarray())

            """"
            Important: based on a discussion with both Fred and James
            The following indexing methods is not supported because the rval
            would be a sparse matrix rather than a sparse vector, which is a
            deviation from numpy indexing rule. This decision is made largely
            for keeping the consistency between numpy and theano.

            f2 = theano.function([x, a, b, c], x[a:b, c])
            r2 = f2(vx, m, n, p)
            t2 = vx[m:n, p]
            assert r2.shape == t2.shape
            assert numpy.all(t2.toarray() == r2.toarray())

            f3 = theano.function([x, a, b, c], x[a, b:c])
            r3 = f3(vx, m, n, p)
            t3 = vx[m, n:p]
            assert r3.shape == t3.shape
            assert numpy.all(t3.toarray() == r3.toarray())

            f5 = theano.function([x], x[1:2,3])
            r5 = f5(vx)
            t5 = vx[1:2, 3]
            assert r5.shape == t5.shape
            assert numpy.all(r5.toarray() == t5.toarray())

            f7 = theano.function([x], x[50])
            r7 = f7(vx)
            t7 = vx[50]
            assert r7.shape == t7.shape
            assert numpy.all(r7.toarray() == t7.toarray())
            """

            f4 = theano.function([x, a, b], x[a:b])
            r4 = f4(vx, m, n)
            t4 = vx[m:n]
            assert r4.shape == t4.shape
            assert numpy.all(t4.toarray() == r4.toarray())
            #-----------------------------------------------------------
            # test cases using int indexing instead of theano variable

            f6 = theano.function([x], x[1:10, 10:20])
            r6 = f6(vx)
            t6 = vx[1:10, 10:20]
            assert r6.shape == t6.shape
            assert numpy.all(r6.toarray() == t6.toarray())

            #----------------------------------------------------------
            # test cases with indexing both with theano variable and int
            f8 = theano.function([x, a, b], x[a:b, 10:20])
            r8 = f8(vx, m, n)
            t8 = vx[m:n, 10:20]
            assert r8.shape == t8.shape
            assert numpy.all(r8.toarray() == t8.toarray())

            f9 = theano.function([x, a, b], x[1:a, 1:b])
            r9 = f9(vx, p, q)
            t9 = vx[1:p, 1:q]
            assert r9.shape == t9.shape
            assert numpy.all(r9.toarray() == t9.toarray())

            #-----------------------------------------------------------
            # Test mixing None and variables
            f10 = theano.function([x, a, b], x[:a, :b])
            r10 = f10(vx, p, q)
            t10 = vx[:p, :q]
            assert r10.shape == t10.shape
            assert numpy.all(r10.toarray() == t10.toarray())

            f11 = theano.function([x, a], x[:, a:])
            r11 = f11(vx, p)
            t11 = vx[:, p:]
            assert r11.shape == t11.shape
            assert numpy.all(r11.toarray() == t11.toarray())

            #------------------------------------------------------------
            # Invalid things
            # The syntax is a bit awkward because assertRaises forbids
            # the [] shortcut for getitem.
            # x[a:b] is not accepted because we don't have sparse vectors
            self.assertRaises(NotImplementedError,
                    x.__getitem__, (slice(a, b), c))

            # x[a:b:step, c:d] is not accepted because scipy silently drops
            # the step (!)
            self.assertRaises(ValueError,
                    x.__getitem__, (slice(a, b, -1), slice(c, d)))
            self.assertRaises(ValueError,
                    x.__getitem__, (slice(a, b), slice(c, d, 2)))

            # Advanced indexing is not supported
            self.assertRaises(ValueError,
                    x.__getitem__, (tensor.ivector('l'), slice(a, b)))

            # Indexing with random things is not supported either
            self.assertRaises(ValueError,
                    x.__getitem__, slice(tensor.fscalar('f'), None))
            self.assertRaises(ValueError,
                    x.__getitem__, (slice(None), slice([1, 3, 4], None)))

    def test_GetItemScalar(self):
        sparse_formats = ('csc', 'csr')
        for format in sparse_formats:
            x = theano.sparse.csc_matrix('x')
            a = theano.tensor.iscalar()
            b = theano.tensor.iscalar()

            m = 50
            n = 42

            vx = as_sparse_format(self.rng.binomial(1, 0.5, (97, 100)),
                                 format).astype(theano.config.floatX)

            f1 = theano.function([x, a, b], x[a, b])
            r1 = f1(vx, 10, 10)
            t1 = vx[10, 10]
            assert r1.shape == t1.shape
            assert numpy.all(t1 == r1)

            f2 = theano.function([x, a], x[50, a])
            r2 = f2(vx, m)
            t2 = vx[50, m]
            assert r2.shape == t2.shape
            assert numpy.all(t2 == r2)

            f3 = theano.function([x, a], x[a, 50])
            r3 = f3(vx, m)
            t3 = vx[m, 50]
            assert r3.shape == t3.shape
            assert numpy.all(t3 == r3)

            f4 = theano.function([x], x[50, 42])
            r4 = f4(vx)
            t4 = vx[m, n]
            assert r3.shape == t3.shape
            assert numpy.all(t4 == r4)


import theano.tensor.tests.test_sharedvar
test_shared_options = theano.tensor.tests.test_sharedvar.makeSharedTester(
    shared_constructor_=theano.sparse.shared,
    dtype_='float64',
    get_value_borrow_true_alias_=True,
    shared_borrow_true_alias_=True,
    set_value_borrow_true_alias_=True,
    set_value_inplace_=False,
    set_cast_value_inplace_=False,
    shared_constructor_accept_ndarray_=False,
    internal_type_=scipy.sparse.csc_matrix,
    test_internal_type_=scipy.sparse.issparse,
    theano_fct_=lambda a: dense_from_sparse(a * 2.),
    ref_fct_=lambda a: numpy.asarray((a * 2).todense()),
    cast_value_=scipy.sparse.csr_matrix,
    name='test_shared_options',
    )


if __name__ == '__main__':
    unittest.main()
