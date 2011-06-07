import time
import unittest

from nose.plugins.skip import SkipTest
import numpy
try:
    import scipy.sparse as sp
    import scipy.sparse
except ImportError:
    pass#the variable enable_sparse will be used to disable the test file.

import theano
from theano import compile
from theano.sparse import enable_sparse
if enable_sparse == False:
    raise SkipTest('Optional package sparse disabled')

from theano.sparse.basic import _is_dense, _is_sparse, _is_dense_variable, _is_sparse_variable
from theano.sparse.basic import _mtypes
from theano.sparse import as_sparse_variable, CSC, CSR, CSM, CSMProperties, SparseType, StructuredDotCSC
from theano.sparse import add, mul, structured_dot, transpose
from theano.sparse import csc_from_dense, csr_from_dense, dense_from_sparse

from theano.tests import unittest_tools as utt
from theano import tensor
from theano.tensor.basic import _allclose


def eval_outputs(outputs):
    return compile.function([], outputs)()[0]

def random_lil(shape, dtype, nnz):
    rval = sp.lil_matrix(shape, dtype=dtype)
    huge = 2**30
    for k in range(nnz):
        # set non-zeros in random locations (row x, col y)
        idx = numpy.random.random_integers(huge,size=len(shape)) % shape
        value = numpy.random.rand()
        #if dtype *int*, value will always be zeros!
        if "int" in dtype:
            value = int(value*100)
        rval.__setitem__(
                idx,
                value)
    return rval

class T_transpose(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_transpose_csc(self):
        sp = scipy.sparse.csc_matrix(scipy.sparse.eye(5,3))
        a = as_sparse_variable(sp)
        self.assertFalse(a.data is sp)
        self.assertTrue(a.data.shape == (5,3))
        self.assertTrue(a.type.dtype == 'float64', a.type.dtype)
        self.assertTrue(a.type.format == 'csc', a.type.format)
        ta = transpose(a)
        self.assertTrue(ta.type.dtype == 'float64', ta.type.dtype)
        self.assertTrue(ta.type.format == 'csr', ta.type.format)

        vta = eval_outputs([ta])
        self.assertTrue(vta.shape == (3,5))
    def test_transpose_csr(self):
        a = as_sparse_variable(scipy.sparse.csr_matrix(scipy.sparse.eye(5,3)))
        self.assertTrue(a.data.shape == (5,3))
        self.assertTrue(a.type.dtype == 'float64')
        self.assertTrue(a.type.format == 'csr')
        ta = transpose(a)
        self.assertTrue(ta.type.dtype == 'float64', ta.type.dtype)
        self.assertTrue(ta.type.format == 'csc', ta.type.format)

        vta = eval_outputs([ta])
        self.assertTrue(vta.shape == (3,5))

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

    def _testSS(self, op, array1 = numpy.array([[1., 0], [3, 0], [0, 6]]),
                array2 = numpy.asarray([[0, 2.], [0, 4], [5, 0]])):
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
            self.assertTrue(val.shape == (3,2))
            if op is add:
                self.assertTrue(numpy.all(val.todense() == (a + b).todense()))
                self.assertTrue(numpy.all(val.todense() == numpy.array([[1., 2], [3, 4], [5, 6]])))
            elif op is mul:
                self.assertTrue(numpy.all(val.todense() == (a.multiply(b)).todense()))
                self.assertTrue(numpy.all(val.todense() == numpy.array([[1, 0], [9, 0], [0, 36]])))

    def _testSD(self, op, array1 = numpy.array([[1., 0], [3, 0], [0, 6]]),
                array2 = numpy.asarray([[0, 2.], [0, 4], [5, 0]])):
        for mtype in _mtypes:
            a = numpy.array(array1)
            aR = tensor.as_tensor_variable(a)
            self.assertFalse(aR.data is a) #constants are copied
            self.assertTrue(_is_dense(a))
            self.assertTrue(_is_dense_variable(aR))

            b = mtype(array2)
            bR = as_sparse_variable(b)
            self.assertFalse(bR.data is b) #constants are copied
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
                self.assertTrue(numpy.all(val == numpy.array([[1., 2], [3, 4], [5, 6]])))
            elif op is mul:
                self.assertTrue(_is_sparse_variable(apb))
                self.assertTrue(numpy.all(val.todense() == (b.multiply(a))))
                self.assertTrue(numpy.all(val.todense() == numpy.array([[1, 0],
[9, 0], [0, 36]])))

    def _testDS(self, op, array1 = numpy.array([[1., 0], [3, 0], [0, 6]]),
                array2 = numpy.asarray([[0, 2.], [0, 4], [5, 0]])):
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
                self.assertTrue(numpy.all(val == numpy.array([[1., 2], [3, 4], [5, 6]])))
            elif op is mul:
                self.assertTrue(_is_sparse_variable(apb))
                self.assertTrue(numpy.all(val.todense() == (a.multiply(b))))
                self.assertTrue(numpy.all(val.todense() == numpy.array([[1, 0],
[9, 0], [0, 36]])))

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

            # mul upcasts the dense input if needed
            self.assertRaises(NotImplementedError, mul, a_sv, b_dv)
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
            self.assertTrue(str(val.dtype)=='float64')
            self.assertTrue(val.format == 'csc')

    if 0:
        def test1(self):
            a = tensor.as_tensor_variable(numpy.random.rand(5))
            s = csr_from_dense(a)
            val = eval_outputs([s])
            self.assertTrue(str(val.dtype)=='float64')
            self.assertTrue(val.format == 'csr')

    if 1:
        def test2(self):
            #call dense_from_sparse
            for t in _mtypes:
                s = t(scipy.sparse.identity(5))
                d = dense_from_sparse(s)
                # s should be copied into the graph as a constant
                s[0,0] = 3.0 # changes s, but not the copy
                val = eval_outputs([d])
                return
                self.assertTrue(str(val.dtype)==s.dtype)
                self.assertTrue(numpy.all(val[0] == [1,0,0,0,0]))

class test_structureddot(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    def test_structureddot_csc_grad(self):

        #shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = sp.csc_matrix(random_lil((4,3), 'float32', 3))

        mat = numpy.asarray(numpy.random.randn(3,2), 'float32')

        def buildgraphCSC(spdata,sym_mat):
            csc = CSC(spdata, spmat.indices[:spmat.size],
                    spmat.indptr, spmat.shape)
            assert csc.type.dtype == 'float32'
            rval = structured_dot(csc, sym_mat)
            assert rval.type.dtype == 'float32'
            return rval

        utt.verify_grad(buildgraphCSC,
                    [spmat.data, mat])

    def test_structureddot_csr_grad(self):

        #shortcut: testing csc in float32, testing csr in float64

        # allocate a random sparse matrix
        spmat = sp.csr_matrix(random_lil((4,3), 'float64', 3))

        mat = numpy.asarray(numpy.random.randn(3,2), 'float64')

        def buildgraph(spdata,sym_mat):
            csr = CSR(spdata, spmat.indices[:spmat.size],
                    spmat.indptr, spmat.shape)
            assert csr.type.dtype == 'float64'
            rval = structured_dot(csr, sym_mat)
            assert rval.type.dtype == 'float64'
            return rval

        utt.verify_grad(buildgraph,
                    [spmat.data, mat])

    def test_upcast(self):

        typenames = 'float32', 'int64', 'int8', 'int32', 'int16', 'float64', 'complex64', 'complex128'
        for dense_dtype in typenames:
            for sparse_dtype in typenames:
                correct_dtype = theano.scalar.upcast(sparse_dtype, dense_dtype)
                a = SparseType('csc', dtype=sparse_dtype)()
                b = tensor.matrix(dtype=dense_dtype)
                d = structured_dot(a,b)
                assert d.type.dtype == correct_dtype

                # compile and run a function

                f = theano.function([a,b],d)

                M,N,K,nnz = (4,3,5,3)
                spmat = sp.csc_matrix(random_lil((M,N), sparse_dtype, nnz))
                # the following madness is necessary to workaround
                # an intc vs. int32 bug.
                # The lil makes an intc on my computer when sparse_dtype
                # is int32.
                spmat.dtype = numpy.dtype(sparse_dtype)
                mat = numpy.asarray(numpy.random.randn(N,K)*9, dtype=dense_dtype)
                print 'DTYPES', sparse_dtype,dense_dtype
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
        # Test that a graph involving structured_dot(assembled_csc_matrix) is optimized to be
        # just a structured_dot_csc Op and no assembly of a csc_matrix.
        #
        # The optimization from structured_dot -> structured_dot_csc is currently disabled,
        # So this test is not expected to pass

        return
        #
        kerns = tensor.Tensor(dtype='int64', broadcastable=[False])('kerns')
        spmat = sp.lil_matrix((4,6), dtype='int64')
        for i in range(5):
            # set non-zeros in random locations (row x, col y)
            x = numpy.floor(numpy.random.rand()*spmat.shape[0])
            y = numpy.floor(numpy.random.rand()*spmat.shape[1])
            spmat[x,y] = numpy.random.rand()*10
        spmat = sp.csc_matrix(spmat)

        images = tensor.Tensor(dtype='float32', broadcastable=[False, False])('images')

        cscmat = CSC(kerns, spmat.indices[:spmat.size], spmat.indptr, spmat.shape)
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
        #print 'kdtype', kernvals.dtype, kernvals.shape, kernvals.ndim, kernvals.dtype.num
        #print 'type of kernvals = ', kernvals.dtype
        bsize = 3
        imvals = 1.0 * numpy.array(numpy.arange(bsize*spmat.shape[1]).\
                reshape(bsize,spmat.shape[1]), dtype='float32')
        outvals = f(kernvals,imvals)
        print outvals

    def test_dot_sparse_sparse(self):
        #test dot for 2 input sparse matrix
        sparse_dtype = 'float64'
        sp_mat = {'csc':sp.csc_matrix,
                  'csr':sp.csr_matrix}

        for sparse_format_a in ['csc','csr']:
            for sparse_format_b in ['csc', 'csr']:
                a = SparseType(sparse_format_a, dtype=sparse_dtype)()
                b = SparseType(sparse_format_b, dtype=sparse_dtype)()
                d = theano.dot(a,b)
                f = theano.function([a,b], theano.Out(d, borrow=True))
                topo = f.maker.env.toposort()
                for M,N,K,nnz in [(4,3,2,3),
                                  (40,30,20,3),
                                  (40,30,20,30),
                                  (400,3000,200,6000),
                                  ]:
                    a_val = sp_mat[sparse_format_a](random_lil((M,N), sparse_dtype, nnz))
                    b_val = sp_mat[sparse_format_b](random_lil((N,K), sparse_dtype, nnz))
                    f(a_val, b_val)

    def test_csc_correct_output_faster_than_scipy(self):
        sparse_dtype = 'float64'
        dense_dtype = 'float64'

        a = SparseType('csc', dtype=sparse_dtype)()
        b = tensor.matrix(dtype=dense_dtype)
        d = theano.dot(a,b)
        f = theano.function([a,b], theano.Out(d, borrow=True))

        for M,N,K,nnz in [(4,3,2,3),
                (40,30,20,3),
                (40,30,20,30),
                (400,3000,200,6000),
                ]:
            spmat = sp.csc_matrix(random_lil((M,N), sparse_dtype, nnz))
            mat = numpy.asarray(numpy.random.randn(N,K), dense_dtype)
            theano_times = []
            scipy_times = []
            for i in xrange(5):
                t0 = time.time()
                theano_result = f(spmat, mat)
                t1 = time.time()
                scipy_result = spmat * mat
                t2 = time.time()

                theano_times.append(t1-t0)
                scipy_times.append(t2-t1)

            theano_time = numpy.min(theano_times)
            scipy_time = numpy.min(scipy_times)

            speedup = scipy_time / theano_time
            print scipy_times
            print theano_times
            print 'M=%(M)s N=%(N)s K=%(K)s nnz=%(nnz)s theano_time=%(theano_time)s speedup=%(speedup)s' % locals()

            # fail if Theano is slower than scipy by more than a certain amount
            overhead_tol = 0.003 # seconds overall
            overhead_rtol = 1.2 # times as long
            self.assertTrue(numpy.allclose(theano_result, scipy_result))
            if not theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
                self.assertFalse(theano_time > overhead_rtol*scipy_time + overhead_tol)

    def test_csr_correct_output_faster_than_scipy(self):

        #contrast with test_grad, we put csr in float32, csc in float64

        sparse_dtype = 'float32'
        dense_dtype = 'float32'

        a = SparseType('csr', dtype=sparse_dtype)()
        b = tensor.matrix(dtype=dense_dtype)
        d = theano.dot(a,b)
        f = theano.function([a,b], d)

        for M,N,K,nnz in [(4,3,2,3),
                (40,30,20,3),
                (40,30,20,30),
                (400,3000,200,6000),
                ]:
            spmat = sp.csr_matrix(random_lil((M,N), sparse_dtype, nnz))
            mat = numpy.asarray(numpy.random.randn(N,K), dense_dtype)
            t0 = time.time()
            theano_result = f(spmat, mat)
            t1 = time.time()
            scipy_result = spmat * mat
            t2 = time.time()

            theano_time = t1-t0
            scipy_time = t2-t1
            #print theano_result
            #print scipy_result
            print 'theano took', theano_time,
            print 'scipy took', scipy_time
            overhead_tol = 0.002 # seconds
            overhead_rtol = 1.1 # times as long
            self.assertTrue(numpy.allclose(theano_result, scipy_result))
            if not theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
                self.assertFalse(theano_time > overhead_rtol*scipy_time + overhead_tol)

def test_shape_i():
    sparse_dtype = 'float32'

    a = SparseType('csr', dtype=sparse_dtype)()
    f = theano.function([a], a.shape[1])
    assert f(sp.csr_matrix(random_lil((100,10), sparse_dtype, 3))) == 10

def test_shape():
    # Test that getting the shape of a sparse variable
    # does not actually create a dense tensor in the process.
    sparse_dtype = 'float32'

    a = SparseType('csr', dtype=sparse_dtype)()
    f = theano.function([a], a.shape)
    assert numpy.all(f(sp.csr_matrix(random_lil((100,10), sparse_dtype, 3)))==(100,10))
    if theano.config.mode!='FAST_COMPILE':
        topo = f.maker.env.toposort()
        assert len(topo)==3
        assert isinstance(topo[0].op,tensor.opt.Shape_i)
        assert isinstance(topo[1].op,tensor.opt.Shape_i)
        assert isinstance(topo[2].op,tensor.opt.MakeVector)

def test_may_share_memory():
    a=scipy.sparse.csc_matrix(scipy.sparse.eye(5,3))
    b=scipy.sparse.csc_matrix(scipy.sparse.eye(4,3))
    as_ar = lambda a: theano._asarray(a, dtype='int32')
    for a_,b_,rep in [(a,a,True),(b,b,True),(a,b,False),
                    (a,a.data,True),(a,a.indptr,True),(a,a.indices,True),(a,as_ar(a.shape),False),
                    (a.data,a,True),(a.indptr,a,True),(a.indices,a,True),(as_ar(a.shape),a,False),
                    (b,b.data,True),(b,b.indptr,True),(b,b.indices,True),(b,as_ar(b.shape),False),
                    (b.data,b,True),(b.indptr,b,True),(b.indices,b,True),(as_ar(b.shape),b,False),
                    (b.data,a,False),(b.indptr,a,False),(b.indices,a,False),(as_ar(b.shape),a,False),
                    ]:

        assert SparseType.may_share_memory(a_,b_)==rep

def test_sparse_shared_memory():
    # Note : There are no inplace ops on sparse matrix yet. If  one is someday implemented, we could test it here.
    a = random_lil((3,4), 'float32', 3).tocsr()
    m1 = random_lil((4,4), 'float32', 3).tocsr()
    m2 = random_lil((4,4), 'float32', 3).tocsr()
    x = SparseType('csr', dtype='float32')()
    y = SparseType('csr', dtype='float32')()

    sdot = theano.sparse.structured_dot
    z = sdot(x*3,m1) + sdot(y*2, m2)

    f = theano.function([theano.In(x,mutable=True),theano.In(y,mutable = True)],z, mode='FAST_RUN')

    def f_(x,y,m1=m1,m2=m2):
        return numpy.dot(x*3,m1) + numpy.dot(y*2,m2)

    assert SparseType.may_share_memory(a,a) #This is trivial
    result  = f(a,a)
    result_ = f_(a,a)
    assert (result_.todense() == result.todense()).all()

import theano.tensor.tests.test_sharedvar
test_shared_options=theano.tensor.tests.test_sharedvar.makeSharedTester(
    shared_constructor_ = theano.sparse.shared,
    dtype_ = 'float64',
    get_value_borrow_true_alias_ = True,
    shared_borrow_true_alias_ = True,
    set_value_borrow_true_alias_ = True,
    set_value_inplace_ = False,
    set_casted_value_inplace_ = False,
    shared_constructor_accept_ndarray_ = False,
    internal_type_ = scipy.sparse.csc_matrix,
    test_internal_type_ = scipy.sparse.issparse,
    theano_fct_ = lambda a: dense_from_sparse(a*2.),
    ref_fct_ = lambda a: numpy.asarray((a*2).todense()),
    cast_value_ = scipy.sparse.csr_matrix)


if __name__ == '__main__':
    unittest.main()
