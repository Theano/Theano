from sparse import *

import unittest
import compile
import gradient

from sparse import _is_dense, _is_sparse, _is_dense_result, _is_sparse_result
from sparse import _mtypes, _mtype_to_str

import random

class T_transpose(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)
    def test_transpose_csc(self):
        sp = sparse.csc_matrix(sparse.speye(5,3))
        a = assparse(sp)
        self.failUnless(a.data is sp)
        self.failUnless(a.data.shape == (5,3))
        self.failUnless(a.dtype == 'float64')
        self.failUnless(a.format == 'csc', a.format)
        ta = transpose(a)
        self.failUnless(ta.dtype == 'float64', ta.dtype)
        self.failUnless(ta.format == 'csr', ta.format)

        vta = compile.eval_outputs([ta])
        self.failUnless(vta.shape == (3,5))
    def test_transpose_csr(self):
        a = assparse(sparse.csr_matrix(sparse.speye(5,3)))
        self.failUnless(a.data.shape == (5,3))
        self.failUnless(a.dtype == 'float64')
        self.failUnless(a.format == 'csr')
        ta = transpose(a)
        self.failUnless(ta.dtype == 'float64', ta.dtype)
        self.failUnless(ta.format == 'csc', ta.format)

        vta = compile.eval_outputs([ta])
        self.failUnless(vta.shape == (3,5))

class T_Add(unittest.TestCase):
    def testSS(self):
        for mtype in _mtypes:
            a = mtype(numpy.array([[1., 0], [3, 0], [0, 6]]))
            aR = assparse(a)
            self.failUnless(aR.data is a)
            self.failUnless(_is_sparse(a))
            self.failUnless(_is_sparse_result(aR))

            b = mtype(numpy.asarray([[0, 2.], [0, 4], [5, 0]]))
            bR = assparse(b)
            self.failUnless(bR.data is b)
            self.failUnless(_is_sparse(b))
            self.failUnless(_is_sparse_result(bR))

            apb = add(aR, bR)
            self.failUnless(_is_sparse_result(apb))

            self.failUnless(apb.dtype == aR.dtype, apb.dtype)
            self.failUnless(apb.dtype == bR.dtype, apb.dtype)
            self.failUnless(apb.format == aR.format, apb.format)
            self.failUnless(apb.format == bR.format, apb.format)

            val = compile.eval_outputs([apb])
            self.failUnless(val.shape == (3,2))
            self.failUnless(numpy.all(val.todense() == (a + b).todense()))
            self.failUnless(numpy.all(val.todense() == numpy.array([[1., 2], [3, 4], [5, 6]])))

    def testSD(self):
        for mtype in _mtypes:
            a = numpy.array([[1., 0], [3, 0], [0, 6]])
            aR = tensor.astensor(a)
            self.failUnless(aR.data is a)
            self.failUnless(_is_dense(a))
            self.failUnless(_is_dense_result(aR))

            b = mtype(numpy.asarray([[0, 2.], [0, 4], [5, 0]]))
            bR = assparse(b)
            self.failUnless(bR.data is b)
            self.failUnless(_is_sparse(b))
            self.failUnless(_is_sparse_result(bR))

            apb = add(aR, bR)
            self.failUnless(_is_dense_result(apb))

            self.failUnless(apb.dtype == aR.dtype, apb.dtype)
            self.failUnless(apb.dtype == bR.dtype, apb.dtype)

            val = compile.eval_outputs([apb])
            self.failUnless(val.shape == (3, 2))
            self.failUnless(numpy.all(val == (a + b)))
            self.failUnless(numpy.all(val == numpy.array([[1., 2], [3, 4], [5, 6]])))

    def testDS(self):
        for mtype in _mtypes:
            a = mtype(numpy.array([[1., 0], [3, 0], [0, 6]]))
            aR = assparse(a)
            self.failUnless(aR.data is a)
            self.failUnless(_is_sparse(a))
            self.failUnless(_is_sparse_result(aR))

            b = numpy.asarray([[0, 2.], [0, 4], [5, 0]])
            bR = tensor.astensor(b)
            self.failUnless(bR.data is b)
            self.failUnless(_is_dense(b))
            self.failUnless(_is_dense_result(bR))

            apb = add(aR, bR)
            self.failUnless(_is_dense_result(apb))

            self.failUnless(apb.dtype == aR.dtype, apb.dtype)
            self.failUnless(apb.dtype == bR.dtype, apb.dtype)

            val = compile.eval_outputs([apb])
            self.failUnless(val.shape == (3, 2))
            self.failUnless(numpy.all(val == (a + b)))
            self.failUnless(numpy.all(val == numpy.array([[1., 2], [3, 4], [5, 6]])))

class T_conversion(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)

    def test0(self):
        a = tensor.astensor(numpy.random.rand(5))
        s = sparse_from_dense(a, 'csc')
        val = compile.eval_outputs([s])
        self.failUnless(str(val.dtype)=='float64')
        self.failUnless(val.format == 'csc')

    def test1(self):
        a = tensor.astensor(numpy.random.rand(5))
        s = sparse_from_dense(a,'csr')
        val = compile.eval_outputs([s])
        self.failUnless(str(val.dtype)=='float64')
        self.failUnless(val.format == 'csr')

    def test2(self):
        for t in _mtypes:
            s = t((2,5))
            d = dense_from_sparse(s)
            s[0,0] = 1.0
            val = compile.eval_outputs([d])
            self.failUnless(str(val.dtype)=='float64')
            self.failUnless(numpy.all(val[0] == [1,0,0,0,0]))


class _testCase_dot(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)

    def test_basicSS(self):
        for mtype in _mtypes:
            x = assparse(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.failUnless(_is_sparse_result(x))

            xT = x.T
            self.failUnless(_is_sparse_result(xT))

            zop = dot(x,xT)
            self.failUnless(_is_sparse_result(zop))
            z = compile.eval_outputs([zop])
            self.failUnless(_is_sparse(z))
            self.failUnless(z.shape == (500,500))
            self.failUnless(type(z) is mtype)

            w = mtype((500,500))
            w[(10, 10)] = 1
            w[(20, 20)] = 4
            self.failUnless(z.shape == w.shape)
            self.failUnless(type(z) == type(w))
            self.failUnless(z.dtype == w.dtype)

            #self.failUnless(z == w)
            self.failUnless(abs(z-w).nnz == 0)

            z = z.todense()
            w = w.todense()
            self.failUnless((z == w).all() == True)

    def test_basicSD(self):
        for mtype in _mtypes:
            x = assparse(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.failUnless(_is_sparse_result(x))

            y = tensor.astensor([[1., 2], [3, 4], [2, 1]])
            self.failUnless(_is_dense_result(y))

            zop = dot(x,y)
            self.failUnless(_is_sparse_result(zop))
            z = compile.eval_outputs([zop])
            self.failUnless(_is_sparse(z))
            self.failUnless(z.shape == (500,2))
            self.failUnless(type(z) is mtype)

            w = mtype((500,2))
            w[(10, 0)] = 3.
            w[(20, 0)] = 4
            w[(10, 1)] = 4
            w[(20, 1)] = 2
            self.failUnless(z.shape == w.shape)
            self.failUnless(type(z) == type(w))
            self.failUnless(z.dtype == w.dtype)

            #self.failUnless(z == w)
            self.failUnless(abs(z-w).nnz == 0)

            z = z.todense()
            w = w.todense()
            self.failUnless((z == w).all() == True)

    def test_basicDS(self):
        for mtype in _mtypes:
            x = assparse(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.failUnless(_is_sparse_result(x))

            y = tensor.astensor([[1., 2], [3, 4], [2, 1]])
            self.failUnless(_is_dense_result(y))

            x.data = x.data.T
            y.data = y.data.T

#            zop = dot(y, x)
            zop = transpose(dot(y, x))
            self.failUnless(_is_sparse_result(zop))
            z = compile.eval_outputs([zop])
            self.failUnless(_is_sparse(z))
            self.failUnless(z.shape == (500,2))
#            self.failUnless(type(z) is mtype)

            w = mtype((500,2))
            w[(10, 0)] = 3.
            w[(20, 0)] = 4
            w[(10, 1)] = 4
            w[(20, 1)] = 2
            self.failUnless(z.shape == w.shape)
            # Type should switch from csr to csc and vice-versa, so don't perform this test
            #self.failUnless(type(z) == type(w))
            self.failUnless(z.dtype == w.dtype)

            # Type should switch from csr to csc and vice-versa, so don't perform this test
            #self.failUnless(z == w)
            self.failUnless(abs(z-w).nnz == 0)

            z = z.todense()
            w = w.todense()
            self.failUnless((z == w).all() == True)

    def test_graph_bprop0(self):
        for mtype in _mtypes:
            x = tensor.Tensor('float64', broadcastable=[False,False], name='x')
            w = SparseResult('float64', _mtype_to_str[mtype])
            xw = dense_from_sparse(dot(w, x))
            y = dense_from_sparse(dot(w.T, xw))
            diff = x-y
            loss = tensor.sum(tensor.sqr(diff))
            gw = tensor.grad(loss, w)
            trainfn = compile.Function([x, w], [y, loss, gw])

            x = numpy.asarray([[1., 2], [3, 4], [2, 1]])
            w = mtype((500,3))
            w[(10, 1)] = 1
            w[(20, 2)] = 2
            lr = 0.001
            y, origloss, gw = trainfn(x, w)
            for epoch in xrange(50):
                y, loss, gw = trainfn(x, w)
                w = w - (lr * gw)

            self.failUnless(origloss > loss)
            self.failUnless('1.0543172285' == str(loss))

    def test_graph_bprop_rand(self):
        for i in range(10):
            xorig = numpy.random.rand(3,2)
            for mtype in _mtypes:
                x = tensor.Tensor('float64', broadcastable=[False,False], name='x')
                w = SparseResult('float64', _mtype_to_str[mtype])
                xw = dense_from_sparse(dot(w, x))
                y = dense_from_sparse(dot(w.T, xw))
                diff = x-y
                loss = tensor.sum(tensor.sqr(diff))
                gw = tensor.grad(loss, w)
                trainfn = compile.Function([x, w], [y, loss, gw])

                x = xorig
                w = mtype((500,3))
                w[(10, 1)] = 1
                w[(20, 2)] = 2
                lr = 0.001
                y, origloss, gw = trainfn(x, w)
                for epoch in xrange(50):
                    y, loss, gw = trainfn(x, w)
                    w = w - (lr * gw)

                self.failUnless(origloss > loss)

if __name__ == '__main__':
    unittest.main()
