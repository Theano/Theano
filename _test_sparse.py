from sparse import *

import unittest
import compile
import gradient

from sparse import _is_dense, _is_sparse, _is_dense_result, _is_sparse_result
from sparse import _mtypes, _mtype_to_str

class T_transpose(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)
    def test_transpose_csc(self):
        sp = sparse.csc_matrix(sparse.speye(5,3))
        a = as_sparse(sp)
        self.failUnless(a.data is sp)
        self.failUnless(a.data.shape == (5,3))
        self.failUnless(a.type.dtype == 'float64', a.type.dtype)
        self.failUnless(a.type.format == 'csc', a.type.format)
        ta = transpose(a)
        self.failUnless(ta.type.dtype == 'float64', ta.type.dtype)
        self.failUnless(ta.type.format == 'csr', ta.type.format)

        vta = compile.eval_outputs([ta])
        self.failUnless(vta.shape == (3,5))
    def test_transpose_csr(self):
        a = as_sparse(sparse.csr_matrix(sparse.speye(5,3)))
        self.failUnless(a.data.shape == (5,3))
        self.failUnless(a.type.dtype == 'float64')
        self.failUnless(a.type.format == 'csr')
        ta = transpose(a)
        self.failUnless(ta.type.dtype == 'float64', ta.type.dtype)
        self.failUnless(ta.type.format == 'csc', ta.type.format)

        vta = compile.eval_outputs([ta])
        self.failUnless(vta.shape == (3,5))

class T_Add(unittest.TestCase):
    def testSS(self):
        for mtype in _mtypes:
            a = mtype(numpy.array([[1., 0], [3, 0], [0, 6]]))
            aR = as_sparse(a)
            self.failUnless(aR.data is a)
            self.failUnless(_is_sparse(a))
            self.failUnless(_is_sparse_result(aR))

            b = mtype(numpy.asarray([[0, 2.], [0, 4], [5, 0]]))
            bR = as_sparse(b)
            self.failUnless(bR.data is b)
            self.failUnless(_is_sparse(b))
            self.failUnless(_is_sparse_result(bR))

            apb = add(aR, bR)
            self.failUnless(_is_sparse_result(apb))

            self.failUnless(apb.type.dtype == aR.type.dtype, apb.type.dtype)
            self.failUnless(apb.type.dtype == bR.type.dtype, apb.type.dtype)
            self.failUnless(apb.type.format == aR.type.format, apb.type.format)
            self.failUnless(apb.type.format == bR.type.format, apb.type.format)

            val = compile.eval_outputs([apb])
            self.failUnless(val.shape == (3,2))
            self.failUnless(numpy.all(val.todense() == (a + b).todense()))
            self.failUnless(numpy.all(val.todense() == numpy.array([[1., 2], [3, 4], [5, 6]])))

    def testSD(self):
        for mtype in _mtypes:
            a = numpy.array([[1., 0], [3, 0], [0, 6]])
            aR = tensor.as_tensor(a)
            self.failUnless(aR.data is a)
            self.failUnless(_is_dense(a))
            self.failUnless(_is_dense_result(aR))

            b = mtype(numpy.asarray([[0, 2.], [0, 4], [5, 0]]))
            bR = as_sparse(b)
            self.failUnless(bR.data is b)
            self.failUnless(_is_sparse(b))
            self.failUnless(_is_sparse_result(bR))

            apb = add(aR, bR)
            self.failUnless(_is_dense_result(apb))

            self.failUnless(apb.type.dtype == aR.type.dtype, apb.type.dtype)
            self.failUnless(apb.type.dtype == bR.type.dtype, apb.type.dtype)

            val = compile.eval_outputs([apb])
            self.failUnless(val.shape == (3, 2))
            self.failUnless(numpy.all(val == (a + b)))
            self.failUnless(numpy.all(val == numpy.array([[1., 2], [3, 4], [5, 6]])))

    def testDS(self):
        for mtype in _mtypes:
            a = mtype(numpy.array([[1., 0], [3, 0], [0, 6]]))
            aR = as_sparse(a)
            self.failUnless(aR.data is a)
            self.failUnless(_is_sparse(a))
            self.failUnless(_is_sparse_result(aR))

            b = numpy.asarray([[0, 2.], [0, 4], [5, 0]])
            bR = tensor.as_tensor(b)
            self.failUnless(bR.data is b)
            self.failUnless(_is_dense(b))
            self.failUnless(_is_dense_result(bR))

            apb = add(aR, bR)
            self.failUnless(_is_dense_result(apb))

            self.failUnless(apb.type.dtype == aR.type.dtype, apb.type.dtype)
            self.failUnless(apb.type.dtype == bR.type.dtype, apb.type.dtype)

            val = compile.eval_outputs([apb])
            self.failUnless(val.shape == (3, 2))
            self.failUnless(numpy.all(val == (a + b)))
            self.failUnless(numpy.all(val == numpy.array([[1., 2], [3, 4], [5, 6]])))

class T_conversion(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)

    def test0(self):
        a = tensor.as_tensor(numpy.random.rand(5))
        s = csc_from_dense(a)
        val = compile.eval_outputs([s])
        self.failUnless(str(val.dtype)=='float64')
        self.failUnless(val.format == 'csc')

    def test1(self):
        a = tensor.as_tensor(numpy.random.rand(5))
        s = csr_from_dense(a)
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
            x = as_sparse(mtype((500,3)))
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
            x = as_sparse(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.failUnless(_is_sparse_result(x))

            y = tensor.as_tensor([[1., 2], [3, 4], [2, 1]])
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
            x = as_sparse(mtype((500,3)))
            x.data[(10, 1)] = 1
            x.data[(20, 2)] = 2
            self.failUnless(_is_sparse_result(x))

            y = tensor.as_tensor([[1., 2], [3, 4], [2, 1]])
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
            x = tensor.matrix('x') #Tensor('float64', broadcastable=[False,False], name='x')
            w = Sparse(dtype = 'float64', format = _mtype_to_str[mtype]).make_result()
            xw = dense_from_sparse(dot(w, x))
            y = dense_from_sparse(dot(w.T, xw))
            diff = x-y
            loss = tensor.sum(tensor.sqr(diff))
            gw = gradient.grad(loss, w)
            trainfn = compile.function([x, w], [y, loss, gw])

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
                x = tensor.matrix('x')
                w = Sparse(dtype = 'float64', format = _mtype_to_str[mtype]).make_result()
                xw = dense_from_sparse(dot(w, x))
                y = dense_from_sparse(dot(w.T, xw))
                diff = x-y
                loss = tensor.sum(tensor.sqr(diff))
                gw = gradient.grad(loss, w)
                trainfn = compile.function([x, w], [y, loss, gw])

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
