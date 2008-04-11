from sparse import *

import unittest
import compile

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
    def test0(self):
        sp_a = sparse.csc_matrix(sparse.speye(5,3))
        a = assparse(sp_a)

        sp_b = sparse.csc_matrix(sparse.speye(5,3))
        b = assparse(sp_b)

        self.failUnless(a.data is sp_a)
        apb = add_s_s(a, b)

        self.failUnless(apb.dtype == a.dtype, apb.dtype)
        self.failUnless(apb.format == a.format, apb.format)

        val = compile.eval_outputs([apb])
        self.failUnless(val.shape == (5,3))
        self.failUnless(numpy.all(val.todense() == (sp_a + sp_b).todense()))

class T_conversion(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)

    def test0(self):
        a = tensor.astensor(numpy.random.rand(5))
        s = sparse_from_dense(a,'csc')
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
        csr = sparse.csr_matrix((2,5))
        d = dense_from_sparse(csr)
        csr[0,0] = 1.0
        val = compile.eval_outputs([d])
        self.failUnless(str(val.dtype)=='float64')
        self.failUnless(numpy.all(val[0] == [1,0,0,0,0]))


class _testCase_dot(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(44)

    def test_basic0(self):
        for mtype in [sparse.csc_matrix, sparse.csr_matrix]:
            x = assparse(mtype(sparse.speye(5,3)))
            y = tensor.astensor(numpy.random.rand(3, 2))

            zop = dot(x,y)
            z = compile.eval_outputs([zop])
            self.failUnless(z.shape == (5,2))
            self.failUnless(type(z) is mtype)

    def test_missing(self):
        raise NotImplementedError('tests commented out')

#    def test_basic1(self):
#        """dot: sparse left"""
#        a = numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0, 0]],
#                dtype='float64')
#        b = numpy.random.rand(5, 3)
#        for mtype in [sparse.csr_matrix, sparse.csc_matrix, sparse.dok_matrix,
#                sparse.lil_matrix]:#, sparse.coo_matrix]:
#            #print type(a), mtype
#            m = mtype(a)
#            ab = m.dot(b)
#            try:
#                z = dot(assparse(m), gof.Result(data=b))
#                self.failUnless(z.data.shape == ab.shape)
#                self.failUnless(type(z.data) == type(ab))
#            except Exception, e:
#                print 'cccc', mtype, e, str(e)
#                raise
#
#    def test_basic2(self):
#        """dot: sparse right"""
#        a = numpy.random.rand(2, 5)
#        b = numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0, 0]],
#                dtype='float64').transpose()
#
#        for mtype in [sparse.csr_matrix, sparse.csc_matrix, sparse.dok_matrix,
#                sparse.lil_matrix]:#, sparse.coo_matrix]:
#            m = mtype(b)
#            ab = m.transpose().dot(a.transpose()).transpose()
#            z = dot(gof.Result(data=a),assparse(mtype(b)))
#            self.failUnless(z.data.shape == ab.shape)
#            self.failUnless(type(z.data) == type(ab))
#
#    def test_graph_bprop0(self):
#        x = tensor.astensor(numpy.random.rand(10,2))
#        w = assparse(sparse.csr_matrix(
#                numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0,0]],dtype='float64')
#            ))
#
#        for epoch in xrange(50):
#            xw = dense_from_sparse(dot(x, w))
#            y = dense_from_sparse(dot(xw, transpose(w)))
#            loss = core.sum(core.sqr(x-y))
#            gy = y-x
#            g = grad.Grad({y:gy})
#            g.bprop()
#            lr = 0.002
#            g(w).data[1,0] = 0
#            g(w).data[1,4] = 0
#            w.data = -lr * g(w).data + w.data
#
#        self.failUnless('3.08560636025' == str(loss.data))
#
#    def test_graph_bprop1(self):
#        x = tensor.astensor(numpy.random.rand(10,2))
#        w = assparse(sparse.csr_matrix(
#                numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0,0]],dtype='float64')
#            ))
#
#        for epoch in xrange(50):
#            xw = dense_from_sparse(dot(x, w))
#            y = dense_from_sparse(dot(xw, transpose(w)))
#            loss = core.sum(core.sqr(x-y))
#            g = grad.grad(loss)
#            lr = 0.001
#
#            g(w).data[1,0] = 0
#            g(w).data[1,4] = 0
#            w.data = -lr * g(w).data + w.data
#
#        self.failUnless('3.08560636025' == str(loss.data))

if __name__ == '__main__':
    unittest.main()
