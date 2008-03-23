from sparse import *

import unittest

class _testCase_transpose(unittest.TestCase):
    def setUp(self):
        core.build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        core.pop_mode()
    def test_transpose(self):
        a = SparseR(sparse.csr_matrix(sparse.speye(5,3)))
        self.failUnless(a.data.shape == (5,3))
        ta = transpose(a)
        self.failUnless(ta.data.shape == (3,5))

class _testCase_dot(unittest.TestCase):
    def setUp(self):
        core.build_eval_mode()
        numpy.random.seed(44)
    def tearDown(self):
        core.pop_mode()
    def test_basic0(self):
        for mtype in [sparse.csc_matrix, sparse.csr_matrix]:
            x = SparseR(mtype(sparse.speye(5,3)))
            y = core.wrap(numpy.random.rand(3, 2))

            z = dot(x,y)
            self.failUnless(z.data.shape == (5,2))
            self.failUnless(type(z.data) is mtype)
    def test_basic1(self):
        """dot: sparse left"""
        a = numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0, 0]],
                dtype='float64')
        b = numpy.random.rand(5, 3)
        for mtype in [sparse.csr_matrix, sparse.csc_matrix, sparse.dok_matrix,
                sparse.lil_matrix]:#, sparse.coo_matrix]:
            #print type(a), mtype
            m = mtype(a)
            ab = m.dot(b)
            try:
                z = dot(SparseR(m),core.ResultBase(data=b))
                self.failUnless(z.data.shape == ab.shape)
                self.failUnless(type(z.data) == type(ab))
            except Exception, e:
                print 'cccc', mtype, e, str(e)
                raise
    def test_basic2(self):
        """dot: sparse right"""
        a = numpy.random.rand(2, 5)
        b = numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0, 0]],
                dtype='float64').transpose()

        for mtype in [sparse.csr_matrix, sparse.csc_matrix, sparse.dok_matrix,
                sparse.lil_matrix]:#, sparse.coo_matrix]:
            m = mtype(b)
            ab = m.transpose().dot(a.transpose()).transpose()
            z = dot(core.ResultBase(data=a),SparseR(mtype(b)))
            self.failUnless(z.data.shape == ab.shape)
            self.failUnless(type(z.data) == type(ab))
    def test_graph_bprop0(self):
        x = core.wrap(numpy.random.rand(10,2))
        w = SparseR(sparse.csr_matrix(numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0,
            0]],dtype='float64')))

        for epoch in xrange(50):
            xw = sparse2dense(dot(x, w))
            y = sparse2dense(dot(xw, transpose(w)))
            loss = core.sum(core.sqr(x-y))
            gy = y-x
            g = grad.Grad({y:gy})
            g.bprop()
            lr = 0.002
            g(w).data[1,0] = 0
            g(w).data[1,4] = 0
            w.data = -lr * g(w).data + w.data

        self.failUnless('3.08560636025' == str(loss.data))

    def test_graph_bprop1(self):
        x = core.wrap(numpy.random.rand(10,2))
        w = SparseR(sparse.csr_matrix(numpy.asarray([[1, 0, 3, 0, 5], [0, 0, -2, 0,
            0]],dtype='float64')))

        for epoch in xrange(50):
            xw = sparse2dense(dot(x, w))
            y = sparse2dense(dot(xw, transpose(w)))
            loss = core.sum(core.sqr(x-y))
            g = grad.grad(loss)
            lr = 0.001

            g(w).data[1,0] = 0
            g(w).data[1,4] = 0
            w.data = -lr * g(w).data + w.data

        self.failUnless('3.08560636025' == str(loss.data))
        
if __name__ == '__main__':
    unittest.main()

