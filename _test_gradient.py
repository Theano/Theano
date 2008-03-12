
#
# UNIT TEST
#
import unittest
import numpy
import compile
import tensor
import tensor_ops as T

from gradient import *

def matrix():
    return tensor.Tensor('float64', [0,0])

def matrices(n):
    return [matrix() for i in xrange(n)]


class _testNone(unitTest.TestCase):
    def test0(self):



class _testCase_matinv:# (unittest.TestCase):
    def setUp(self):
        numpy.random.seed(1)
    def matinv(self,dim):
        # symbolic program
        a,b = matrices(2)
        ab = T.dot(a,b)
        diff = ab - tensor.tensor(numpy.identity(dim))
        ssdiff = T.sum((diff**2.0))
        g = grad(ssdiff,None, tensor.tensor(numpy.ones(1)))

        # compilation to function
        fn = compile.Function([a,b], [ssdiff,g(b)])

        # use the function
        w = numpy.random.rand(dim,dim)
        wi = numpy.random.rand(dim,dim)
        for i in xrange(300):
            ssd, gw = fn(w,wi)
            #print ssdiff
            if i == 0:
                str0 = str(ssd)
            wi -= 0.4 * gw

        return str0, str(ssd)

    def test_matinv(self):
        """Matrix inversion by gradient descent (eval mode)"""
        self.assertEqual(('2.67327580893', '0.000438649434819'), self.matinv(3))


class _testCase_old:#(unittest.TestCase):

    class posneg(T._TensorOp):
        nout=2
        def impl(x): return x, -x
        def grad(x, gpos, gneg): return gpos - gneg

    class posnegzero(T._TensorOp):
        nout=3
        def impl(x): return x, -x, 0.0
        def grad(x, gpos, gneg, gzero): return gpos - gneg

    def setUp(self):
        numpy.random.seed(1)

    def test_grad_wrt_ndarray_pointer(self):
        """Grad indexing by un-wrapped ndarray"""
        a = numpy.ones((4, 4))
        b = numpy.ones((4, 4))
        c = numpy.ones((4, 4))
        expr = core.sum(core.dot(core.add(a, b), c))
        g = grad(expr)
        g[a]

    def test_bprop_call_order(self):
        """Ensure call before bprop is illegal"""
        a = numpy.ones((3,3,3))
        b = core.exp(a)
        gb = Grad({b:wrappers.wrap(a)})
        try:
            gb(a)
            self.assertEqual('should have raised',0)
        except Exception, e:
            self.assertEqual(str(e), 'Grad.__call__ only makes sense after a bprop')
            return
        self.assertEqual('should have caught, returned',0)

    def test_undefined_grad0(self):
        """Make sure posneg works with fully specified gradients"""

        a = numpy.ones((3,3,3))
        b,c = _testCase.posneg(a)

        g = Grad({b:wrappers.wrap(a),c:wrappers.wrap(a)})
        g.bprop()
        max = numpy.max(g(a))
        min = numpy.min(g(a))
        self.assertEqual(max, min)
        self.assertEqual(max, 0.0)

    def test_undefined_grad1(self):
        """Propagate undefined values through posneg's first gradient"""

        a = numpy.ones((3,3,3))
        b,c = _testCase.posneg(a)

        gb = Grad({b:wrappers.wrap(a)})
        try:
            gb.bprop()
            self.assertEqual('should have raised',0)
        except UndefinedError:
            return
        self.assertEqual("Should have been error", 0)
    
    def test_undefined_grad2(self):
        """Propagate undefined values through posneg's second gradient"""

        a = numpy.ones((3,3,3))
        b,c = _testCase.posneg(a)
        gc = Grad({c:wrappers.wrap(a)})
        try:
            gc.bprop()
            self.assertEqual('should have raised',0)
        except UndefinedError:
            return
        self.assertEqual("Should have been error", 0)

    def test_undefined_grad3(self):
        """Ignore undefined values properly"""

        a = numpy.ones((3,3,3))
        b,c,d = _testCase.posnegzero(a)
        #print b, c, d
        g = Grad({b:wrappers.wrap(a), c:wrappers.wrap(a)})
        g.bprop()
        max = numpy.max(g(a))
        min = numpy.min(g(a))
        self.assertEqual(max, min)
        self.assertEqual(max, 0.0)

    def test_repeat_bprop(self):
        """Refuse to repeat bprop"""

        a = numpy.ones((3,3,3))
        b,c,d = _testCase.posnegzero(a)
        #print b, c, d
        g = Grad({b:wrappers.wrap(a), c:wrappers.wrap(a)})
        g.bprop()
        try:
            g.bprop()
            self.assertEqual('should have raised')
        except Exception, e:
            self.assertEqual(str(e), 'bprop has already been done. Consider calling with maybe_redo=True.')
            return
        self.assertEqual('should have caught')

    def test_repeat_bprop1(self):
        """Force repeat bprop"""

        a = numpy.ones((3,3,3))
        z = numpy.zeros((3,3,3))
        b,c,d = _testCase.posnegzero(a)
        #print b, c, d
        g = Grad({b:wrappers.wrap(a), c:wrappers.wrap(z)})
        g.bprop()
        g.bprop(maybe_redo=True)
        max = numpy.max(g(a))
        min = numpy.min(g(a))
        self.assertEqual(max, min)
        self.assertEqual(max, 2.0)

    def tearDown(self):
        core.pop_mode()

if __name__ == '__main__':
    unittest.main()
