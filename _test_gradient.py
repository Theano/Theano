
#
# UNIT TEST
#
import unittest
import numpy
import tensor_ops as T
import tensor
import gof

from gradient import *
import gradient

class posneg(T.TensorOp):
    nout=2
    def impl(self, x): return x, -x
    def grad(self, x, (gpos, gneg)): return gpos - gneg

class posnegzero(T.TensorOp):
    nout=3
    def impl(self, x): return x, -x, 0.0
    def grad(self, x, (gpos, gneg, gzero)): return gpos - gneg

class _test_grad_sources_inputs(unittest.TestCase):
    def test_retNone1(self): 
        """Test that it is not ok to return None from op.grad()"""
        class retNone(gof.op.Op):
            def __init__(self, arg):
                self.inputs = [gof.result.ResultBase()]
                self.outputs = [gof.result.ResultBase()]
            def grad(self, x, gz):
                pass
        a = retNone(5)
        try:
            grad_sources_inputs([(a.out, 1)], None)
        except ValueError, e:
            self.failUnless(e[0] is gradient._msg_retNone)
            return
        self.fail()
    def test_retNone1_b(self): 
        """Test that it is ok to return [None] from op.grad()"""
        class retNone(gof.op.Op):
            def __init__(self, arg):
                self.inputs = arg
                self.outputs = [gof.result.ResultBase()]
            def grad(self, x, gz):
                return [None]
        i = gof.result.ResultBase()
        a = retNone([i])
        g = grad_sources_inputs([(a.out, 1)], None)
        self.failUnless(not i in g)

    def test_wrong_rval_len1(self): 
        """Test that it is not ok to return the wrong number of gradients"""
        class retNone(gof.op.Op):
            def __init__(self, arg):
                self.inputs = arg
                self.outputs = [gof.result.ResultBase()]
            def grad(self, inputs, gz):
                return [None]

        i = gof.result.ResultBase()
        j = gof.result.ResultBase()
        a1 = retNone([i])
        g = grad_sources_inputs([(a1.out, 1)], None)
        a2 = retNone([i,j])
        try:
            g = grad_sources_inputs([(a2.out, 1)], None)
        except ValueError, e:
            self.failUnless(e[0] is gradient._msg_badlen)
            return
        self.fail()


    def test_stop_on_all_none(self):
        """Test that op.grad() is not called when output grads are all None"""
        class retNone(gof.op.Op):
            def __init__(self, arg, tst):
                self.inputs = arg
                self.outputs = [gof.result.ResultBase()]
                self.tst = tst
            def grad(self, inputs, gz):
                self.tst.fail()

        i = gof.result.ResultBase()
        a1 = retNone([i],self)
        g = grad_sources_inputs([(a1.out, None)], None)

    def test_no_invalid_graph(self):
        """Test that bprop fails on an invalid graph"""
        raise NotImplementedError()
    def test_1in_1out(self):
        """Test grad is called correctly for a 1-to-1 op"""
        gval = gof.result.ResultBase()
        class O(gof.op.Op):
            def __init__(self):
                self.inputs = [gof.result.ResultBase()]
                self.outputs = [gof.result.ResultBase()]
            def grad(self, x, gz):
                return gval
        a1 = O()
        g = grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval)

    def test_1in_Nout(self):
        """Test grad is called correctly for a 1-to-many op"""
        gval = gof.result.ResultBase()
        class O(gof.op.Op):
            def __init__(self):
                self.inputs = [gof.result.ResultBase()]
                self.outputs = [gof.result.ResultBase(),gof.result.ResultBase()]
            def grad(self, x, (gz1, gz2)):
                return gval
        a1 = O()
        g = grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval)
    def test_Nin_1out(self):
        """Test grad is called correctly for a many-to-1 op"""
        gval0 = gof.result.ResultBase()
        gval1 = gof.result.ResultBase()
        class O(gof.op.Op):
            def __init__(self):
                self.inputs = [gof.result.ResultBase(),gof.result.ResultBase()]
                self.outputs = [gof.result.ResultBase()]
            def grad(self, (x0,x1), gz):
                return (gval0, gval1)
        a1 = O()
        g = grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval0)
        self.failUnless(g[a1.inputs[1]] is gval1)
    def test_Nin_Nout(self):
        """Test grad is called correctly for a many-to-many op"""
        gval0 = gof.result.ResultBase()
        gval1 = gof.result.ResultBase()
        class O(gof.op.Op):
            def __init__(self):
                self.inputs = [gof.result.ResultBase(),gof.result.ResultBase()]
                self.outputs = [gof.result.ResultBase(),gof.result.ResultBase()]
            def grad(self, (x0,x1), (gz0,gz1)):
                return gval0, gval1
        a1 = O()
        g = grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval0)
        self.failUnless(g[a1.inputs[1]] is gval1)
    def test_some_None_ograds(self):
        """Test grad is called when some output gradients are None"""
        class O(gof.op.Op):
            def __init__(self, arg, tst):
                self.inputs = arg
                self.outputs = [gof.result.ResultBase(),gof.result.ResultBase()]
                self.tst = tst
            def grad(self, inputs, g_out):
                return [1]
        i = gof.result.ResultBase()
        a1 = O([i],self)
        g = grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[i] is 1)

    def test_some_None_igrads(self):
        """Test that traversal works properly when an op return some None"""
        class O(gof.op.Op):
            def __init__(self, arg, tst, grad_ok):
                self.inputs = arg
                self.outputs = [gof.result.ResultBase(),gof.result.ResultBase()]
                self.tst = tst
                self.grad_ok = grad_ok
            def grad(self, inputs, g_out):
                if not self.grad_ok:
                    self.tst.fail()
                else:
                    return [1, None]
        i = gof.result.ResultBase()
        j = gof.result.ResultBase()
        k = gof.result.ResultBase()
        a1 = O([i,j],self,True)
        a2 = O([a1.outputs[1], k], self, True)
        g = grad_sources_inputs([(a2.outputs[0], 1)], None)
        self.failUnless(g[i] is 1 and j not in g and k not in g)

        a1 = O([i,j],self,True)
        a2 = O([k, a1.outputs[1]], self, True)
        g = grad_sources_inputs([(a2.outputs[0], 1)], None)
        self.failUnless(g[k] is 1 and i not in g and j not in g)

    def test_inputs(self):
        """Test that passing inputs shortens the traversal"""
        class O(gof.op.Op):
            def __init__(self, arg, tst, grad_ok):
                self.inputs = arg
                self.outputs = [gof.result.ResultBase(),gof.result.ResultBase()]
                self.tst = tst
                self.grad_ok = grad_ok
            def grad(self, inputs, (g0,g1)):
                if not self.grad_ok:
                    self.tst.fail()
                else:
                    if g1:
                        return [g0, g0+g1]
                    else:
                        return [g0, g0]
        i = gof.result.ResultBase()
        j = gof.result.ResultBase()
        k = gof.result.ResultBase()
        a1 = O([i,j],self,True)
        a2 = O([k,a1.outputs[1]], self, True)
        g = grad_sources_inputs([(a2.outputs[0], 1), (a1.outputs[1],4),
            (a1.outputs[0], 3), (a1.outputs[0], 3)], a1.outputs)
        self.failUnless(g[a2.inputs[0]] == 1)
        self.failUnless(g[a2.inputs[1]] == 5)
        self.failUnless(g[a1.outputs[0]] == 6)
        self.failUnless(g[a1.outputs[1]] == 5)
        self.failUnless(a1.inputs[0] not in g)
        self.failUnless(a1.inputs[1] not in g)

    def test_multiple_sources(self):
        """Test that passing multiple sources works"""
        class O(gof.op.Op):
            def __init__(self, arg, tst, grad_ok):
                self.inputs = arg
                self.outputs = [gof.result.ResultBase(),gof.result.ResultBase()]
                self.tst = tst
                self.grad_ok = grad_ok
            def grad(self, inputs, (g0,g1)):
                if not self.grad_ok:
                    self.tst.fail()
                else:
                    if g1:
                        return [g0, g0+g1]
                    else:
                        return [g0, g0]
        i = gof.result.ResultBase()
        j = gof.result.ResultBase()
        k = gof.result.ResultBase()
        a1 = O([i,j],self,True)
        a2 = O([k,a1.outputs[1]], self, True)
        g = grad_sources_inputs([(a2.outputs[0], 1), (a1.outputs[1],4),
            (a1.outputs[0], 3), (a1.outputs[0], 3)], None)
        self.failUnless(g[a2.inputs[0]] == 1)
        self.failUnless(g[a2.inputs[1]] == 5)
        self.failUnless(g[a1.outputs[0]] == 6)
        self.failUnless(g[a1.outputs[1]] == 5)
        self.failUnless(g[a1.inputs[0]] == 6)
        self.failUnless(g[a1.inputs[1]] == 11)


class _test_grad(unittest.TestCase):
    class O(gof.op.Op):
        def __init__(self):
            self.inputs = [gof.result.ResultBase(),gof.result.ResultBase()]
            self.outputs = [gof.result.ResultBase(),gof.result.ResultBase()]
            self.gval0 = gof.result.ResultBase()
            self.gval1 = gof.result.ResultBase()
        def grad(self, (x0,x1), (gz0,gz1)):
            return self.gval0, self.gval1

    def test_1param(self):
        """grad: Test passing a single result param"""
        a1 = _test_grad.O()
        self.failUnless(a1.gval0 is grad(a1.outputs[0], a1.inputs[0]))

    def test_Nparam(self):
        """grad: Test passing multiple result params"""
        a1 = _test_grad.O()
        g0,g1 = grad(a1.outputs[0], a1.inputs)
        self.failUnless(a1.gval0 is g0)
        self.failUnless(a1.gval1 is g1)

    def test_1None_rval(self):
        """grad: Test returning a single None from grad"""
        a1 = _test_grad.O()
        self.failUnless(None is grad(a1.outputs[0], a1.outputs[1]))
        self.failUnless(None is grad(a1.outputs[0], 'wtf'))
    def test_NNone_rval(self):
        """grad: Test returning some Nones from grad"""
        a1 = _test_grad.O()
        g0,g1,g2 = grad(a1.outputs[0], a1.inputs + ['wtf'])
        self.failUnless(a1.gval0 is g0)
        self.failUnless(a1.gval1 is g1)
        self.failUnless(None is g2)




def matrix():
    return tensor.Tensor('float64', [0,0])

def matrices(n):
    return [matrix() for i in xrange(n)]


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

    def tearDown(self):
        core.pop_mode()

if __name__ == '__main__':
    unittest.main()
