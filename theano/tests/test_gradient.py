
#
# UNIT TEST
#
import unittest
import numpy
from theano import gof

from theano.gradient import *
from theano import gradient


def _grad_sources_inputs(*args):
    # warn_type was introduced after this code, it complains throughout for nothing.
    return grad_sources_inputs(warn_type=False, *args)

class test_grad_sources_inputs(unittest.TestCase):
    def test_retNone1(self): 
        """Test that it is not ok to return None from op.grad()"""
        class retNone(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic()]
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, (x, ), (gz, )):
                pass
        a = retNone().make_node()
        try:
            _grad_sources_inputs([(a.out, 1)], None)
        except ValueError, e:
            self.failUnless(e[0] is gradient._msg_retType)
            return
        self.fail()
    def test_retNone1_b(self): 
        """Test that it is ok to return [None] from op.grad()"""
        class retNone(gof.op.Op):
            def make_node(self, *inputs):
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, (x, ), (gz, )):
                return [None]
        i = gof.generic()
        a = retNone().make_node(i)
        g = _grad_sources_inputs([(a.out, 1)], None)
        self.failUnless(not i in g)

    def test_wrong_rval_len1(self): 
        """Test that it is not ok to return the wrong number of gradients"""
        class retNone(gof.op.Op):
            def make_node(self, *inputs):
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, (gz, )):
                return [None]

        i = gof.generic()
        j = gof.generic()
        a1 = retNone().make_node(i)
        g = _grad_sources_inputs([(a1.out, 1)], None)
        a2 = retNone().make_node(i,j)
        try:
            g = _grad_sources_inputs([(a2.out, 1)], None)
        except ValueError, e:
            self.failUnless(e[0] is gradient._msg_badlen)
            return
        self.fail()


    def test_stop_on_all_none(self):
        """Test that op.grad() is not called when output grads are all None"""
        class retNone(gof.op.Op):
            def __init__(self, tst):
                self.tst = tst
            def make_node(self, *inputs):
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, (gz, )):
                self.tst.fail()

        i = gof.generic()
        a1 = retNone(self).make_node(i)
        g = _grad_sources_inputs([(a1.out, None)], None)

    def test_1in_1out(self):
        """Test grad is called correctly for a 1-to-1 op"""
        gval = gof.generic()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic()]
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, (x, ), (gz, )):
                return gval,
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval)

    def test_1in_Nout(self):
        """Test grad is called correctly for a 1-to-many op"""
        gval = gof.generic()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic()]
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, (x, ), (gz1, gz2)):
                return gval,
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval)
    def test_Nin_1out(self):
        """Test grad is called correctly for a many-to-1 op"""
        gval0 = gof.generic()
        gval1 = gof.generic()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic(),gof.generic()]
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, (x0,x1), (gz, )):
                return (gval0, gval1)
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval0)
        self.failUnless(g[a1.inputs[1]] is gval1)
    def test_Nin_Nout(self):
        """Test grad is called correctly for a many-to-many op"""
        gval0 = gof.generic()
        gval1 = gof.generic()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic(),gof.generic()]
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, (x0,x1), (gz0,gz1)):
                return gval0, gval1
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.failUnless(g[a1.inputs[0]] is gval0)
        self.failUnless(g[a1.inputs[1]] is gval1)
    def test_some_None_ograds(self):
        """Test grad is called when some output gradients are None"""
        class O(gof.op.Op):
            def __init__(self, tst):
                self.tst = tst
            def make_node(self, *inputs):
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, g_out):
                return [1]
        i = gof.generic()
        a1 = O(self).make_node(i)
        g = grad_sources_inputs([(a1.outputs[0], 1)], None, warn_type=False)
        self.failUnless(g[i] is 1)

    def test_some_None_igrads(self):
        """Test that traversal works properly when an op return some None"""
        class O(gof.op.Op):
            def __init__(self, tst, grad_ok):
                self.tst = tst
                self.grad_ok = grad_ok
            def make_node(self, *inputs):
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, g_out):
                if not self.grad_ok:
                    self.tst.fail()
                else:
                    return [1, None]
        i = gof.generic()
        j = gof.generic()
        k = gof.generic()
        a1 = O(self, True).make_node(i,j)
        a2 = O(self, True).make_node(a1.outputs[1], k)
        g = grad_sources_inputs([(a2.outputs[0], 1)], None, warn_type=False)
        self.failUnless(g[i] is 1 and j not in g and k not in g)

        a1 = O(self, True).make_node(i,j)
        a2 = O(self, True).make_node(k, a1.outputs[1])
        g = _grad_sources_inputs([(a2.outputs[0], 1)], None)
        self.failUnless(g[k] is 1 and i not in g and j not in g)

    def test_inputs(self):
        """Test that passing inputs shortens the traversal"""
        class O(gof.op.Op):
            def __init__(self, tst, grad_ok):
                self.tst = tst
                self.grad_ok = grad_ok
            def make_node(self, *inputs):
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, (g0,g1)):
                if not self.grad_ok:
                    self.tst.fail()
                else:
                    if g1:
                        return [g0, g0+g1]
                    else:
                        return [g0, g0]
        i = gof.generic()
        j = gof.generic()
        k = gof.generic()
        a1 = O(self, True).make_node(i,j)
        a2 = O(self, True).make_node(k,a1.outputs[1])
        g = _grad_sources_inputs([(a2.outputs[0], 1), (a1.outputs[1],4),
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
            def __init__(self, tst, grad_ok):
                self.tst = tst
                self.grad_ok = grad_ok
            def make_node(self, *inputs):
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, (g0,g1)):
                if not self.grad_ok:
                    self.tst.fail()
                else:
                    if g1:
                        return [g0, g0+g1]
                    else:
                        return [g0, g0]
        i = gof.generic()
        j = gof.generic()
        k = gof.generic()
        a1 = O(self,True).make_node(i,j)
        a2 = O(self,True).make_node(k,a1.outputs[1])
        g = _grad_sources_inputs([(a2.outputs[0], 1), (a1.outputs[1],4),
            (a1.outputs[0], 3), (a1.outputs[0], 3)], None)
        self.failUnless(g[a2.inputs[0]] == 1)
        self.failUnless(g[a2.inputs[1]] == 5)
        self.failUnless(g[a1.outputs[0]] == 6)
        self.failUnless(g[a1.outputs[1]] == 5)
        self.failUnless(g[a1.inputs[0]] == 6)
        self.failUnless(g[a1.inputs[1]] == 11)


if __name__ == '__main__':
    unittest.main()
