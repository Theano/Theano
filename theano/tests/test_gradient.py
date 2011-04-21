
#
# UNIT TEST
#
import unittest
import numpy
import theano
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
            def grad(self, inp, grads):
                x, = inp
                gz, = grads
                pass
        a = retNone().make_node()
        try:
            _grad_sources_inputs([(a.out, 1)], None)
        except ValueError, e:
            self.assertTrue(e[0] is gradient._msg_retType)
            return
        self.fail()
    def test_retNone1_b(self):
        """Test that it is ok to return [None] from op.grad()"""
        class retNone(gof.op.Op):
            def make_node(self, *inputs):
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                return [None]
        i = gof.generic()
        a = retNone().make_node(i)
        g = _grad_sources_inputs([(a.out, 1)], None)
        self.assertTrue(not i in g)

    def test_wrong_rval_len1(self):
        """Test that it is not ok to return the wrong number of gradients"""
        class retNone(gof.op.Op):
            def make_node(self, *inputs):
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, grads):
                return [None]

        i = gof.generic()
        j = gof.generic()
        a1 = retNone().make_node(i)
        g = _grad_sources_inputs([(a1.out, 1)], None)
        a2 = retNone().make_node(i,j)
        try:
            g = _grad_sources_inputs([(a2.out, 1)], None)
        except ValueError, e:
            self.assertTrue(e[0] is gradient._msg_badlen)
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
            def grad(self, inputs, grads):
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
            def grad(self, inp, grads):
                return gval,
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.assertTrue(g[a1.inputs[0]] is gval)

    def test_1in_Nout(self):
        """Test grad is called correctly for a 1-to-many op"""
        gval = gof.generic()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic()]
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                x, = inp
                gz1, gz2 = grads
                return gval,
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.assertTrue(g[a1.inputs[0]] is gval)
    def test_Nin_1out(self):
        """Test grad is called correctly for a many-to-1 op"""
        gval0 = gof.generic()
        gval1 = gof.generic()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic(),gof.generic()]
                outputs = [gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                x0, x1 = inp
                gz, = grads
                return (gval0, gval1)
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.assertTrue(g[a1.inputs[0]] is gval0)
        self.assertTrue(g[a1.inputs[1]] is gval1)
    def test_Nin_Nout(self):
        """Test grad is called correctly for a many-to-many op"""
        gval0 = gof.generic()
        gval1 = gof.generic()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [gof.generic(),gof.generic()]
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                return gval0, gval1
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], 1)], None)
        self.assertTrue(g[a1.inputs[0]] is gval0)
        self.assertTrue(g[a1.inputs[1]] is gval1)
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
        self.assertTrue(g[i] is 1)

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
        self.assertTrue(g[i] is 1 and j not in g and k not in g)

        a1 = O(self, True).make_node(i,j)
        a2 = O(self, True).make_node(k, a1.outputs[1])
        g = _grad_sources_inputs([(a2.outputs[0], 1)], None)
        self.assertTrue(g[k] is 1 and i not in g and j not in g)

    def test_inputs(self):
        """Test that passing inputs shortens the traversal"""
        class O(gof.op.Op):
            def __init__(self, tst, grad_ok):
                self.tst = tst
                self.grad_ok = grad_ok
            def make_node(self, *inputs):
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, grads):
                g0, g1 = grads
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
        self.assertTrue(g[a2.inputs[0]] == 1)
        self.assertTrue(g[a2.inputs[1]] == 5)
        self.assertTrue(g[a1.outputs[0]] == 6)
        self.assertTrue(g[a1.outputs[1]] == 5)
        self.assertTrue(a1.inputs[0] not in g)
        self.assertTrue(a1.inputs[1] not in g)

    def test_multiple_sources(self):
        """Test that passing multiple sources works"""
        class O(gof.op.Op):
            def __init__(self, tst, grad_ok):
                self.tst = tst
                self.grad_ok = grad_ok
            def make_node(self, *inputs):
                outputs = [gof.generic(),gof.generic()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, grads):
                g0, g1 = grads
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
        self.assertTrue(g[a2.inputs[0]] == 1)
        self.assertTrue(g[a2.inputs[1]] == 5)
        self.assertTrue(g[a1.outputs[0]] == 6)
        self.assertTrue(g[a1.outputs[1]] == 5)
        self.assertTrue(g[a1.inputs[0]] == 6)
        self.assertTrue(g[a1.inputs[1]] == 11)

def test_unimplemented_grad():
    a = theano.tensor.vector()
    b = theano.gradient.unimplemented_grad(theano.tensor.add, 1, a)
    f = theano.function([a], b)
    try:
        f([1,2,3])
        assert 0
    except NotImplementedError:
        pass

if __name__ == '__main__':
    unittest.main()
