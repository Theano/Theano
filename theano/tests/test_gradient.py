
#
# UNIT TEST
#
import unittest
import theano
from theano import gof

from theano.gradient import grad_sources_inputs
from theano import gradient
from theano.tensor.nnet.Conv3D import conv3D
from theano import config


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

def test_unimplemented_grad_func():
    #tests that function compilation catches unimplemented grads in the graph
    a = theano.tensor.vector()
    b = theano.gradient.grad_not_implemented(theano.tensor.add, 0, a)
    try:
        f = theano.function([a], b, on_unused_input = 'ignore')
        assert 0
        #Note: it's important that the NotImplementedGradOp is caught
        #at COMPILATION time, not execution time.
        #If the uncomputable variable is, for example, multiplied by 0,
        #it could be optimized out of the final graph.
    except TypeError:
        pass

def test_undefined_grad_func():
    #tests that function compilation catches undefined grads in the graph
    a = theano.tensor.vector()
    b = theano.gradient.grad_undefined(theano.tensor.add, 0, a)
    try:
        f = theano.function([a],b, on_unused_input = 'ignore')
        assert 0
        #Note: it's important that the GradUndefinedOp is caught at
        #COMPILATION time, not execution time.
        #If the uncomputable variable is, for example, multiplied by0,
        #it could be optimized out of the final graph
    except TypeError:
        pass

def test_unimplemented_grad_grad():
    #tests that unimplemented grads are caught in the grad method

    class DummyOp(gof.Op):
        def make_node(self, x):
            return gof.Apply(self, [x], [x.type()])

        def grad(self, inputs, output_grads):
            return [ theano.gradient.grad_not_implemented(self, 0, inputs[0]) ]

    a = theano.tensor.scalar()
    b = DummyOp()(a)

    try:
        g = theano.gradient.grad(b,a)
        assert False
    except TypeError:
        pass

def test_undefined_grad_grad():
    #tests that undefined grads are caught in the grad method

    V = theano.tensor.TensorType(dtype=config.floatX,
            broadcastable = (False,False,False,False,False))()
    W = theano.tensor.TensorType(dtype=config.floatX,
            broadcastable = (False, False, False, False, False))()
    b = theano.tensor.vector()
    d = theano.tensor.ivector()

    Z = conv3D(V,W,b,d)

    try:
        g = theano.gradient.grad(Z.sum(),d)
        assert False
    except TypeError:
        pass

def test_grad_name():
    A = theano.tensor.matrix('A')
    x = theano.tensor.vector('x')
    f = theano.tensor.dot(x,theano.tensor.dot(A,x))
    f.name = 'f'
    g = theano.tensor.grad(f,x)
    assert g.name == '(df/dx)'

if __name__ == '__main__':
    unittest.main()
