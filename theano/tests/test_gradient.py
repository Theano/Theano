
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
import numpy as np

one = theano.tensor.as_tensor_variable(1.)

def _grad_sources_inputs(*args):
    # warn_type was introduced after this code, it complains throughout for nothing.
    return grad_sources_inputs(warn_type=False, *args)

class test_grad_sources_inputs(unittest.TestCase):

    def test_retNone1(self):
        """Test that it is not ok to return None from op.grad()"""
        class retNone(gof.op.Op):
            def make_node(self):
                inputs = [theano.tensor.vector()]
                outputs = [theano.tensor.vector()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                x, = inp
                gz, = grads
                pass
        a = retNone().make_node()
        try:
            _grad_sources_inputs([(a.out, one)], None)
        except TypeError, e:
            return
        self.fail()

    def test_wrong_rval_len1(self):
        """Test that it is not ok to return the wrong number of gradient terms"""
        class retOne(gof.op.Op):
            def make_node(self, *inputs):
                outputs = [theano.tensor.vector()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, grads):
                return [ inputs[0].zeros_like() ]

        i = theano.tensor.vector()
        j = theano.tensor.vector()
        a1 = retOne().make_node(i)
        g = _grad_sources_inputs([(a1.out, one)], None)
        a2 = retOne().make_node(i,j)
        try:
            g = _grad_sources_inputs([(a2.out, one)], None)
        except ValueError, e:
            return
        self.fail()

    def test_1in_1out(self):
        """Test grad is called correctly for a 1-to-1 op"""
        gval = theano.tensor.matrix()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [theano.tensor.matrix()]
                outputs = [theano.tensor.matrix()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                return gval,
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], one)], None)
        self.assertTrue(g[a1.inputs[0]] is gval)

    def test_1in_Nout(self):
        """Test grad is called correctly for a 1-to-many op"""
        gval = theano.tensor.matrix()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [theano.tensor.matrix()]
                outputs = [theano.tensor.scalar(),theano.tensor.scalar()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                x, = inp
                gz1, gz2 = grads
                return gval,
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], one)], None)
        self.assertTrue(g[a1.inputs[0]] is gval)

    def test_Nin_1out(self):
        """Test grad is called correctly for a many-to-1 op"""
        gval0 = theano.tensor.scalar()
        gval1 = theano.tensor.scalar()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [theano.tensor.scalar(), theano.tensor.scalar()]
                outputs = [theano.tensor.matrix()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                x0, x1 = inp
                gz, = grads
                return (gval0, gval1)
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], one)], None)
        self.assertTrue(g[a1.inputs[0]] is gval0)
        self.assertTrue(g[a1.inputs[1]] is gval1)

    def test_Nin_Nout(self):
        """Test grad is called correctly for a many-to-many op"""
        gval0 = theano.tensor.matrix()
        gval1 = theano.tensor.matrix()
        class O(gof.op.Op):
            def make_node(self):
                inputs = [theano.tensor.matrix(),theano.tensor.matrix()]
                outputs = [theano.tensor.matrix(),theano.tensor.matrix()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inp, grads):
                return gval0, gval1
        a1 = O().make_node()
        g = _grad_sources_inputs([(a1.outputs[0], one)], None)
        self.assertTrue(g[a1.inputs[0]] is gval0)
        self.assertTrue(g[a1.inputs[1]] is gval1)

    def test_some_None_ograds(self):
        """Test grad is called when some output gradients are None"""
        class O(gof.op.Op):
            def __init__(self, tst):
                self.tst = tst
            def make_node(self, *inputs):
                outputs = [theano.tensor.matrix(),theano.tensor.matrix()]
                return gof.Apply(self, inputs, outputs)
            def grad(self, inputs, g_out):
                return [one]
        i = theano.tensor.matrix()
        a1 = O(self).make_node(i)
        g = grad_sources_inputs([(a1.outputs[0], one)], None, warn_type=False)
        self.assertTrue(g[i] is one)

def test_unimplemented_grad_func():
    # tests that function compilation catches unimplemented grads in the graph
    a = theano.tensor.vector()
    b = theano.gradient.grad_not_implemented(theano.tensor.add, 0, a)
    try:
        f = theano.function([a], b, on_unused_input = 'ignore')
        assert 0
    except TypeError:
        pass

def test_undefined_grad_func():
    #tests that function compilation catches undefined grads in the graph
    a = theano.tensor.vector()
    b = theano.gradient.grad_undefined(theano.tensor.add, 0, a)
    try:
        f = theano.function([a],b, on_unused_input = 'ignore')
        assert 0
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



def test_grad_duplicate_input():

    #test that the grad works when a variable
    #appears in more than one place in a node's input list

    def output(x):
        return (x*x)

    rng = np.random.RandomState([2012,8,28])

    vx = rng.randn(2)

    theano.tests.unittest_tools.verify_grad(output,[vx])

def test_grad_quadratic():

    #test the gradient on a tiny graph

    def cost(x,A):
        return theano.tensor.dot(x,theano.tensor.dot(A,x))

    rng = np.random.RandomState([2012,8,28])

    vx = rng.randn(2)
    vA = rng.randn(2,2)

    theano.tests.unittest_tools.verify_grad(cost,[vx,vA])


def test_grad_quadratic_vector():

    #test the gradient on a small graph

    def output(x,A):
        return theano.tensor.dot(x*x,A)

    rng = np.random.RandomState([2012,8,28])

    vx = rng.randn(2)
    vA = rng.randn(2,2)

    theano.tests.unittest_tools.verify_grad(output,[vx,vA])


def test_grad_cubic():

    #test the gradient on a bigger graph

    def cost(x,A):
        return theano.tensor.dot(x*x,theano.tensor.dot(A,x))

    rng = np.random.RandomState([2012,8,28])

    vx = rng.randn(2)
    vA = rng.randn(2,2)

    theano.tests.unittest_tools.verify_grad(cost,[vx,vA])

def test_grad_grad_quadratic():

    #test the gradient on a graph constructed using the gradient

    def output(x,A):
        orig_cost = theano.tensor.dot(x,theano.tensor.dot(A,x))
        return theano.gradient.grad(orig_cost, x)

    rng = np.random.RandomState([2012,8,28])

    vx = rng.randn(2)
    vA = rng.randn(2,2)

    theano.tests.unittest_tools.verify_grad(output,[vx,vA])

def test_grad_grad_cubic():

    #test the gradient on a bigger graph constructed using the gradient

    def output(x,A):
        orig_cost = theano.tensor.dot(x*x,theano.tensor.dot(A,x))
        return theano.gradient.grad(orig_cost, x)

    rng = np.random.RandomState([2012,8,28])

    vx = rng.randn(2)
    vA = rng.randn(2,2)

    theano.tests.unittest_tools.verify_grad(output,[vx,vA])


def test_grad_disconnected():

    #tests corner cases of gradient for shape and alloc

    x = theano.tensor.vector(name = 'x')
    total = x.sum()
    total.name = 'total'
    num_elements = x.shape[0]
    num_elements.name = 'num_elements'
    silly_vector = theano.tensor.alloc( total / num_elements, num_elements)
    silly_vector.name = 'silly_vector'
    cost = silly_vector.sum()
    cost.name = 'cost'
    #note that cost simplifies to be the same as "total"
    g = gradient.grad(cost, x, add_names = False)
    #we still need to pass in x because it determines the shape of the output
    f = theano.function([x],g)
    rng = np.random.RandomState([2012,9,5])
    x = np.cast[x.dtype](rng.randn(3))
    g = f(x)
    assert np.allclose(g,np.ones(x.shape,dtype=x.dtype))


if __name__ == '__main__':
    unittest.main()
