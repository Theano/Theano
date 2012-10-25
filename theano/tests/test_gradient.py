
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
from theano.gof.null_type import NullType

one = theano.tensor.as_tensor_variable(1.)


class testgrad_sources_inputs(unittest.TestCase):

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
        self.assertRaises(TypeError, grad_sources_inputs, [(a.out, one)], None)

    def test_wrong_rval_len1(self):
        """Test that it is not ok to return the wrong number of gradient terms
        """
        class retOne(gof.op.Op):
            def make_node(self, *inputs):
                outputs = [theano.tensor.vector()]
                return gof.Apply(self, inputs, outputs)

            def grad(self, inputs, grads):
                return [inputs[0].zeros_like()]

        i = theano.tensor.vector()
        j = theano.tensor.vector()
        a1 = retOne().make_node(i)
        grad_sources_inputs([(a1.out, one)], None)
        a2 = retOne().make_node(i, j)
        self.assertRaises(ValueError, grad_sources_inputs,
                [(a2.out, one)], None)

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
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
        self.assertTrue(g[a1.inputs[0]] is gval)

    def test_1in_Nout(self):
        """Test grad is called correctly for a 1-to-many op"""
        gval = theano.tensor.matrix()

        class O(gof.op.Op):
            def make_node(self):
                inputs = [theano.tensor.matrix()]
                outputs = [theano.tensor.scalar(), theano.tensor.scalar()]
                return gof.Apply(self, inputs, outputs)

            def grad(self, inp, grads):
                x, = inp
                gz1, gz2 = grads
                return gval,
        a1 = O().make_node()
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
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
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
        self.assertTrue(g[a1.inputs[0]] is gval0)
        self.assertTrue(g[a1.inputs[1]] is gval1)

    def test_Nin_Nout(self):
        """Test grad is called correctly for a many-to-many op"""
        gval0 = theano.tensor.matrix()
        gval1 = theano.tensor.matrix()

        class O(gof.op.Op):
            def make_node(self):
                inputs = [theano.tensor.matrix(), theano.tensor.matrix()]
                outputs = [theano.tensor.matrix(), theano.tensor.matrix()]
                return gof.Apply(self, inputs, outputs)

            def grad(self, inp, grads):
                return gval0, gval1
        a1 = O().make_node()
        g = grad_sources_inputs([(a1.outputs[0], one)], None)
        self.assertTrue(g[a1.inputs[0]] is gval0)
        self.assertTrue(g[a1.inputs[1]] is gval1)


class test_grad(unittest.TestCase):

    def test_unimplemented_grad_func(self):
        # tests that function compilation catches unimplemented grads
        # in the graph
        a = theano.tensor.vector()
        b = theano.gradient.grad_not_implemented(theano.tensor.add, 0, a)
        self.assertRaises(TypeError, theano.function,
                [a], b, on_unused_input='ignore')

    def test_undefined_grad_func(self):
        #tests that function compilation catches undefined grads in the graph
        a = theano.tensor.vector()
        b = theano.gradient.grad_undefined(theano.tensor.add, 0, a)
        self.assertRaises(TypeError, theano.function,
                [a], b, on_unused_input='ignore')

    def test_unimplemented_grad_grad(self):
        #tests that unimplemented grads are caught in the grad method

        class DummyOp(gof.Op):
            def make_node(self, x):
                return gof.Apply(self, [x], [x.type()])

            def grad(self, inputs, output_grads):
                return [theano.gradient.grad_not_implemented(
                            self, 0, inputs[0])]

        a = theano.tensor.scalar()
        b = DummyOp()(a)

        self.assertRaises(TypeError, theano.gradient.grad, b, a)

    def test_undefined_grad_grad(self):
        #tests that undefined grads are caught in the grad method

        V = theano.tensor.TensorType(dtype=config.floatX,
                broadcastable=(False, False, False, False, False))()
        W = theano.tensor.TensorType(dtype=config.floatX,
                broadcastable=(False, False, False, False, False))()
        b = theano.tensor.vector()
        d = theano.tensor.ivector()

        Z = conv3D(V, W, b, d)

        self.assertRaises(TypeError, theano.gradient.grad, Z.sum(), d)

    def test_grad_name(self):
        A = theano.tensor.matrix('A')
        x = theano.tensor.vector('x')
        f = theano.tensor.dot(x, theano.tensor.dot(A, x))
        f.name = 'f'
        g = theano.tensor.grad(f, x)
        assert g.name == '(df/dx)'

    def test_grad_duplicate_input(self):

        #test that the grad works when a variable
        #appears in more than one place in a node's input list

        def output(x):
            return (x * x)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)

        theano.tests.unittest_tools.verify_grad(output, [vx])

    def test_grad_quadratic(self):

        #test the gradient on a tiny graph

        def cost(x, A):
            return theano.tensor.dot(x, theano.tensor.dot(A, x))

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(cost, [vx, vA])

    def test_grad_quadratic_vector(self):

        #test the gradient on a small graph

        def output(x, A):
            return theano.tensor.dot(x * x, A)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(output, [vx, vA])

    def test_grad_cubic(self):

        #test the gradient on a bigger graph

        def cost(x, A):
            return theano.tensor.dot(x * x, theano.tensor.dot(A, x))

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(cost, [vx, vA])

    def test_grad_grad_quadratic(self):

        #test the gradient on a graph constructed using the gradient

        def output(x, A):
            orig_cost = theano.tensor.dot(x, theano.tensor.dot(A, x))
            return theano.gradient.grad(orig_cost, x)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(output, [vx, vA])

    def test_grad_grad_cubic(self):

        #test the gradient on a bigger graph constructed using the gradient

        def output(x, A):
            orig_cost = theano.tensor.dot(x * x, theano.tensor.dot(A, x))
            return theano.gradient.grad(orig_cost, x)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(output, [vx, vA])

    def test_grad_int(self):

        # tests that the gradient with respect to an integer
        # is the same as the gradient with respect to a float

        W = theano.tensor.matrix()
        b = theano.tensor.vector()

        def make_grad_func(X):
            Z = theano.tensor.dot(X, W) + b
            H = theano.tensor.nnet.sigmoid(Z)
            cost = H.sum()
            g = gradient.grad(cost, X)
            return theano.function([X, W, b], g, on_unused_input='ignore')

        int_func = make_grad_func(theano.tensor.imatrix())
        #we have to use float64 as the float type to get the results to match
        #using an integer for the input makes all the later functions use
        #float64
        float_func = make_grad_func(theano.tensor.matrix(dtype='float64'))

        m = 5
        d = 3
        n = 4
        rng = np.random.RandomState([2012, 9, 5])

        int_type = theano.tensor.imatrix().dtype
        float_type = 'float64'

        X = np.cast[int_type](rng.randn(m, d) * 127.)
        W = np.cast[W.dtype](rng.randn(d, n))
        b = np.cast[b.dtype](rng.randn(n))

        int_result = int_func(X, W, b)
        float_result = float_func(np.cast[float_type](X), W, b)

        assert np.allclose(int_result, float_result), (
                int_result, float_result)

    def test_grad_disconnected(self):

        #tests corner cases of gradient for shape and alloc

        x = theano.tensor.vector(name='x')
        total = x.sum()
        total.name = 'total'
        num_elements = x.shape[0]
        num_elements.name = 'num_elements'
        silly_vector = theano.tensor.alloc(total / num_elements, num_elements)
        silly_vector.name = 'silly_vector'
        cost = silly_vector.sum()
        cost.name = 'cost'
        #note that cost simplifies to be the same as "total"
        g = gradient.grad(cost, x, add_names=False)
        #we still need to pass in x because it determines the shape of
        #the output
        f = theano.function([x], g)
        rng = np.random.RandomState([2012, 9, 5])
        x = np.cast[x.dtype](rng.randn(3))
        g = f(x)
        assert np.allclose(g, np.ones(x.shape, dtype=x.dtype))

    def test_disconnected_nan(self):

        # test that connection_pattern can prevent getting NaN

        # Op1 has two outputs, f and g
        # x is connected to f but not to g
        class Op1(theano.gof.Op):
            def make_node(self, x):
                return theano.Apply(self, inputs=[x],
                        outputs=[x.type(), theano.tensor.scalar()])

            def connection_pattern(self, node):
                return [[True, False]]

            def grad(self, inputs, output_grads):
                return [inputs[0].zeros_like()]

        # Op2 has two inputs, f and g
        # Its gradient with respect to g is not defined
        class Op2(theano.gof.Op):
            def make_node(self, f, g):
                return theano.Apply(self, inputs=[f, g],
                        outputs=[theano.tensor.scalar()])

            def grad(self, inputs, output_grads):
                return [inputs[0].zeros_like(), NullType()()]

        x = theano.tensor.vector()
        f, g = Op1()(x)
        cost = Op2()(f, g)

        # cost is differentiable wrt x
        # but we can't tell that without using Op1's connection pattern
        # looking at the theano graph alone, g is an ancestor of cost
        # and has x as an ancestor, so we must compute its gradient

        g = gradient.grad(cost, x)

        # If we made it to here without an exception, then the
        # connection_pattern functionality worked correctly

    def test_sum_disconnected(self):

        # Tests that we can add DisconnectedType to other terms correctly
        x = theano.tensor.scalar()
        y = x * 2.
        z = x + 1.
        cost = y + z
        theano.tensor.grad(cost, x, consider_constant=[y, z])
        # In an earlier version of theano, the above line would have failed
        # while trying to add two DisconnectedTypes

    def test_output_grad_on_int(self):
        # If the g_cost argument is specified when x has a discrete dtype,
        # g_cost should be equivalent to 0.
        x = theano.tensor.iscalar('x')
        y = x * 2

        # Should work:
        c0 = theano.tensor.constant(0)
        theano.grad(y, x, g_cost=c0)
        theano.grad(y, x, g_cost=y.zeros_like())
        theano.grad(y, x, g_cost=y.zeros_like().astype('float64'))

        # Should raise ValueError
        c1 = theano.tensor.constant(1)
        self.assertRaises(ValueError, theano.grad, y, x, g_cost=c1)
        s0 = theano.shared(np.zeros((), dtype='int8'))
        self.assertRaises(ValueError, theano.grad, y, x, g_cost=s0)

    def test_downcast_dtype(self):
        # Test that the gradient of a cost wrt a float32 variable does not
        # get upcasted to float64.
        # x has dtype float32, regardless of the value of floatX
        x = theano.tensor.fscalar('x')
        y = x * 2
        z = theano.tensor.lscalar('z')

        c = y + z
        dc_dx, dc_dy, dc_dz, dc_dc = theano.grad(c, [x, y, z, c])
        # The dtype of dc_dy and dc_dz can be either float32 or float64,
        # that might depend on floatX, but is not specified.
        assert dc_dc.dtype in ('float32', 'float64')
        assert dc_dz.dtype in ('float32', 'float64')
        assert dc_dy.dtype in ('float32', 'float64')

        # When the output gradient of y is passed to op.grad, it should
        # be downcasted to float32, so dc_dx should also be float32
        assert dc_dx.dtype == 'float32'


if __name__ == '__main__':
    unittest.main()
