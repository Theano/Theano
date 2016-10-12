from __future__ import absolute_import, print_function, division
from collections import OrderedDict
#
# UNIT TEST
#
import unittest

import numpy as np
from six.moves import xrange

import theano
from theano import gof
from theano.compat import izip
from theano.tests import unittest_tools as utt

from theano import gradient
from theano import config
from theano.gof.null_type import NullType

one = theano.tensor.as_tensor_variable(1.)


def grad_sources_inputs(sources, inputs):
    """
    This implements the old grad_sources_inputs function in terms of
    the new interface so the tests don't need to be rewritten.
    """
    if inputs is None:
        inputs = theano.gof.graph.inputs([source[0] for source in sources])
    return dict(izip(inputs, theano.gradient.grad(cost=None, known_grads=dict(sources),
                                                  wrt=inputs, consider_constant=inputs)))


class testgrad_sources_inputs(unittest.TestCase):

    def test_retNone1(self):
        """Test that it is not ok to return None from op.grad()"""
        class retNone(gof.op.Op):
            __props__ = ()

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
            __props__ = ()

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
        self.assertRaises(ValueError, grad_sources_inputs, [(a2.out, one)], None)

    def test_1in_1out(self):
        """Test grad is called correctly for a 1-to-1 op"""
        gval = theano.tensor.matrix()

        class O(gof.op.Op):
            __props__ = ()

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
            __props__ = ()

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
            __props__ = ()

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
            __props__ = ()

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
        self.assertRaises(TypeError, theano.function, [a], b, on_unused_input='ignore')

    def test_undefined_grad_func(self):
        # tests that function compilation catches undefined grads in the graph
        a = theano.tensor.vector()
        b = theano.gradient.grad_undefined(theano.tensor.add, 0, a)
        self.assertRaises(TypeError, theano.function, [a], b, on_unused_input='ignore')

    def test_unimplemented_grad_grad(self):
        # tests that unimplemented grads are caught in the grad method

        class DummyOp(gof.Op):
            __props__ = ()

            def make_node(self, x):
                return gof.Apply(self, [x], [x.type()])

            def grad(self, inputs, output_grads):
                return [theano.gradient.grad_not_implemented(self, 0, inputs[0])]

        a = theano.tensor.scalar()
        b = DummyOp()(a)

        self.assertRaises(TypeError, theano.gradient.grad, b, a)

    def test_undefined_grad_grad(self):
        # tests that undefined grads are caught in the grad method

        class DummyOp(gof.Op):
            __props__ = ()

            def make_node(self, x):
                return gof.Apply(self, [x], [x.type()])

            def grad(self, inputs, output_grads):
                return [theano.gradient.grad_undefined(self, 0, inputs[0])]

        a = theano.tensor.scalar()
        b = DummyOp()(a)

        self.assertRaises(TypeError, theano.gradient.grad, b, a)

    def test_grad_name(self):
        A = theano.tensor.matrix('A')
        x = theano.tensor.vector('x')
        f = theano.tensor.dot(x, theano.tensor.dot(A, x))
        f.name = 'f'
        g = theano.tensor.grad(f, x)
        assert g.name == '(df/dx)'

    def test_grad_duplicate_input(self):

        # test that the grad works when a variable
        # appears in more than one place in a node's input list

        def output(x):
            return (x * x)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)

        theano.tests.unittest_tools.verify_grad(output, [vx])

    def test_grad_quadratic(self):

        # test the gradient on a tiny graph

        def cost(x, A):
            return theano.tensor.dot(x, theano.tensor.dot(A, x))

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(cost, [vx, vA])

    def test_grad_quadratic_vector(self):

        # test the gradient on a small graph

        def output(x, A):
            return theano.tensor.dot(x * x, A)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(output, [vx, vA])

    def test_grad_cubic(self):

        # test the gradient on a bigger graph

        def cost(x, A):
            return theano.tensor.dot(x * x, theano.tensor.dot(A, x))

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(cost, [vx, vA])

    def test_grad_grad_quadratic(self):

        # test the gradient on a graph constructed using the gradient

        def output(x, A):
            orig_cost = theano.tensor.dot(x, theano.tensor.dot(A, x))
            return theano.gradient.grad(orig_cost, x)

        rng = np.random.RandomState([2012, 8, 28])

        vx = rng.randn(2)
        vA = rng.randn(2, 2)

        theano.tests.unittest_tools.verify_grad(output, [vx, vA])

    def test_grad_grad_cubic(self):

        # test the gradient on a bigger graph constructed using the gradient

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
        # we have to use float64 as the float type to get the results to match
        # using an integer for the input makes all the later functions use
        # float64
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

        assert np.allclose(int_result, float_result), (int_result, float_result)

    def test_grad_disconnected(self):

        # tests corner cases of gradient for shape and alloc

        x = theano.tensor.vector(name='x')
        total = x.sum()
        total.name = 'total'
        num_elements = x.shape[0]
        num_elements.name = 'num_elements'
        silly_vector = theano.tensor.alloc(total / num_elements, num_elements)
        silly_vector.name = 'silly_vector'
        cost = silly_vector.sum()
        cost.name = 'cost'
        # note that cost simplifies to be the same as "total"
        g = gradient.grad(cost, x, add_names=False)
        # we still need to pass in x because it determines the shape of
        # the output
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
            __props__ = ()

            def make_node(self, x):
                return theano.Apply(self, inputs=[x], outputs=[x.type(), theano.tensor.scalar()])

            def connection_pattern(self, node):
                return [[True, False]]

            def grad(self, inputs, output_grads):
                return [inputs[0].zeros_like()]

        # Op2 has two inputs, f and g
        # Its gradient with respect to g is not defined
        class Op2(theano.gof.Op):
            __props__ = ()

            def make_node(self, f, g):
                return theano.Apply(self, inputs=[f, g], outputs=[theano.tensor.scalar()])

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

    def test_grad_constant(self):

        # Test that the gradient handles Constants and consider_constant variables
        # consistently

        x = theano.tensor.scalar()
        y = theano.tensor.scalar()
        z_x = x + y
        z_one = one + y
        g_x = theano.tensor.grad(z_x, x, consider_constant=[x])
        g_one = theano.tensor.grad(z_one, one)

        f = theano.function([x, y], [g_x, g_one])

        g_x, g_one = f(1, .5)

        if not np.allclose(g_x, g_one):
            raise AssertionError("Gradient using consider constant is " +
                                 str(g_x) +
                                 " but gradient with respect to the same Constant is " +
                                 str(g_one))


def test_known_grads():

    # Tests that the grad method with no known_grads
    # matches what happens if you put its own known_grads
    # in for each variable

    full_range = theano.tensor.arange(10)
    x = theano.tensor.scalar('x')
    t = theano.tensor.iscalar('t')
    ft = full_range[t]
    ft.name = 'ft'
    coeffs = theano.tensor.vector('c')
    ct = coeffs[t]
    ct.name = 'ct'
    p = x ** ft
    p.name = 'p'
    y = ct * p
    y.name = 'y'
    cost = theano.tensor.sqr(y)
    cost.name = 'cost'

    layers = [[cost], [y], [ct, p], [ct, x, ft], [coeffs, t, full_range, x]]

    inputs = [coeffs, t, x]

    rng = np.random.RandomState([2012, 11, 15])
    values = [rng.randn(10), rng.randint(10), rng.randn()]
    values = [np.cast[ipt.dtype](value) for ipt, value in zip(inputs, values)]

    true_grads = theano.tensor.grad(cost, inputs, disconnected_inputs='ignore')
    true_grads = theano.function(inputs, true_grads)
    true_grads = true_grads(*values)

    for layer in layers:
        first = theano.tensor.grad(cost, layer, disconnected_inputs='ignore')
        known = OrderedDict(izip(layer, first))
        full = theano.tensor.grad(cost=None, known_grads=known, wrt=inputs, disconnected_inputs='ignore')
        full = theano.function(inputs, full)
        full = full(*values)
        assert len(true_grads) == len(full)
        for a, b, var in zip(true_grads, full, inputs):
            if not np.allclose(a, b):
                print('Failure')
                print(a)
                print(b)
                print(var)
                print(layer)
                for v in known:
                    print(v, ':', theano.function(inputs, known[v])(*values))
                assert False


def test_dxdx():

    # Tests that the gradient of a scalar with respect to itself is 1
    # I use an integer in this case because people keep changing this
    # gradient to be 0 on integers but according to our interpretation
    # of the gradient as defined in the Op contract, it should be 1.
    # If you feel the need to change this unit test you are probably
    # modifying the Op contract and should definitely get the approval
    # of multiple people on theano-dev.

    x = theano.tensor.iscalar()
    g = theano.tensor.grad(x, x)

    g = g.eval({x: 12})

    assert np.allclose(g, 1.)


def test_known_grads_integers():

    # Tests that known_grads works on integers

    x = theano.tensor.iscalar()
    g_expected = theano.tensor.scalar()

    g_grad = theano.gradient.grad(cost=None, known_grads={x: g_expected}, wrt=x)

    f = theano.function([g_expected], g_grad)

    x = -3
    gv = np.cast[theano.config.floatX](.6)

    g_actual = f(gv)

    assert np.allclose(g_actual, gv)


def test_undefined_cost_grad():

        # Tests that if we say the cost is not differentiable via the
        # known_grads mechanism, it is treated as such by the rest of the
        # system.
        # This is so that Ops that are built around minigraphs like OpFromGraph
        # and scan can implement Op.grad by passing ograds to known_grads

        x = theano.tensor.iscalar()
        y = theano.tensor.iscalar()
        cost = x + y
        assert cost.dtype in theano.tensor.discrete_dtypes
        try:
            theano.tensor.grad(cost, [x, y], known_grads={cost: NullType()()})
        except theano.gradient.NullTypeGradError:
            return
        raise AssertionError("An undefined gradient has been ignored.")


def test_disconnected_cost_grad():

        # Tests that if we say the cost is disconnected via the
        # known_grads mechanism, it is treated as such by the rest of the
        # system.
        # This is so that Ops that are built around minigraphs like OpFromGraph
        # and scan can implement Op.grad by passing ograds to known_grads

        x = theano.tensor.iscalar()
        y = theano.tensor.iscalar()
        cost = x + y
        assert cost.dtype in theano.tensor.discrete_dtypes
        try:
            theano.tensor.grad(cost, [x, y], known_grads={cost: gradient.DisconnectedType()()}, disconnected_inputs='raise')
        except theano.gradient.DisconnectedInputError:
            return
        raise AssertionError("A disconnected gradient has been ignored.")


def test_subgraph_grad():

    # Tests that the grad method with no known_grads
    # matches what happens if you use successive subgraph_grads

    x = theano.tensor.fvector('x')
    t = theano.tensor.fvector('t')
    w1 = theano.shared(np.random.randn(3, 4))
    w2 = theano.shared(np.random.randn(4, 2))
    a1 = theano.tensor.tanh(theano.tensor.dot(x, w1))
    a2 = theano.tensor.tanh(theano.tensor.dot(a1, w2))
    cost2 = theano.tensor.sqr(a2 - t).sum()
    cost2 += theano.tensor.sqr(w2.sum())
    cost1 = theano.tensor.sqr(w1.sum())

    params = [[w2], [w1]]
    costs = [cost2, cost1]
    grad_ends = [[a1], [x]]

    inputs = [t, x]
    rng = np.random.RandomState([2012, 11, 15])
    values = [rng.randn(2), rng.randn(3)]
    values = [np.cast[ipt.dtype](value) for ipt, value in zip(inputs, values)]

    wrt = [w2, w1]
    cost = cost2 + cost1
    true_grads = theano.grad(cost, wrt)
    true_grads = theano.function(inputs, true_grads)
    true_grads = true_grads(*values)
    next_grad = None
    param_grads = []
    for i in xrange(2):
        param_grad, next_grad = theano.subgraph_grad(
            wrt=params[i], end=grad_ends[i],
            start=next_grad, cost=costs[i]
        )
        next_grad = OrderedDict(izip(grad_ends[i], next_grad))
        param_grads.extend(param_grad)

    pgrads = theano.function(inputs, param_grads)
    pgrads = pgrads(*values)

    for true_grad, pgrad in zip(true_grads, pgrads):
        assert(np.sum(np.abs(true_grad - pgrad)) < 0.00001)


class TestConsiderConstant(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_op_removed(self):
        x = theano.tensor.matrix('x')
        y = x * gradient.consider_constant(x)
        f = theano.function([x], y)
        # need to refer to theano.gradient.consider_constant_ here,
        # theano.gradient.consider_constant is a wrapper function!
        assert gradient.consider_constant_ not in \
            [node.op for node in f.maker.fgraph.toposort()]

    def test_grad(self):
        T = theano.tensor
        a = np.asarray(self.rng.randn(5, 5),
                       dtype=config.floatX)

        x = T.matrix('x')

        expressions_gradients = [
            (x * gradient.consider_constant(x), x),
            (x * gradient.consider_constant(T.exp(x)), T.exp(x)),
            (gradient.consider_constant(x), T.constant(0.)),
            (x**2 * gradient.consider_constant(x), 2 * x**2),
        ]

        for expr, expr_grad in expressions_gradients:
            g = gradient.grad(expr.sum(), x)
            # gradient according to theano
            f = theano.function([x], g, on_unused_input='ignore')
            # desired gradient
            f2 = theano.function([x], expr_grad, on_unused_input='ignore')

            assert np.allclose(f(a), f2(a))


class TestZeroGrad(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_op_removed(self):
        x = theano.tensor.matrix('x')
        y = x * gradient.zero_grad(x)
        f = theano.function([x], y)
        # need to refer to theano.gradient.zero_grad here,
        # theano.gradient.zero_grad is a wrapper function!
        assert gradient.zero_grad_ not in \
            [node.op for node in f.maker.fgraph.toposort()]

    def test_grad(self):
        T = theano.tensor
        a = np.asarray(self.rng.randn(5, 5),
                       dtype=config.floatX)

        x = T.matrix('x')

        expressions_gradients = [
            (x * gradient.zero_grad(x), x),
            (x * gradient.zero_grad(T.exp(x)), T.exp(x)),
            (gradient.zero_grad(x), T.constant(0.)),
            (x**2 * gradient.zero_grad(x), 2 * x**2),
        ]

        for expr, expr_grad in expressions_gradients:
            g = gradient.grad(expr.sum(), x)
            # gradient according to theano
            f = theano.function([x], g, on_unused_input='ignore')
            # desired gradient
            f2 = theano.function([x], expr_grad, on_unused_input='ignore')

            assert np.allclose(f(a), f2(a))


class TestDisconnectedGrad(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_op_removed(self):
        x = theano.tensor.matrix('x')
        y = x * gradient.disconnected_grad(x)
        f = theano.function([x], y)
        # need to refer to theano.gradient.disconnected_grad here,
        # theano.gradient.disconnected_grad is a wrapper function!
        assert gradient.disconnected_grad_ not in \
            [node.op for node in f.maker.fgraph.toposort()]

    def test_grad(self):
        T = theano.tensor
        a = np.asarray(self.rng.randn(5, 5),
                       dtype=config.floatX)

        x = T.matrix('x')

        expressions_gradients = [
            (x * gradient.disconnected_grad(x), x),
            (x * gradient.disconnected_grad(T.exp(x)), T.exp(x)),
            (x**2 * gradient.disconnected_grad(x), 2 * x**2),
        ]

        for expr, expr_grad in expressions_gradients:
            g = gradient.grad(expr.sum(), x)
            # gradient according to theano
            f = theano.function([x], g, on_unused_input='ignore')
            # desired gradient
            f2 = theano.function([x], expr_grad, on_unused_input='ignore')

            assert np.allclose(f(a), f2(a))

    def test_connection_pattern(self):
        T = theano.tensor
        x = T.matrix('x')
        y = gradient.disconnected_grad(x)

        connection_pattern = y.owner.op.connection_pattern(y.owner)
        assert connection_pattern == [[False]]

    def test_disconnected_paths(self):
        # Test that taking gradient going through a disconnected
        # path rasises an exception
        T = theano.tensor
        a = np.asarray(self.rng.randn(5, 5),
                       dtype=config.floatX)

        x = T.matrix('x')

        # This MUST raise a DisconnectedInputError error.
        # This also rasies an additional warning from gradients.py.
        self.assertRaises(gradient.DisconnectedInputError, gradient.grad,
                          gradient.disconnected_grad(x).sum(), x)

        # This MUST NOT raise a DisconnectedInputError error.
        y = gradient.grad((x + gradient.disconnected_grad(x)).sum(), x)

        a = T.matrix('a')
        b = T.matrix('b')
        y = a + gradient.disconnected_grad(b)
        # This MUST raise a DisconnectedInputError error.
        # This also rasies an additional warning from gradients.py.
        self.assertRaises(gradient.DisconnectedInputError,
                          gradient.grad, y.sum(), b)

        # This MUST NOT raise a DisconnectedInputError error.
        gradient.grad(y.sum(), a)


def test_grad_clip():
    x = theano.tensor.scalar()

    z = theano.tensor.grad(gradient.grad_clip(x, -1, 1)**2, x)
    z2 = theano.tensor.grad(x**2, x)

    f = theano.function([x], outputs=[z, z2])

    if theano.config.mode != "FAST_COMPILE":
        topo = f.maker.fgraph.toposort()
        assert not any([isinstance(node.op, gradient.GradClip)
                        for node in topo])
    out = f(2.)
    assert np.allclose(out, (1, 4))
    assert not np.allclose(out[0], out[1])

if __name__ == '__main__':
    unittest.main()
