""" test code snippet in the Theano tutorials.
"""

import os
import shutil
import unittest

from nose.plugins.skip import SkipTest
import numpy
from numpy import array

import theano
import theano.tensor as T
from theano import function

from theano import config
from theano.tests import unittest_tools as utt
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams


class T_extending(unittest.TestCase):
    # All tests here belong to files in
    # http://deeplearning.net/software/theano/extending
    # Theano/doc/extending/*.txt
    # Any change you do here also add it to the tutorial!
    # This belongs to an entire folder since code-snippets are connected
    # from one file to another .. and they do not make sense on their
    # own.

    def test_extending_1(self):

        # Note that we shadow Python's function ``filter`` with this
        # definition.
        def filter(x, strict=False, allow_downcast=None):
            if strict:
                if isinstance(x, float):
                    return x
                else:
                    raise TypeError('Expected a float!')
            else:
                return float(x)

        def values_eq_approx(x, y, tolerance=1e-4):
            return abs(x - y) / (abs(x) + abs(y)) < tolerance


        from theano import gof

        double = gof.Type()
        double.filter = filter
        double.values_eq_approx = values_eq_approx


        from theano import gof

        class Double(gof.Type):

            def filter(self, x, strict=False):
                if strict and not isinstance(x, float):
                    raise TypeError('Expected a float!')
                return float(x)

            def values_eq_approx(self, x, y, tolerance=1e-4):
                return abs(x - y) / (abs(x) + abs(y)) < tolerance

            # Added to make those tests pass in DebugMode
            @staticmethod
            def may_share_memory(a, b):
                return a is b

        double = Double()


        def __eq__(self, other):
            return type(self) is Double and type(other) is Double



        from theano import gof

        class Double(gof.Type):

            def filter(self, x, strict=False, allow_downcast=None):
                if strict and not isinstance(x, float):
                    raise TypeError('Expected a float!')
                return float(x)

            def values_eq_approx(self, x, y, tolerance=1e-4):
                return abs(x - y) / (abs(x) + abs(y)) < tolerance

            def __str__(self):
                return "double"

            # Added to make those tests pass in DebugMode
            @staticmethod
            def may_share_memory(a, b):
                return a is b

        double = Double()


        from theano import gof
        mul = gof.Op()

        def make_node(x, y):
            if x.type != double or y.type != double:
                raise TypeError('mul only works on doubles')
            return gof.Apply(mul, [x, y], [double()])
        mul.make_node = make_node

        def perform(node, inputs, output_storage):
            x, y = inputs[0], inputs[1]
            z = output_storage[0]
            z[0] = x * y
        mul.perform = perform

        x, y = double('x'), double('y')
        z = mul(x, y)
        f = theano.function([x, y], z)
        assert f(5, 6) == 30.0
        assert f(5.6, 6.7) == 37.519999999999996

        x = double('x')
        self.assertRaises(AttributeError, mul, x, 2)


        def make_node(x, y):
            if isinstance(x, (int, float)):
                x = gof.Constant(double, x)
            if isinstance(y, (int, float)):
                y = gof.Constant(double, y)
            if x.type != double or y.type != double:
                raise TypeError('mul only works on doubles')
            return gof.Apply(mul, [x, y], [double()])
        mul.make_node = make_node


        x = double('x')
        z = mul(x, 2)
        f = theano.function([x], z)
        assert f(10) == 20.0
        assert f(3.4) == 6.7999999999999998

        from theano import gof
        class BinaryDoubleOp(gof.Op):
            def __init__(self, name, fn):
                self.name = name
                self.fn = fn

            def __eq__(self, other):
                return type(self) == type(other) and (self.name == other.name) and (self.fn == other.fn)

            def __hash__(self):
                return hash(type(self)) ^ hash(self.name) ^ hash(self.fn)

            def make_node(self, x, y):
                if isinstance(x, (int, float)):
                    x = gof.Constant(double, x)
                if isinstance(y, (int, float)):
                    y = gof.Constant(double, y)
                if x.type != double or y.type != double:
                    raise TypeError('%s only works on doubles' % self.name)
                return gof.Apply(self, [x, y], [double()])

            def perform(self, node, inp, out):
                x, y = inp
                z, = out
                z[0] = self.fn(x, y)

            def __str__(self):
                return self.name

        add = BinaryDoubleOp(name='add',
                             fn=lambda x, y: x + y)

        sub = BinaryDoubleOp(name='sub',
                             fn=lambda x, y: x - y)

        mul = BinaryDoubleOp(name='mul',
                             fn=lambda x, y: x * y)

        div = BinaryDoubleOp(name='div',
                             fn=lambda x, y: x / y)

    def test_extending_2(self):
        '''
         This test fails in DebugMode for the same reasons the test in
         tensor/tests/test_basic.py:T_scalarfromtensor.test0
         fails on debug mode ( as much as I could tell - Razvan )
        '''
        from theano import gof

        class Double(gof.Type):

            def filter(self, x, strict=False, allow_downcast=None):
                if strict and not isinstance(x, float):
                    raise TypeError('Expected a float!')
                return float(x)

            def values_eq_approx(self, x, y, tolerance=1e-4):
                return abs(x - y) / (abs(x) + abs(y)) < tolerance

            def __str__(self):
                return "double"

            # Added to make those tests pass in DebugMode
            @staticmethod
            def may_share_memory(a, b):
                return a is b

        double = Double()

        class BinaryDoubleOp(gof.Op):
            def __init__(self, name, fn):
                self.name = name
                self.fn = fn

            def __eq__(self, other):
                return type(self) == type(other) and (self.name == other.name) and (self.fn == other.fn)

            def __hash__(self):
                return hash(type(self)) ^ hash(self.name) ^ hash(self.fn)

            def make_node(self, x, y):
                if isinstance(x, (int, float)):
                    x = gof.Constant(double, x)
                if isinstance(y, (int, float)):
                    y = gof.Constant(double, y)
                if x.type != double or y.type != double:
                    raise TypeError('%s only works on doubles' % self.name)
                return gof.Apply(self, [x, y], [double()])

            def perform(self, node, inp, out):
                x, y = inp
                z, = out
                z[0] = self.fn(x, y)

            def __str__(self):
                return self.name

        add = BinaryDoubleOp(name='add',
                             fn=lambda x, y: x + y)

        sub = BinaryDoubleOp(name='sub',
                             fn=lambda x, y: x - y)

        mul = BinaryDoubleOp(name='mul',
                             fn=lambda x, y: x * y)

        div = BinaryDoubleOp(name='div',
                             fn=lambda x, y: x / y)

        def c_declare(name, sub, check_input=True):
            return """
            double %(name)s;
            """ % dict(name=name)
        double.c_declare = c_declare


        def c_init(name, sub):
            return """
            %(name)s = 0.0;
            """ % dict(name = name)
        double.c_init = c_init

        def c_extract(name, sub, check_input=True):
            if(check_input):
                pre = """
                if (!PyFloat_Check(py_%(name)s)) {
                    PyErr_SetString(PyExc_TypeError, "expected a float");
                    %(fail)s
                }""" % dict(name = name, fail = sub['fail'])
            else:
                pre = ""
            return pre + """
            %(name)s = PyFloat_AsDouble(py_%(name)s);
            """ % dict(name = name, fail = sub['fail'])
        double.c_extract = c_extract

        def c_sync( name, sub):
            return """
            Py_XDECREF(py_%(name)s);
            py_%(name)s = PyFloat_FromDouble(%(name)s);
            if (!py_%(name)s) {
                printf("PyFloat_FromDouble failed on: %%f\\n", %(name)s);
                Py_XINCREF(Py_None);
                py_%(name)s = Py_None;
            }
            """ % dict(name = name)
        double.c_sync = c_sync

        def c_cleanup(name, sub):
            return ""
        double.c_cleanup = c_cleanup


        from theano import function

        x, y, z = double('x'), double('y'), double('z')
        a = add(x, y)
        b = mul(a, z)
        f = function([x, y, z], b)
        assert f(1.0, 2.0, 3.0) == 9.0


        from theano import gof
        class Double(gof.Type):

            def filter(self, x, strict=False, allow_downcast=None):
                if strict and not isinstance(x, float):
                    raise TypeError('Expected a float!')
                return float(x)

            def values_eq_approx(self, x, y, tolerance=1e-4):
                return abs(x - y) / (x + y) < tolerance

            def __str__(self):
                return "double"

            def c_declare(self, name, sub, check_input=True):
                return """
                double %(name)s;
                """ % dict(name = name)

            def c_init(self, name, sub):
                return """
                %(name)s = 0.0;
                """ % dict(name = name)

            def c_extract(self, name, sub, check_input=True):
                if(check_input):
                    pre = """
                    if (!PyFloat_Check(py_%(name)s)) {
                        PyErr_SetString(PyExc_TypeError, "expected a float");
                        %(fail)s
                    }
                    """ % dict(sub, name=name)
                else:
                    pre = ""
                return pre + """
                %(name)s = PyFloat_AsDouble(py_%(name)s);
                """ % dict(sub, name=name)

            def c_sync(self, name, sub):
                return """
                Py_XDECREF(py_%(name)s);
                py_%(name)s = PyFloat_FromDouble(%(name)s);
                if (!py_%(name)s) {
                    printf("PyFloat_FromDouble failed on: %%f\\n", %(name)s);
                    Py_XINCREF(Py_None);
                    py_%(name)s = Py_None;
                }
                """ % dict(name = name)

            def c_cleanup(self, name, sub):
                return ""

            # Added to make those tests pass in DebugMode
            @staticmethod
            def may_share_memory(a, b):
                return a is b

        double = Double()


        def c_code(node, name, input_names, output_names, sub):
            x_name, y_name = input_names[0], input_names[1]
            output_name = output_names[0]
            return """
            %(output_name)s = %(x_name)s * %(y_name)s;
            """ % locals()
        mul.c_code = c_code


        from theano import gof
        class BinaryDoubleOp(gof.Op):

            def __init__(self, name, fn, ccode):
                self.name = name
                self.fn = fn
                self.ccode = ccode

            def make_node(self, x, y):
                if isinstance(x, (int, float)):
                    x = gof.Constant(double, x)
                if isinstance(y, (int, float)):
                    y = gof.Constant(double, y)
                if x.type != double or y.type != double:
                    raise TypeError('%s only works on doubles' % self.name)
                return gof.Apply(self, [x, y], [double()])

            def perform(self, node, inp, out):
                x, y = inp
                z, = out
                z[0] = self.fn(x, y)

            def __str__(self):
                return self.name

            def c_code(self, node, name, inp, out, sub):
                x, y = inp
                z, = out
                return self.ccode % locals()


        add = BinaryDoubleOp(name='add',
                            fn=lambda x, y: x + y,
                            ccode="%(z)s = %(x)s + %(y)s;")

        sub = BinaryDoubleOp(name='sub',
                            fn=lambda x, y: x - y,
                            ccode="%(z)s = %(x)s - %(y)s;")

        mul = BinaryDoubleOp(name='mul',
                            fn=lambda x, y: x * y,
                            ccode="%(z)s = %(x)s * %(y)s;")

        div = BinaryDoubleOp(name='div',
                            fn=lambda x, y: x / y,
                            ccode="%(z)s = %(x)s / %(y)s;")


        from theano.gof import toolbox

        class Simplify(gof.Optimizer):
            def add_requirements(self, fgraph):
                fgraph.attach_feature(toolbox.ReplaceValidate())
            def apply(self, fgraph):
                for node in fgraph.toposort():
                    if node.op == div:
                        x, y = node.inputs
                        z = node.outputs[0]
                        if x.owner and x.owner.op == mul:
                            a, b = x.owner.inputs
                            if y == a:
                                fgraph.replace_validate(z, b)
                            elif y == b:
                                fgraph.replace_validate(z, a)

        simplify = Simplify()
        x = double('x')
        y = double('y')
        z = double('z')
        a = add(z, mul(div(mul(y, x), y), div(z, x)))
        e = gof.FunctionGraph([x, y, z], [a])
        simplify.optimize(e)

        class LocalSimplify(gof.LocalOptimizer):
            def transform(self, node):
                if node.op == div:
                    x, y = node.inputs
                    if x.owner and x.owner.op == mul:
                        a, b = x.owner.inputs
                        if y == a:
                            return [b]
                        elif y == b:
                            return [a]
                return False
            def tracks(self):
                # This should be needed for the EquilibriumOptimizer
                # but it isn't now
                # TODO: do this and explain it
                return [] # that's not what you should do

        local_simplify = LocalSimplify()

        x = double('x')
        y = double('y')
        z = double('z')
        a = add(z, mul(div(mul(y, x), y), div(z, x)))
        e = gof.FunctionGraph([x, y, z], [a])
        simplify = gof.TopoOptimizer(local_simplify)
        simplify.optimize(e)

    def test_as_op(self):
        import theano
        import numpy
        from theano.compile.ops import as_op

        def infer_shape_numpy_dot(node, input_shapes):
            ashp, bshp = input_shapes
            return [ashp[:-1] + bshp[-1:]]

        @as_op(itypes=[theano.tensor.fmatrix, theano.tensor.fmatrix],
               otypes=[theano.tensor.fmatrix],
               infer_shape=infer_shape_numpy_dot)
        def numpy_add(a, b):
            return numpy.add(a, b)

        def infer_shape_numpy_add_sub(node, input_shapes):
            ashp, bshp = input_shapes
            # Both inputs should have that same shape, so we just
            # return one of them.
            return [ashp[0]]

        @as_op(itypes=[theano.tensor.fmatrix, theano.tensor.fmatrix],
               otypes=[theano.tensor.fmatrix],
               infer_shape=infer_shape_numpy_add_sub)
        def numpy_add(a, b):
            return numpy.add(a, b)

        @as_op(itypes=[theano.tensor.fmatrix, theano.tensor.fmatrix],
               otypes=[theano.tensor.fmatrix],
               infer_shape=infer_shape_numpy_add_sub)
        def numpy_sub(a, b):
            return numpy.sub(a, b)


class T_introduction(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/tutorial/introduction.html
    # Theano/doc/tutorial/introduction.txt
    # Any change you do here also add it to the tutorial !
    def test_introduction_1(self):

        import theano
        from theano import tensor

        # declare two symbolic floating-point scalars
        a = tensor.dscalar()
        b = tensor.dscalar()

        # create a simple expression
        c = a + b

        # convert the expression into a callable object that takes (a,b)
        # values as input and computes a value for c
        f = theano.function([a,b], c)

        # bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
        assert 4.0 == f(1.5, 2.5)


class T_adding(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/tutorial/adding.html
    # Theano/doc/tutorial/adding.txt
    # Any change you do here also add it to the tutorial !


    def test_adding_1(self):
        import theano.tensor as T
        from theano import function
        x = T.dscalar('x')
        y = T.dscalar('y')
        z = x + y
        f = function([x, y], z)
        assert f(2, 3) == numpy.array(5.0)
        assert f(16.3, 12.1) == numpy.array(28.4)

    def test_adding_2(self):
        x = T.dmatrix('x')
        y = T.dmatrix('y')
        z = x + y
        f = function([x, y], z)
        assert numpy.all(f([[1, 2], [3, 4]], [[10, 20], [30, 40]]) ==
                         numpy.array([[ 11.,  22.],[ 33.,  44.]]))

        assert numpy.all(f(numpy.array([[1, 2], [3, 4]])
                           , numpy.array([[10, 20], [30, 40]])) ==
                         numpy.array([[ 11.,  22.], [ 33.,  44.]]))



class T_examples(unittest.TestCase):
    # All tests here belog to
    # http://deeplearning.net/software/theano/tutorial/examples.html
    # Theano/doc/tutorial/examples.txt
    # Any change you do here also add it to the tutorial !

    def test_examples_1(self):
        x = T.dmatrix('x')
        s = 1 / (1 + T.exp(-x))
        logistic = function([x], s)
        assert numpy.allclose( logistic([[0, 1], [-1, -2]]),
                         array([[ 0.5       ,  0.73105858],
                                [ 0.26894142,  0.11920292]]))




    def test_examples_2(self):

        x = T.dmatrix('x')
        s2 = (1 + T.tanh(x / 2)) / 2
        logistic2 = function([x], s2)
        assert numpy.allclose(logistic2([[0, 1], [-1, -2]]),
                    array([[ 0.5       ,  0.73105858],
                          [ 0.26894142,  0.11920292]]))

    def test_examples_3(self):
        a, b = T.dmatrices('a', 'b')
        diff         = a - b
        abs_diff     = abs(diff)
        diff_squared = diff**2
        f = function([a, b], [diff, abs_diff, diff_squared])
        elems = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])
        assert numpy.all( elems[0] == array([[ 1.,  0.],[-1., -2.]]))
        assert numpy.all( elems[1] == array([[ 1.,  0.],[ 1.,  2.]]))
        assert numpy.all( elems[2] == array([[ 1.,  0.],[ 1.,  4.]]))

    def test_examples_4(self):
        from theano import pp
        x = T.dscalar('x')
        y = x**2
        gy = T.grad(y, x)
        pp(gy)  # print out the gradient prior to optimization
        '((fill((x ** 2), 1.0) * 2) * (x ** (2 - 1)))'
        f = function([x], gy)
        assert f(4)    ==  array(8.0)
        assert f(94.2) == array(188.40000000000001)


    def test_examples_5(self):

        x = T.dmatrix('x')
        s = T.sum(1 / (1 + T.exp(-x)))
        gs = T.grad(s, x)
        dlogistic = function([x], gs)
        assert numpy.allclose( dlogistic([[0, 1], [-1, -2]]),
                         array([[ 0.25      ,  0.19661193],
                               [ 0.19661193,  0.10499359]]))


    def test_examples_6(self):

        from theano import Param
        x, y = T.dscalars('x', 'y')
        z = x + y
        f = function([x, Param(y, default=1)], z)
        assert f(33)    == array(34.0)
        assert f(33, 2) == array(35.0)


    def test_examples_7(self):
        from theano import Param
        x, y, w = T.dscalars('x', 'y', 'w')
        z = (x + y) * w
        f = function([x, Param(y, default=1), Param(w, default=2, name='w_by_name')], z)
        assert f(33)                   == array(68.0)
        assert f(33, 2)                == array(70.0)
        assert f(33, 0, 1)             == array(33.0)
        assert f(33, w_by_name=1)      == array(34.0)
        assert f(33, w_by_name=1, y=0) == array(33.0)


    def test_examples_8(self):
        from theano import shared
        # Force the dtype to int64 to work correctly on 32 bit computer.
        # Otherwise, it create by default a int32 on 32 bit computer.
        state = shared(0)
        inc = T.iscalar('inc')
        accumulator = function([inc], state, updates=[(state, state+inc)])

        assert state.get_value()       == array(0)
        assert accumulator(1)          == array(0)
        assert state.get_value()       == array(1)
        assert accumulator(300)        == array(1)
        assert state.get_value()       == array(301)

        state.set_value(-1)
        assert accumulator(3)          == array(-1)
        assert state.get_value()       == array(2)

        decrementor = function([inc], state, updates=[(state, state-inc)])
        assert decrementor(2)          == array(2)
        assert state.get_value()       == array(0)

        fn_of_state = state * 2 + inc
        # The type of foo must match the shared variable we are replacing
        # with the ``givens``
        foo = T.scalar(dtype=state.dtype)
        skip_shared = function([inc, foo], fn_of_state,
                               givens=[(state, foo)])
        assert skip_shared(1, 3)       == array(7)
        assert state.get_value()       == array(0)


    def test_examples_9(self):

        from theano.tensor.shared_randomstreams import RandomStreams
        srng = RandomStreams(seed=234)
        rv_u = srng.uniform((2,2))
        rv_n = srng.normal((2,2))
        f = function([], rv_u)
        g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
        nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)


        f_val0 = f()
        f_val1 = f()  #different numbers from f_val0
        assert numpy.all(f_val0 != f_val1)

        g_val0 = g()  # different numbers from f_val0 and f_val1
        g_val1 = g()  # same numbers as g_val0 !!!

        assert numpy.all(g_val0 == g_val1)
        assert numpy.all(g_val0 != f_val0)
        assert numpy.all(g_val0 != f_val1)

        nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
        assert numpy.allclose(nearly_zeros(), [[0.,0.],[0.,0.]])

        rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
        rng_val.seed(89234)                         # seeds the generator
        rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng

        srng.seed(902340)  # seeds rv_u and rv_n with different seeds each
        state_after_v0 = rv_u.rng.get_value().get_state()
        nearly_zeros()       # this affects rv_u's generator
        v1 = f()
        rng = rv_u.rng.get_value(borrow=True)
        rng.set_state(state_after_v0)
        rv_u.rng.set_value(rng, borrow=True)
        v2 = f()             # v2 != v1
        v3 = f()             # v3 == v1
        assert numpy.all(v1 != v2)
        assert numpy.all(v1 == v3)

    def test_copy_random_state(self):

        class Graph():
            def __init__(self, seed=123):
                self.rng = RandomStreams(seed)
                self.y = self.rng.uniform(size=(1,))

        g1 = Graph(seed=123)
        f1 = theano.function([], g1.y)

        g2 = Graph(seed=987)
        f2 = theano.function([], g2.y)

        #print 'By default, the two functions are out of sync.'
        v1 =  f1()
        v2 =  f2()

        def copy_random_state(g1, g2):
            if isinstance(g1.rng, MRG_RandomStreams):
                g2.rng.rstate = g1.rng.rstate
            for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
                su2[0].set_value(su1[0].get_value())

        #print 'We now copy the state of the theano random number generators.'
        copy_random_state(g1, g2)
        v3 = f1()
        v4 = f2()
        assert numpy.allclose(v1, 0.72803009)
        assert numpy.allclose(v2, 0.55056769)
        assert numpy.allclose(v3, 0.59044123)
        assert numpy.allclose(v4, 0.59044123)

    def test_examples_real_example(self):
        rng = numpy.random

        N = 400
        feats = 784
        D = (rng.randn(N, feats).astype(config.floatX),
             rng.randint(size=N, low=0, high=2).astype(config.floatX))
        training_steps = 10000
        if config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            training_steps = 10

        # Declare Theano symbolic variables
        x = T.matrix("x")
        y = T.vector("y")
        # The *.03 have been added to have DebugMode don't complain
        w = theano.shared(rng.randn(feats).astype(config.floatX) * .03,
                          name="w")
        b = theano.shared(numpy.asarray(0., dtype=config.floatX),
                          name="b")
        print "Initial model:"
        print w.get_value(), b.get_value()

        # Construct Theano expression graph
        p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
        prediction = p_1 > 0.5                    # The prediction thresholded
        xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
        cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
        gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                                  # (we shall return to this in a
                                                  # following section of this tutorial)

        # Compile
        train = theano.function(
            inputs=[x,y],
            outputs=[prediction, xent],
            updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
        predict = theano.function(inputs=[x], outputs=prediction)

        # Train
        for i in range(training_steps):
            pred, err = train(D[0], D[1])

        print "Final model:"
        print w.get_value(), b.get_value()
        print "target values for D:", D[1]
        print "prediction on D:", predict(D[0])

        # A user reported that this happened on the mailig list.
        assert not numpy.isnan(b.get_value()).any()
        assert not numpy.isnan(w.get_value()).any()


class T_aliasing(unittest.TestCase):
    # All tests here belog to
    # http://deeplearning.net/software/theano/tutorial/aliasing.html
    # Theano/doc/tutorial/aliasing.txt
    # Any change you do here also add it to the tutorial !

    def test_aliasing_1(self):

        import numpy, theano
        np_array = numpy.ones(2, dtype='float32')

        s_default = theano.shared(np_array)
        s_false   = theano.shared(np_array, borrow=False)
        s_true    = theano.shared(np_array, borrow=True)

        np_array += 1 # now it is an array of 2.0 s

        assert numpy.all(s_default.get_value() == array([1.0, 1.0]))
        assert numpy.all(s_false.get_value()   == array([1.0, 1.0]))
        assert numpy.all(s_true.get_value()    == array([2.0, 2.0]))


    def test_aliasing_2(self):

        import numpy, theano
        np_array = numpy.ones(2, dtype='float32')

        s = theano.shared(np_array)

        v_false = s.get_value(borrow=False) # N.B. borrow default is False
        v_true = s.get_value(borrow=True)

        v_internal = s.get_value(borrow=True, return_internal_type=True)


        s.set_value(
            # some_inplace_fn
            s.get_value(borrow=True).__imul__(2),
            borrow=True)



    def test_aliasing_3(self):

        import theano, theano.tensor

        x = theano.tensor.matrix()
        y = 2*x
        f = theano.function([theano.In(x, borrow=True)], theano.Out(y, borrow=True))


class T_loading_and_saving(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
    # Theano/doc/tutorial/loading_and_saving.txt
    # Any change you do here also add it to the tutorial !

    def test_loading_and_saving_1(self):

        import cPickle
        import theano, theano.tensor

        x = theano.tensor.matrix()
        y = 2*x
        my_obj =  theano.function([theano.In(x, borrow=True)]
                                  , theano.Out(y, borrow=True))

        mode_instance = theano.compile.mode.get_mode(None)
        if not isinstance(mode_instance, theano.compile.debugmode.DebugMode):
            # Here, we work in a temporary directory in order not to clutter
            # the Theano repository. Code relative to creating that dir and
            # removing it afterwards should _not_ be backported to the tutorial.
            from tempfile import mkdtemp
            origdir = os.getcwd()
            tmpdir = None
            try:
                tmpdir = mkdtemp()
                os.chdir(tmpdir)

                f = open('obj.save', 'wb')
                cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()


                f = open('obj.save', 'rb')
                loaded_obj = cPickle.load(f)
                f.close()

                obj1 = my_obj
                obj2 = my_obj
                obj3 = my_obj

                f = open('objects.save', 'wb')
                for obj in [obj1, obj2, obj3]:
                    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

                f = open('objects.save', 'rb')
                loaded_objects = []
                for i in range(3):
                    loaded_objects.append(cPickle.load(f))
                f.close()
            finally:
                # Get back to the orinal dir, and temporary one.
                os.chdir(origdir)
                if tmpdir is not None:
                    shutil.rmtree(tmpdir)


class T_modes(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/tutorial/modes.html
    # Theano/doc/tutorial/modes.txt
    # Any change you do here also add it to the tutorial !

    def test_modes_1(self):

        x = T.dvector('x')

        f = theano.function([x], 10*x, mode='DEBUG_MODE')

        assert numpy.all(f([5]) == [50.])
        assert numpy.all(f([0]) == [0.])
        assert numpy.all(f([7]) == [70.])


class T_using_gpu(unittest.TestCase):
    # All tests here belog to
    # http://deeplearning.net/software/theano/tutorial/using_gpu.html
    # Theano/doc/tutorial/using_gpu.txt
    # Any change you do here also add it to the tutorial !

    def test_using_gpu_1(self):
        # I'm checking if this compiles and runs
        from theano import function, config, shared, sandbox
        import theano.tensor as T
        import numpy
        import time

        vlen = 10 * 30 * 70  # 10 x #cores x # threads per core
        iters = 10

        rng = numpy.random.RandomState(22)
        x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
        f = function([], T.exp(x))
        # print f.maker.fgraph.toposort()
        t0 = time.time()
        for i in xrange(iters):
            r = f()
        t1 = time.time()
        print 'Looping %d times took' % iters, t1 - t0, 'seconds'
        print 'Result is', r
        if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
            print 'Used the cpu'
        else:
            print 'Used the gpu'
        if theano.config.device.find('gpu') > -1:
            assert not numpy.any( [isinstance(x.op,T.Elemwise) for x in f.maker.fgraph.toposort()])
        else:
            assert numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()])

    def test_using_gpu_2(self):
        if theano.config.device.find('gpu') > -1:

            from theano import function, config, shared, sandbox
            import theano.tensor as T
            import numpy
            import time

            vlen = 10 * 30 * 70  # 10 x #cores x # threads per core
            iters = 10

            rng = numpy.random.RandomState(22)
            x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
            f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
            # print f.maker.fgraph.toposort()
            t0 = time.time()
            for i in xrange(iters):
                r = f()
            t1 = time.time()
            print 'Looping %d times took' % iters, t1 - t0, 'seconds'
            print 'Result is', r
            print 'Numpy result is', numpy.asarray(r)
            if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
                print 'Used the cpu'
            else:
                print 'Used the gpu'

            assert not numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()])

    def test_using_gpu_3(self):

        if theano.config.device.find('gpu') > -1:

            from theano import function, config, shared, sandbox, Out
            import theano.tensor as T
            import numpy
            import time

            vlen = 10 * 30 * 70  # 10 x #cores x # threads per core
            iters = 10

            rng = numpy.random.RandomState(22)
            x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
            f = function([],
                    Out(sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)),
                        borrow=True))
            # print f.maker.fgraph.toposort()
            t0 = time.time()
            for i in xrange(iters):
                r = f()
            t1 = time.time()
            print 'Looping %d times took' % iters, t1 - t0, 'seconds'
            print 'Result is', r
            print 'Numpy result is', numpy.asarray(r)
            if numpy.any([isinstance(x.op, T.Elemwise)
                          for x in f.maker.fgraph.toposort()]):
                print 'Used the cpu'
            else:
                print 'Used the gpu'

            assert not numpy.any([isinstance(x.op, T.Elemwise)
                                  for x in f.maker.fgraph.toposort()])

    def test_using_gpu_pycudaop(self):
        import theano.misc.pycuda_init
        if not theano.misc.pycuda_init.pycuda_available:
            raise SkipTest("Pycuda not installed. Skip test of theano op"
                           " with pycuda code.")
        from pycuda.compiler import SourceModule
        import theano.sandbox.cuda as cuda

        import theano.sandbox.cuda as cuda_ndarray
        if not cuda_ndarray.cuda_available:
            raise SkipTest('Optional package cuda disabled')

        class PyCUDADoubleOp(theano.Op):
            def __eq__(self, other):
                return type(self) == type(other)

            def __hash__(self):
                return hash(type(self))

            def __str__(self):
                return self.__class__.__name__

            def make_node(self, inp):
                inp = cuda.basic_ops.gpu_contiguous(
                    cuda.basic_ops.as_cuda_ndarray_variable(inp))
                assert inp.dtype == "float32"
                return theano.Apply(self, [inp], [inp.type()])

            def make_thunk(self, node, storage_map, _, _2):
                mod = SourceModule("""
    __global__ void my_fct(float * i0, float * o0, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<size){
        o0[i] = i0[i]*2;
    }
  }""")
                pycuda_fct = mod.get_function("my_fct")
                inputs = [storage_map[v] for v in node.inputs]
                outputs = [storage_map[v] for v in node.outputs]

                def thunk():
                    z = outputs[0]
                    if z[0] is None or z[0].shape != inputs[0][0].shape:
                        z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
                        grid = (int(numpy.ceil(inputs[0][0].size / 512.)), 1)
                        pycuda_fct(inputs[0][0], z[0],
                                   numpy.intc(inputs[0][0].size),
                                   block=(512, 1, 1), grid=grid)
                return thunk
        x = theano.tensor.fmatrix()
        f = theano.function([x], PyCUDADoubleOp()(x))
        xv = numpy.ones((4, 5), dtype="float32")
        assert numpy.allclose(f(xv), xv*2)
        # print numpy.asarray(f(xv))


# Used in T_fibby
class Fibby(theano.Op):

    """
    An arbitrarily generalized Fibbonacci sequence
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def make_node(self, x):
        x_ = theano.tensor.as_tensor_variable(x)
        assert x_.ndim == 1
        return theano.Apply(self,
            inputs=[x_],
            outputs=[x_.type()])
        # using x_.type() is dangerous, it copies x's broadcasting
        # behaviour

    def perform(self, node, inputs, output_storage):
        x, = inputs
        y = output_storage[0][0] = x.copy()
        for i in range(2, len(x)):
            y[i] = y[i - 1] * y[i - 2] + x[i]

    def c_code(self, node, name, inames, onames, sub):
        x, = inames
        y, = onames
        fail = sub['fail']
        return """
            Py_XDECREF(%(y)s);
            %(y)s = (PyArrayObject*)PyArray_FromArray(
                    %(x)s, 0, NPY_ARRAY_ENSURECOPY);
            if (!%(y)s)
                %(fail)s;
            {//New scope needed to make compilation work
                dtype_%(y)s * y = (dtype_%(y)s*)PyArray_DATA(%(y)s);
                dtype_%(x)s * x = (dtype_%(x)s*)PyArray_DATA(%(x)s);
                for (int i = 2; i < PyArray_DIMS(%(x)s)[0]; ++i)
                    y[i] = y[i-1]*y[i-2] + x[i];
            }
        """ % locals()

    def c_code_cache_version(self):
        return (1,)


class T_fibby(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/extending/fibby.html
    # Theano/doc/extending/fibby.txt
    # Any change you do here also add it to the tutorial !

    def test_fibby_1(self):

        # The definition of class Fibby is done outside of the test,
        # so the object can be pickled.
        fibby = Fibby()

        from theano.tensor.opt import (get_scalar_constant_value,
                                       NotScalarConstantError)

        # Remove any fibby(zeros(...))
        @theano.tensor.opt.register_specialize
        @theano.gof.local_optimizer([fibby])
        def fibby_of_zero(node):
            if node.op == fibby:
                x = node.inputs[0]
                try:
                    if numpy.all(0 == get_scalar_constant_value(x)):
                        return [x]
                except NotScalarConstantError:
                    pass

        # Test it does not apply when not needed
        x = T.dvector()
        f = function([x], fibby(x))
        #theano.printing.debugprint(f)

        # We call the function to make sure it runs.
        # If you run in DebugMode, it will compare the C and Python outputs.
        f(numpy.random.rand(5))
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Fibby)

        # Test that the optimization gets applied.
        f_zero = function([], fibby(T.zeros([5])))
        #theano.printing.debugprint(f_zero)

        # If you run in DebugMode, it will compare the output before
        # and after the optimization.
        f_zero()

        # Check that the optimization removes the Fibby Op.
        # For security, the Theano memory interface ensures that the output
        # of the function is always memory not aliased to the input.
        # That is why there is a DeepCopyOp op.
        topo = f_zero.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, theano.compile.ops.DeepCopyOp)


class T_graphstructures(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/extending/graphstructures.html
    # Theano/doc/extending/graphstructures.txt
    # Any change you do here also add it to the tutorial !

    def test_graphstructures_1(self):

        x = T.dmatrix('x')
        y = T.dmatrix('y')
        z = x + y

        x = T.matrix('x')
        y = T.matrix('y')
        z = T.matrix('z')

        # create 2 Variables (one for 'e', one intermediate for y*z)
        # create 2 Apply instances (one for '+', one for '*')
        e = x + y * z

        from theano.tensor import add, mul, Apply, Variable, TensorType

        # Instantiate a type that represents a matrix of doubles
        float64_matrix = TensorType(dtype='float64',               # double
                                    broadcastable=(False, False))  # matrix

        # We make the Variable instances we need.
        x = Variable(type=float64_matrix, name='x')
        y = Variable(type=float64_matrix, name='y')
        z = Variable(type=float64_matrix, name='z')

        # This is the Variable that we want to symbolically represents y*z
        mul_variable = Variable(type=float64_matrix)
        assert mul_variable.owner is None

        # Instantiate a symbolic multiplication
        node_mul = Apply(op=mul,
                         inputs=[y, z],
                         outputs=[mul_variable])
        # Fields 'owner' and 'index' are set by Apply
        assert mul_variable.owner is node_mul
        # 'index' is the position of mul_variable in mode_mul's outputs
        assert mul_variable.index == 0

        # This is the Variable that we want to symbolically represents x+(y*z)
        add_variable = Variable(type=float64_matrix)
        assert add_variable.owner is None

        # Instantiate a symbolic addition
        node_add = Apply(op=add,
                         inputs=[x, mul_variable],
                         outputs=[add_variable])
        # Fields 'owner' and 'index' are set by Apply
        assert add_variable.owner is node_add
        assert add_variable.index == 0

        e = add_variable

        # We have access to x, y and z through pointers
        assert e.owner.inputs[0] is x
        assert e.owner.inputs[1] is mul_variable
        assert e.owner.inputs[1].owner.inputs[0] is y
        assert e.owner.inputs[1].owner.inputs[1] is z


class T_scan(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/tutorial/loop.html
    # Theano/doc/tutorial/loop.txt
    # Any change you do here also add it to the tutorial !

    def test_elemwise(self):
        # defining the tensor variables
        X = T.matrix("X")
        W = T.matrix("W")
        b_sym = T.vector("b_sym")

        results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym),
                                       sequences=X)
        compute_elementwise = theano.function(inputs=[X, W, b_sym],
                                              outputs=[results])

        # test values
        x = numpy.eye(2, dtype=theano.config.floatX)
        w = numpy.ones((2, 2), dtype=theano.config.floatX)
        b = numpy.ones((2), dtype=theano.config.floatX)
        b[1] = 2

        print "Scan results:", compute_elementwise(x, w, b)[0]

        # comparison with numpy
        print "Numpy results:", numpy.tanh(x.dot(w) + b)

    def test_sequence(self):
        # define tensor variables
        X = T.vector("X")
        W = T.matrix("W")
        b_sym = T.vector("b_sym")
        U = T.matrix("U")
        Y = T.matrix("Y")
        V = T.matrix("V")
        P = T.matrix("P")

        results, updates = theano.scan(
            lambda y, p, x_tm1: T.tanh(T.dot(x_tm1, W) +
                                       T.dot(y, U) + T.dot(p, V)),
            sequences=[Y, P[::-1]], outputs_info=[X])

        compute_seq = theano.function(inputs=[X, W, Y, U, P, V],
                                      outputs=[results])

        # test values
        x = numpy.zeros((2), dtype=theano.config.floatX)
        x[1] = 1
        w = numpy.ones((2, 2), dtype=theano.config.floatX)
        y = numpy.ones((5, 2), dtype=theano.config.floatX)
        y[0, :] = -3
        u = numpy.ones((2, 2), dtype=theano.config.floatX)
        p = numpy.ones((5, 2), dtype=theano.config.floatX)
        p[0, :] = 3
        v = numpy.ones((2, 2), dtype=theano.config.floatX)

        print "Scan results", compute_seq(x, w, y, u, p, v)[0]

        # comparison with numpy
        x_res = numpy.zeros((5, 2), dtype=theano.config.floatX)
        x_res[0] = numpy.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
        for i in range(1, 5):
            x_res[i] = numpy.tanh(x_res[i-1].dot(w) +
                                  y[i].dot(u) + p[4-i].dot(v))

        print "Numpy results:", x_res

    def test_norm(self):
        # define tensor variable
        X = T.matrix("X")
        results, updates = theano.scan(lambda x_i: T.sqrt((x_i**2).sum()),
                                       sequences=[X])
        compute_norm_lines = theano.function(inputs=[X], outputs=[results])

        results, updates = theano.scan(lambda x_i: T.sqrt((x_i**2).sum()),
                                       sequences=[X.T])
        compute_norm_cols = theano.function(inputs=[X], outputs=[results])

        # test value
        x = numpy.diag(numpy.arange(1, 6, dtype=theano.config.floatX), 1)
        print "Scan results:", compute_norm_lines(x)[0], \
                            compute_norm_cols(x)[0]

        # comparison with numpy
        print "Numpy results:", numpy.sqrt((x**2).sum(1)), \
                            numpy.sqrt((x**2).sum(0))

    def test_trace(self):
        # define tensor variable
        X = T.matrix("X")
        results, updates = theano.scan(lambda i, j, t_f: T.cast(X[i,j] +
                                                                t_f, theano.config.floatX),
                                       sequences=[T.arange(X.shape[0]),
                                                  T.arange(X.shape[1])],
                                       outputs_info=numpy.asarray(
                                           0., dtype=theano.config.floatX))

        result = results[-1]
        compute_trace = theano.function(inputs=[X], outputs=[result])

        # test value
        x = numpy.eye(5, dtype=theano.config.floatX)
        x[0] = numpy.arange(5, dtype=theano.config.floatX)
        print "Scan results:", compute_trace(x)[0]

        # comparison with numpy
        print "Numpy results:", numpy.diagonal(x).sum()

    def test_taps(self):
        # define tensor variables
        X = T.matrix("X")
        W = T.matrix("W")
        b_sym = T.vector("b_sym")
        U = T.matrix("U")
        V = T.matrix("V")
        n_sym = T.iscalar("n_sym")

        results, updates = theano.scan(
            lambda x_tm2,x_tm1: T.dot(x_tm2,U) + T.dot(x_tm1,V) + T.tanh(T.dot(x_tm1,W) + b_sym),
            n_steps=n_sym,
            outputs_info=[dict(initial=X, taps=[-2, -1])])

        compute_seq2 = theano.function(inputs=[X, U, V, W, b_sym, n_sym],
                                       outputs=[results])

        # test values
        x = numpy.zeros((2, 2), dtype=theano.config.floatX)
        # the initial value must be able to return x[-2]
        x[1, 1] = 1
        w = 0.5 * numpy.ones((2, 2), dtype=theano.config.floatX)
        u = 0.5 * (numpy.ones((2, 2), dtype=theano.config.floatX) -
                   numpy.eye(2, dtype=theano.config.floatX))
        v = 0.5 * numpy.ones((2, 2), dtype=theano.config.floatX)
        n = 10
        b = numpy.ones((2), dtype=theano.config.floatX)

        print "Scan results:", compute_seq2(x, u, v, w, b, n)

        # comparison with numpy
        x_res = numpy.zeros((10, 2), dtype=theano.config.floatX)
        x_res[0] = x[0].dot(u) + x[1].dot(v) + numpy.tanh(x[1].dot(w) + b)
        x_res[1] = x[1].dot(u) + x_res[0].dot(v) \
                        + numpy.tanh(x_res[0].dot(w) + b)
        x_res[2] = x_res[0].dot(u) + x_res[1].dot(v) \
                   + numpy.tanh(x_res[1].dot(w) + b)
        for i in range(2, 10):
            x_res[i] = (x_res[i-2].dot(u) + x_res[i-1].dot(v) +
                        numpy.tanh(x_res[i-1].dot(w) + b))

        print "Numpy results:", x_res

    def test_jacobian(self):
        # define tensor variables
        v = T.vector()
        A = T.matrix()
        y = T.tanh(T.dot(v, A))
        results, updates = theano.scan(lambda i: T.grad(y[i], v),
                                       sequences=[T.arange(y.shape[0])])
        compute_jac_t = theano.function([A, v], [results],
                                        allow_input_downcast=True)  # shape (d_out, d_in)

        # test values
        x = numpy.eye(5)[0]
        w = numpy.eye(5, 3)
        w[2] = numpy.ones((3))
        print "Scan results:", compute_jac_t(w, x)[0]

        # compare with numpy
        print "Numpy results:", ((1 - numpy.tanh(x.dot(w))**2)*w).T

    def test_accumulator(self):
        # define shared variables
        k = theano.shared(0)
        n_sym = T.iscalar("n_sym")

        results, updates = theano.scan(lambda: {k: (k + 1)}, n_steps=n_sym)
        accumulator = theano.function([n_sym], [], updates=updates,
                                      allow_input_downcast=True)

        print "Before 5 steps:", k.get_value()
        accumulator(5)
        print "After 5 steps:", k.get_value()

    def test_random(self):
        # define tensor variables
        X = T.matrix("X")
        W = T.matrix("W")
        b_sym = T.vector("b_sym")

        # define shared random stream
        trng = T.shared_randomstreams.RandomStreams(1234)
        d = trng.binomial(size=W[1].shape)

        results, updates = theano.scan(lambda v: T.tanh(T.dot(v, W) + b_sym) * d,
                                       sequences=X)
        compute_with_bnoise = theano.function(inputs=[X, W, b_sym],
                                              outputs=[results],
                                              updates=updates,
                                              allow_input_downcast = True)
        x = numpy.eye(10,2)
        w = numpy.ones((2,2))
        b = numpy.ones((2))

        print compute_with_bnoise(x, w, b)


class T_typedlist(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/library/typed_list.html
    # Theano/doc/library/typed_list.txt
    # Any change you do here must also be done in the documentation !

    def test_typedlist_basic(self):
        import theano.typed_list

        tl = theano.typed_list.TypedListType(theano.tensor.fvector)()
        v = theano.tensor.fvector()
        o = theano.typed_list.append(tl, v)
        f = theano.function([tl, v], o)
        output = f([[1, 2, 3], [4, 5]], [2])

        # Validate ouput is as expected
        expected_output = [numpy.array([1,2,3], dtype="float32"),
                           numpy.array([4,5], dtype="float32"),
                           numpy.array([2], dtype="float32")]

        assert len(output) == len(expected_output)
        for i in range(len(output)):
            utt.assert_allclose(output[i], expected_output[i])

    def test_typedlist_with_scan(self):
        import theano.typed_list

        a = theano.typed_list.TypedListType(theano.tensor.fvector)()
        l = theano.typed_list.length(a)
        s, _ = theano.scan(fn=lambda i, tl: tl[i].sum(),
                        non_sequences=[a],
                        sequences=[theano.tensor.arange(l, dtype='int64')])

        f = theano.function([a], s)
        output = f([[1, 2, 3], [4, 5]])

        # Validate ouput is as expected
        expected_output = numpy.array([6, 9], dtype="float32")
        utt.assert_allclose(output, expected_output)
