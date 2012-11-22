""" test code snippet in the Theano tutorials.
"""

import os, shutil, unittest
import theano
import theano.tensor as T
from theano import function
import numpy
from numpy import array

from theano import config
from theano.tests  import unittest_tools as utt


class T_extending(unittest.TestCase):
    ## All tests here belong to files in
    ## http://deeplearning.net/software/theano/extending
    ## Theano/doc/extending/*.txt
    ## Any change you do here also add it to the tutorial!
    ## This belongs to an entire folder since code-snippets are connected
    ## from one file to another .. and they do not make sense on their
    ## own.

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
        assert f(5, 6)     == 30.0
        assert f(5.6, 6.7) ==  37.519999999999996

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

        add = BinaryDoubleOp(name = 'add',
                            fn = lambda x, y: x + y)

        sub = BinaryDoubleOp(name = 'sub',
                            fn = lambda x, y: x - y)

        mul = BinaryDoubleOp(name = 'mul',
                            fn = lambda x, y: x * y)

        div = BinaryDoubleOp(name = 'div',
                            fn = lambda x, y: x / y)


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

        add = BinaryDoubleOp(name = 'add',
                            fn = lambda x, y: x + y)

        sub = BinaryDoubleOp(name = 'sub',
                            fn = lambda x, y: x - y)

        mul = BinaryDoubleOp(name = 'mul',
                            fn = lambda x, y: x * y)

        div = BinaryDoubleOp(name = 'div',
                            fn = lambda x, y: x / y)

        def c_declare(name, sub):
            return """
            double %(name)s;
            """ % dict(name = name)
        double.c_declare = c_declare


        def c_init(name, sub):
            return """
            %(name)s = 0.0;
            """ % dict(name = name)
        double.c_init = c_init



        def c_extract(name, sub):
            return """
            if (!PyFloat_Check(py_%(name)s)) {
                PyErr_SetString(PyExc_TypeError, "expected a float");
                %(fail)s
            }
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

            def c_declare(self, name, sub):
                return """
                double %(name)s;
                """ % dict(name = name)

            def c_init(self, name, sub):
                return """
                %(name)s = 0.0;
                """ % dict(name = name)

            def c_extract(self, name, sub):
                return """
                if (!PyFloat_Check(py_%(name)s)) {
                    PyErr_SetString(PyExc_TypeError, "expected a float");
                    %(fail)s
                }
                %(name)s = PyFloat_AsDouble(py_%(name)s);
                """ % dict(sub, name = name)

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


        add = BinaryDoubleOp(name = 'add',
                            fn = lambda x, y: x + y,
                            ccode = "%(z)s = %(x)s + %(y)s;")

        sub = BinaryDoubleOp(name = 'sub',
                            fn = lambda x, y: x - y,
                            ccode = "%(z)s = %(x)s - %(y)s;")

        mul = BinaryDoubleOp(name = 'mul',
                            fn = lambda x, y: x * y,
                            ccode = "%(z)s = %(x)s * %(y)s;")

        div = BinaryDoubleOp(name = 'div',
                            fn = lambda x, y: x / y,
                            ccode = "%(z)s = %(x)s / %(y)s;")


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



class T_introduction(unittest.TestCase):
    ## All tests here belong to
    ## http://deeplearning.net/software/theano/tutorial/introduction.html
    ## Theano/doc/tutorial/introduction.txt
    ## Any change you do here also add it to the tutorial !
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
    ## All tests here belong to
    ## http://deeplearning.net/software/theano/tutorial/adding.html
    ## Theano/doc/tutorial/adding.txt
    ## Any change you do here also add it to the tutorial !


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
    ## All tests here belog to
    ## http://deeplearning.net/software/theano/tutorial/examples.html
    ## Theano/doc/tutorial/examples.txt
    ## Any change you do here also add it to the tutorial !

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
        state = shared(numpy.int64(0))
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
        foo = T.lscalar()    # the type (lscalar) must match the shared variable we
                            # are replacing with the ``givens`` list
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
        assert numpy.all(v1 != v2)


class T_aliasing(unittest.TestCase):
    ## All tests here belog to
    ## http://deeplearning.net/software/theano/tutorial/aliasing.html
    ## Theano/doc/tutorial/aliasing.txt
    ## Any change you do here also add it to the tutorial !

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
            ## some_inplace_fn
            s.get_value(borrow=True).__imul__(2),
            borrow=True)



    def test_aliasing_3(self):

        import theano, theano.tensor

        x = theano.tensor.matrix()
        y = 2*x
        f = theano.function([theano.In(x, borrow=True)], theano.Out(y, borrow=True))



class T_loading_and_saving(unittest.TestCase):
    ## All tests here belong to
    ## http://deeplearning.net/software/theano/tutorial/loading_and_saving.html
    ## Theano/doc/tutorial/loading_and_saving.txt
    ## Any change you do here also add it to the tutorial !

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

                f = file('obj.save', 'wb')
                cPickle.dump(my_obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()


                f = file('obj.save', 'rb')
                loaded_obj = cPickle.load(f)
                f.close()

                obj1 = my_obj
                obj2 = my_obj
                obj3 = my_obj

                f = file('objects.save', 'wb')
                for obj in [obj1, obj2, obj3]:
                    cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
                f.close()

                f = file('objects.save', 'rb')
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
    ## All tests here belog to
    ## http://deeplearning.net/software/theano/tutorial/modes.html
    ## Theano/doc/tutorial/modes.txt
    ## Any change you do here also add it to the tutorial !

    def test_modes_1(self):

        x = T.dvector('x')

        f = theano.function([x], 10*x, mode='DEBUG_MODE')

        assert numpy.all(f([5]) == [50.])
        assert numpy.all(f([0]) == [0.] )
        assert numpy.all(f([7]) == [70.])

class T_using_gpu(unittest.TestCase):
    ## All tests here belog to
    ## http://deeplearning.net/software/theano/tutorial/using_gpu.html
    ## Theano/doc/tutorial/using_gpu.txt
    ## Any change you do here also add it to the tutorial !


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

        if theano.config.device.find('gpu') >-1:

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
            if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
                print 'Used the cpu'
            else:
                print 'Used the gpu'

            assert not numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()])


class T_fibby(unittest.TestCase):
    ## All tests here belong to
    ## http://deeplearning.net/software/theano/extending/fibby.html
    ## Theano/doc/extending/fibby.txt
    ## Any change you do here also add it to the tutorial !

    def test_fibby_1(self):

        class Fibby(theano.Op):

            """
            An arbitrarily generalized Fibbonacci sequence
            """

            def __eq__(self, other):
                return type(self) == type(other)

            def __hash__(self):
                return hash(type(self))

            def make_node(self, x):
                x_ = tensor.as_tensor_variable(x)
                return theano.Apply(self,
                    inputs=[x_],
                    outputs=[x_.type()])
                # using x_.type() is dangerous, it copies x's broadcasting behaviour

            def perform(self, node, inputs, output_storage):
                x, = inputs
                y = output_storage[0][0] = x.copy()
                for i in range(2,len(x)):
                    y[i] = y[i-1] * y[i-2] + x[i]

            def c_code(self, node, name, inames, onames, sub):
                x, = inames
                y, = onames
                fail = sub['fail']
                return """
                    Py_XDECREF(%(y)s);
                    %(y)s = (PyArrayObject*)PyArray_FromArray(
                            %(x)s, 0, NPY_ARRAY_ENSURECOPY);
                    if (!(%y)s) %(fail)s;
                    dtype_%(y)s * y = (dtype_%(y)s*)%(y)s->data;
                    dtype_%(x)s * x = (dtype_%(x)s*)%(x)s->data;
                    for (int i = 2; i < %(x)s->dimensions[0]; ++i)
                        y[i] = y[i-1]*y[i-2] + x[i];
                """ % locals()

        fibby = Fibby()


        # Remove any fibby(zeros(...))
        @theano.tensor.opt.register_specialize
        @theano.gof.local_optimizer([fibby])
        def fibby_of_zero(node):
            if node.op == fibby:
                x = node.inputs[0]
                try:
                    if numpy.all(0 == get_scalar_constant_value(x)):
                        return [x]
                except TypeError:
                    pass



class T_graphstructures(unittest.TestCase):
    ## All tests here belong to
    ## http://deeplearning.net/software/theano/extending/graphstructures.html
    ## Theano/doc/extending/graphstructures.txt
    ## Any change you do here also add it to the tutorial !

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
        float64_matrix = TensorType(dtype = 'float64',              # double
                                    broadcastable = (False, False)) # matrix

        # We make the Variable instances we need.
        x = Variable(type = float64_matrix, name = 'x')
        y = Variable(type = float64_matrix, name = 'y')
        z = Variable(type = float64_matrix, name = 'z')

        # This is the Variable that we want to symbolically represents y*z
        mul_variable = Variable(type = float64_matrix)
        assert mul_variable.owner is None

        # Instantiate a symbolic multiplication
        node_mul = Apply(op = mul,
                         inputs = [y, z],
                         outputs = [mul_variable])
        # Fields 'owner' and 'index' are set by Apply
        assert mul_variable.owner is node_mul
        # 'index' is the position of mul_variable in mode_mul's outputs
        assert mul_variable.index == 0

        # This is the Variable that we want to symbolically represents x+(y*z)
        add_variable = Variable(type = float64_matrix)
        assert add_variable.owner is None

        # Instantiate a symbolic addition
        node_add = Apply(op = add,
                         inputs = [x, mul_variable],
                         outputs = [add_variable])
        # Fields 'owner' and 'index' are set by Apply
        assert add_variable.owner is node_add
        assert add_variable.index == 0

        e = add_variable

        # We have access to x, y and z through pointers
        assert e.owner.inputs[0] is x
        assert e.owner.inputs[1] is mul_variable
        assert e.owner.inputs[1].owner.inputs[0] is y
        assert e.owner.inputs[1].owner.inputs[1] is z
