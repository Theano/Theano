from __future__ import absolute_import, print_function, division
import six.moves.cPickle as pickle
import os
import shutil
import tempfile
import unittest

import numpy as np

import theano
from theano.compile.io import In


def test_function_dump():
    v = theano.tensor.vector()
    fct1 = theano.function([v], v + 1)

    try:
        tmpdir = tempfile.mkdtemp()
        fname = os.path.join(tmpdir, 'test_function_dump.pkl')
        theano.function_dump(fname, [v], v + 1)
        with open(fname, 'rb') as f:
            l = pickle.load(f)
    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir)

    fct2 = theano.function(**l)
    x = [1, 2, 3]
    assert np.allclose(fct1(x), fct2(x))


class TestFunctionIn(unittest.TestCase):

    def test_in_strict(self):

        a = theano.tensor.dvector()
        b = theano.shared(7)
        out = a + b

        f = theano.function([In(a, strict=False)], out)
        # works, rand generates float64 by default
        f(np.random.rand(8))
        # works, casting is allowed
        f(np.array([1, 2, 3, 4], dtype='int32'))

        f = theano.function([In(a, strict=True)], out)
        try:
            # fails, f expects float64
            f(np.array([1, 2, 3, 4], dtype='int32'))
        except TypeError:
            pass

    def test_explicit_shared_input(self):
        # This is not a test of the In class per se, but the In class relies
        # on the fact that shared variables cannot be explicit inputs
        a = theano.shared(1.0)
        self.assertRaises(TypeError, theano.function, [a], a + 1)

    def test_in_shared_variable(self):
        # Ensure that an error is raised if the In wrapped is used to wrap
        # a shared variable
        a = theano.shared(1.0)
        a_wrapped = In(a, update=a + 1)
        self.assertRaises(TypeError, theano.function, [a_wrapped])

    def test_in_mutable(self):
        a = theano.tensor.dvector()
        a_out = a * 2  # assuming the op which makes this "in place" triggers

        # using mutable=True will let f change the value in aval
        f = theano.function([In(a, mutable=True)], a_out, mode='FAST_RUN')
        aval = np.random.rand(10)
        aval2 = aval.copy()
        assert np.all(f(aval) == (aval2 * 2))
        assert not np.all(aval == aval2)

        # using mutable=False should leave the input untouched
        f = theano.function([In(a, mutable=False)], a_out, mode='FAST_RUN')
        aval = np.random.rand(10)
        aval2 = aval.copy()
        assert np.all(f(aval) == (aval2 * 2))
        assert np.all(aval == aval2)

    def test_in_update(self):
        a = theano.tensor.dscalar('a')
        f = theano.function([In(a, value=0.0, update=a + 1)], a,
                            mode='FAST_RUN')

        # Ensure that, through the executions of the function, the state of the
        # input is persistent and is updated as it should
        assert f() == 0.0
        assert f() == 1.0
        assert f() == 2.0

    def test_in_update_wrong_dtype(self):
        # Ensure that an error is raised if an In-wrapped variables has
        # an update of a different type
        a = theano.tensor.dscalar('a')
        b = theano.tensor.dvector('b')
        self.assertRaises(TypeError, In, a, update=b)

    def test_in_update_shared(self):
        # Test that using both In() with updates and shared variables with
        # updates in the same function behaves as expected
        shared_var = theano.shared(1.0)
        a = theano.tensor.dscalar('a')
        a_wrapped = In(a, value=0.0, update=shared_var)
        f = theano.function([a_wrapped], [], updates={shared_var: a},
                            mode='FAST_RUN')

        # Ensure that, through the executions of the function, the state of
        # the input and the shared variable are appropriate (after N execution,
        # the values have swapped N times). This allows testing that the
        # changes occur at the same time and one doesn't overwrite the other.
        for i in range(5):
            f()
            assert np.allclose(shared_var.get_value(), i % 2)

    def test_in_allow_downcast_int(self):
        a = theano.tensor.wvector('a')  # int16
        b = theano.tensor.bvector('b')  # int8
        c = theano.tensor.bscalar('c')  # int8
        f = theano.function([In(a, allow_downcast=True),
                             In(b, allow_downcast=False),
                             In(c, allow_downcast=None)],
                            (a + b + c))

        # Both values are in range. Since they're not ndarrays (but lists),
        # they will be converted, and their value checked.
        assert np.all(f([3], [6], 1) == 10)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        self.assertRaises(TypeError, f, [3], np.array([6], dtype='int16'),
                          1)

        # Value too big for a, silently ignored
        assert np.all(f([2 ** 20], np.ones(1, dtype='int8'), 1) == 2)

        # Value too big for b, raises TypeError
        self.assertRaises(TypeError, f, [3], [312], 1)

        # Value too big for c, raises TypeError
        self.assertRaises(TypeError, f, [3], [6], 806)

    def test_in_allow_downcast_floatX(self):
        a = theano.tensor.fscalar('a')
        b = theano.tensor.fscalar('b')
        c = theano.tensor.fscalar('c')

        f = theano.function([In(a, allow_downcast=True),
                             In(b, allow_downcast=False),
                             In(c, allow_downcast=None)],
                            (a + b + c))

        # If the values can be accurately represented, everything is OK
        assert np.all(f(0, 0, 0) == 0)

        # If allow_downcast is True, idem
        assert np.allclose(f(0.1, 0, 0), 0.1)

        # If allow_downcast is False, nope
        self.assertRaises(TypeError, f, 0, 0.1, 0)

        # If allow_downcast is None, it should work iff floatX=float32
        if theano.config.floatX == 'float32':
            assert np.allclose(f(0, 0, 0.1), 0.1)
        else:
            self.assertRaises(TypeError, f, 0, 0, 0.1)

    def test_in_allow_downcast_vector_floatX(self):
        a = theano.tensor.fvector('a')
        b = theano.tensor.fvector('b')
        c = theano.tensor.fvector('c')

        f = theano.function([In(a, allow_downcast=True),
                             In(b, allow_downcast=False),
                             In(c, allow_downcast=None)],
                            (a + b + c))

        # If the values can be accurately represented, everything is OK
        z = [0]
        assert np.all(f(z, z, z) == 0)

        # If allow_downcast is True, idem
        assert np.allclose(f([0.1], z, z), 0.1)

        # If allow_downcast is False, nope
        self.assertRaises(TypeError, f, z, [0.1], z)

        # If allow_downcast is None, like False
        self.assertRaises(TypeError, f, z, z, [0.1])
