from __future__ import absolute_import, print_function, division
import unittest

from nose.plugins.skip import SkipTest
import numpy as np

import theano
from theano.tensor import dmatrix, iscalar, lscalar, dmatrices
from theano import tensor

from theano.compile import In
from theano.compile import pfunc
from theano.compile import shared
from theano.compile import config


def data_of(s):
    """Return the raw value of a shared variable"""
    return s.container.storage[0]


class Test_pfunc(unittest.TestCase):

    def test_doc(self):
        """Ensure the code given in pfunc.txt works as expected"""

        # Example #1.
        a = lscalar()
        b = shared(1)
        f1 = pfunc([a], (a + b))
        f2 = pfunc([In(a, value=44)], a + b, updates={b: b + 1})
        self.assertTrue(b.get_value() == 1)
        self.assertTrue(f1(3) == 4)
        self.assertTrue(f2(3) == 4)
        self.assertTrue(b.get_value() == 2)
        self.assertTrue(f1(3) == 5)
        b.set_value(0)
        self.assertTrue(f1(3) == 3)

        # Example #2.
        a = tensor.lscalar()
        b = shared(7)
        f1 = pfunc([a], a + b)
        f2 = pfunc([a], a * b)
        self.assertTrue(f1(5) == 12)
        b.set_value(8)
        self.assertTrue(f1(5) == 13)
        self.assertTrue(f2(4) == 32)

    def test_shared(self):

        # CHECK: two functions (f1 and f2) can share w
        w = shared(np.random.rand(2, 2), 'w')
        wval = w.get_value(borrow=False)

        x = dmatrix()
        out1 = w + x
        out2 = w * x
        f1 = pfunc([x], [out1])
        f2 = pfunc([x], [out2])
        xval = np.random.rand(2, 2)
        assert np.all(f1(xval) == xval + wval)
        assert np.all(f2(xval) == xval * wval)

        # CHECK: updating a shared value
        f3 = pfunc([x], out1, updates=[(w, (w - 1))])
        # f3 changes the value of w
        assert np.all(f3(xval) == xval + wval)
        # this same value is read by f1
        assert np.all(f1(xval) == xval + (wval - 1))

        w.set_value(w.get_value(borrow=True) * 10, borrow=True)
        # this same value is read by f1
        assert np.all(f1(xval) == xval + w.get_value(borrow=True))

    def test_no_shared_as_input(self):
        """Test that shared variables cannot be used as function inputs."""
        w_init = np.random.rand(2, 2)
        w = shared(w_init.copy(), 'w')
        try:
            pfunc([w], theano.tensor.sum(w * w))
            assert False
        except TypeError as e:
            msg = 'Cannot use a shared variable (w) as explicit input'
            if str(e).find(msg) < 0:
                raise

    def test_default_container(self):
        # Ensure it is possible to (implicitly) use a shared variable in a
        # function, as a 'state' that can be updated at will.

        rng = np.random.RandomState(1827)
        w_init = rng.rand(5)
        w = shared(w_init.copy(), 'w')
        reg = theano.tensor.sum(w * w)
        f = pfunc([], reg)

        assert f() == np.sum(w_init * w_init)
        # Change the value of w and ensure the output changes accordingly.
        w.set_value(w.get_value(borrow=True) + 1.0, borrow=True)
        assert f() == np.sum((w_init + 1) ** 2)

    def test_default_scalar_container(self):
        # Similar in spirit to test_default_container, but updating a scalar
        # variable. This is a sanity check for non mutable types.
        x = shared(0.0, 'x')
        f = pfunc([], x)
        assert f() == 0
        x.set_value(x.get_value(borrow=True) + 1, borrow=True)
        assert f() == 1

    def test_param_strict(self):

        a = tensor.dvector()
        b = shared(7)
        out = a + b

        f = pfunc([In(a, strict=False)], [out])
        # works, rand generates float64 by default
        f(np.random.rand(8))
        # works, casting is allowed
        f(np.array([1, 2, 3, 4], dtype='int32'))

        f = pfunc([In(a, strict=True)], [out])
        try:
            # fails, f expects float64
            f(np.array([1, 2, 3, 4], dtype='int32'))
        except TypeError:
            pass

    def test_param_mutable(self):
        a = tensor.dvector()
        a_out = a * 2  # assuming the op which makes this "in place" triggers

        # using mutable=True will let fip change the value in aval
        fip = pfunc([In(a, mutable=True)], [a_out], mode='FAST_RUN')
        aval = np.random.rand(10)
        aval2 = aval.copy()
        assert np.all(fip(aval) == (aval2 * 2))
        assert not np.all(aval == aval2)

        # using mutable=False should leave the input untouched
        f = pfunc([In(a, mutable=False)], [a_out], mode='FAST_RUN')
        aval = np.random.rand(10)
        aval2 = aval.copy()
        assert np.all(f(aval) == (aval2 * 2))
        assert np.all(aval == aval2)

    def test_shared_mutable(self):
        bval = np.arange(5)
        b = shared(bval)
        b_out = b * 2

        # shared vars copy args.
        assert b.get_value(borrow=True) is not bval
        # so we do this to get at the underlying data
        bval = data_of(b)

        # by default, shared are not mutable unless doing an explicit update
        f = pfunc([], [b_out], mode='FAST_RUN')
        assert (f() == np.arange(5) * 2).all()
        assert np.all(b.get_value(borrow=True) == np.arange(5))

        # using updates, b is now a mutable parameter
        f = pfunc([], [b_out], updates=[(b, b_out)], mode='FAST_RUN')
        assert (f() == (np.arange(5) * 2)).all()
        # because of the update
        assert (b.get_value(borrow=True) == (np.arange(5) * 2)).all()
        assert (bval == (np.arange(5) * 2)).all()  # because of mutable=True

        # do not depend on updates being in-place though!
        bval = np.arange(5)
        b.set_value(bval, borrow=True)
        bval = data_of(b)
        f = pfunc([], [b_out], updates=[(b, (b_out + 3))], mode='FAST_RUN')
        assert (f() == (np.arange(5) * 2)).all()
        # because of the update
        assert (b.get_value(borrow=True) == ((np.arange(5) * 2) + 3)).all()
        # bval got modified to something...
        assert not (bval == np.arange(5)).all()
        # ... but not to b.value !
        assert not (bval == b.get_value(borrow=True)).all()

    def test_param_allow_downcast_int(self):
        a = tensor.wvector('a')  # int16
        b = tensor.bvector('b')  # int8
        c = tensor.bscalar('c')  # int8
        f = pfunc([In(a, allow_downcast=True),
                   In(b, allow_downcast=False),
                   In(c, allow_downcast=None)],
                  (a + b + c))

        # Both values are in range. Since they're not ndarrays (but lists),
        # they will be converted, and their value checked.
        assert np.all(f([3], [6], 1) == 10)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        self.assertRaises(TypeError, f,
                          [3], np.array([6], dtype='int16'), 1)

        # Value too big for a, silently ignored
        assert np.all(f([2 ** 20], np.ones(1, dtype='int8'), 1) == 2)

        # Value too big for b, raises TypeError
        self.assertRaises(TypeError, f, [3], [312], 1)

        # Value too big for c, raises TypeError
        self.assertRaises(TypeError, f, [3], [6], 806)

    def test_param_allow_downcast_floatX(self):
        a = tensor.fscalar('a')
        b = tensor.fscalar('b')
        c = tensor.fscalar('c')

        f = pfunc([In(a, allow_downcast=True),
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
        if config.floatX == 'float32':
            assert np.allclose(f(0, 0, 0.1), 0.1)
        else:
            self.assertRaises(TypeError, f, 0, 0, 0.1)

    def test_param_allow_downcast_vector_floatX(self):
        a = tensor.fvector('a')
        b = tensor.fvector('b')
        c = tensor.fvector('c')

        f = pfunc([In(a, allow_downcast=True),
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

    def test_allow_input_downcast_int(self):
        a = tensor.wvector('a')  # int16
        b = tensor.bvector('b')  # int8
        c = tensor.bscalar('c')  # int8

        f = pfunc([a, b, c], (a + b + c), allow_input_downcast=True)
        # Value too big for a, b, or c, silently ignored
        assert f([2 ** 20], [1], 0) == 1
        assert f([3], [312], 0) == 59
        assert f([3], [1], 806) == 42

        g = pfunc([a, b, c], (a + b + c), allow_input_downcast=False)
        # All values are in range. Since they're not ndarrays (but lists
        # or scalars), they will be converted, and their value checked.
        assert np.all(g([3], [6], 0) == 9)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        self.assertRaises(TypeError, g,
                          [3], np.array([6], dtype='int16'), 0)

        # Value too big for b, raises TypeError
        self.assertRaises(TypeError, g, [3], [312], 0)

        h = pfunc([a, b, c], (a + b + c))  # Default: allow_input_downcast=None
        # Everything here should behave like with False
        assert np.all(h([3], [6], 0) == 9)
        self.assertRaises(TypeError, h,
                          [3], np.array([6], dtype='int16'), 0)
        self.assertRaises(TypeError, h, [3], [312], 0)

    def test_allow_downcast_floatX(self):
        a = tensor.fscalar('a')
        b = tensor.fvector('b')

        f = pfunc([a, b], (a + b), allow_input_downcast=True)
        g = pfunc([a, b], (a + b), allow_input_downcast=False)
        h = pfunc([a, b], (a + b), allow_input_downcast=None)

        # If the values can be accurately represented, OK
        assert np.all(f(0, [0]) == 0)
        assert np.all(g(0, [0]) == 0)
        assert np.all(h(0, [0]) == 0)

        # For the vector: OK iff allow_input_downcast is True
        assert np.allclose(f(0, [0.1]), 0.1)
        self.assertRaises(TypeError, g, 0, [0.1])
        self.assertRaises(TypeError, h, 0, [0.1])

        # For the scalar: OK if allow_input_downcast is True,
        # or None and floatX==float32
        assert np.allclose(f(0.1, [0]), 0.1)
        self.assertRaises(TypeError, g, 0.1, [0])
        if config.floatX == 'float32':
            assert np.allclose(h(0.1, [0]), 0.1)
        else:
            self.assertRaises(TypeError, h, 0.1, [0])

    def test_update(self):
        """Test update mechanism in different settings."""

        # Simple value assignment.
        x = shared(0)
        assign = pfunc([], [], updates={x: 3})
        assign()
        self.assertTrue(x.get_value() == 3)

        # Basic increment function.
        x.set_value(0)
        inc = pfunc([], [], updates={x: x + 1})
        inc()
        self.assertTrue(x.get_value() == 1)

        # Increment by a constant value.
        x.set_value(-1)
        y = shared(2)
        inc_by_y = pfunc([], [], updates={x: x + y})
        inc_by_y()
        self.assertTrue(x.get_value() == 1)

    def test_update_err_broadcast(self):
        # Test that broadcastable dimensions raise error
        data = np.random.rand(10, 10).astype('float32')
        output_var = shared(name="output", value=data)

        # the update_var has type matrix, and the update expression
        # is a broadcasted scalar, and that should be allowed.
        self.assertRaises(TypeError, theano.function, inputs=[], outputs=[],
                          updates={output_var: output_var.sum().dimshuffle('x', 'x')})

    def test_duplicate_updates(self):
        x, y = dmatrices('x', 'y')
        z = shared(np.ones((2, 3)))
        self.assertRaises(ValueError, theano.function, [x, y], [z],
                          updates=[(z, (z + x + y)), (z, (z - x))])

    def test_givens(self):
        x = shared(0)
        assign = pfunc([], x, givens={x: 3})
        assert assign() == 3
        assert x.get_value(borrow=True) == 0

        y = tensor.ivector()
        f = pfunc([y], (y * x), givens={x: 6})
        assert np.all(f([1, 1, 1]) == [6, 6, 6])
        assert x.get_value() == 0

        z = tensor.ivector()
        c = z * y
        f = pfunc([y], (c + 7),
                  givens={z: theano._asarray([4, 4, 4], dtype='int32')})
        assert np.all(f([1, 1, 1]) == [11, 11, 11])
        assert x.get_value() == 0

    def test_clone0(self):
        x = shared(np.asarray([4, 4, 4]))
        y = shared(np.asarray([4, 4, 4]))
        z = shared(np.asarray([2, 2, 2]))
        up = pfunc([], [], updates={
            x: (x * 5),
            y: ((x * 5) + y),
            z: (((x * 5) + y) ** z)})

        up()
        assert np.all(x.get_value() == 20)
        assert np.all(y.get_value() == 24)
        assert np.all(z.get_value() == (24 ** 2))

    def test_default_updates(self):
        x = shared(0)
        x.default_update = x + 1

        f = pfunc([], [x])
        f()
        assert x.get_value() == 1

        del x.default_update
        f()
        assert x.get_value() == 2

        g = pfunc([], [x])
        g()
        assert x.get_value() == 2

    def test_no_default_updates(self):
        x = shared(0)
        y = shared(1)
        x.default_update = x + 2

        # Test that the default update is taken into account in the right cases
        f1 = pfunc([], [x], no_default_updates=True)
        f1()
        assert x.get_value() == 0

        f2 = pfunc([], [x], no_default_updates=[x])
        f2()
        assert x.get_value() == 0

        f3 = pfunc([], [x], no_default_updates=[x, y])
        f3()
        assert x.get_value() == 0

        f4 = pfunc([], [x], no_default_updates=[y])
        f4()
        assert x.get_value() == 2

        f5 = pfunc([], [x], no_default_updates=[])
        f5()
        assert x.get_value() == 4

        f5 = pfunc([], [x], no_default_updates=False)
        f5()
        assert x.get_value() == 6

        self.assertRaises(TypeError, pfunc, [], [x], no_default_updates=(x))
        self.assertRaises(TypeError, pfunc, [], [x], no_default_updates=x)
        self.assertRaises(TypeError, pfunc, [], [x],
                          no_default_updates='canard')

        # Mix explicit updates and no_default_updates
        g1 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=True)
        g1()
        assert x.get_value() == 5

        g2 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[x])
        g2()
        assert x.get_value() == 4

        g3 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[x, y])
        g3()
        assert x.get_value() == 3

        g4 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[y])
        g4()
        assert x.get_value() == 2

        g5 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=[])
        g5()
        assert x.get_value() == 1

        g5 = pfunc([], [x], updates=[(x, (x - 1))], no_default_updates=False)
        g5()
        assert x.get_value() == 0

    def test_default_updates_expressions(self):
        x = shared(0)
        y = shared(1)
        a = lscalar('a')

        z = a * x
        x.default_update = x + y

        f1 = pfunc([a], z)
        f1(12)
        assert x.get_value() == 1

        f2 = pfunc([a], z, no_default_updates=True)
        assert f2(7) == 7
        assert x.get_value() == 1

        f3 = pfunc([a], z, no_default_updates=[x])
        assert f3(9) == 9
        assert x.get_value() == 1

    def test_default_updates_multiple(self):
        x = shared(0)
        y = shared(1)

        x.default_update = x - 1
        y.default_update = y + 1

        f1 = pfunc([], [x, y])
        f1()
        assert x.get_value() == -1
        assert y.get_value() == 2

        f2 = pfunc([], [x, y], updates=[(x, (x - 2))], no_default_updates=[y])
        f2()
        assert x.get_value() == -3
        assert y.get_value() == 2

        f3 = pfunc([], [x, y], updates=[(x, (x - 2))], no_default_updates=True)
        f3()
        assert x.get_value() == -5
        assert y.get_value() == 2

        f4 = pfunc([], [x, y], updates=[(y, (y - 2))])
        f4()
        assert x.get_value() == -6
        assert y.get_value() == 0

    def test_default_updates_chained(self):
        x = shared(2)
        y = shared(1)
        z = shared(-1)

        x.default_update = x - y
        y.default_update = z
        z.default_update = z - 1

        f1 = pfunc([], [x])
        f1()
        assert x.get_value() == 1
        assert y.get_value() == -1
        assert z.get_value() == -2

        f2 = pfunc([], [x, y])
        f2()
        assert x.get_value() == 2
        assert y.get_value() == -2
        assert z.get_value() == -3

        f3 = pfunc([], [y])
        f3()
        assert x.get_value() == 2
        assert y.get_value() == -3
        assert z.get_value() == -4

        f4 = pfunc([], [x, y], no_default_updates=[x])
        f4()
        assert x.get_value() == 2
        assert y.get_value() == -4
        assert z.get_value() == -5

        f5 = pfunc([], [x, y, z], no_default_updates=[z])
        f5()
        assert x.get_value() == 6
        assert y.get_value() == -5
        assert z.get_value() == -5

    def test_default_updates_input(self):
        x = shared(0)
        y = shared(1)
        if theano.configdefaults.python_int_bitwidth() == 32:
            a = iscalar('a')
        else:
            a = lscalar('a')

        x.default_update = y
        y.default_update = y + a

        f1 = pfunc([], x, no_default_updates=True)
        f1()
        assert x.get_value() == 0
        assert y.get_value() == 1

        f2 = pfunc([], x, no_default_updates=[x])
        f2()
        assert x.get_value() == 0
        assert y.get_value() == 1

        f3 = pfunc([], x, no_default_updates=[y])
        f3()
        assert x.get_value() == 1
        assert y.get_value() == 1

        f4 = pfunc([a], x)
        f4(2)
        assert x.get_value() == 1
        assert y.get_value() == 3

        f5 = pfunc([], x, updates={y: (y - 1)})
        f5()
        assert x.get_value() == 3
        assert y.get_value() == 2

        # a is needed as input if y.default_update is used
        self.assertRaises(theano.gof.MissingInputError, pfunc, [], x)

    def test_default_updates_partial_graph(self):
        a = shared(0)
        a.default_update = a + 1  # Increment a each time it is used
        b = 2 * a
        # Use only the tip of the graph, a is not used
        f = pfunc([b], b)
        assert a.get_value() == 0
        f(21)
        assert a.get_value() == 0

    def test_givens_replaces_shared_variable(self):
        a = shared(1., 'a')
        a.default_update = a + 3.
        b = tensor.dscalar('b')
        c = a + 10
        f = pfunc([b], c, givens={a: b})

        assert len(f.maker.fgraph.inputs) == 1
        assert len(f.maker.fgraph.outputs) == 1

    def test_givens_replaces_shared_variable2(self):
        a = shared(1., 'a')
        a.default_update = a + 3
        c = a + 10
        f = pfunc([], c, givens={a: (a + 10)})

        assert f() == 21
        assert f() == 34

    def test_duplicate_inputs(self):
        x = theano.tensor.lscalar('x')
        self.assertRaises(theano.compile.UnusedInputError,
                          theano.function, [x, x, x], x)

    def test_update_same(self):
        # There was a bug in CVM, triggered when a shared variable
        # was its own update expression.
        a = shared(1., 'a')
        b = shared(np.ones((2, 3)), 'b')

        # The order of the variables is not determined, so we try
        # both shared variables.
        # TODO: explain the above comment. By "not determined" does
        # this mean "not deterministic"?
        # This test originally wrote the updates using dictionaries,
        # and iterating over the dictionary was not deterministic.
        # Is that all the comment above meant, or is the CVM intended
        # to add extra non-determinism? Or is the CVM meant to
        # deterministically but arbitrarily pick an order for the updates?
        f = theano.function([], [], updates=[(a, a), (b, (2 * b))])
        g = theano.function([], [], updates=[(a, (a * 2)), (b, b)])

        f()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()
        g()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()

    def test_update_equiv(self):
        # Like test_update_same, but the update expression is simplified until
        # it is found to be equal to the original variable
        a = shared(1., 'a')
        b = shared(np.ones((2, 3)), 'b')

        # See comment in test_update_same about why we try both
        # shared variables.
        f = theano.function([], [], updates=[(a, a), (b, (2 * b - b))])
        g = theano.function([], [], updates=[(a, (a * 2 - a)), (b, b)])

        f()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()
        g()
        assert a.get_value(borrow=True).shape == (), a.get_value()
        assert b.get_value(borrow=True).shape == (2, 3), b.get_value()


class Test_aliasing_rules(unittest.TestCase):
    """
    1. Theano manages its own memory space, which typically does not overlap
    with the memory of normal python variables that the user uses.

    2. shared variables are allocated in this memory space, as are the
    temporaries used for Function evalution.

    3. Physically, this managed memory space may be spread across the host,
    on a GPU device(s), or even on a remote machine.

    4. Theano assumes that shared variables are never aliased to one another,
    and tries to make it impossible to accidentally alias them.

    5. Theano's managed data is constant while Theano Functions are not running
    and theano library code is not running.

    6. The default behaviour of Function is to return user-space values for
    outputs, but this can be overridden (borrow=True) for better performance,
    in which case the returned value may be aliased to managed memory, and
    potentially invalidated by the next Theano Function call or call to theano
    library code.
    """

    def shared(self, x):
        return tensor._shared(x)

    def test_shared_constructor_copies(self):
        # shared constructor makes copy
        # (rule #2)
        orig_a = np.zeros((2, 2))
        A = self.shared(orig_a)
        assert not np.may_share_memory(orig_a, data_of(A))

        # rule #2 reading back from theano-managed memory
        assert not np.may_share_memory(A.get_value(borrow=False),
                                       data_of(A))

    def test_sparse_input_aliasing_affecting_inplace_operations(self):
        ##
        # Note this test will never fail because I am not aware of any
        # inplace op on sparse variables
        try:
            import scipy.sparse as sp
        except ImportError:
            # The variable enable_sparse will be used to disable the test file.
            pass

        from theano.sparse import enable_sparse
        if not enable_sparse:
            raise SkipTest('Optional package sparse disabled')

        from theano import sparse

        # Note: to trigger this bug with theano rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable (so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op (here dot)

        x = sparse.SparseType('csc', dtype='float64')()
        y = sparse.SparseType('csc', dtype='float64')()
        f = theano.function([theano.In(x, mutable=True),
                             theano.In(y, mutable=True)],
                            (x + y) + (x + y))
        # Test 1. If the same variable is given twice

        # Compute bogus values
        m = sp.csc_matrix(np.asarray(
            [[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]], dtype='float64'))
        bogus_vals = f(m, m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        m = sp.csc_matrix(np.asarray(
            [[1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0],
             [0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0],
             [0, 0, 0, 0, 1]], dtype='float64'))
        m_copy = m.copy()
        vals = f(m, m_copy)

        assert np.allclose(vals.todense(), bogus_vals.todense())

    def test_input_aliasing_affecting_inplace_operations(self):

        # Note: to trigger this bug with theano rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable (so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op (here dot)
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        m1 = theano.tensor.dmatrix()
        m2 = theano.tensor.dmatrix()
        f = theano.function([theano.In(x, mutable=True),
                             theano.In(y, mutable=True),
                             theano.In(m1, mutable=True),
                             theano.In(m2, mutable=True)],
                            theano.dot((x * 2), m1) + theano.dot((y * 3), m2))
        # Test 1. If the same variable is given twice

        # Compute bogus values
        v = np.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = np.asarray([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]], dtype='float64')
        bogus_vals = f(v, v, m, m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        v = np.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = np.asarray([[1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]], dtype='float64')
        m_copy = m.copy()
        v_copy = v.copy()
        vals = f(v, v_copy, m, m_copy)

        assert np.allclose(vals, bogus_vals)

    def test_partial_input_aliasing_affecting_inplace_operations(self):

        # Note: to trigger this bug with theano rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable ( so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op ( here dot )
        x = theano.tensor.dvector()
        y = theano.tensor.dvector()
        z = theano.tensor.dvector()
        m1 = theano.tensor.dmatrix()
        m2 = theano.tensor.dmatrix()
        m3 = theano.tensor.dmatrix()

        # Test 2. If variables only partial overlap
        #   more exactly we care about the case when we have a,b,c
        #   and a shares memory with b, b shares memory with c, but
        #   c does not share memory with a

        f = theano.function(
            [theano.In(x, mutable=True),
             theano.In(y, mutable=True),
             theano.In(z, mutable=True),
             theano.In(m1, mutable=True),
             theano.In(m2, mutable=True),
             theano.In(m3, mutable=True)],
            (theano.dot((x * 2), m1) + theano.dot((y * 3), m2) +
             theano.dot((z * 4), m3)))

        # Compute bogus values
        v = np.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = np.asarray([[1, 0],
                        [0, 1]], dtype='float64')
        bogus_vals = f(v[:2], v[1:3], v[2:4], m, m, m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        v = np.asarray([1, 2, 3, 4, 5], dtype='float64')
        m = np.asarray([[1, 0],
                        [0, 1]], dtype='float64')
        m_copy1 = m.copy()
        v_copy1 = v.copy()
        m_copy2 = m.copy()
        v_copy2 = v.copy()
        vals = f(v[:2], v_copy1[1:3], v_copy2[2:4], m, m_copy1, m_copy2)

        assert np.allclose(vals, bogus_vals)

    def test_potential_output_aliasing_induced_by_updates(self):

        A = self.shared(np.zeros((2, 2)))
        B = self.shared(np.zeros((2, 2)))
        C = np.zeros((2, 2))
        D = tensor.dmatrix()
        DD = D + 5

        f = pfunc([D], [], updates=[(A, D), (B, D)])
        f(C)

        assert not np.may_share_memory(data_of(A), data_of(B))
        f = pfunc([D], [], updates=[(A, D[:]), (B, D)])
        f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        f = pfunc([D], [], updates=[(A, (D + 5)), (B, D[:])])
        f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))

        f = pfunc([D], [], updates=[(A, (D + 5)), (B, D)])
        f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))

        f = pfunc([D], DD, updates=[(A, DD[:1]), (B, DD)])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

        f = pfunc([D], DD, updates=[(A, DD[:1]), (B, (DD[:1] * 2))])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

        f = pfunc([D], (DD * 4),
                  updates=[(A, (DD[:1] * 3)), (B, (DD[:1] * 2))])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

        f = pfunc([D], (DD * 4),
                  updates=[(A, (DD[:1] * 3)), (B, (DD[:1] * 3))])
        R = f(C)
        assert not np.may_share_memory(data_of(A), data_of(B))
        assert not np.may_share_memory(R, data_of(B))
        assert not np.may_share_memory(R, data_of(A))

    def test_no_aliasing_0(self):
        # B is a shared variable, A is updated with B's contents
        # we need A to be copied to avoid aliasing
        A = self.shared(np.zeros((2, 2)) + .5)
        B = self.shared(np.zeros((2, 2)) - .5)
        f = pfunc([], [], updates=[(A, B)])
        f()
        assert not np.may_share_memory(data_of(A), data_of(B))

    def test_no_aliasing_1(self):
        # B is a shared variable, A is updated with B's contents
        # since B is being updated as well, we don't need to copy anything
        # to avoid aliasing shared variables.
        A = self.shared(np.zeros((2, 2)) + .5)
        B = self.shared(np.zeros((2, 2)) - .5)
        C = tensor.dmatrix()
        f = pfunc([C], [], updates=[(A, B), (B, C)])
        z = np.zeros((2, 2))
        f(z)
        assert not np.may_share_memory(data_of(A), data_of(B))
        # Theano tries to maintain its own memory space.
        assert not np.may_share_memory(z, data_of(B))
        assert np.all(data_of(B) == z)

    def test_no_aliasing_2(self):
        # B and A take one another's values
        # no copying is necessary since each one is updated.
        orig_a = np.zeros((2, 2)) + .5
        orig_b = np.zeros((2, 2)) - .5
        A = self.shared(orig_a)
        B = self.shared(orig_b)

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = pfunc([], [], updates=[(A, B), (B, A)])
        f()
        # correctness
        assert np.all(data_of(A) == -.5)
        assert np.all(data_of(B) == +.5)

        # shared vars may not be aliased
        assert not np.may_share_memory(data_of(A), data_of(B))

        # theano should have been smart enough to not make copies
        assert np.may_share_memory(data_of(A), data_of_b)
        assert np.may_share_memory(data_of(B), data_of_a)

    def test_no_aliasing_2b(self):
        # B and A take one another's values
        # no copying is necessary since each one is updated.
        # The twist one `test_no_aliasing_2` is that each shared var is updated
        # with a view of the other one.

        orig_a = np.zeros((2, 2)) + .5
        orig_b = np.zeros((2, 2)) - .5
        A = self.shared(orig_a)
        B = self.shared(orig_b)

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = pfunc([], [], updates=[(A, B[:, ::-1]), (B, A.T)])
        # theano.printing.debugprint(f)
        f()
        # correctness (doesn't actually test the view...)
        assert np.all(data_of(A) == -.5)
        assert np.all(data_of(B) == +.5)

        # shared vars may not be aliased
        assert not np.may_share_memory(data_of(A), data_of(B))

        # theano should have been smart enough to not make copies
        if theano.config.mode not in [
                'DebugMode', 'DEBUG_MODE', 'FAST_COMPILE']:
            # We don't ask DebugMode and FAST_COMPILE not to make copy.
            # We have the right to do so.
            assert np.all(data_of(A) < 5)
            data_of_b += 10
            assert np.all(data_of(A) > 5)
            data_of_b -= 10

            assert np.all(data_of(B) < 5)
            data_of_a += 10
            assert np.all(data_of(B) > 5)
            data_of_a -= 10

            # N.B. may_share_memory is what we mean, but does it work?
            assert np.may_share_memory(data_of(A), data_of_b)
            assert np.may_share_memory(data_of(B), data_of_a)

            # N.B. This pattern could form a memory leak - each shared
            # variable always points to a view, and that view gets
            # further and further from the (e.g. data_of_a) with each
            # call.  The memory leak is in the increasing number of view
            # objects forming a chain to the underlying data.


class Test_rebuild_strict(unittest.TestCase):
    def test1(self):
        # Test fix for error reported at
        # https://groups.google.com/d/topic/theano-users/BRK0UEB72XA/discussion
        w = tensor.imatrix()
        x, y = tensor.ivectors('x', 'y')
        z = x * y
        f = theano.function([w, y], z, givens=[(x, w)], rebuild_strict=False)
        z_val = f(np.ones((3, 5), dtype='int32'), np.arange(5, dtype='int32'))
        assert z_val.ndim == 2
        assert np.all(z_val == np.ones((3, 5)) * np.arange(5))


if __name__ == '__main__':
    theano.config.mode = 'FAST_COMPILE'
    Test_pfunc().test_default_scalar_container()
