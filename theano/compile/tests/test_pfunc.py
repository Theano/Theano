import copy
import unittest

from nose.plugins.skip import SkipTest
import numpy

import theano
from theano.tensor import dmatrix, iscalar, lscalar, dmatrices
from theano import tensor

from theano.compile.sharedvalue import *
from theano.compile.pfunc import *


def data_of(s):
    """Return the raw value of a shared variable"""
    return s.container.storage[0]


class Test_pfunc(unittest.TestCase):

    def test_doc(self):
        """Ensure the code given in pfunc.txt works as expected"""

        # Example #1.
        a = lscalar()
        b = shared(1)
        f1 = pfunc([a], a+b)
        f2 = pfunc([Param(a, default=44)], a + b, updates={b: b + 1})
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
        w = shared(numpy.random.rand(2,2), 'w')
        wval = w.get_value(borrow=False)

        x = dmatrix()
        out1 = w + x
        out2 = w * x
        f1 = pfunc([x],[out1])
        f2 = pfunc([x],[out2])
        xval = numpy.random.rand(2,2)
        assert numpy.all(f1(xval) == xval + wval)
        assert numpy.all(f2(xval) == xval * wval)

        # CHECK: updating a shared value
        f3 = pfunc([x], out1, updates=[(w, w-1)])
        assert numpy.all(f3(xval) == xval + wval) # f3 changes the value of w
        assert numpy.all(f1(xval) == xval + (wval-1)) # this same value is read by f1

        w.set_value(w.get_value(borrow=True) * 10, borrow=True)
        # this same value is read by f1
        assert numpy.all(f1(xval) == xval + w.get_value(borrow=True))

    def test_no_shared_as_input(self):
        """Test that shared variables cannot be used as function inputs."""
        w_init = numpy.random.rand(2,2)
        w = shared(w_init.copy(), 'w')
        try:
            f = pfunc([w], theano.tensor.sum(w * w))
            assert False
        except TypeError, e:
            msg = 'Cannot use a shared variable (w) as explicit input'
            if str(e).find(msg) < 0:
                raise

    def test_default_container(self):
        # Ensure it is possible to (implicitly) use a shared variable in a
        # function, as a 'state' that can be updated at will.

        rng = numpy.random.RandomState(1827)
        w_init = rng.rand(5)
        w = shared(w_init.copy(), 'w')
        reg = theano.tensor.sum(w*w)
        f = pfunc([], reg)

        assert f() == numpy.sum(w_init * w_init)
        # Change the value of w and ensure the output changes accordingly.
        w.set_value(w.get_value(borrow=True) + 1.0, borrow=True)
        assert f() == numpy.sum((w_init+1)**2)

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

        f = pfunc([Param(a, strict=False)], [out])
        f(numpy.random.rand(8)) # works, rand generates float64 by default
        f(numpy.array([1,2,3,4], dtype='int32')) # works, casting is allowed

        f = pfunc([Param(a, strict=True)], [out])
        try:
            f(numpy.array([1,2,3,4], dtype='int32')) # fails, f expects float64
        except TypeError:
            pass

    def test_param_mutable(self):
        a = tensor.dvector()
        b = shared(7)
        out = a + b

        a_out = a * 2 # assuming the op which makes this "in place" triggers

        # using mutable=True will let fip change the value in aval
        fip = pfunc([Param(a, mutable=True)], [a_out], mode='FAST_RUN')
        aval = numpy.random.rand(10)
        aval2 = aval.copy()
        assert numpy.all( fip(aval) == aval2*2 )
        assert not numpy.all( aval == aval2 )

        # using mutable=False should leave the input untouched
        f = pfunc([Param(a, mutable=False)], [a_out], mode='FAST_RUN')
        aval = numpy.random.rand(10)
        aval2 = aval.copy()
        assert numpy.all( f(aval) == aval2*2 )
        assert numpy.all( aval == aval2 )

    def test_shared_mutable(self):
        bval = numpy.arange(5)
        b = shared(bval)
        b_out = b * 2

        assert b.get_value(borrow=True) is not bval # shared vars copy args.
        bval = data_of(b)          # so we do this to get at the underlying data

        # by default, shared are not mutable unless doing an explicit update
        f = pfunc([], [b_out], mode='FAST_RUN')
        assert (f() ==  numpy.arange(5) * 2).all()
        assert numpy.all(b.get_value(borrow=True) == numpy.arange(5))

        # using updates, b is now a mutable parameter
        f = pfunc([], [b_out], updates=[(b, b_out)], mode='FAST_RUN')
        assert (f() == numpy.arange(5)*2 ).all()
        # because of the update
        assert (b.get_value(borrow=True) == numpy.arange(5)*2).all()
        assert (bval == numpy.arange(5)*2).all() # because of mutable=True

        # do not depend on updates being in-place though!
        bval = numpy.arange(5)
        b.set_value(bval, borrow=True)
        bval = data_of(b)
        f = pfunc([], [b_out], updates=[(b, b_out+3)], mode='FAST_RUN')
        assert (f() == numpy.arange(5)*2 ).all()
        # because of the update
        assert (b.get_value(borrow=True) == ((numpy.arange(5)*2)+3)).all()
        # bval got modified to something...
        assert not (bval == numpy.arange(5)).all()
        # ... but not to b.value !
        assert not (bval == b.get_value(borrow=True)).all()

    def test_param_allow_downcast_int(self):
        a = tensor.wvector('a') # int16
        b = tensor.bvector('b') # int8
        c = tensor.bscalar('c') # int8
        f = pfunc([Param(a, allow_downcast=True),
                   Param(b, allow_downcast=False),
                   Param(c, allow_downcast=None)],
                  a+b+c)

        # Both values are in range. Since they're not ndarrays (but lists),
        # they will be converted, and their value checked.
        assert numpy.all(f([3], [6], 1) == 10)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        self.assertRaises(TypeError, f,
                [3], numpy.array([6], dtype='int16'), 1)

        # Value too big for a, silently ignored
        assert numpy.all(f([2**20], numpy.ones(1, dtype='int8'), 1) == 2)

        # Value too big for b, raises TypeError
        self.assertRaises(TypeError, f, [3], [312], 1)

        # Value too big for c, raises TypeError
        self.assertRaises(TypeError, f, [3], [6], 806)

    def test_param_allow_downcast_floatX(self):
        a = tensor.fscalar('a')
        b = tensor.fscalar('b')
        c = tensor.fscalar('c')

        f = pfunc([Param(a, allow_downcast=True),
                   Param(b, allow_downcast=False),
                   Param(c, allow_downcast=None)],
                  a+b+c)

        # If the values can be accurately represented, everything is OK
        assert numpy.all(f(0, 0, 0) == 0)

        # If allow_downcast is True, idem
        assert numpy.allclose(f(0.1, 0, 0), 0.1)

        # If allow_downcast is False, nope
        self.assertRaises(TypeError, f, 0, 0.1, 0)

        # If allow_downcast is None, it should work iff floatX=float32
        if config.floatX == 'float32':
            assert numpy.allclose(f(0, 0, 0.1), 0.1)
        else:
            self.assertRaises(TypeError, f, 0, 0, 0.1)

    def test_param_allow_downcast_vector_floatX(self):
        a = tensor.fvector('a')
        b = tensor.fvector('b')
        c = tensor.fvector('c')

        f = pfunc([Param(a, allow_downcast=True),
                   Param(b, allow_downcast=False),
                   Param(c, allow_downcast=None)],
                  a+b+c)

        # If the values can be accurately represented, everything is OK
        z = [0]
        assert numpy.all(f(z, z, z) == 0)

        # If allow_downcast is True, idem
        assert numpy.allclose(f([0.1], z, z), 0.1)

        # If allow_downcast is False, nope
        self.assertRaises(TypeError, f, z, [0.1], z)

        # If allow_downcast is None, like False
        self.assertRaises(TypeError, f, z, z, [0.1])

    def test_allow_input_downcast_int(self):
        a = tensor.wvector('a') # int16
        b = tensor.bvector('b') # int8
        c = tensor.bscalar('c') # int8

        f = pfunc([a, b, c], a+b+c, allow_input_downcast=True)
        # Value too big for a, b, or c, silently ignored
        assert f([2**20], [1], 0) == 1
        assert f([3], [312], 0) == 59
        assert f([3], [1], 806) == 42

        g = pfunc([a, b, c], a+b+c, allow_input_downcast=False)
        # All values are in range. Since they're not ndarrays (but lists
        # or scalars), they will be converted, and their value checked.
        assert numpy.all(g([3], [6], 0) == 9)

        # Values are in range, but a dtype too large has explicitly been given
        # For performance reasons, no check of the data is explicitly performed
        # (It might be OK to change this in the future.)
        self.assertRaises(TypeError, g,
                [3], numpy.array([6], dtype='int16'), 0)

        # Value too big for b, raises TypeError
        self.assertRaises(TypeError, g, [3], [312], 0)

        h = pfunc([a, b, c], a+b+c) # Default: allow_input_downcast=None
        # Everything here should behave like with False
        assert numpy.all(h([3], [6], 0) == 9)
        self.assertRaises(TypeError, h,
                [3], numpy.array([6], dtype='int16'), 0)
        self.assertRaises(TypeError, h, [3], [312], 0)

    def test_allow_downcast_floatX(self):
        a = tensor.fscalar('a')
        b = tensor.fvector('b')

        f = pfunc([a, b], a+b, allow_input_downcast=True)
        g = pfunc([a, b], a+b, allow_input_downcast=False)
        h = pfunc([a, b], a+b, allow_input_downcast=None)

        # If the values can be accurately represented, OK
        assert numpy.all(f(0, [0]) == 0)
        assert numpy.all(g(0, [0]) == 0)
        assert numpy.all(h(0, [0]) == 0)

        # For the vector: OK iff allow_input_downcast is True
        assert numpy.allclose(f(0, [0.1]), 0.1)
        self.assertRaises(TypeError, g, 0, [0.1])
        self.assertRaises(TypeError, h, 0, [0.1])

        # For the scalar: OK if allow_input_downcast is True,
        # or None and floatX==float32
        assert numpy.allclose(f(0.1, [0]), 0.1)
        self.assertRaises(TypeError, g, 0.1, [0])
        if config.floatX == 'float32':
            assert numpy.allclose(h(0.1, [0]), 0.1)
        else:
            self.assertRaises(TypeError, h, 0.1, [0])

    def test_update(self):
        """Test update mechanism in different settings."""

        # Simple value assignment.
        x = shared(0)
        assign = pfunc([], [], updates = {x: 3})
        assign()
        self.assertTrue(x.get_value() == 3)

        # Basic increment function.
        x.set_value(0)
        inc = pfunc([], [], updates = {x: x + 1})
        inc()
        self.assertTrue(x.get_value() == 1)

        # Increment by a constant value.
        x.set_value(-1)
        y = shared(2)
        inc_by_y = pfunc([], [], updates = {x: x + y})
        inc_by_y()
        self.assertTrue(x.get_value() == 1)

    def test_duplicate_updates(self):
        x, y = dmatrices('x', 'y')
        z = shared(numpy.ones((2,3)))
        self.assertRaises(ValueError, theano.function, [x,y], [z], updates=[(z, z+x+y), (z, z-x)])

    def test_givens(self):
        x = shared(0)
        assign = pfunc([], x, givens = {x: 3})
        assert assign() == 3
        assert x.get_value(borrow=True) == 0

        y = tensor.ivector()
        f = pfunc([y], y*x, givens = {x : 6})
        assert numpy.all(f([1,1,1]) == [6,6,6])
        assert x.get_value() == 0

        z = tensor.ivector()
        c = z*y
        f = pfunc([y], c+7, givens = {z : theano._asarray([4,4,4], dtype='int32')})
        assert numpy.all(f([1,1,1]) == [11,11,11])
        assert x.get_value() == 0

    def test_clone0(self):
        x = shared(numpy.asarray([4,4,4]))
        y = shared(numpy.asarray([4,4,4]))
        z = shared(numpy.asarray([2,2,2]))
        up = pfunc([], [], updates = {x: (x*5), y:(x*5)+y, z: ((x*5)+y)**z})

        up()
        print x.get_value(borrow=True)
        assert numpy.all(x.get_value()==20)
        assert numpy.all(y.get_value()==24)
        assert numpy.all(z.get_value()==24**2)

    def test_default_updates(self):
        x = shared(0)
        x.default_update = x+1

        f = pfunc([], [x])
        f()
        print x.get_value()
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
        x.default_update = x+2

        # Test that the default update is taken into account in the right cases
        f1 = pfunc([], [x], no_default_updates=True)
        f1()
        print x.get_value()
        assert x.get_value() == 0

        f2 = pfunc([], [x], no_default_updates=[x])
        f2()
        print x.get_value()
        assert x.get_value() == 0

        f3 = pfunc([], [x], no_default_updates=[x, y])
        f3()
        print x.get_value()
        assert x.get_value() == 0

        f4 = pfunc([], [x], no_default_updates=[y])
        f4()
        print x.get_value()
        assert x.get_value() == 2

        f5 = pfunc([], [x], no_default_updates=[])
        f5()
        print x.get_value()
        assert x.get_value() == 4

        f5 = pfunc([], [x], no_default_updates=False)
        f5()
        print x.get_value()
        assert x.get_value() == 6

        self.assertRaises(TypeError, pfunc, [], [x], no_default_updates=(x))
        self.assertRaises(TypeError, pfunc, [], [x], no_default_updates=x)
        self.assertRaises(TypeError, pfunc, [], [x], no_default_updates='canard')

        # Mix explicit updates and no_default_updates
        g1 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=True)
        g1()
        print x.get_value()
        assert x.get_value() == 5

        g2 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[x])
        g2()
        print x.get_value()
        assert x.get_value() == 4

        g3 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[x, y])
        g3()
        print x.get_value()
        assert x.get_value() == 3

        g4 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[y])
        g4()
        print x.get_value()
        assert x.get_value() == 2

        g5 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[])
        g5()
        print x.get_value()
        assert x.get_value() == 1

        g5 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=False)
        g5()
        print x.get_value()
        assert x.get_value() == 0

    def test_default_updates_expressions(self):
        x = shared(0)
        y = shared(1)
        a = lscalar('a')

        z = a*x
        x.default_update = x+y

        f1 = pfunc([a], z)
        f1(12)
        print x
        assert x.get_value() == 1

        f2 = pfunc([a], z, no_default_updates=True)
        assert f2(7) == 7
        print x
        assert x.get_value() == 1

        f3 = pfunc([a], z, no_default_updates=[x])
        assert f3(9) == 9
        print x
        assert x.get_value() == 1

    def test_default_updates_multiple(self):
        x = shared(0)
        y = shared(1)

        x.default_update = x-1
        y.default_update = y+1

        f1 = pfunc([], [x,y])
        f1()
        assert x.get_value() == -1
        assert y.get_value() == 2

        f2 = pfunc([], [x,y], updates=[(x, x-2)], no_default_updates=[y])
        f2()
        assert x.get_value() == -3
        assert y.get_value() == 2

        f3 = pfunc([], [x,y], updates=[(x, x-2)], no_default_updates=True)
        f3()
        assert x.get_value() == -5
        assert y.get_value() == 2

        f4 = pfunc([], [x,y], updates=[(y, y-2)])
        f4()
        assert x.get_value() == -6
        assert y.get_value() == 0

    def test_default_updates_chained(self):
        x = shared(2)
        y = shared(1)
        z = shared(-1)

        x.default_update = x-y
        y.default_update = z
        z.default_update = z-1

        f1 = pfunc([], [x])
        f1()
        print x.get_value(), y.get_value(), z.get_value()
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

        f4 = pfunc([], [x,y], no_default_updates=[x])
        f4()
        assert x.get_value() == 2
        assert y.get_value() == -4
        assert z.get_value() == -5

        f5 = pfunc([], [x,y,z], no_default_updates=[z])
        f5()
        assert x.get_value() == 6
        assert y.get_value() == -5
        assert z.get_value() == -5


    def test_default_updates_input(self):
        x = shared(0)
        y = shared(1)
        if theano.gof.cmodule.local_bitwidth()==32:
            a = iscalar('a')
        else:
            a = lscalar('a')

        x.default_update = y
        y.default_update = y+a

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

        f5 = pfunc([], x, updates={y:y-1})
        f5()
        assert x.get_value() == 3
        assert y.get_value() == 2

        # a is needed as input if y.default_update is used
        self.assertRaises(TypeError, pfunc, [], x)

    def test_default_updates_partial_graph(self):
        a = shared(0)
        a.default_update = a+1 # Increment a each time it is used
        b = 2*a
        # Use only the tip of the graph, a is not used
        f = pfunc([b], b)
        print 'a.get_value() =', a.get_value()
        assert a.get_value() == 0
        f(21)
        print 'a.get_value() =', a.get_value()
        assert a.get_value() == 0


    def test_givens_replaces_shared_variable(self):
        a = shared(1.,'a')
        a.default_update = a+3.
        b = tensor.dscalar('b')
        c = a + 10
        f = pfunc([b],c, givens = {a:b})

        assert len(f.maker.env.inputs) == 1
        assert len(f.maker.env.outputs) == 1

    def test_givens_replaces_shared_variable2(self):
        a = shared(1.,'a')
        a.default_update = a+3
        c = a+ 10
        f = pfunc([],c, givens = { a: a+10} )

        assert f() == 21
        assert f() == 34






class Test_aliasing_rules(unittest.TestCase):
    """
    1. Theano manages its own memory space, which typically does not overlap with the memory of
    normal python variables that the user uses.

    2. shared variables are allocated in this memory space, as are the temporaries used for
    Function evalution.

    3. Physically, this managed memory space may be spread across the host, on a GPU device(s),
    or even on a remote machine.

    4. Theano assumes that shared variables are never aliased to one another, and tries to make
    it impossible to accidentally alias them.

    5. Theano's managed data is constant while Theano Functions are not running and theano
    library code is not running.

    6. The default behaviour of Function is to return user-space values for outputs, but this
    can be overridden (borrow=True) for better performance, in which case the returned value
    may be aliased to managed memory, and potentially invalidated by the next Theano Function
    call or call to theano library code.
    """

    def shared(self, x):
        return tensor._shared(x)

    def test_shared_constructor_copies(self):
        # shared constructor makes copy
        # (rule #2)
        orig_a = numpy.zeros((2,2))
        A = self.shared(orig_a)
        assert not numpy.may_share_memory(orig_a, data_of(A))

        # rule #2 reading back from theano-managed memory
        assert not numpy.may_share_memory(A.get_value(borrow=False), data_of(A))

    def test_sparse_input_aliasing_affecting_inplace_operations(self):
        ##
        ## Note this test will never fail because I am not aware of any
        ## inplace op on sparse variables
        try:
            import scipy.sparse as sp
        except ImportError:
            # The variable enable_sparse will be used to disable the test file.
            pass

        from theano.sparse import enable_sparse
        if enable_sparse == False:
            raise SkipTest('Optional package sparse disabled')

        from theano import sparse

        ## Note: to trigger this bug with theano rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable (so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op (here dot)

        x  = sparse.SparseType('csc', dtype = 'float64')()
        y  = sparse.SparseType('csc', dtype = 'float64')()
        f = theano.function( [theano.In(x,  mutable = True),
                              theano.In(y, mutable = True)],
                                (x+y)+(x+y))
        ## Test 1. If the same variable is given twice

        # Compute bogus values
        m = sp.csc_matrix(numpy.asarray([[1,0,0,0,0],
                           [0,1,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,1,0],
                           [0,0,0,0,1]], dtype = 'float64'))
        bogus_vals =  f(m,m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        m = sp.csc_matrix(numpy.asarray([[1,0,0,0,0],
                           [0,1,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,1,0],
                           [0,0,0,0,1]], dtype = 'float64'))
        m_copy = m.copy()
        vals =  f(m,m_copy)

        assert numpy.allclose(vals.todense(), bogus_vals.todense())

    def test_input_aliasing_affecting_inplace_operations(self):

        ## Note: to trigger this bug with theano rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable (so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op (here dot)
        x  = theano.tensor.dvector()
        y  = theano.tensor.dvector()
        m1 = theano.tensor.dmatrix()
        m2 = theano.tensor.dmatrix()
        f = theano.function( [theano.In(x,  mutable = True),
                              theano.In(y,  mutable = True),
                              theano.In(m1, mutable = True),
                              theano.In(m2, mutable = True)],
                            theano.dot(x*2,m1)+theano.dot(y*3,m2))
        ## Test 1. If the same variable is given twice

        # Compute bogus values
        v = numpy.asarray( [1,2,3,4,5], dtype = 'float64')
        m = numpy.asarray([[1,0,0,0,0],
                           [0,1,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,1,0],
                           [0,0,0,0,1]], dtype = 'float64')
        bogus_vals =  f(v,v,m,m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        v = numpy.asarray( [1,2,3,4,5], dtype = 'float64')
        m = numpy.asarray([[1,0,0,0,0],
                           [0,1,0,0,0],
                           [0,0,1,0,0],
                           [0,0,0,1,0],
                           [0,0,0,0,1]], dtype = 'float64')
        m_copy = m.copy()
        v_copy = v.copy()
        vals =  f(v,v_copy,m,m_copy)

        assert numpy.allclose(vals, bogus_vals)

    def test_partial_input_aliasing_affecting_inplace_operations(self):

        ## Note: to trigger this bug with theano rev 4586:2bc6fc7f218b,
        #        you need to make in inputs mutable ( so that inplace
        #        operations are used) and to break the elemwise composition
        #        with some non-elemwise op ( here dot )
        x  = theano.tensor.dvector()
        y  = theano.tensor.dvector()
        z  = theano.tensor.dvector()
        m1 = theano.tensor.dmatrix()
        m2 = theano.tensor.dmatrix()
        m3 = theano.tensor.dmatrix()

        ## Test 2. If variables only partial overlap
        #   more exactly we care about the case when we have a,b,c
        #   and a shares memory with b, b shares memory with c, but
        #   c does not share memory with a



        f = theano.function( [theano.In(x,  mutable = True),
                              theano.In(y,  mutable = True),
                              theano.In(z,  mutable = True),
                              theano.In(m1, mutable = True),
                              theano.In(m2, mutable = True),
                              theano.In(m3, mutable = True)],
                            theano.dot(x*2,m1)+theano.dot(y*3,m2)+theano.dot(z*4,m3))
        # Compute bogus values
        v = numpy.asarray( [1,2,3,4,5], dtype = 'float64')
        m = numpy.asarray([[1,0],
                           [0,1]], dtype = 'float64')
        bogus_vals =  f(v[:2],v[1:3],v[2:4],m,m,m)
        # Since we used inplace operation v and m may be corrupted
        # so we need to recreate them

        v = numpy.asarray( [1,2,3,4,5], dtype = 'float64')
        m = numpy.asarray([[1,0],
                           [0,1]], dtype = 'float64')
        m_copy1 = m.copy()
        v_copy1 = v.copy()
        m_copy2 = m.copy()
        v_copy2 = v.copy()
        vals =  f(v[:2],v_copy1[1:3],v_copy2[2:4],m,m_copy1, m_copy2)

        assert numpy.allclose(vals, bogus_vals)



    def test_potential_output_aliasing_induced_by_updates(self):

        A = self.shared(numpy.zeros((2,2)))
        B = self.shared(numpy.zeros((2,2)))
        C = numpy.zeros((2,2))
        D = tensor.dmatrix()
        DD = D + 5

        f = pfunc([D], [], updates=[ (A,D), (B,D) ])
        f(C)

        assert not numpy.may_share_memory(data_of(A),data_of(B))
        f = pfunc([D], [], updates=[ (A,D[:]), (B,D) ])
        f(C)
        assert not numpy.may_share_memory(data_of(A),data_of(B))
        f = pfunc([D], [], updates=[ (A,D+5), (B,D[:]) ])
        f(C)
        assert not numpy.may_share_memory(data_of(A),data_of(B))

        f = pfunc([D], [], updates=[ (A,D+5), (B,D) ])
        f(C)
        assert not numpy.may_share_memory(data_of(A),data_of(B))

        f = pfunc([D], DD, updates=[ (A,DD[:1]), (B,DD) ])
        R=f(C)
        assert not numpy.may_share_memory(data_of(A),data_of(B))
        assert not numpy.may_share_memory(R,data_of(B))
        assert not numpy.may_share_memory(R,data_of(A))

        f = pfunc([D], DD, updates=[ (A,DD[:1]), (B,DD[:1]*2) ])
        R=f(C)
        assert not numpy.may_share_memory(data_of(A),data_of(B))
        assert not numpy.may_share_memory(R,data_of(B))
        assert not numpy.may_share_memory(R,data_of(A))

        f = pfunc([D], DD*4, updates=[ (A,DD[:1]*3), (B,DD[:1]*2) ])
        R=f(C)
        assert not numpy.may_share_memory(data_of(A),data_of(B))
        assert not numpy.may_share_memory(R,data_of(B))
        assert not numpy.may_share_memory(R,data_of(A))

        f = pfunc([D], DD*4, updates=[ (A,DD[:1]*3), (B,DD[:1]*3) ])
        R=f(C)
        assert not numpy.may_share_memory(data_of(A),data_of(B))
        assert not numpy.may_share_memory(R,data_of(B))
        assert not numpy.may_share_memory(R,data_of(A))

    def test_no_aliasing_0(self):
        # B is a shared variable, A is updated with B's contents
        # we need A to be copied to avoid aliasing
        A = self.shared(numpy.zeros((2,2))+.5)
        B = self.shared(numpy.zeros((2,2))-.5)
        f = pfunc([], [], updates=[(A,B)])
        f()
        assert not numpy.may_share_memory(data_of(A), data_of(B))

    def test_no_aliasing_1(self):
        # B is a shared variable, A is updated with B's contents
        # since B is being updated as well, we don't need to copy anything to avoid aliasing
        # shared variables.
        A = self.shared(numpy.zeros((2,2))+.5)
        B = self.shared(numpy.zeros((2,2))-.5)
        C = tensor.dmatrix()
        f = pfunc([C], [], updates=[ (A,B), (B,C) ])
        z = numpy.zeros((2,2))
        f(z)
        assert not numpy.may_share_memory(data_of(A),data_of(B))
        assert not numpy.may_share_memory(z,data_of(B)) # Theano tries to maintain its own memory space.
        assert numpy.all(data_of(B) == z)

    def test_no_aliasing_2(self):
        # B and A take one another's values
        # no copying is necessary since each one is updated.
        orig_a = numpy.zeros((2,2))+.5
        orig_b = numpy.zeros((2,2))-.5
        A = self.shared(orig_a)
        B = self.shared(orig_b)
        C = tensor.dmatrix()

        z = numpy.zeros((2,2))

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = pfunc([C], [], updates=[(A,B),(B,A)])
        f(z)
        # correctness
        assert numpy.all(data_of(A) == -.5)
        assert numpy.all(data_of(B) == +.5)

        # shared vars may not be aliased
        assert not numpy.may_share_memory(data_of(A), data_of(B))

        # theano should have been smart enough to not make copies
        assert numpy.may_share_memory(data_of(A), data_of_b)
        assert numpy.may_share_memory(data_of(B), data_of_a)

    def test_no_aliasing_2b(self):
        # B and A take one another's values
        # no copying is necessary since each one is updated.
        # The twist one `test_no_aliasing_2` is that each shared var is updated with a view of
        # the other one.

        orig_a = numpy.zeros((2,2))+.5
        orig_b = numpy.zeros((2,2))-.5
        A = self.shared(orig_a)
        B = self.shared(orig_b)
        C = tensor.dmatrix()

        z = numpy.zeros((2,2))

        data_of_a = data_of(A)
        data_of_b = data_of(B)

        f = pfunc([C], [], updates=[(A,B[:,::-1]),(B,A.T)])
        theano.printing.debugprint(f)
        f(z)
        # correctness (doesn't actually test the view...)
        assert numpy.all(data_of(A) == -.5)
        assert numpy.all(data_of(B) == +.5)

        # shared vars may not be aliased
        assert not numpy.may_share_memory(data_of(A), data_of(B))

        # theano should have been smart enough to not make copies
        if theano.config.mode not in ['DebugMode', 'DEBUG_MODE', 'FAST_COMPILE']:
            #we don't ask DebugMode and FAST_COMPILE to don't make copy. We have the right to do so.
            assert numpy.all(data_of(A) < 5)
            data_of_b += 10
            assert numpy.all(data_of(A) > 5)
            data_of_b -= 10

            assert numpy.all(data_of(B) < 5)
            data_of_a += 10
            print data_of(B)
            assert numpy.all(data_of(B) > 5)
            data_of_a -= 10

            # N.B. may_share_memory is what we mean, but does it work?
            assert numpy.may_share_memory(data_of(A), data_of_b)
            assert numpy.may_share_memory(data_of(B), data_of_a)

            # N.B. This pattern could form a memory leak - each shared variable always points to a
            # view, and that view gets further and further from the (e.g. data_of_a) with each
            # call.  The memory leak is in the increasing number of view objects forming a chain to
            # the underlying data.



if __name__ == '__main__':
    theano.config.mode = 'FAST_COMPILE'
    Test_pfunc().test_default_scalar_container()
