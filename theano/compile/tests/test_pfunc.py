import numpy
import unittest
import copy
import theano
from theano.tensor import Tensor, dmatrix, dvector, lscalar, dmatrices
from theano import tensor

from theano.compile.sharedvalue import *
from theano.compile.pfunc import *

class Test_pfunc(unittest.TestCase):

    def test_doc(self):
        """Ensure the code given in pfunc.txt works as expected"""

        # Example #1.
        a = lscalar()
        b = shared(1)
        f1 = pfunc([a], a+b)
        f2 = pfunc([Param(a, default=44)], a + b, updates={b: b + 1})
        self.failUnless(b.value == 1)
        self.failUnless(f1(3) == 4)
        self.failUnless(f2(3) == 4)
        self.failUnless(b.value == 2)
        self.failUnless(f1(3) == 5)
        b.value = 0
        self.failUnless(f1(3) == 3)

        # Example #2.
        a = tensor.lscalar()
        b = shared(7)
        f1 = pfunc([a], a + b)
        f2 = pfunc([a], a * b)
        self.failUnless(f1(5) == 12)
        b.value = 8
        self.failUnless(f1(5) == 13)
        self.failUnless(f2(4) == 32)

    def test_shared(self):

        # CHECK: two functions (f1 and f2) can share w
        w = shared(numpy.random.rand(2,2), 'w')
        wval = copy.copy(w.value)

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

        w.value *= 10
        assert numpy.all(f1(xval) == xval + w.value) # this same value is read by f1

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
        w.value += 1.0
        assert f() == numpy.sum((w_init+1)**2)

    def test_default_scalar_container(self):
        # Similar in spirit to test_default_container, but updating a scalar
        # variable. This is a sanity check for non mutable types.
        x = shared(0.0, 'x')
        f = pfunc([], x)
        assert f() == 0
        x.value += 1
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
        assert b.value is bval
        b_out = b * 2

        # by default, shared are not mutable unless doing an explicit update
        f = pfunc([], [b_out], mode='FAST_RUN')
        assert (f() ==  numpy.arange(5) * 2).all()
        assert numpy.all( b.value == numpy.arange(5))

        # using updates, b is now a mutable parameter
        f = pfunc([], [b_out], updates=[(b, b_out)], mode='FAST_RUN')
        assert (f() == numpy.arange(5)*2 ).all()
        assert ( b.value == numpy.arange(5)*2).all() # because of the update
        assert ( bval == numpy.arange(5)*2).all() # because of mutable=True

        # do not depend on updates being in-place though!
        bval = numpy.arange(5)
        b.value = bval
        f = pfunc([], [b_out], updates=[(b, b_out+3)], mode='FAST_RUN')
        assert ( f() == numpy.arange(5)*2 ).all()
        assert (b.value == ((numpy.arange(5)*2)+3)).all() # because of the update
        # bval got modified to something...
        assert not (bval == numpy.arange(5)).all()
        # ... but not to b.value !
        assert not (bval == b.value).all()

    def test_update(self):
        """Test update mechanism in different settings."""

        # Simple value assignment.
        x = shared(0)
        assign = pfunc([], [], updates = {x: 3})
        assign()
        self.failUnless(x.value == 3)

        # Same but using a mutable constant to show how it can be used to
        # modify the update value after the function is created.
        x.value = 0
        y = numpy.ones((), dtype='int64')
        assign_mutable = pfunc([], [], updates = {x: y})
        assign_mutable()
        self.failUnless(x.value == 1)
        y.fill(4)
        assign_mutable()
        self.failUnless(x.value == 4)

        # Basic increment function.
        x.value = 0
        inc = pfunc([], [], updates = {x: x + 1})
        inc()
        self.failUnless(x.value == 1)

        # Increment by a constant value.
        x.value = -1
        y = shared(2)
        inc_by_y = pfunc([], [], updates = {x: x + y})
        inc_by_y()
        self.failUnless(x.value == 1)

    def test_duplicate_updates(self):
        x, y = dmatrices('x', 'y')
        z = shared(numpy.ones((2,3)))
        self.failUnlessRaises(ValueError, theano.function, [x,y], [z], updates=[(z, z+x+y), (z, z-x)])

    def test_givens(self):
        x = shared(0)
        assign = pfunc([], x, givens = {x: 3})
        assert assign() == 3
        assert x.value == 0

        y = tensor.ivector()
        f = pfunc([y], y*x, givens = {x : 6})
        assert numpy.all(f([1,1,1]) == [6,6,6])
        assert x.value == 0

        z = tensor.ivector()
        c = z*y
        f = pfunc([y], c+7, givens = {z : theano._asarray([4,4,4], dtype='int32')})
        assert numpy.all(f([1,1,1]) == [11,11,11])
        assert x.value == 0

    def test_clone0(self):
        x = shared(numpy.asarray([4,4,4]))
        y = shared(numpy.asarray([4,4,4]))
        z = shared(numpy.asarray([2,2,2]))
        up = pfunc([], [], updates = {x: (x*5), y:(x*5)+y, z: ((x*5)+y)**z})

        up()
        print x.value
        assert numpy.all(x.value==20)
        assert numpy.all(y.value==24)
        assert numpy.all(z.value==24**2)

    def test_default_updates(self):
        x = shared(0)
        x.default_update = x+1

        f = pfunc([], [x])
        f()
        print x.value
        assert x.value == 1

        del x.default_update
        f()
        assert x.value == 2

        g = pfunc([], [x])
        g()
        assert x.value == 2

    def test_no_default_updates(self):
        x = shared(0)
        y = shared(1)
        x.default_update = x+2

        # Test that the default update is taken into account in the right cases
        f1 = pfunc([], [x], no_default_updates=True)
        f1()
        print x.value
        assert x.value == 0

        f2 = pfunc([], [x], no_default_updates=[x])
        f2()
        print x.value
        assert x.value == 0

        f3 = pfunc([], [x], no_default_updates=[x, y])
        f3()
        print x.value
        assert x.value == 0

        f4 = pfunc([], [x], no_default_updates=[y])
        f4()
        print x.value
        assert x.value == 2

        f5 = pfunc([], [x], no_default_updates=[])
        f5()
        print x.value
        assert x.value == 4

        f5 = pfunc([], [x], no_default_updates=False)
        f5()
        print x.value
        assert x.value == 6

        self.failUnlessRaises(TypeError, pfunc, [], [x], no_default_updates=(x))
        self.failUnlessRaises(TypeError, pfunc, [], [x], no_default_updates=x)
        self.failUnlessRaises(TypeError, pfunc, [], [x], no_default_updates='canard')

        # Mix explicit updates and no_default_updates
        g1 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=True)
        g1()
        print x.value
        assert x.value == 5

        g2 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[x])
        g2()
        print x.value
        assert x.value == 4

        g3 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[x, y])
        g3()
        print x.value
        assert x.value == 3

        g4 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[y])
        g4()
        print x.value
        assert x.value == 2

        g5 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=[])
        g5()
        print x.value
        assert x.value == 1

        g5 = pfunc([], [x], updates=[(x,x-1)], no_default_updates=False)
        g5()
        print x.value
        assert x.value == 0

    def test_default_updates_expressions(self):
        x = shared(0)
        y = shared(1)
        a = lscalar('a')

        z = a*x
        x.default_update = x+y

        f1 = pfunc([a], z)
        f1(12)
        print x
        assert x.value == 1

        f2 = pfunc([a], z, no_default_updates=True)
        assert f2(7) == 7
        print x
        assert x.value == 1

        f3 = pfunc([a], z, no_default_updates=[x])
        assert f3(9) == 9
        print x
        assert x.value == 1

    def test_default_updates_multiple(self):
        x = shared(0)
        y = shared(1)

        x.default_update = x-1
        y.default_update = y+1

        f1 = pfunc([], [x,y])
        f1()
        assert x.value == -1
        assert y.value == 2

        f2 = pfunc([], [x,y], updates=[(x, x-2)], no_default_updates=[y])
        f2()
        assert x.value == -3
        assert y.value == 2

        f3 = pfunc([], [x,y], updates=[(x, x-2)], no_default_updates=True)
        f3()
        assert x.value == -5
        assert y.value == 2

        f4 = pfunc([], [x,y], updates=[(y, y-2)])
        f4()
        assert x.value == -6
        assert y.value == 0

    def test_default_updates_chained(self):
        x = shared(2)
        y = shared(1)
        z = shared(-1)

        x.default_update = x-y
        y.default_update = z
        z.default_update = z-1

        f1 = pfunc([], [x])
        f1()
        print x.value, y.value, z.value
        assert x.value == 1
        assert y.value == -1
        assert z.value == -2

        f2 = pfunc([], [x, y])
        f2()
        assert x.value == 2
        assert y.value == -2
        assert z.value == -3

        f3 = pfunc([], [y])
        f3()
        assert x.value == 2
        assert y.value == -3
        assert z.value == -4

        f4 = pfunc([], [x,y], no_default_updates=[x])
        f4()
        assert x.value == 2
        assert y.value == -4
        assert z.value == -5

        f5 = pfunc([], [x,y,z], no_default_updates=[z])
        f5()
        assert x.value == 6
        assert y.value == -5
        assert z.value == -5


    def test_default_updates_input(self):
        x = shared(0)
        y = shared(1)
        a = lscalar('a')

        x.default_update = y
        y.default_update = y+a

        f1 = pfunc([], x, no_default_updates=True)
        f1()
        assert x.value == 0
        assert y.value == 1

        f2 = pfunc([], x, no_default_updates=[x])
        f2()
        assert x.value == 0
        assert y.value == 1

        f3 = pfunc([], x, no_default_updates=[y])
        f3()
        assert x.value == 1
        assert y.value == 1

        f4 = pfunc([a], x)
        f4(2)
        assert x.value == 1
        assert y.value == 3

        f5 = pfunc([], x, updates={y:y-1})
        f5()
        assert x.value == 3
        assert y.value == 2

        # a is needed as input if y.default_update is used
        self.failUnlessRaises(TypeError, pfunc, [], x)

    def test_default_updates_partial_graph(self):
        a = shared(0)
        a.default_update = a+1 # Increment a each time it is used
        b = 2*a
        # Use only the tip of the graph, a is not used
        f = pfunc([b], b)
        print 'a.value =', a.value
        assert a.value == 0
        f(21)
        print 'a.value =', a.value
        assert a.value == 0


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






def data_of(s):
    """Return the raw value of a shared variable"""
    return s.container.storage[0]

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
        return tensor.shared(x)

    def test_shared_constructor_copies(self):
        # shared constructor makes copy
        # (rule #2)
        orig_a = numpy.zeros((2,2))
        A = self.shared(orig_a)
        assert not numpy.may_share_memory(orig_a, data_of(A))

        # rule #2 reading back from theano-managed memory
        assert not numpy.may_share_memory(A.value, data_of(A))
        

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

