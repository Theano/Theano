__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy 

from theano.tensor import raw_random
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

from theano import tensor
from theano import compile, gof

from theano.tests import unittest_tools

class T_SharedRandomStreams(unittest.TestCase):

    def test_tutorial(self):
        srng = RandomStreams(seed=234)
        rv_u = srng.uniform((2,2))
        rv_n = srng.normal((2,2))
        f = function([], rv_u, updates=[rv_u.update])
        g = function([], rv_n)                              #omitting rv_n.update
        nearly_zeros = function([], rv_u + rv_u - 2 * rv_u, updates=[rv_u.update])

        assert numpy.all(f() != f())
        assert numpy.all(g() == g())
        assert numpy.all(abs(nearly_zeros()) < 1e-5)

        assert isinstance(rv_u.rng.value, numpy.random.RandomState)

    def test_basics(self):
        random = RandomStreams(234)
        fn = function([], random.uniform((2,2)), updates=random.updates())
        gn = function([], random.normal((2,2)), updates=random.updates())

        fn_val0 = fn()
        fn_val1 = fn()

        gn_val0 = gn()

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.all(fn_val0 == numpy_val0)
        print fn_val0
        print numpy_val0
        print fn_val1
        print numpy_val1
        assert numpy.all(fn_val1 == numpy_val1)

    def test_seed_fn(self):
        random = RandomStreams(234)
        fn = function([], random.uniform((2,2)), updates=random.updates())

        random.seed(888)

        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(888).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed))  #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_getitem(self):

        random = RandomStreams(234)
        out = random.uniform((2,2))
        fn = function([], out, updates=random.updates())

        random.seed(888)

        rng = numpy.random.RandomState()
        rng.set_state(random[out.rng].get_state()) #tests getitem

        fn_val0 = fn()
        fn_val1 = fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_setitem(self):

        random = RandomStreams(234)
        out = random.uniform((2,2))
        fn = function([], out, updates=random.updates())

        random.seed(888)

        rng = numpy.random.RandomState(823874)
        random[out.rng] = numpy.random.RandomState(823874)

        fn_val0 = fn()
        fn_val1 = fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_permutation(self):
        """Test that RandomStreams.uniform generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        random = RandomStreams(234)
        fn = function([], random.permutation((20,), 10), updates=random.updates())

        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        # rng.permutation outputs one vector at a time, so we iterate.
        numpy_val0 = numpy.asarray([rng.permutation(10) for i in range(20)])
        numpy_val1 = numpy.asarray([rng.permutation(10) for i in range(20)])

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_multinomial(self):
        """Test that RandomStreams.multinomial generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        random = RandomStreams(234)
        fn = function([], random.multinomial((4,4), 1, [0.1]*10), updates=random.updates())

        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.multinomial(1, [0.1]*10, size=(4,4))
        numpy_val1 = rng.multinomial(1, [0.1]*10, size=(4,4))

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_shuffle_row_elements(self):
        """Test that RandomStreams.shuffle_row_elements generates the right results"""
        # Check over two calls to see if the random state is correctly updated.

        # On matrices, for each row, the elements of that row should be shuffled.
        # Note that this differs from numpy.random.shuffle, where all the elements
        # of the matrix are shuffled.
        random = RandomStreams(234)
        m_input = tensor.dmatrix()
        f = function([m_input], random.shuffle_row_elements(m_input), updates=random.updates())

        val_rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        in_mval = val_rng.uniform(-2, 2, size=(20,5))
        fn_mval0 = f(in_mval)
        fn_mval1 = f(in_mval)
        print in_mval[0]
        print fn_mval0[0]
        print fn_mval1[0]
        assert not numpy.all(in_mval == fn_mval0)
        assert not numpy.all(in_mval == fn_mval1)
        assert not numpy.all(fn_mval0 == fn_mval1)

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed))
        numpy_mval0 = in_mval.copy()
        numpy_mval1 = in_mval.copy()
        for row in numpy_mval0:
            rng.shuffle(row)
        for row in numpy_mval1:
            rng.shuffle(row)

        assert numpy.all(numpy_mval0 == fn_mval0)
        assert numpy.all(numpy_mval1 == fn_mval1)

        # On vectors, the behaviour is the same as numpy.random.shuffle,
        # except that it does not work in place, but returns a shuffled vector.
        random1 = RandomStreams(234)
        v_input = tensor.dvector()
        f1 = function([v_input], random1.shuffle_row_elements(v_input))

        in_vval = val_rng.uniform(-3, 3, size=(12,))
        fn_vval = f1(in_vval)
        numpy_vval = in_vval.copy()
        vrng = numpy.random.RandomState(int(rng_seed))
        vrng.shuffle(numpy_vval)
        print in_vval
        print fn_vval
        print numpy_vval
        assert numpy.all(numpy_vval == fn_vval)

        # Trying to shuffle a vector with function that should shuffle
        # matrices, or vice versa, raises a TypeError
        self.assertRaises(TypeError, f1, in_mval)
        self.assertRaises(TypeError, f, in_vval)

if __name__ == '__main__':
    from theano.tests import main
    main("test_randomstreams")
