__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy 

from theano.tensor.randomstreams import RandomStreams, raw_random
from theano.compile import Module, Method, Member
from theano.tests import unittest_tools

from theano import tensor
from theano import compile, gof


class T_RandomStreams(unittest.TestCase):
    def test_basics(self):
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.uniform((2,2)))
        m.gn = Method([], m.random.normal((2,2)))
        made = m.make()
        made.random.initialize()

        fn_val0 = made.fn()
        fn_val1 = made.fn()

        gn_val0 = made.gn()

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_seed_in_initialize(self):
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.uniform((2,2)))
        made = m.make()
        made.random.initialize(seed=888)

        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(888).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed))  #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_seed_fn(self):
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.uniform((2,2)))
        made = m.make()
        made.random.initialize(seed=789)

        made.random.seed(888)

        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(888).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed))  #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_getitem(self):

        m = Module()
        m.random = RandomStreams(234)
        out = m.random.uniform((2,2))
        m.fn = Method([], out)
        made = m.make()
        made.random.initialize(seed=789)

        made.random.seed(888)

        rng = numpy.random.RandomState()
        rng.set_state(made.random[out.rng].get_state())

        fn_val0 = made.fn()
        fn_val1 = made.fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_setitem(self):

        m = Module()
        m.random = RandomStreams(234)
        out = m.random.uniform((2,2))
        m.fn = Method([], out)
        made = m.make()
        made.random.initialize(seed=789)

        made.random.seed(888)

        rng = numpy.random.RandomState(823874)
        made.random[out.rng] = numpy.random.RandomState(823874)

        fn_val0 = made.fn()
        fn_val1 = made.fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_multiple(self):
        M = Module()
        M.random = RandomStreams(234)
        out = M.random.uniform((2,2))
        M.m2 = Module()
        M.m2.random = M.random
        out2 = M.m2.random.uniform((2,2))
        M.fn = Method([], out)
        M.m2.fn2 = Method([], out2)
        m = M.make()
        m.random.initialize()
        m.m2.initialize()

        assert m.random is m.m2.random

    def test_ndim(self):
        """Test that the behaviour of 'ndim' optional parameter"""
        # 'ndim' is an optional integer parameter, specifying the length
        # of the 'shape', placed as first argument.

        # ndim not specified, OK
        m1 = Module()
        m1.random = RandomStreams(234)
        m1.fn = Method([], m1.random.uniform((2,2)))
        made1 = m1.make()
        made1.random.initialize()

        # ndim specified, consistent with shape, OK
        m2 = Module()
        m2.random = RandomStreams(234)
        m2.fn = Method([], m2.random.uniform(2, (2,2)))
        made2 = m2.make()
        made2.random.initialize()

        val1 = made1.fn()
        val2 = made2.fn()
        assert numpy.all(val1 == val2)

        # ndim specified, inconsistent with shape, should raise ValueError
        m3 = Module()
        m3.random = RandomStreams(234)
        m3.fn = Method([], m3.random.uniform(1, (2,2)))
        made3 = m3.make()
        made3.random.initialize()
        self.assertRaises(ValueError, made3.fn)

    def test_uniform(self):
        """Test that RandomStreams.uniform generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.uniform((2,2), -1, 1))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()
        print fn_val0
        print fn_val1

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        numpy_val0 = rng.uniform(-1, 1, size=(2,2))
        numpy_val1 = rng.uniform(-1, 1, size=(2,2))
        print numpy_val0
        print numpy_val1

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_normal(self):
        """Test that RandomStreams.normal generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.normal((2,2), -1, 2))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.normal(-1, 2, size=(2,2))
        numpy_val1 = rng.normal(-1, 2, size=(2,2))

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_random_integers(self):
        """Test that RandomStreams.random_integers generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.random_integers((20,20), -5, 5))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.random_integers(-5, 5, size=(20,20))
        numpy_val1 = rng.random_integers(-5, 5, size=(20,20))

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_permutation(self):
        """Test that RandomStreams.uniform generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.permutation((20,), 10))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

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
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.multinomial((20,20), 1, [0.1]*10))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(234).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.multinomial(1, [0.1]*10, size=(20,20))
        numpy_val1 = rng.multinomial(1, [0.1]*10, size=(20,20))

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_shuffle_row_elements(self):
        """Test that RandomStreams.shuffle_row_elements generates the right results"""
        # Check over two calls to see if the random state is correctly updated.

        # On matrices, for each row, the elements of that row should be shuffled.
        # Note that this differs from numpy.random.shuffle, where all the elements
        # of the matrix are shuffled.
        mm = Module()
        mm.random = RandomStreams(234)
        m_input = tensor.dmatrix()
        mm.f = Method([m_input], mm.random.shuffle_row_elements(m_input))
        mmade = mm.make()
        mmade.random.initialize()

        val_rng = numpy.random.RandomState(unittest_tools.fetch_seed())
        in_mval = val_rng.uniform(-2, 2, size=(20,5))
        fn_mval0 = mmade.f(in_mval)
        fn_mval1 = mmade.f(in_mval)
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
        vm = Module()
        vm.random = RandomStreams(234)
        v_input = tensor.dvector()
        vm.f = Method([v_input], vm.random.shuffle_row_elements(v_input))
        vmade = vm.make()
        vmade.random.initialize()

        in_vval = val_rng.uniform(-3, 3, size=(12,))
        fn_vval = vmade.f(in_vval)
        numpy_vval = in_vval.copy()
        vrng = numpy.random.RandomState(int(rng_seed))
        vrng.shuffle(numpy_vval)
        print in_vval
        print fn_vval
        print numpy_vval
        assert numpy.all(numpy_vval == fn_vval)

        # Trying to shuffle a vector with function that should shuffle
        # matrices, or vice versa, raises a TypeError
        self.assertRaises(TypeError, vmade.f, in_mval)
        self.assertRaises(TypeError, mmade.f, in_vval)


if __name__ == '__main__':
    from theano.tests import main
    main("test_randomstreams")
