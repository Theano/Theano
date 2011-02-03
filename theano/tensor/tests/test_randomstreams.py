__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy

from theano.tensor.randomstreams import RandomStreams, raw_random
from theano.compile import Module, Method, Member
from theano.tests import unittest_tools as utt

from theano import tensor
from theano import compile, config, gof


class T_RandomStreams(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_basics(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        m.fn = Method([], m.random.uniform((2,2)))
        m.gn = Method([], m.random.normal((2,2)))
        made = m.make()
        made.random.initialize()

        fn_val0 = made.fn()
        fn_val1 = made.fn()

        gn_val0 = made.gn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_seed_in_initialize(self):
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.uniform((2,2)))
        made = m.make()
        made.random.initialize(seed=utt.fetch_seed())

        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed))  #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_seed_fn(self):
        m = Module()
        m.random = RandomStreams(234)
        m.fn = Method([], m.random.uniform((2,2)))
        made = m.make()
        made.random.initialize(seed=789)

        made.random.seed(utt.fetch_seed())

        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed))  #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_getitem(self):

        m = Module()
        m.random = RandomStreams(234)
        out = m.random.uniform((2,2))
        m.fn = Method([], out)
        made = m.make()
        made.random.initialize(seed=789)

        made.random.seed(utt.fetch_seed())

        rng = numpy.random.RandomState()
        rng.set_state(made.random[out.rng].get_state())

        fn_val0 = made.fn()
        fn_val1 = made.fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_setitem(self):

        m = Module()
        m.random = RandomStreams(234)
        out = m.random.uniform((2,2))
        m.fn = Method([], out)
        made = m.make()

        #as a distraction, install various seeds
        made.random.initialize(seed=789)
        made.random.seed(888)

        # then replace the rng of the stream we care about via setitem
        realseed = utt.fetch_seed()
        rng = numpy.random.RandomState(realseed)
        made.random[out.rng] = numpy.random.RandomState(realseed)

        print made.fn()
        print rng.uniform(size=(2,2))

        fn_val0 = made.fn()
        fn_val1 = made.fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_multiple(self):
        M = Module()
        M.random = RandomStreams(utt.fetch_seed())
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
        # of the 'shape', passed as a keyword argument.

        # ndim not specified, OK
        m1 = Module()
        m1.random = RandomStreams(utt.fetch_seed())
        m1.fn = Method([], m1.random.uniform((2,2)))
        made1 = m1.make()
        made1.random.initialize()

        # ndim specified, consistent with shape, OK
        m2 = Module()
        m2.random = RandomStreams(utt.fetch_seed())
        m2.fn = Method([], m2.random.uniform((2,2), ndim=2))
        made2 = m2.make()
        made2.random.initialize()

        val1 = made1.fn()
        val2 = made2.fn()
        assert numpy.all(val1 == val2)

        # ndim specified, inconsistent with shape, should raise ValueError
        m3 = Module()
        m3.random = RandomStreams(utt.fetch_seed())
        self.assertRaises(ValueError, m3.random.uniform, (2,2), ndim=1)

    def test_uniform(self):
        """Test that RandomStreams.uniform generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        m.fn = Method([], m.random.uniform((2,2), -1, 1))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()
        print fn_val0
        print fn_val1

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        numpy_val0 = rng.uniform(-1, 1, size=(2,2))
        numpy_val1 = rng.uniform(-1, 1, size=(2,2))
        print numpy_val0
        print numpy_val1

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_normal(self):
        """Test that RandomStreams.normal generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        m.fn = Method([], m.random.normal((2,2), -1, 2))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.normal(-1, 2, size=(2,2))
        numpy_val1 = rng.normal(-1, 2, size=(2,2))

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_random_integers(self):
        """Test that RandomStreams.random_integers generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        m.fn = Method([], m.random.random_integers((20,20), -5, 5))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.random_integers(-5, 5, size=(20,20))
        numpy_val1 = rng.random_integers(-5, 5, size=(20,20))

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_permutation(self):
        """Test that RandomStreams.permutation generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        m.fn = Method([], m.random.permutation((20,), 10))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
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
        m.random = RandomStreams(utt.fetch_seed())
        m.fn = Method([], m.random.multinomial((20,20), 1, [0.1]*10))

        made = m.make()
        made.random.initialize()
        fn_val0 = made.fn()
        fn_val1 = made.fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.multinomial(1, [0.1]*10, size=(20,20))
        numpy_val1 = rng.multinomial(1, [0.1]*10, size=(20,20))

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_shuffle_row_elements(self):
        """Test that RandomStreams.shuffle_row_elements generates the right results"""
        # Check over two calls to see if the random state is correctly updated.
        # On matrices, for each row, the elements of that row should be
        # shuffled.
        # Note that this differs from numpy.random.shuffle, where all the
        # elements of the matrix are shuffled.
        mm = Module()
        mm.random = RandomStreams(utt.fetch_seed())
        m_input = tensor.dmatrix()
        mm.f = Method([m_input], mm.random.shuffle_row_elements(m_input))
        mmade = mm.make()
        mmade.random.initialize()

        # Generate the elements to be shuffled
        val_rng = numpy.random.RandomState(utt.fetch_seed()+42)
        in_mval = val_rng.uniform(-2, 2, size=(20,5))
        fn_mval0 = mmade.f(in_mval)
        fn_mval1 = mmade.f(in_mval)
        print in_mval[0]
        print fn_mval0[0]
        print fn_mval1[0]
        assert not numpy.all(in_mval == fn_mval0)
        assert not numpy.all(in_mval == fn_mval1)
        assert not numpy.all(fn_mval0 == fn_mval1)

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
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
        vm.random = RandomStreams(utt.fetch_seed())
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

    def test_symbolic_shape(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        shape = tensor.lvector()
        out = m.random.uniform(size=shape, ndim=2)
        m.f = Method([shape], out)
        made = m.make()
        made.random.initialize()

        assert made.f([2,3]).shape == (2,3)
        assert made.f([4,8]).shape == (4,8)

        self.assertRaises(ValueError, made.f, [4])
        self.assertRaises(ValueError, made.f, [4,3,4,5])

    def test_default_shape(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        m.f = Method([], m.random.uniform())
        m.g = Method([], m.random.multinomial())
        made = m.make()
        made.random.initialize()

        #seed_rng is generator for generating *seeds* for RandomStates
        seed_rng = numpy.random.RandomState(utt.fetch_seed())
        uniform_rng = numpy.random.RandomState(int(seed_rng.randint(2**30)))
        multinomial_rng = numpy.random.RandomState(int(seed_rng.randint(2**30)))

        val0 = made.f()
        val1 = made.f()
        numpy_val0 = uniform_rng.uniform()
        numpy_val1 = uniform_rng.uniform()
        assert numpy.allclose(val0, numpy_val0)
        assert numpy.allclose(val1, numpy_val1)

        for i in range(10): # every test has 50% chance of passing even with non-matching random states
            val2 = made.g()
            numpy_val2 = multinomial_rng.multinomial(n=1, pvals=[.5, .5])
            assert numpy.all(val2 == numpy_val2)

    def test_vector_arguments(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        low = tensor.vector()
        out = m.random.uniform(low=low, high=1)
        assert out.ndim == 1
        m.f = Method([low], out)

        high = tensor.vector()
        outb = m.random.uniform(low=low, high=high)
        assert outb.ndim == 1
        m.fb = Method([low, high], outb)

        size = tensor.lvector()
        outc = m.random.uniform(low=low, high=high, size=size, ndim=1)
        m.fc = Method([low, high, size], outc)

        made = m.make()
        made.random.initialize()

        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        low_val0 = numpy.asarray([-5, .5, 0, 1], dtype=config.floatX)
        low_val1 = numpy.asarray([.9], dtype=config.floatX)
        val0 = made.f(low_val0)
        val1 = made.f(low_val1)
        numpy_val0 = numpy_rng.uniform(low=low_val0, high=1)
        numpy_val1 = numpy_rng.uniform(low=low_val1, high=1)
        assert numpy.allclose(val0, numpy_val0)
        assert numpy.allclose(val1, numpy_val1)

        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val0b = made.fb([-4., -2], [-1, 0])
        val1b = made.fb([-4.], [-1])
        numpy_val0b = numpy_rng.uniform(low=[-4., -2], high=[-1, 0])
        numpy_val1b = numpy_rng.uniform(low=[-4.], high=[-1])
        assert numpy.allclose(val0b, numpy_val0b)
        assert numpy.allclose(val1b, numpy_val1b)
        self.assertRaises(ValueError, made.fb, [-4., -2], [-1, 0, 1])
        #TODO: do we want that?
        #self.assertRaises(ValueError, made.fb, [-4., -2], [-1])

        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val0c = made.fc([-4., -2], [-1, 0], [2])
        val1c = made.fc([-4.], [-1], [1])
        numpy_val0c = numpy_rng.uniform(low=[-4., -2], high=[-1, 0])
        numpy_val1c = numpy_rng.uniform(low=[-4.], high=[-1])
        assert numpy.allclose(val0c, numpy_val0c)
        assert numpy.allclose(val1c, numpy_val1c)
        self.assertRaises(ValueError, made.fc, [-4., -2], [-1, 0], [1])
        self.assertRaises(ValueError, made.fc, [-4., -2], [-1, 0], [1,2])
        self.assertRaises(ValueError, made.fc, [-4., -2], [-1, 0], [2,1])
        self.assertRaises(ValueError, made.fc, [-4., -2], [-1], [1])
        #TODO: do we want that?
        #self.assertRaises(ValueError, made.fc, [-4., -2], [-1], [2])

    def test_broadcast_arguments(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        low = tensor.vector()
        high = tensor.col()
        out = m.random.uniform(low=low, high=high)
        assert out.ndim == 2
        m.f = Method([low, high], out)
        made = m.make()
        made.random.initialize()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        numpy_rng = numpy.random.RandomState(int(rng_seed))
        low_vals = [
                numpy.asarray([-5, .5, 0, 1], dtype=config.floatX),
                numpy.asarray([.9], dtype=config.floatX),
                numpy.asarray([-5, .5, 0, 1], dtype=config.floatX) ]
        high_vals = [
                numpy.asarray([[1.]], dtype=config.floatX),
                numpy.asarray([[1.], [1.1], [1.5]], dtype=config.floatX),
                numpy.asarray([[1.], [1.1], [1.5]], dtype=config.floatX) ]

        val0 = made.f(low_vals[0], high_vals[0])
        val1 = made.f(low_vals[1], high_vals[1])
        val2 = made.f(low_vals[2], high_vals[2])

        numpy_val0 = numpy_rng.uniform(low=low_vals[0], high=high_vals[0])
        numpy_val1 = numpy_rng.uniform(low=low_vals[1], high=high_vals[1])
        numpy_val2 = numpy_rng.uniform(low=low_vals[2], high=high_vals[2])

        assert numpy.allclose(val0, numpy_val0)
        assert numpy.allclose(val1, numpy_val1)
        assert numpy.allclose(val2, numpy_val2)

    def test_uniform_vector(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        low = tensor.vector()
        high = tensor.vector()
        out = m.random.uniform(low=low, high=high)
        assert out.ndim == 1
        m.f = Method([low, high], out)
        # Specifying the size explicitly
        m.g = Method([low, high],
                m.random.uniform(low=low, high=high, size=(3,)))
        made = m.make()
        made.random.initialize()

        low_val = numpy.asarray([.1, .2, .3], dtype=config.floatX)
        high_val = numpy.asarray([1.1, 2.2, 3.3], dtype=config.floatX)
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = made.f(low_val, high_val)
        numpy_val0 = numpy_rng.uniform(low=low_val, high=high_val)
        assert numpy.allclose(val0, numpy_val0)

        # arguments of size (2,)
        val1 = made.f(low_val[:-1], high_val[:-1])
        numpy_val1 = numpy_rng.uniform(low=low_val[:-1], high=high_val[:-1])
        assert numpy.allclose(val1, numpy_val1)

        # Specifying the size explicitly
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val2 = made.g(low_val, high_val)
        numpy_val2 = numpy_rng.uniform(low=low_val, high=high_val, size=(3,))
        assert numpy.allclose(val2, numpy_val2)
        self.assertRaises(ValueError, made.g, low_val[:-1], high_val[:-1])

    def test_binomial_vector(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        n = tensor.lvector()
        prob = tensor.vector()
        out = m.random.binomial(n=n, p=prob)
        assert out.ndim == 1
        m.f = Method([n, prob], out)
        # Specifying the size explicitly
        m.g = Method([n, prob],
                m.random.binomial(n=n, p=prob, size=(3,)))
        made = m.make()
        made.random.initialize()

        n_val = [1, 2, 3]
        prob_val = numpy.asarray([.1, .2, .3], dtype=config.floatX)
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = made.f(n_val, prob_val)
        numpy_val0 = numpy_rng.binomial(n=n_val, p=prob_val)
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        val1 = made.f(n_val[:-1], prob_val[:-1])
        numpy_val1 = numpy_rng.binomial(n=n_val[:-1], p=prob_val[:-1])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val2 = made.g(n_val, prob_val)
        numpy_val2 = numpy_rng.binomial(n=n_val, p=prob_val, size=(3,))
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, made.g, n_val[:-1], prob_val[:-1])

    def test_normal_vector(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        avg = tensor.vector()
        std = tensor.vector()
        out = m.random.normal(avg=avg, std=std)
        assert out.ndim == 1
        m.f = Method([avg, std], out)
        # Specifying the size explicitly
        m.g = Method([avg, std],
                m.random.normal(avg=avg, std=std, size=(3,)))
        made = m.make()
        made.random.initialize()

        avg_val = [1, 2, 3]
        std_val = numpy.asarray([.1, .2, .3], dtype=config.floatX)
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = made.f(avg_val, std_val)
        numpy_val0 = numpy_rng.normal(loc=avg_val, scale=std_val)
        assert numpy.allclose(val0, numpy_val0)

        # arguments of size (2,)
        val1 = made.f(avg_val[:-1], std_val[:-1])
        numpy_val1 = numpy_rng.normal(loc=avg_val[:-1], scale=std_val[:-1])
        assert numpy.allclose(val1, numpy_val1)

        # Specifying the size explicitly
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val2 = made.g(avg_val, std_val)
        numpy_val2 = numpy_rng.normal(loc=avg_val, scale=std_val, size=(3,))
        assert numpy.allclose(val2, numpy_val2)
        self.assertRaises(ValueError, made.g, avg_val[:-1], std_val[:-1])

    def test_random_integers_vector(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        low = tensor.lvector()
        high = tensor.lvector()
        out = m.random.random_integers(low=low, high=high)
        assert out.ndim == 1
        m.f = Method([low, high], out)
        # Specifying the size explicitly
        m.g = Method([low, high],
                m.random.random_integers(low=low, high=high, size=(3,)))
        made = m.make()
        made.random.initialize()

        low_val = [100, 200, 300]
        high_val = [110, 220, 330]
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = made.f(low_val, high_val)
        numpy_val0 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val, high_val)])
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        val1 = made.f(low_val[:-1], high_val[:-1])
        numpy_val1 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val[:-1], high_val[:-1])])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val2 = made.g(low_val, high_val)
        numpy_val2 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val, high_val)])
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, made.g, low_val[:-1], high_val[:-1])

    # Vectorized permutation don't make sense: the only parameter, n,
    # controls one dimension of the returned tensor.

    def test_multinomial_vector(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        n = tensor.lvector()
        pvals = tensor.matrix()
        out = m.random.multinomial(n=n, pvals=pvals)
        assert out.ndim == 2
        m.f = Method([n, pvals], out)
        # Specifying the size explicitly
        m.g = Method([n, pvals],
                m.random.multinomial(n=n, pvals=pvals, size=(3,)))
        made = m.make()
        made.random.initialize()

        n_val = [1, 2, 3]
        pvals_val = numpy.asarray([[.1, .9], [.2, .8], [.3, .7]],
                dtype=config.floatX)
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = made.f(n_val, pvals_val)
        numpy_val0 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val, pvals_val)])
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        val1 = made.f(n_val[:-1], pvals_val[:-1])
        numpy_val1 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val[:-1], pvals_val[:-1])])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val2 = made.g(n_val, pvals_val)
        numpy_val2 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val, pvals_val)])
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, made.g, n_val[:-1], pvals_val[:-1])

    def test_dtype(self):
        m = Module()
        m.random = RandomStreams(utt.fetch_seed())
        low = tensor.lscalar()
        high = tensor.lscalar()
        out = m.random.random_integers(low=low, high=high, size=(20,), dtype='int8')
        assert out.dtype == 'int8'
        m.f = Method([low, high], out)
        made = m.make()
        made.random.initialize()

        val0 = made.f(0, 9)
        assert val0.dtype == 'int8'

        val1 = made.f(255, 257)
        assert val1.dtype == 'int8'
        assert numpy.all(abs(val1) <= 1)


if __name__ == '__main__':
    from theano.tests import main
    main("test_randomstreams")
