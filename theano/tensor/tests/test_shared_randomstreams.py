__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy

from theano.tensor import raw_random
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function, shared

from theano import tensor
from theano import compile, config, gof

from theano.tests import unittest_tools as utt

class T_SharedRandomStreams(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_tutorial(self):
        srng = RandomStreams(seed=234)
        rv_u = srng.uniform((2,2))
        rv_n = srng.normal((2,2))
        f = function([], rv_u)
        g = function([], rv_n, no_default_updates=True)    #Not updating rv_n.rng
        nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)

        assert numpy.all(f() != f())
        assert numpy.all(g() == g())
        assert numpy.all(abs(nearly_zeros()) < 1e-5)

        assert isinstance(rv_u.rng.get_value(borrow=True),
                numpy.random.RandomState)

    def test_basics(self):
        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.uniform((2,2)), updates=random.updates())
        gn = function([], random.normal((2,2)), updates=random.updates())

        fn_val0 = fn()
        fn_val1 = fn()

        gn_val0 = gn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.allclose(fn_val0, numpy_val0)
        print fn_val0
        print numpy_val0
        print fn_val1
        print numpy_val1
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_seed_fn(self):
        random = RandomStreams(234)
        fn = function([], random.uniform((2,2)), updates=random.updates())

        random.seed(utt.fetch_seed())

        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed))  #int() is for 32bit

        #print fn_val0
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        #print numpy_val0

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_getitem(self):

        random = RandomStreams(234)
        out = random.uniform((2,2))
        fn = function([], out, updates=random.updates())

        random.seed(utt.fetch_seed())

        rng = numpy.random.RandomState()
        rng.set_state(random[out.rng].get_state()) #tests getitem

        fn_val0 = fn()
        fn_val1 = fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_setitem(self):

        random = RandomStreams(234)
        out = random.uniform((2,2))
        fn = function([], out, updates=random.updates())

        random.seed(888)

        rng = numpy.random.RandomState(utt.fetch_seed())
        random[out.rng] = numpy.random.RandomState(utt.fetch_seed())

        fn_val0 = fn()
        fn_val1 = fn()
        numpy_val0 = rng.uniform(size=(2,2))
        numpy_val1 = rng.uniform(size=(2,2))
        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_ndim(self):
        """Test that the behaviour of 'ndim' optional parameter"""
        # 'ndim' is an optional integer parameter, specifying the length
        # of the 'shape', passed as a keyword argument.

        # ndim not specified, OK
        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.uniform((2,2)))

        # ndim specified, consistent with shape, OK
        random2 = RandomStreams(utt.fetch_seed())
        fn2 = function([], random2.uniform((2,2), ndim=2))

        val1 = fn()
        val2 = fn2()
        assert numpy.all(val1 == val2)

        # ndim specified, inconsistent with shape, should raise ValueError
        random3 = RandomStreams(utt.fetch_seed())
        self.assertRaises(ValueError, random3.uniform, (2,2), ndim=1)

    def test_uniform(self):
        """Test that RandomStreams.uniform generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.uniform((2,2), -1, 1))
        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.uniform(-1, 1, size=(2,2))
        numpy_val1 = rng.uniform(-1, 1, size=(2,2))

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_normal(self):
        """Test that RandomStreams.normal generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.

        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.normal((2,2), -1, 2))
        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.normal(-1, 2, size=(2,2))
        numpy_val1 = rng.normal(-1, 2, size=(2,2))

        assert numpy.allclose(fn_val0, numpy_val0)
        assert numpy.allclose(fn_val1, numpy_val1)

    def test_random_integers(self):
        """Test that RandomStreams.random_integers generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.random_integers((20, 20), -5, 5))
        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.random_integers(-5, 5, size=(20,20))
        numpy_val1 = rng.random_integers(-5, 5, size=(20,20))

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)
    
    def test_choice(self):
        """Test that RandomStreams.choice generates the same results as numpy"""
        # numpy.random.choice is only available for numpy versions >= 1.7
        major, minor, _ = numpy.version.short_version.split('.')
        if (int(major), int(minor)) < (1, 7):
            raise utt.SkipTest('choice requires at NumPy version >= 1.7 '
                               '(%s)' % numpy.__version__)
        
        # Check over two calls to see if the random state is correctly updated.
        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.choice((11, 8), 10, 1, 0))
        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        rng = numpy.random.RandomState(int(rng_seed)) #int() is for 32bit
        numpy_val0 = rng.choice(10, (11, 8), True, None)
        numpy_val1 = rng.choice(10, (11, 8), True, None)

        assert numpy.all(fn_val0 == numpy_val0)
        assert numpy.all(fn_val1 == numpy_val1)

    def test_permutation(self):
        """Test that RandomStreams.permutation generates the same results as numpy"""
        # Check over two calls to see if the random state is correctly updated.
        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.permutation((20,), 10), updates=random.updates())

        fn_val0 = fn()
        fn_val1 = fn()

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
        random = RandomStreams(utt.fetch_seed())
        fn = function([], random.multinomial((4,4), 1, [0.1]*10), updates=random.updates())

        fn_val0 = fn()
        fn_val1 = fn()

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
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
        random = RandomStreams(utt.fetch_seed())
        m_input = tensor.dmatrix()
        f = function([m_input], random.shuffle_row_elements(m_input), updates=random.updates())

        # Generate the elements to be shuffled
        val_rng = numpy.random.RandomState(utt.fetch_seed()+42)
        in_mval = val_rng.uniform(-2, 2, size=(20,5))
        fn_mval0 = f(in_mval)
        fn_mval1 = f(in_mval)
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
        random1 = RandomStreams(utt.fetch_seed())
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

    def test_default_updates(self):
        # Basic case: default_updates
        random_a = RandomStreams(utt.fetch_seed())
        out_a = random_a.uniform((2,2))
        fn_a = function([], out_a)
        fn_a_val0 = fn_a()
        fn_a_val1 = fn_a()
        assert not numpy.all(fn_a_val0 == fn_a_val1)

        nearly_zeros = function([], out_a + out_a - 2 * out_a)
        assert numpy.all(abs(nearly_zeros()) < 1e-5)

        # Explicit updates #1
        random_b = RandomStreams(utt.fetch_seed())
        out_b = random_b.uniform((2,2))
        fn_b = function([], out_b, updates=random_b.updates())
        fn_b_val0 = fn_b()
        fn_b_val1 = fn_b()
        assert numpy.all(fn_b_val0 == fn_a_val0)
        assert numpy.all(fn_b_val1 == fn_a_val1)

        # Explicit updates #2
        random_c = RandomStreams(utt.fetch_seed())
        out_c = random_c.uniform((2,2))
        fn_c = function([], out_c, updates=[out_c.update])
        fn_c_val0 = fn_c()
        fn_c_val1 = fn_c()
        assert numpy.all(fn_c_val0 == fn_a_val0)
        assert numpy.all(fn_c_val1 == fn_a_val1)

        # No updates at all
        random_d = RandomStreams(utt.fetch_seed())
        out_d = random_d.uniform((2,2))
        fn_d = function([], out_d, no_default_updates=True)
        fn_d_val0 = fn_d()
        fn_d_val1 = fn_d()
        assert numpy.all(fn_d_val0 == fn_a_val0)
        assert numpy.all(fn_d_val1 == fn_d_val0)

        # No updates for out
        random_e = RandomStreams(utt.fetch_seed())
        out_e = random_e.uniform((2,2))
        fn_e = function([], out_e, no_default_updates=[out_e.rng])
        fn_e_val0 = fn_e()
        fn_e_val1 = fn_e()
        assert numpy.all(fn_e_val0 == fn_a_val0)
        assert numpy.all(fn_e_val1 == fn_e_val0)

    def test_symbolic_shape(self):
        random = RandomStreams(utt.fetch_seed())
        shape = tensor.lvector()
        f = function([shape], random.uniform(size=shape, ndim=2))

        assert f([2,3]).shape == (2,3)
        assert f([4,8]).shape == (4,8)
        self.assertRaises(ValueError, f, [4])
        self.assertRaises(ValueError, f, [4,3,4,5])

    def test_mixed_shape(self):
        # Test when the provided shape is a tuple of ints and scalar vars
        random = RandomStreams(utt.fetch_seed())
        shape0 = tensor.lscalar()
        shape = (shape0, 3)
        f = function([shape0], random.uniform(size=shape, ndim=2))
        assert f(2).shape == (2, 3)
        assert f(8).shape == (8, 3)

        g = function([shape0], random.uniform(size=shape))
        assert g(2).shape == (2, 3)
        assert g(8).shape == (8, 3)

    def test_mixed_shape_bcastable(self):
        # Test when the provided shape is a tuple of ints and scalar vars
        random = RandomStreams(utt.fetch_seed())
        shape0 = tensor.lscalar()
        shape = (shape0, 1)
        u = random.uniform(size=shape, ndim=2)
        assert u.broadcastable == (False, True)
        f = function([shape0], u)
        assert f(2).shape == (2, 1)
        assert f(8).shape == (8, 1)

        v = random.uniform(size=shape)
        assert v.broadcastable == (False, True)
        g = function([shape0], v)
        assert g(2).shape == (2, 1)
        assert g(8).shape == (8, 1)

    def test_default_shape(self):
        random = RandomStreams(utt.fetch_seed())
        f = function([], random.uniform())
        g = function([], random.multinomial())

        #seed_rng is generator for generating *seeds* for RandomStates
        seed_rng = numpy.random.RandomState(utt.fetch_seed())
        uniform_rng = numpy.random.RandomState(int(seed_rng.randint(2**30)))
        multinomial_rng = numpy.random.RandomState(int(seed_rng.randint(2**30)))

        val0 = f()
        val1 = f()
        numpy_val0 = uniform_rng.uniform()
        numpy_val1 = uniform_rng.uniform()
        assert numpy.allclose(val0, numpy_val0)
        assert numpy.allclose(val1, numpy_val1)

        for i in range(10): # every test has 50% chance of passing even with non-matching random states
            val2 = g()
            numpy_val2 = multinomial_rng.multinomial(n=1, pvals=[.5, .5])
            assert numpy.all(val2 == numpy_val2)

    def test_vector_arguments(self):
        random = RandomStreams(utt.fetch_seed())
        low = tensor.dvector()
        out = random.uniform(low=low, high=1)
        assert out.ndim == 1
        f = function([low], out)

        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val0 = f([-5, .5, 0, 1])
        val1 = f([.9])
        numpy_val0 = numpy_rng.uniform(low=[-5, .5, 0, 1], high=1)
        numpy_val1 = numpy_rng.uniform(low=[.9], high=1)
        assert numpy.all(val0 == numpy_val0)
        assert numpy.all(val1 == numpy_val1)

        high = tensor.vector()
        outb = random.uniform(low=low, high=high)
        assert outb.ndim == 1
        fb = function([low, high], outb)

        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val0b = fb([-4., -2], [-1, 0])
        val1b = fb([-4.], [-1])
        numpy_val0b = numpy_rng.uniform(low=[-4., -2], high=[-1, 0])
        numpy_val1b = numpy_rng.uniform(low=[-4.], high=[-1])
        assert numpy.all(val0b == numpy_val0b)
        assert numpy.all(val1b == numpy_val1b)
        self.assertRaises(ValueError, fb, [-4., -2], [-1, 0, 1])
        #TODO: do we want that?
        #self.assertRaises(ValueError, fb, [-4., -2], [-1])

        size = tensor.lvector()
        outc = random.uniform(low=low, high=high, size=size, ndim=1)
        fc = function([low, high, size], outc)

        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        val0c = fc([-4., -2], [-1, 0], [2])
        val1c = fc([-4.], [-1], [1])
        numpy_val0c = numpy_rng.uniform(low=[-4., -2], high=[-1, 0])
        numpy_val1c = numpy_rng.uniform(low=[-4.], high=[-1])
        assert numpy.all(val0c == numpy_val0c)
        assert numpy.all(val1c == numpy_val1c)
        self.assertRaises(ValueError, fc, [-4., -2], [-1, 0], [1])
        self.assertRaises(ValueError, fc, [-4., -2], [-1, 0], [1,2])
        self.assertRaises(ValueError, fc, [-4., -2], [-1, 0], [2,1])
        self.assertRaises(ValueError, fc, [-4., -2], [-1], [1])
        #TODO: do we want that?
        #self.assertRaises(ValueError, fc, [-4., -2], [-1], [2])

    def test_broadcast_arguments(self):
        random = RandomStreams(utt.fetch_seed())
        low = tensor.dvector()
        high = tensor.dcol()
        out = random.uniform(low=low, high=high)
        assert out.ndim == 2
        f = function([low, high], out)

        rng_seed = numpy.random.RandomState(utt.fetch_seed()).randint(2**30)
        numpy_rng = numpy.random.RandomState(int(rng_seed))
        val0 = f([-5, .5, 0, 1], [[1.]])
        val1 = f([.9], [[1.], [1.1], [1.5]])
        val2 = f([-5, .5, 0, 1], [[1.], [1.1], [1.5]])

        numpy_val0 = numpy_rng.uniform(low=[-5, .5, 0, 1], high=[1.])
        numpy_val1 = numpy_rng.uniform(low=[.9], high=[[1.], [1.1], [1.5]])
        numpy_val2 = numpy_rng.uniform(low=[-5, .5, 0, 1], high=[[1.], [1.1], [1.5]])

        assert numpy.all(val0 == numpy_val0)
        assert numpy.all(val1 == numpy_val1)
        assert numpy.all(val2 == numpy_val2)

    def test_uniform_vector(self):
        random = RandomStreams(utt.fetch_seed())
        low = tensor.dvector()
        high = tensor.dvector()
        out = random.uniform(low=low, high=high)
        assert out.ndim == 1
        f = function([low, high], out)

        low_val = [.1, .2, .3]
        high_val = [1.1, 2.2, 3.3]
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = f(low_val, high_val)
        numpy_val0 = numpy_rng.uniform(low=low_val, high=high_val)
        print 'THEANO', val0
        print 'NUMPY', numpy_val0
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        val1 = f(low_val[:-1], high_val[:-1])
        numpy_val1 = numpy_rng.uniform(low=low_val[:-1], high=high_val[:-1])
        print 'THEANO', val1
        print 'NUMPY', numpy_val1
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = function([low, high], random.uniform(low=low, high=high, size=(3,)))
        val2 = g(low_val, high_val)
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        numpy_val2 = numpy_rng.uniform(low=low_val, high=high_val, size=(3,))
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, low_val[:-1], high_val[:-1])

    def test_binomial_vector(self):
        random = RandomStreams(utt.fetch_seed())
        n = tensor.lvector()
        prob = tensor.vector()
        out = random.binomial(n=n, p=prob)
        assert out.ndim == 1
        f = function([n, prob], out)

        n_val = [1, 2, 3]
        prob_val = numpy.asarray([.1, .2, .3], dtype=config.floatX)
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = f(n_val, prob_val)
        numpy_val0 = numpy_rng.binomial(n=n_val, p=prob_val)
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        val1 = f(n_val[:-1], prob_val[:-1])
        numpy_val1 = numpy_rng.binomial(n=n_val[:-1], p=prob_val[:-1])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = function([n, prob], random.binomial(n=n, p=prob, size=(3,)))
        val2 = g(n_val, prob_val)
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        numpy_val2 = numpy_rng.binomial(n=n_val, p=prob_val, size=(3,))
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, n_val[:-1], prob_val[:-1])

    def test_normal_vector(self):
        random = RandomStreams(utt.fetch_seed())
        avg = tensor.dvector()
        std = tensor.dvector()
        out = random.normal(avg=avg, std=std)
        assert out.ndim == 1
        f = function([avg, std], out)

        avg_val = [1, 2, 3]
        std_val = [.1, .2, .3]
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = f(avg_val, std_val)
        numpy_val0 = numpy_rng.normal(loc=avg_val, scale=std_val)
        assert numpy.allclose(val0, numpy_val0)

        # arguments of size (2,)
        val1 = f(avg_val[:-1], std_val[:-1])
        numpy_val1 = numpy_rng.normal(loc=avg_val[:-1], scale=std_val[:-1])
        assert numpy.allclose(val1, numpy_val1)

        # Specifying the size explicitly
        g = function([avg, std], random.normal(avg=avg, std=std, size=(3,)))
        val2 = g(avg_val, std_val)
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        numpy_val2 = numpy_rng.normal(loc=avg_val, scale=std_val, size=(3,))
        assert numpy.allclose(val2, numpy_val2)
        self.assertRaises(ValueError, g, avg_val[:-1], std_val[:-1])

    def test_random_integers_vector(self):
        random = RandomStreams(utt.fetch_seed())
        low = tensor.lvector()
        high = tensor.lvector()
        out = random.random_integers(low=low, high=high)
        assert out.ndim == 1
        f = function([low, high], out)

        low_val = [100, 200, 300]
        high_val = [110, 220, 330]
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = f(low_val, high_val)
        numpy_val0 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val, high_val)])
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        val1 = f(low_val[:-1], high_val[:-1])
        numpy_val1 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val[:-1], high_val[:-1])])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = function([low, high], random.random_integers(low=low, high=high, size=(3,)))
        val2 = g(low_val, high_val)
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        numpy_val2 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val, high_val)])
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, low_val[:-1], high_val[:-1])

    # Vectorized permutation don't make sense: the only parameter, n,
    # controls one dimension of the returned tensor.

    def test_multinomial_vector(self):
        random = RandomStreams(utt.fetch_seed())
        n = tensor.lvector()
        pvals = tensor.matrix()
        out = random.multinomial(n=n, pvals=pvals)
        assert out.ndim == 2
        f = function([n, pvals], out)

        n_val = [1, 2, 3]
        pvals_val = [[.1, .9], [.2, .8], [.3, .7]]
        pvals_val = numpy.asarray(pvals_val, dtype=config.floatX)
        seed_gen = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))

        # Arguments of size (3,)
        val0 = f(n_val, pvals_val)
        numpy_val0 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val, pvals_val)])
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        val1 = f(n_val[:-1], pvals_val[:-1])
        numpy_val1 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val[:-1], pvals_val[:-1])])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = function([n, pvals], random.multinomial(n=n, pvals=pvals, size=(3,)))
        val2 = g(n_val, pvals_val)
        numpy_rng = numpy.random.RandomState(int(seed_gen.randint(2**30)))
        numpy_val2 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val, pvals_val)])
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, n_val[:-1], pvals_val[:-1])

    def test_dtype(self):
        random = RandomStreams(utt.fetch_seed())
        low = tensor.lscalar()
        high = tensor.lscalar()
        out = random.random_integers(low=low, high=high, size=(20,), dtype='int8')
        assert out.dtype == 'int8'
        f = function([low, high], out)

        val0 = f(0, 9)
        assert val0.dtype == 'int8'

        val1 = f(255, 257)
        assert val1.dtype == 'int8'
        assert numpy.all(abs(val1) <= 1)

    def test_default_dtype(self):
        random = RandomStreams(utt.fetch_seed())
        low = tensor.dscalar()
        high = tensor.dscalar()

        # Should not silently downcast from low and high
        out0 = random.uniform(low=low, high=high, size=(42,))
        assert out0.dtype == 'float64'
        f0 = function([low, high], out0)
        val0 = f0(-2.1, 3.1)
        assert val0.dtype == 'float64'

        # Should downcast, since asked explicitly
        out1 = random.uniform(low=low, high=high, size=(42,), dtype='float32')
        assert out1.dtype == 'float32'
        f1 = function([low, high], out1)
        val1 = f1(-1.1, 1.1)
        assert val1.dtype == 'float32'

        # Should use floatX
        lowf = tensor.fscalar()
        highf = tensor.fscalar()
        outf = random.uniform(low=lowf, high=highf, size=(42,))
        assert outf.dtype == config.floatX
        ff = function([lowf, highf], outf)
        valf = ff(numpy.float32(-0.1), numpy.float32(0.3))
        assert valf.dtype == config.floatX

    def test_shared_constructor_borrow(self):
        rng = numpy.random.RandomState(123)
        s_rng_default = shared(rng)
        s_rng_True = shared(rng, borrow=True)
        s_rng_False = shared(rng, borrow=False)


        # test borrow contract: that False means a copy must have been made
        assert s_rng_default.container.storage[0] is not rng
        assert s_rng_False.container.storage[0] is not rng

        # test current implementation: that True means a copy was not made
        assert s_rng_True.container.storage[0] is rng

        # ensure that all the random number generators are in the same state
        v = rng.randn()
        v0 = s_rng_default.container.storage[0].randn()
        v1 = s_rng_False.container.storage[0].randn()
        assert v == v0 == v1

    def test_get_value_borrow(self):

        rng = numpy.random.RandomState(123)
        s_rng = shared(rng)

        r_ = s_rng.container.storage[0]
        r_T = s_rng.get_value(borrow=True)
        r_F = s_rng.get_value(borrow=False)

        #the contract requires that borrow=False returns a copy
        assert r_ is not r_F

        # the current implementation allows for True to return the real thing
        assert r_ is r_T

        #either way, the rngs should all be in the same state
        assert r_.rand() == r_F.rand()

    def test_get_value_internal_type(self):
        rng = numpy.random.RandomState(123)
        s_rng = shared(rng)

        # there is no special behaviour required of return_internal_type
        # this test just ensures that the flag doesn't screw anything up
        # by repeating the get_value_borrow test.
        r_ = s_rng.container.storage[0]
        r_T = s_rng.get_value(borrow=True, return_internal_type=True)
        r_F = s_rng.get_value(borrow=False, return_internal_type=True)

        #the contract requires that borrow=False returns a copy
        assert r_ is not r_F

        # the current implementation allows for True to return the real thing
        assert r_ is r_T

        #either way, the rngs should all be in the same state
        assert r_.rand() == r_F.rand()

    def test_set_value_borrow(self):
        rng = numpy.random.RandomState(123)

        s_rng = shared(rng)

        new_rng = numpy.random.RandomState(234234)

        # Test the borrow contract is respected:
        # assigning with borrow=False makes a copy
        s_rng.set_value(new_rng, borrow=False)
        assert new_rng is not s_rng.container.storage[0]
        assert new_rng.randn() == s_rng.container.storage[0].randn()

        # Test that the current implementation is actually borrowing when it can.
        rr = numpy.random.RandomState(33)
        s_rng.set_value(rr, borrow=True)
        assert rr is s_rng.container.storage[0]

    def test_multiple_rng_aliasing(self):
        """
        Test that when we have multiple random number generators, we do not alias
        the state_updates member. `state_updates` can be useful when attempting to
        copy the (random) state between two similar theano graphs. The test is
        meant to detect a previous bug where state_updates was initialized as a
        class-attribute, instead of the __init__ function.
        """
        rng1 = RandomStreams(1234)
        rng2 = RandomStreams(2392)
        assert rng1.state_updates is not rng2.state_updates
        assert rng1.gen_seedgen is not rng2.gen_seedgen

    def test_random_state_transfer(self):
        """
        Test that random state can be transferred from one theano graph to another.
        """
        class Graph:
            def __init__(self, seed=123):
                self.rng = RandomStreams(seed)
                self.y = self.rng.uniform(size=(1,))
        g1 = Graph(seed=123)
        f1 = function([], g1.y)
        g2 = Graph(seed=987)
        f2 = function([], g2.y)

        for (su1, su2) in zip(g1.rng.state_updates, g2.rng.state_updates):
            su2[0].set_value(su1[0].get_value())

        numpy.testing.assert_array_almost_equal(f1(), f2(), decimal=6)


if __name__ == '__main__':
    from theano.tests import main
    main("test_shared_randomstreams")
