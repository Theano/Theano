from __future__ import absolute_import, print_function, division
import numpy
import pickle

from theano.tests import unittest_tools as utt

from theano.tensor.raw_random import *
from theano.tensor import (raw_random, ivector, dvector, iscalar, dcol,
                           dtensor3)
from theano import tensor

from theano import compile, config, gof

__docformat__ = "restructuredtext en"


class T_random_function(utt.InferShapeTester):
    def setUp(self):
        utt.seed_rng()

    def test_basic_usage(self):
        rf = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector)
        assert not rf.inplace
        assert getattr(rf, 'destroy_map', {}) == {}

        rng_R = random_state_type()

        # If calling RandomFunction directly, all args have to be specified,
        # because shape will have to be moved to the end
        post_r, out = rf(rng_R, (4,), 0., 1.)

        assert out.type == tensor.dvector

        f = compile.function([rng_R], out)

        rng_state0 = numpy.random.RandomState(utt.fetch_seed())

        f_0 = f(rng_state0)
        f_1 = f(rng_state0)

        assert numpy.all(f_0 == f_1)

    def test_inplace_norun(self):
        rf = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector,
                            inplace=True)
        assert rf.inplace
        assert getattr(rf, 'destroy_map', {}) != {}

    def test_args(self):
        """Test that arguments to RandomFunction are honored"""
        rf2 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector)
        rf4 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector,
                             inplace=True)
        rng_R = random_state_type()

        # use make_node to override some of the self.args
        post_r2, out2 = rf2(rng_R, (4,), -2, 2)  # NOT INPLACE
        post_r4, out4 = rf4(rng_R, (4,), -4, 4)  # INPLACE
        post_r2_4, out2_4 = rf2(rng_R, (4, ), -4.0, 2)  # NOT INPLACE
        post_r2_4_4, out2_4_4 = rf2(rng_R, (4, ), -4.0, 4.0)  # NOT INPLACE

        # configure out4 to be computed inplace
        # The update expression means that the random state rng_R will
        # be maintained by post_r4
        f = compile.function(
                [compile.In(rng_R,
                            value=numpy.random.RandomState(utt.fetch_seed()),
                            update=post_r4,
                            mutable=True)],
                [out2, out4, out2_4, out2_4_4],
                accept_inplace=True)

        f2, f4, f2_4, f2_4_4 = f()
        f2b, f4b, f2_4b, f2_4_4b = f()

        # print f2
        # print f4
        # print f2_4
        # print f2_4_4

        # print f2b
        # print f4b
        # print f2_4b
        # print f2_4_4b

        # setting bounds is same as multiplying by 2
        assert numpy.allclose(f2 * 2, f4), (f2, f4)

        # retrieving from non-inplace generator
        # is same as inplace one for first call
        assert numpy.allclose(f2_4_4, f4), (f2_4_4, f4)

        # f4 changes from call to call, that the update has worked
        assert not numpy.allclose(f4, f4b), (f4, f4b)

    def test_inplace_optimization(self):
        """Test that FAST_RUN includes the random_make_inplace optimization"""
        #inplace = False
        rf2 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector)
        rng_R = random_state_type()

        # If calling RandomFunction directly, all args have to be specified,
        # because shape will have to be moved to the end
        post_r2, out2 = rf2(rng_R, (4,), 0., 1.)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r2,
                    mutable=True)],
                out2,
                mode='FAST_RUN')  # DEBUG_MODE can't pass the id-based
                                  # test below

        # test that the RandomState object stays the same from function call to
        # function call, but that the values returned change from call to call.

        id0 = id(f[rng_R])
        val0 = f()
        assert id0 == id(f[rng_R])
        val1 = f()
        assert id0 == id(f[rng_R])

        assert not numpy.allclose(val0, val1)

    def test_no_inplace(self):
        """Test that when not running inplace, the RandomState is
        not updated"""
        rf = RandomFunction('uniform', tensor.dvector)
        rng_R = random_state_type()

        post_r, out = rf(rng_R, (3,), 0., 1.)
        f = compile.function([rng_R], [post_r, out])
        rng = numpy.random.RandomState(utt.fetch_seed())

        rng0, val0 = f(rng)
        rng_ = numpy.random.RandomState(utt.fetch_seed())
        # rng should still be in a fresh state
        self.assertTrue(rng_R.type.values_eq(rng, rng_))
        # rng0 should be in an updated state
        self.assertFalse(rng_R.type.values_eq(rng, rng0))

        f2 = compile.function(
                [compile.In(rng_R,
                    value=rng,
                    update=post_r,
                    mutable=False)],
                [post_r, out])
        rng2, val2 = f2()
        # rng should be in a fresh state
        self.assertTrue(rng_R.type.values_eq(rng, rng_))
        # rng2 should be in an updated state
        self.assertFalse(rng_R.type.values_eq(rng, rng2))
        # The updated state should be the same for both functions
        self.assertTrue(rng_R.type.values_eq(rng2, rng0))

        rng3, val3 = f2()
        # rng2 should not have changed
        self.assertTrue(rng_R.type.values_eq(rng2, rng0))
        # rng3 should be an updated again version of rng2
        self.assertFalse(rng_R.type.values_eq(rng3, rng2))
        self.assertFalse(rng_R.type.values_eq(rng3, rng))

    def test_random_function_ndim(self):
        """Test that random_function helper function accepts argument ndim"""
        rng_R = random_state_type()

        # ndim is an optional argument indicating the length of the 'shape'
        # ndim not specified, OK
        post_out4, out4 = uniform(rng_R, (4,))

        # ndim specified, consistent with shape, OK
        post_out1_4, out1_4 = uniform(rng_R, (4, ), ndim=1)
        post_out2_4_4, out2_4_4 = uniform(rng_R, (4, 4), ndim=2)

        # ndim specified, but not compatible with shape
        self.assertRaises(ValueError, uniform, rng_R, (4,), ndim=2)

        f_ok = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_out2_4_4,
                    mutable=True)],
                [out4, out1_4, out2_4_4],
                accept_inplace=True)

        # The correct cases should execute properly
        o4, o1_4, o2_4_4 = f_ok()

        # Check the sanity of the answers
        self.assertTrue(numpy.allclose(o4, o1_4))
        self.assertTrue(numpy.allclose(o4, o2_4_4[0]))

    def test_random_function_noshape_args(self):
        '''Test if random_function helper works with args but without shape'''
        rng_R = random_state_type()

        # No shape, default args -> OK
        post_out, out = uniform(rng_R, size=None, ndim=2)
        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_out,
                    mutable=True)],
                [out],
                accept_inplace=True)
        o, = f()

        # No shape, args that have to be broadcasted -> OK
        low = tensor.TensorType(dtype='float64',
                broadcastable=(False, True, True))()
        high = tensor.TensorType(dtype='float64',
                broadcastable=(True, True, True, False))()
        post_out2, out2 = uniform(rng_R, size=None, ndim=2, low=low, high=high)
        self.assertEqual(out2.ndim, 4)
        self.assertEqual(out2.broadcastable, (True, False, True, False))

        g = compile.function(
                [low,
                 high,
                 compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_out2,
                    mutable=True)],
                [out2],
                accept_inplace=True)
        low_v = [[[3]], [[4]], [[-5]]]
        high_v = [[[[5, 8]]]]
        o2, = g(low_v, high_v)
        self.assertEqual(o2.shape, (1, 3, 1, 2))

    def test_random_function_noshape_noargs(self):
        '''Test if random_function helper works without args or shape'''
        rng_R = random_state_type()

        # No shape, no args -> TypeError
        self.assertRaises(TypeError, poisson, rng_R, size=None, ndim=2)

    def test_random_function_ndim_added(self):
        """Test that random_function helper function accepts ndim_added as
        keyword argument"""
        # If using numpy's uniform distribution, ndim_added should be 0,
        # because the shape provided as argument is the output shape.
        # Specifying a different ndim_added will change the Op's output ndim,
        # so numpy.uniform will produce a result of incorrect shape,
        # and a ValueError should be raised.
        def ndim_added_deco(ndim_added):
            def randomfunction(random_state, size=(), low=0.0, high=0.0,
                               ndim=None):
                ndim, size, bcast = raw_random._infer_ndim_bcast(ndim, size)
                if ndim_added < 0:
                    bcast = bcast[:ndim_added]
                else:
                    bcast = bcast + ((False,) * ndim_added)
                assert len(bcast) == ndim + ndim_added
                op = RandomFunction('uniform',
                        tensor.TensorType(dtype='float64',
                        broadcastable=bcast),
                        ndim_added=ndim_added)
                return op(random_state, size, low, high)
            return randomfunction

        uni_1 = ndim_added_deco(1)
        uni_0 = ndim_added_deco(0)
        uni_m1 = ndim_added_deco(-1)

        rng_R = random_state_type()

        p_uni11, uni11 = uni_1(rng_R, size=(4,))
        p_uni12, uni12 = uni_1(rng_R, size=(3, 4))
        p_uni01, uni01 = uni_0(rng_R, size=(4,))
        p_uni02, uni02 = uni_0(rng_R, size=(3, 4))
        p_unim11, unim11 = uni_m1(rng_R, size=(4,))
        p_unim12, unim12 = uni_m1(rng_R, size=(3, 4))

        self.assertEqual(uni11.ndim, 2)
        self.assertEqual(uni12.ndim, 3)
        self.assertEqual(uni01.ndim, 1)
        self.assertEqual(uni02.ndim, 2)
        self.assertEqual(unim11.ndim, 0)
        self.assertEqual(unim12.ndim, 1)

        f11 = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=p_uni11, mutable=True)],
                [uni11], accept_inplace=True)
        f12 = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=p_uni12, mutable=True)],
                [uni12], accept_inplace=True)
        fm11 = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=p_unim11, mutable=True)],
                [unim11], accept_inplace=True)
        fm12 = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=p_unim12, mutable=True)],
                [unim12], accept_inplace=True)
        f0 = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=p_uni02, mutable=True)],
                [uni01, uni02], accept_inplace=True)
        self.assertRaises(ValueError, f11)
        self.assertRaises(ValueError, f12)
        self.assertRaises(ValueError, fm11)
        self.assertRaises(ValueError, fm12)
        u01, u02 = f0()
        self.assertTrue(numpy.allclose(u01, u02[0]))

    def test_uniform(self):
        """Test that raw_random.uniform generates the same results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters
        post_r, out = uniform(rng_R, (4,), -2.0, 2.0)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.uniform(-2.0, 2.0, size=(4,))
        numpy_val1 = numpy_rng.uniform(-2.0, 2.0, size=(4,))
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_binomial(self):
        """Test that raw_random.binomial generates the same results
        as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters, and larger dimensions because of
        # the integer nature of the result
        post_r, bin = binomial(rng_R, (7, 12), 5, 0.8)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [bin], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.binomial(5, 0.8, size=(7, 12))
        numpy_val1 = numpy_rng.binomial(5, 0.8, size=(7, 12))
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

    def test_normal(self):
        """Test that raw_random.normal generates the same results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters
        post_r, out = normal(rng_R, (2, 3), 4.0, 2.0)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.normal(4.0, 2.0, size=(2, 3))
        numpy_val1 = numpy_rng.normal(4.0, 2.0, size=(2, 3))
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_random_integers(self):
        """Test that raw_random.random_integers generates the same
        results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters, and larger dimensions because of
        # the integer nature of the result
        post_r, out = random_integers(rng_R, (11, 8), -3, 16)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.random_integers(-3, 16, size=(11, 8))
        numpy_val1 = numpy_rng.random_integers(-3, 16, size=(11, 8))
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_permutation_helper(self):
        """Test that raw_random.permutation_helper generates the same
        results as numpy,
        and that the 'ndim_added' keyword behaves correctly."""
        # permutation_helper needs "ndim_added=1", because its output
        # is one dimension more than its "shape" argument (and there's
        # no way to determine that automatically).
        # Check the working case, over two calls to see if the random
        # state is correctly updated.
        rf = RandomFunction(permutation_helper, tensor.imatrix, 8,
                            ndim_added=1)
        rng_R = random_state_type()
        post_r, out = rf(rng_R, (7,), 8)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0 = f()
        val1 = f()
        # numpy_rng.permutation outputs one vector at a time,
        # so we call it iteratively to generate all the samples.
        numpy_val0 = numpy.asarray([numpy_rng.permutation(8)
                                    for i in range(7)])
        numpy_val1 = numpy.asarray([numpy_rng.permutation(8)
                                    for i in range(7)])
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

        # This call lacks "ndim_added=1", so ndim_added defaults to 0.
        # A ValueError should be raised.
        rf0 = RandomFunction(permutation_helper, tensor.imatrix, 8)
        post_r0, out0 = rf0(rng_R, (7,), 8)
        f0 = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r0, mutable=True)],
                [out0], accept_inplace=True)
        self.assertRaises(ValueError, f0)

        # Here, ndim_added is 2 instead of 1. A ValueError should be raised.
        rf2 = RandomFunction(permutation_helper, tensor.imatrix, 8,
                             ndim_added=2)
        post_r2, out2 = rf2(rng_R, (7,), 8)
        f2 = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r2, mutable=True)],
                [out2], accept_inplace=True)
        self.assertRaises(ValueError, f2)
    
    def test_choice(self):
        """Test that raw_random.choice generates the same
        results as numpy."""
        # numpy.random.choice is only available for numpy versions >= 1.7
        major, minor, _ = numpy.version.short_version.split('.')
        if (int(major), int(minor)) < (1, 7):
            raise utt.SkipTest('choice requires at NumPy version >= 1.7 '
                               '(%s)' % numpy.__version__)
        
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters, and larger dimensions because of
        # the integer nature of the result
        post_r, out = choice(rng_R, (11, 8), 10, 1, 0)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.choice(10, (11, 8), True, None)
        numpy_val1 = numpy_rng.choice(10, (11, 8), True, None)
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_poisson(self):
        """Test that raw_random.poisson generates the same
        results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters, and larger dimensions because of
        # the integer nature of the result
        post_r, out = poisson(rng_R, lam=5, size=(11, 8))

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.poisson(5, size=(11, 8))
        numpy_val1 = numpy_rng.poisson(5, size=(11, 8))
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_permutation(self):
        """Test that raw_random.permutation generates the same
        results as numpy."""
        rng_R = random_state_type()
        post_r, out = permutation(rng_R, size=(9,), n=6)
        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        # Check over two calls to see if the random state is correctly updated.
        # numpy_rng.permutation outputs one vector at a time,
        # so we call it iteratively to generate all the samples.
        val0 = f()
        val1 = f()
        numpy_val0 = numpy.asarray([numpy_rng.permutation(6)
                                    for i in range(9)])
        numpy_val1 = numpy.asarray([numpy_rng.permutation(6)
                                    for i in range(9)])
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

        # Test that we can generate a list: have size=None or ().
        for ndim in [1, None]:
            post_r, out = permutation(rng_R, n=10, size=None, ndim=ndim)
            inp = compile.In(rng_R,
                             value=numpy.random.RandomState(utt.fetch_seed()),
                             update=post_r, mutable=True)
            f = theano.function([inp], out)
            o = f()
            assert o.shape == (10,)
            assert (numpy.sort(o) == numpy.arange(10)).all()
        # Wrong number of dimensions asked
        self.assertRaises(TypeError, permutation, rng_R, size=None, ndim=2)

    def test_multinomial(self):
        """Test that raw_random.multinomial generates the same
        results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        post_r, out = multinomial(rng_R, (7, 3), 6, [0.2] * 5)

        f = compile.function(
                [compile.In(rng_R,
                    value=numpy.random.RandomState(utt.fetch_seed()),
                    update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        val0, = f()
        val1, = f()
        numpy_val0 = numpy_rng.multinomial(6, [0.2] * 5, (7, 3))
        numpy_val1 = numpy_rng.multinomial(6, [0.2] * 5, (7, 3))
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

        self.assertTrue(val0.shape == (7, 3, 5))
        self.assertTrue(val1.shape == (7, 3, 5))

    def test_symbolic_shape(self):
        rng_R = random_state_type()
        shape = tensor.lvector()
        post_r, out = uniform(rng_R, shape, ndim=2)
        f = compile.function([rng_R, shape], out)
        rng_state0 = numpy.random.RandomState(utt.fetch_seed())

        assert f(rng_state0, [2, 3]).shape == (2, 3)
        assert f(rng_state0, [4, 8]).shape == (4, 8)

        self.assertRaises(ValueError, f, rng_state0, [4])
        self.assertRaises(ValueError, f, rng_state0, [4, 3, 4, 5])

    def test_mixed_shape(self):
        # Test when the provided shape is a tuple of ints and scalar vars
        rng_R = random_state_type()
        shape0 = tensor.lscalar()
        shape = (shape0, 3)
        post_r, u = uniform(rng_R, size=shape, ndim=2)
        f = compile.function([rng_R, shape0], u)
        rng_state0 = numpy.random.RandomState(utt.fetch_seed())

        assert f(rng_state0, 2).shape == (2, 3)
        assert f(rng_state0, 8).shape == (8, 3)

        post_r, v = uniform(rng_R, size=shape)
        g = compile.function([rng_R, shape0], v)
        assert g(rng_state0, 2).shape == (2, 3)
        assert g(rng_state0, 8).shape == (8, 3)

    def test_mixed_shape_bcastable(self):
        # Test when the provided shape is a tuple of ints and scalar vars
        rng_R = random_state_type()
        shape0 = tensor.lscalar()
        shape = (shape0, 1)
        post_r, u = uniform(rng_R, size=shape, ndim=2)
        assert u.broadcastable == (False, True)
        f = compile.function([rng_R, shape0], u)
        rng_state0 = numpy.random.RandomState(utt.fetch_seed())

        assert f(rng_state0, 2).shape == (2, 1)
        assert f(rng_state0, 8).shape == (8, 1)

        post_r, v = uniform(rng_R, size=shape)
        assert v.broadcastable == (False, True)
        g = compile.function([rng_R, shape0], v)
        assert g(rng_state0, 2).shape == (2, 1)
        assert g(rng_state0, 8).shape == (8, 1)

    def test_default_shape(self):
        rng_R = random_state_type()
        post_r, out = uniform(rng_R)
        f = compile.function([rng_R], [post_r, out], accept_inplace=True)

        rng_state0 = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        post0, val0 = f(rng_state0)
        post1, val1 = f(post0)
        numpy_val0 = numpy.asarray(numpy_rng.uniform(),
                                   dtype=theano.config.floatX)
        numpy_val1 = numpy.asarray(numpy_rng.uniform(),
                                   dtype=theano.config.floatX)

        assert numpy.all(val0 == numpy_val0)
        assert numpy.all(val1 == numpy_val1)

        post_r, out = multinomial(rng_R)
        g = compile.function([rng_R], [post_r, out], accept_inplace=True)
        post2, val2 = g(post1)
        numpy_val2 = numpy.asarray(numpy_rng.multinomial(n=1, pvals=[.5, .5]),
                dtype=theano.config.floatX)

        assert numpy.all(val2 == numpy_val2)

    def test_vector_arguments(self):
        rng_R = random_state_type()
        low = tensor.vector()
        post_r, out = uniform(rng_R, low=low, high=1)
        assert out.ndim == 1
        f = compile.function([rng_R, low], [post_r, out], accept_inplace=True)

        def as_floatX(thing):
            return numpy.asarray(thing, dtype=theano.config.floatX)

        rng_state0 = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        post0, val0 = f(rng_state0, [-5, .5, 0, 1])
        post1, val1 = f(post0, as_floatX([.9]))
        numpy_val0 = as_floatX(numpy_rng.uniform(low=[-5, .5, 0, 1], high=1))
        numpy_val1 = as_floatX(numpy_rng.uniform(low=as_floatX([.9]), high=1))

        assert numpy.all(val0 == numpy_val0)
        assert numpy.all(val1 == numpy_val1)

        high = tensor.vector()
        post_rb, outb = uniform(rng_R, low=low, high=high)
        assert outb.ndim == 1
        fb = compile.function([rng_R, low, high], [post_rb, outb],
                              accept_inplace=True)

        post0b, val0b = fb(post1, [-4., -2], [-1, 0])
        post1b, val1b = fb(post0b, [-4.], [-1])
        numpy_val0b = as_floatX(numpy_rng.uniform(low=[-4., -2], high=[-1, 0]))
        numpy_val1b = as_floatX(numpy_rng.uniform(low=[-4.], high=[-1]))
        assert numpy.all(val0b == numpy_val0b)
        assert numpy.all(val1b == numpy_val1b)
        self.assertRaises(ValueError, fb, post1b, [-4., -2], [-1, 0, 1])
        # TODO: do we want that?
        #self.assertRaises(ValueError, fb, post1b, [-4., -2], [-1])

        size = tensor.lvector()
        post_rc, outc = uniform(rng_R, low=low, high=high, size=size, ndim=1)
        fc = compile.function([rng_R, low, high, size], [post_rc, outc],
                              accept_inplace=True)
        post0c, val0c = fc(post1b, [-4., -2], [-1, 0], [2])
        post1c, val1c = fc(post0c, [-4.], [-1], [1])
        numpy_val0c = as_floatX(numpy_rng.uniform(low=[-4., -2], high=[-1, 0]))
        numpy_val1c = as_floatX(numpy_rng.uniform(low=[-4.], high=[-1]))
        assert numpy.all(val0c == numpy_val0c)
        assert numpy.all(val1c == numpy_val1c)
        self.assertRaises(ValueError, fc, post1c, [-4., -2], [-1, 0], [1])
        self.assertRaises(ValueError, fc, post1c, [-4., -2], [-1, 0], [1, 2])
        self.assertRaises(ValueError, fc, post1c, [-4., -2], [-1, 0], [2, 1])
        self.assertRaises(ValueError, fc, post1c, [-4., -2], [-1], [1])
        # TODO: do we want that?
        #self.assertRaises(ValueError, fc, post1c, [-4., -2], [-1], [2])

    def test_broadcast_arguments(self):
        rng_R = random_state_type()
        low = tensor.dvector()
        high = tensor.dcol()
        post_r, out = uniform(rng_R, low=low, high=high)
        assert out.ndim == 2
        f = compile.function([rng_R, low, high], [post_r, out],
                             accept_inplace=True)

        rng_state0 = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())
        post0, val0 = f(rng_state0, [-5, .5, 0, 1], [[1.]])
        post1, val1 = f(post0, [.9], [[1.], [1.1], [1.5]])
        post2, val2 = f(post1, [-5, .5, 0, 1], [[1.], [1.1], [1.5]])

        numpy_val0 = numpy_rng.uniform(low=[-5, .5, 0, 1], high=[1.])
        numpy_val1 = numpy_rng.uniform(low=[.9], high=[[1.], [1.1], [1.5]])
        numpy_val2 = numpy_rng.uniform(low=[-5, .5, 0, 1],
                                       high=[[1.], [1.1], [1.5]])

        assert numpy.all(val0 == numpy_val0), (val0, numpy_val0)
        assert numpy.all(val1 == numpy_val1)
        assert numpy.all(val2 == numpy_val2)

    def test_uniform_vector(self):
        rng_R = random_state_type()
        low = tensor.vector()
        high = tensor.vector()
        post_r, out = uniform(rng_R, low=low, high=high)
        assert out.ndim == 1
        f = compile.function([rng_R, low, high], [post_r, out],
                             accept_inplace=True)

        def as_floatX(thing):
            return numpy.asarray(thing, dtype=theano.config.floatX)
        low_val = as_floatX([.1, .2, .3])
        high_val = as_floatX([1.1, 2.2, 3.3])
        rng = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())

        # Arguments of size (3,)
        rng0, val0 = f(rng, low_val, high_val)
        numpy_val0 = as_floatX(numpy_rng.uniform(low=low_val, high=high_val))
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        rng1, val1 = f(rng0, low_val[:-1], high_val[:-1])
        numpy_val1 = as_floatX(numpy_rng.uniform(low=low_val[:-1],
                                                 high=high_val[:-1]))
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = compile.function([rng_R, low, high],
                uniform(rng_R, low=low, high=high, size=(3,)),
                accept_inplace=True)
        rng2, val2 = g(rng1, low_val, high_val)
        numpy_val2 = as_floatX(numpy_rng.uniform(low=low_val, high=high_val,
                                                 size=(3,)))
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, rng2, low_val[:-1], high_val[:-1])

    def test_binomial_vector(self):
        rng_R = random_state_type()
        n = tensor.lvector()
        prob = tensor.vector()
        post_r, out = binomial(rng_R, n=n, p=prob)
        assert out.ndim == 1
        f = compile.function([rng_R, n, prob], [post_r, out],
                             accept_inplace=True)

        n_val = [1, 2, 3]
        prob_val = numpy.asarray([.1, .2, .3], dtype=config.floatX)
        rng = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())

        # Arguments of size (3,)
        rng0, val0 = f(rng, n_val, prob_val)
        numpy_val0 = numpy_rng.binomial(n=n_val, p=prob_val)
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        rng1, val1 = f(rng0, n_val[:-1], prob_val[:-1])
        numpy_val1 = numpy_rng.binomial(n=n_val[:-1], p=prob_val[:-1])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = compile.function([rng_R, n, prob],
                binomial(rng_R, n=n, p=prob, size=(3,)),
                accept_inplace=True)
        rng2, val2 = g(rng1, n_val, prob_val)
        numpy_val2 = numpy_rng.binomial(n=n_val, p=prob_val, size=(3,))
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, rng2, n_val[:-1], prob_val[:-1])

    def test_normal_vector(self):
        rng_R = random_state_type()
        avg = tensor.vector()
        std = tensor.vector()
        post_r, out = normal(rng_R, avg=avg, std=std)
        assert out.ndim == 1
        f = compile.function([rng_R, avg, std], [post_r, out],
                             accept_inplace=True)

        def as_floatX(thing):
            return numpy.asarray(thing, dtype=theano.config.floatX)

        avg_val = [1, 2, 3]
        std_val = as_floatX([.1, .2, .3])
        rng = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())

        # Arguments of size (3,)
        rng0, val0 = f(rng, avg_val, std_val)
        numpy_val0 = as_floatX(numpy_rng.normal(loc=as_floatX(avg_val),
            scale=as_floatX(std_val)))
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        rng1, val1 = f(rng0, avg_val[:-1], std_val[:-1])
        numpy_val1 = numpy.asarray(numpy_rng.normal(loc=avg_val[:-1],
                                                    scale=std_val[:-1]),
                dtype=theano.config.floatX)
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = compile.function([rng_R, avg, std],
                normal(rng_R, avg=avg, std=std, size=(3,)),
                accept_inplace=True)
        rng2, val2 = g(rng1, avg_val, std_val)
        numpy_val2 = numpy.asarray(numpy_rng.normal(loc=avg_val, scale=std_val,
                                                    size=(3,)),
                dtype=theano.config.floatX)
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, rng2, avg_val[:-1], std_val[:-1])

    def test_random_integers_vector(self):
        rng_R = random_state_type()
        low = tensor.lvector()
        high = tensor.lvector()
        post_r, out = random_integers(rng_R, low=low, high=high)
        assert out.ndim == 1
        f = compile.function([rng_R, low, high], [post_r, out],
                             accept_inplace=True)

        low_val = [100, 200, 300]
        high_val = [110, 220, 330]
        rng = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())

        # Arguments of size (3,)
        rng0, val0 = f(rng, low_val, high_val)
        numpy_val0 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val, high_val)])
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        rng1, val1 = f(rng0, low_val[:-1], high_val[:-1])
        numpy_val1 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val[:-1], high_val[:-1])])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = compile.function([rng_R, low, high],
                random_integers(rng_R, low=low, high=high, size=(3,)),
                accept_inplace=True)
        rng2, val2 = g(rng1, low_val, high_val)
        numpy_val2 = numpy.asarray([numpy_rng.random_integers(low=lv, high=hv)
            for lv, hv in zip(low_val, high_val)])
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, rng2, low_val[:-1], high_val[:-1])

    # Vectorized permutation don't make sense: the only parameter, n,
    # controls one dimension of the returned tensor.

    def test_multinomial_vector(self):
        rng_R = random_state_type()
        n = tensor.lvector()
        pvals = tensor.matrix()
        post_r, out = multinomial(rng_R, n=n, pvals=pvals)
        assert out.ndim == 2
        f = compile.function([rng_R, n, pvals], [post_r, out],
                             accept_inplace=True)

        n_val = [1, 2, 3]
        pvals_val = [[.1, .9], [.2, .8], [.3, .7]]
        pvals_val = numpy.asarray(pvals_val, dtype=config.floatX)
        rng = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())

        # Arguments of size (3,)
        rng0, val0 = f(rng, n_val, pvals_val)
        numpy_val0 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val, pvals_val)])
        assert numpy.all(val0 == numpy_val0)

        # arguments of size (2,)
        rng1, val1 = f(rng0, n_val[:-1], pvals_val[:-1])
        numpy_val1 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val[:-1], pvals_val[:-1])])
        assert numpy.all(val1 == numpy_val1)

        # Specifying the size explicitly
        g = compile.function([rng_R, n, pvals],
                multinomial(rng_R, n=n, pvals=pvals, size=(3,)),
                accept_inplace=True)
        rng2, val2 = g(rng1, n_val, pvals_val)
        numpy_val2 = numpy.asarray([numpy_rng.multinomial(n=nv, pvals=pv)
            for nv, pv in zip(n_val, pvals_val)])
        assert numpy.all(val2 == numpy_val2)
        self.assertRaises(ValueError, g, rng2, n_val[:-1], pvals_val[:-1])

    def test_multinomial_tensor3_a(self):
        # Test the examples given in the multinomial documentation regarding
        # tensor3 objects
        rng_R = random_state_type()
        n = 9
        pvals = tensor.dtensor3()
        post_r, out = multinomial(rng_R, n=n, pvals=pvals, size=(1, -1))
        assert out.ndim == 3
        assert out.broadcastable == (True, False, False)

        f = compile.function([rng_R, pvals], [post_r, out],
                             accept_inplace=True)

        rng = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())

        pvals_val = numpy.asarray([[[.1, .9], [.2, .8], [.3, .7]]])
        assert pvals_val.shape == (1, 3, 2)

        new_rng, draw = f(rng, pvals_val)
        assert draw.shape == (1, 3, 2)
        assert numpy.allclose(draw.sum(axis=2), 9)

    def test_multinomial_tensor3_b(self):
        # Test the examples given in the multinomial documentation regarding
        # tensor3 objects
        rng_R = random_state_type()
        n = 9
        pvals = tensor.dtensor3()
        post_r, out = multinomial(rng_R, n=n, pvals=pvals, size=(10, 1, -1))
        assert out.ndim == 4
        assert out.broadcastable == (False, True, False, False)

        f = compile.function([rng_R, pvals], [post_r, out],
                             accept_inplace=True)

        rng = numpy.random.RandomState(utt.fetch_seed())
        numpy_rng = numpy.random.RandomState(utt.fetch_seed())

        pvals_val = numpy.asarray([[[.1, .9], [.2, .8], [.3, .7]]])
        assert pvals_val.shape == (1, 3, 2)

        out_rng, draw = f(rng, pvals_val)
        assert draw.shape == (10, 1, 3, 2)
        assert numpy.allclose(draw.sum(axis=3), 9)

    def test_dtype(self):
        rng_R = random_state_type()
        low = tensor.lscalar()
        high = tensor.lscalar()
        post_r, out = random_integers(rng_R, low=low, high=high, size=(20, ),
                                      dtype='int8')
        assert out.dtype == 'int8'
        f = compile.function([rng_R, low, high], [post_r, out])

        rng = numpy.random.RandomState(utt.fetch_seed())
        rng0, val0 = f(rng, 0, 9)
        assert val0.dtype == 'int8'

        rng1, val1 = f(rng0, 255, 257)
        assert val1.dtype == 'int8'
        assert numpy.all(abs(val1) <= 1)

    def test_dtype_normal_uniform_687(self):
        # Regression test for #687.
        rng_R = random_state_type()
        assert uniform(rng_R, low=tensor.constant(0, dtype='float64'),
                       dtype='float32')[1].dtype == 'float32'

        assert normal(rng_R, avg=tensor.constant(0, dtype='float64'),
                      dtype='float32')[1].dtype == 'float32'

    def setUp(self):
        super(T_random_function, self).setUp()

    def test_infer_shape(self):
        rng_R = random_state_type()
        rng_R_val = numpy.random.RandomState(utt.fetch_seed())

        # no shape specified, default args
        post_r, out = uniform(rng_R)
        self._compile_and_check([rng_R], [out], [rng_R_val],
                             RandomFunction)

        post_r, out = uniform(rng_R, size=None, ndim=2)
        self._compile_and_check([rng_R], [out], [rng_R_val],
                                RandomFunction)

        """
        #infer_shape don't work for multinomial.
        #The parameter ndim_added is set to 1 and in this case, the infer_shape
        #inplementation don't know how to infer the shape
        post_r, out = multinomial(rng_R)

        self._compile_and_check([rng_R], [out], [rng_R_val],
                                RandomFunction)
        """

        # no shape specified, args have to be broadcasted
        low = tensor.TensorType(dtype='float64',
                broadcastable=(False, True, True))()
        high = tensor.TensorType(dtype='float64',
                broadcastable=(True, True, True, False))()
        post_r, out = uniform(rng_R, size=None, ndim=2, low=low, high=high)
        low_val = [[[3]], [[4]], [[-5]]]
        high_val = [[[[5, 8]]]]
        self._compile_and_check([rng_R, low, high], [out],
                                [rng_R_val, low_val, high_val],
                                RandomFunction)

        # multinomial, specified shape
        """
        #infer_shape don't work for multinomial
        n = iscalar()
        pvals = dvector()
        size_val = (7, 3)
        n_val = 6
        pvals_val = [0.2] * 5
        post_r, out = multinomial(rng_R, size=size_val, n=n, pvals=pvals,
                                  ndim=2)

        self._compile_and_check([rng_R, n, pvals], [out],
                                [rng_R_val, n_val, pvals_val],
                                RandomFunction)
        """

        # uniform vector low and high
        low = dvector()
        high = dvector()
        post_r, out = uniform(rng_R, low=low, high=1)
        low_val = [-5, .5, 0, 1]
        self._compile_and_check([rng_R, low], [out], [rng_R_val, low_val],
                          RandomFunction)

        low_val = [.9]
        self._compile_and_check([rng_R, low], [out], [rng_R_val, low_val],
                          RandomFunction)

        post_r, out = uniform(rng_R, low=low, high=high)
        low_val = [-4., -2]
        high_val = [-1, 0]
        self._compile_and_check([rng_R, low, high], [out], [rng_R_val, low_val,
                                high_val], RandomFunction)

        low_val = [-4.]
        high_val = [-1]
        self._compile_and_check([rng_R, low, high], [out], [rng_R_val, low_val,
                                high_val], RandomFunction)

        # uniform broadcasting low and high
        low = dvector()
        high = dcol()
        post_r, out = uniform(rng_R, low=low, high=high)
        low_val = [-5, .5, 0, 1]
        high_val = [[1.]]
        self._compile_and_check([rng_R, low, high], [out], [rng_R_val, low_val,
                                high_val], RandomFunction)

        low_val = [.9]
        high_val = [[1.], [1.1], [1.5]]
        self._compile_and_check([rng_R, low, high], [out], [rng_R_val, low_val,
                                high_val], RandomFunction)

        low_val = [-5, .5, 0, 1]
        high_val = [[1.], [1.1], [1.5]]
        self._compile_and_check([rng_R, low, high], [out], [rng_R_val, low_val,
                                high_val], RandomFunction)

        # uniform with vector slice
        low = dvector()
        high = dvector()
        post_r, out = uniform(rng_R, low=low, high=high)
        low_val = [.1, .2, .3]
        high_val = [1.1, 2.2, 3.3]
        size_val = (3, )
        self._compile_and_check([rng_R, low, high], [out],
                                [rng_R_val, low_val[:-1],
                                high_val[:-1]], RandomFunction)

        # uniform with explicit size and size implicit in parameters
        # NOTE 1: Would it be desirable that size could also be supplied
        # as a Theano variable?
        post_r, out = uniform(rng_R, size=size_val, low=low, high=high)
        self._compile_and_check([rng_R, low, high], [out], [rng_R_val, low_val,
                                high_val], RandomFunction)

        # binomial with vector slice
        n = ivector()
        prob = dvector()
        post_r, out = binomial(rng_R, n=n, p=prob)
        n_val = [1, 2, 3]
        prob_val = [.1, .2, .3]
        size_val = (3, )
        self._compile_and_check([rng_R, n, prob], [out],
                                [rng_R_val, n_val[:-1],
                                prob_val[:-1]], RandomFunction)

        # binomial with explicit size and size implicit in parameters
        # cf. NOTE 1
        post_r, out = binomial(rng_R, n=n, p=prob, size=size_val)
        self._compile_and_check([rng_R, n, prob], [out], [rng_R_val, n_val,
                                prob_val], RandomFunction)

        # normal with vector slice
        avg = dvector()
        std = dvector()
        post_r, out = normal(rng_R, avg=avg, std=std)
        avg_val = [1, 2, 3]
        std_val = [.1, .2, .3]
        size_val = (3, )
        self._compile_and_check([rng_R, avg, std], [out],
                                [rng_R_val, avg_val[:-1],
                                std_val[:-1]], RandomFunction)

        # normal with explicit size and size implicit in parameters
        # cf. NOTE 1
        post_r, out = normal(rng_R, avg=avg, std=std, size=size_val)
        self._compile_and_check([rng_R, avg, std], [out], [rng_R_val, avg_val,
                                std_val], RandomFunction)

        # multinomial with tensor-3 probabilities
        """
        #multinomial infer_shape don't work.
        pvals = dtensor3()
        n = iscalar()
        post_r, out = multinomial(rng_R, n=n, pvals=pvals, size=(1, -1))
        pvals_val = [[[.1, .9], [.2, .8], [.3, .7]]]
        n_val = 9

        self._compile_and_check([rng_R, n, pvals], [out],
                                [rng_R_val, n_val,
                                pvals_val], RandomFunction)

        post_r, out = multinomial(rng_R, n=n, pvals=pvals, size=(10, 1, -1))

        self._compile_and_check([rng_R, n, pvals], [out],
                                [rng_R_val, n_val,
                                pvals_val], RandomFunction)
        """

    def test_pkl(self):
        # Test pickling of RandomFunction.
        # binomial was created by calling RandomFunction on a string,
        # random_integers by calling it on a function.
        rng_r = random_state_type()
        mode = None
        if theano.config.mode in ["DEBUG_MODE", "DebugMode"]:
            mode = 'FAST_COMPILE'
        post_bin_r, bin_sample = binomial(rng_r, (3, 5), 1, .3)
        f = theano.function([rng_r], [post_bin_r, bin_sample], mode=mode)
        pkl_f = pickle.dumps(f)

        post_int_r, int_sample = random_integers(rng_r, (3, 5), -1, 8)
        g = theano.function([rng_r], [post_int_r, int_sample], mode=mode)
        pkl_g = pickle.dumps(g)
        pickle.loads(pkl_g)


if __name__ == '__main__':
    from theano.tests import main
    main("test_raw_random")
