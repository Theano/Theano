__docformat__ = "restructuredtext en"
import sys
import unittest
import numpy as N
from theano.tests import unittest_tools

from theano.tensor.raw_random import *
from theano.tensor import raw_random 

from theano import tensor

from theano import compile, gof

class T_random_function(unittest.TestCase):
    def test_basic_usage(self):
        rf = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector)
        assert not rf.inplace
        assert getattr(rf, 'destroy_map', {}) == {}

        rng_R = random_state_type()

        post_r, out = rf(rng_R, (4,))

        assert out.type == tensor.dvector

        f = compile.function([rng_R], out)

        rng_state0 = numpy.random.RandomState(55)

        f_0 = f(rng_state0)
        f_1 = f(rng_state0)

        assert numpy.all(f_0 == f_1)

    def test_inplace_norun(self):
        rf = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector, inplace=True)
        assert rf.inplace
        assert getattr(rf, 'destroy_map', {}) != {}

    def test_args(self):
        """Test that arguments to RandomFunction are honored"""
        rf2 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector)
        rf4 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector, inplace=True)
        rng_R = random_state_type()

        # use make_node to override some of the self.args
        post_r2, out2 = rf2(rng_R, (4,), -2, 2)
        post_r2_4, out2_4 = rf2(rng_R, (4,), -4.0, 2)
        post_r2_4_4, out2_4_4 = rf2(rng_R, (4,), -4.0, 4.0)
        post_r4, out4 = rf4(rng_R, (4,), -4, 4)

        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r4, mutable=True)], 
                [out2, out4, out2_4, out2_4_4], 
                accept_inplace=True)

        f2, f4, f2_4, f2_4_4 = f()
        f2b, f4b, f2_4b, f2_4_4b = f()

        assert numpy.allclose(f2*2, f4)
        assert numpy.allclose(f2_4_4, f4)
        assert not numpy.allclose(f4, f4b)

    def test_inplace_optimization(self):
        """Test that FAST_RUN includes the random_make_inplace optimization"""
        #inplace = False
        rf2 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector)
        rng_R = random_state_type()

        # use make_node to override some of the self.args
        post_r2, out2 = rf2(rng_R, (4,))

        f = compile.function(
                [compile.In(rng_R, 
                    value=numpy.random.RandomState(55),
                    update=post_r2, 
                    mutable=True)], 
                out2,
                mode='FAST_RUN') #DEBUG_MODE can't pass the id-based test below

        # test that the RandomState object stays the same from function call to function call,
        # but that the values returned change from call to call.

        id0 = id(f[rng_R])
        val0 = f()
        assert id0 == id(f[rng_R])
        val1 = f()
        assert id0 == id(f[rng_R])

        assert not numpy.allclose(val0, val1)

    def test_random_function_ndim(self):
        """Test that random_function helper function accepts argument ndim"""
        rng_R = random_state_type()

        # ndim is an optional argument indicating the length of the 'shape'
        # ndim not specified, OK
        post_out4,      out4    =   uniform(rng_R, (4,))

        # ndim specified, consistent with shape, OK
        post_out1_4,    out1_4  =   uniform(rng_R, (4,), ndim=1)
        post_out2_4_4,  out2_4_4=   uniform(rng_R, (4, 4), ndim=2)

        # ndim specified, but not compatible with shape
        self.assertRaises(ValueError, uniform, rng_R, (4,), ndim=2)

        f_ok = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_out2_4_4, mutable=True)],
                [out4, out1_4, out2_4_4],
                accept_inplace=True)

        # The correct cases should execute properly
        o4, o1_4, o2_4_4 = f_ok()

        # Check the sanity of the answers
        self.assertTrue(numpy.allclose(o4, o1_4))
        self.assertTrue(numpy.allclose(o4, o2_4_4[0]))

    def test_random_function_ndim_added(self):
        """Test that random_function helper function accepts ndim_added as keyword argument"""
        # If using numpy's uniform distribution, ndim_added should be 0,
        # because the shape provided as argument is the output shape.
        # Specifying a different ndim_added will change the Op's output ndim,
        # so numpy.uniform will produce a result of incorrect shape,
        # and a ValueError should be raised.
        def ndim_added_deco(ndim_added):
            def randomfunction(random_state, size=(), low=0.0, high=0.0, ndim=None):
                ndim, size = raw_random._infer_ndim(ndim, size)
                op = RandomFunction('uniform', 
                        tensor.TensorType(dtype = 'float64', broadcastable =
                            (False,)*(ndim+ndim_added)),
                        ndim_added=ndim_added)
                return op(random_state, size, low, high)
            return randomfunction

        uni_1 = ndim_added_deco(1)
        uni_0 = ndim_added_deco(0)
        uni_m1 = ndim_added_deco(-1)

        #uni_1 = random_function(numpy.random.RandomState.uniform, 'float64', -2.0, 2.0, ndim_added=1)
        #uni_0 = random_function(numpy.random.RandomState.uniform, 'float64', -2.0, 2.0, ndim_added=0)
        #uni_m1 = random_function(numpy.random.RandomState.uniform, 'float64', -2.0, 2.0, ndim_added=-1)
        rng_R = random_state_type()

        p_uni11, uni11 = uni_1(rng_R, size=(4,))
        p_uni12, uni12 = uni_1(rng_R, size=(3,4))
        p_uni01, uni01 = uni_0(rng_R, size=(4,))
        p_uni02, uni02 = uni_0(rng_R, size=(3,4))
        p_unim11, unim11 = uni_m1(rng_R, size=(4,))
        p_unim12, unim12 = uni_m1(rng_R, size=(3,4))

        self.assertEqual(uni11.ndim, 2)
        self.assertEqual(uni12.ndim, 3)
        self.assertEqual(uni01.ndim, 1)
        self.assertEqual(uni02.ndim, 2)
        self.assertEqual(unim11.ndim, 0)
        self.assertEqual(unim12.ndim, 1)

        f11 = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=p_uni11, mutable=True)],
                [uni11], accept_inplace=True)
        f12 = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=p_uni12, mutable=True)],
                [uni12], accept_inplace=True)
        fm11 = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=p_unim11, mutable=True)],
                [unim11], accept_inplace=True)
        fm12 = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=p_unim12, mutable=True)],
                [unim12], accept_inplace=True)
        f0 = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=p_uni02, mutable=True)],
                [uni01, uni02], accept_inplace=True)
        self.assertRaises(ValueError, f11)
        self.assertRaises(ValueError, f12)
        self.assertRaises(ValueError, fm11)
        self.assertRaises(ValueError, fm12)
        u01, u02 = f0()
        print u01
        print u02
        self.assertTrue(numpy.allclose(u01, u02[0]))

    def test_uniform(self):
        """Test that raw_random.uniform generates the same results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters
        post_r, out = uniform(rng_R, (4,), -2.0, 2.0)

        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(55)
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.uniform(-2.0, 2.0, size=(4,))
        numpy_val1 = numpy_rng.uniform(-2.0, 2.0, size=(4,))
        print val0
        print numpy_val0
        print val1
        print numpy_val1
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_binomial(self):
        """Test that raw_random.binomial generates the same results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters, and larger dimensions because of
        # the integer nature of the result
        post_r, bin = binomial(rng_R, (7,12), 5, 0.8)

        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r, mutable=True)],
                [bin], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(55)
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.binomial(5, 0.8, size=(7,12))
        numpy_val1 = numpy_rng.binomial(5, 0.8, size=(7,12))
        print val0
        print numpy_val0
        print val1
        print numpy_val1
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

    def test_normal(self):
        """Test that raw_random.normal generates the same results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters
        post_r, out = normal(rng_R, (2,3), 4.0, 2.0)

        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(55)
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.normal(4.0, 2.0, size=(2,3))
        numpy_val1 = numpy_rng.normal(4.0, 2.0, size=(2,3))
        print val0
        print numpy_val0
        print val1
        print numpy_val1
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_random_integers(self):
        """Test that raw_random.random_integers generates the same results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        # Use non-default parameters, and larger dimensions because of
        # the integer nature of the result
        post_r, out = random_integers(rng_R, (11,8), -3, 16)

        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(55)
        val0 = f()
        val1 = f()
        numpy_val0 = numpy_rng.random_integers(-3, 16, size=(11,8))
        numpy_val1 = numpy_rng.random_integers(-3, 16, size=(11,8))
        print val0
        print numpy_val0
        print val1
        print numpy_val1
        self.assertTrue(numpy.allclose(val0, numpy_val0))
        self.assertTrue(numpy.allclose(val1, numpy_val1))

    def test_permutation_helper(self):
        """Test that raw_random.permutation_helper generates the same results as numpy,
        and that the 'ndim_added' keyword behaves correctly."""
        # permutation_helper needs "ndim_added=1", because its output
        # is one dimension more than its "shape" argument (and there's
        # no way to determine that automatically).
        # Check the working case, over two calls to see if the random
        # state is correctly updated.
        rf = RandomFunction(permutation_helper, tensor.imatrix, 8, ndim_added=1)
        rng_R = random_state_type()
        post_r, out = rf(rng_R, (7,), 8)

        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(55)
        val0 = f()
        val1 = f()
        # numpy_rng.permutation outputs one vector at a time,
        # so we call it iteratively to generate all the samples.
        numpy_val0 = numpy.asarray([numpy_rng.permutation(8) for i in range(7)])
        numpy_val1 = numpy.asarray([numpy_rng.permutation(8) for i in range(7)])
        print val0
        print numpy_val0
        print val1
        print numpy_val1
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

        # This call lacks "ndim_added=1", so ndim_added defaults to 0.
        # A ValueError should be raised.
        rf0 = RandomFunction(permutation_helper, tensor.imatrix, 8)
        post_r0, out0 = rf0(rng_R, (7,), 8)
        f0 = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r0, mutable=True)],
                [out0], accept_inplace=True)
        self.assertRaises(ValueError, f0)

        # Here, ndim_added is 2 instead of 1. A ValueError should be raised.
        rf2 = RandomFunction(permutation_helper, tensor.imatrix, 8, ndim_added=2)
        post_r2, out2 = rf2(rng_R, (7,), 8)
        f2 = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r2, mutable=True)],
                [out2], accept_inplace=True)
        self.assertRaises(ValueError, f2)

    def test_permutation(self):
        """Test that raw_random.permutation generates the same results as numpy."""
        rng_R = random_state_type()
        post_r, out = permutation(rng_R, size=(9,), n=6)
        print 'OUT NDIM', out.ndim
        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(55)
        # Check over two calls to see if the random state is correctly updated.
        # numpy_rng.permutation outputs one vector at a time,
        # so we call it iteratively to generate all the samples.
        val0 = f()
        val1 = f()
        numpy_val0 = numpy.asarray([numpy_rng.permutation(6) for i in range(9)])
        numpy_val1 = numpy.asarray([numpy_rng.permutation(6) for i in range(9)])
        print val0
        print numpy_val0
        print val1
        print numpy_val1
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

    def test_multinomial(self):
        """Test that raw_random.multinomial generates the same results as numpy."""
        # Check over two calls to see if the random state is correctly updated.
        rng_R = random_state_type()
        post_r, out = multinomial(rng_R, (7,3), 6, [0.2]*5)

        f = compile.function(
                [compile.In(rng_R, value=numpy.random.RandomState(55), update=post_r, mutable=True)],
                [out], accept_inplace=True)

        numpy_rng = numpy.random.RandomState(55)
        val0, = f()
        val1, = f()
        numpy_val0 = numpy_rng.multinomial(6, [0.2]*5, (7,3))
        numpy_val1 = numpy_rng.multinomial(6, [0.2]*5, (7,3))
        print val0
        print numpy_val0
        print val1
        print numpy_val1
        self.assertTrue(numpy.all(val0 == numpy_val0))
        self.assertTrue(numpy.all(val1 == numpy_val1))

        self.assertTrue(val0.shape == (7,3,5))
        self.assertTrue(val1.shape == (7,3,5))


    def test_symbolic_shape(self):
        rng_R = random_state_type()

        shape = tensor.lvector()

        post_r, out = uniform(rng_R, shape, ndim=2)

        f = compile.function([rng_R, shape], out)

        rng_state0 = numpy.random.RandomState(55)

        assert f(rng_state0, [2,3]).shape == (2,3)
        assert f(rng_state0, [4,8]).shape == (4,8)

        self.assertRaises(ValueError, f, rng_state0, [4])
        self.assertRaises(ValueError, f, rng_state0, [4,3,4,5])

if __name__ == '__main__':
    from theano.tests import main
    main("test_raw_random")
