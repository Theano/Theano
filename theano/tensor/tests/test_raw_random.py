__docformat__ = "restructuredtext en"
import sys
import unittest
import numpy as N
from theano.tests import unittest_tools

from theano.tensor.raw_random import *

from theano import tensor

from theano import compile, gof

class T_random_function(unittest.TestCase):
    def test_basic_usage(self):
        rf = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector, -2.0, 2.0)
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
        rf = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector, -2.0, 2.0,
                inplace=True)
        assert rf.inplace
        assert getattr(rf, 'destroy_map', {}) != {}

    def test_args(self):
        """Test that arguments to RandomFunction are honored"""
        rf2 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector, -2.0, 2.0)
        rf4 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector, -4.0, 4.0,
                inplace=True)
        rng_R = random_state_type()

        # use make_node to override some of the self.args
        post_r2, out2 = rf2(rng_R, (4,))
        post_r2_4, out2_4 = rf2(rng_R, (4,), -4.0)
        post_r2_4_4, out2_4_4 = rf2(rng_R, (4,), -4.0, 4.0)
        post_r4, out4 = rf4(rng_R, (4,))

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
        rf2 = RandomFunction(numpy.random.RandomState.uniform, tensor.dvector, -2.0, 2.0)
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


if __name__ == '__main__':
    from theano.tests import main
    main("test_raw_random")
