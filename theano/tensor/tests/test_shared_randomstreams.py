__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy 

from theano.tensor import raw_random
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function

from theano import tensor
from theano import compile, gof


class T_RandomStreams(unittest.TestCase):

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


if __name__ == '__main__':
    from theano.tests import main
    main("test_randomstreams")
