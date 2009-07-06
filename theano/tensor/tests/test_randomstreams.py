__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy 
from theano.tensor.randomstreams import RandomStreams, raw_random, getRandomStream, randstream_singleton
from theano.compile import Module, Method, Member

import theano
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



    def test_singleton(self):

        moda = Module()
        moda.randa = getRandomStream(12)
        a = moda.randa.uniform((2,2))
        moda.fn = Method([], a)
        imoda = moda.make()
        imoda.randa.initialize()

        modb = Module()
        modb.randb = getRandomStream()
        b = modb.randb.uniform((2,2))
        modb.fn = Method([], b)
        imodb = modb.make()
        imodb.randb.initialize()

        avals1 = imoda.fn()
        bvals1 = imodb.fn()
       
        modc = Module()
        modc.randc = getRandomStream(12, force_new=True)
        a2 = modc.randc.uniform((2,2))
        b2 = modc.randc.uniform((2,2))
        modc.fna = Method([], a2)
        modc.fnb = Method([], b2)
        imodc = modc.make()
        imodc.randc.initialize()

        avals2 = imodc.fna()
        bvals2 = imodc.fnb()

        assert (avals1 == avals2).all()
        assert (bvals1 == bvals2).all()

if __name__ == '__main__':
    from theano.tests import main
    main("test_randomstreams")
