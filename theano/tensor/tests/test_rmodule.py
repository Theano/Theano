__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy as N

from theano.tensor.rmodule import *

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
        rng = numpy.random.RandomState(rng_seed)

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
        rng = numpy.random.RandomState(rng_seed)

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
        rng = numpy.random.RandomState(rng_seed)

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




class T_test_module(unittest.TestCase):
    def test_state_propagation(self):
        if 1:
            print >> sys.stderr, "RModule deprecated"
        else:
            x = tensor.vector()
            rk = RandomKit('rk', 1000)
            f = compile.function([x, (rk, [gof.Container(r = gof.generic, storage = [123], name='bla')])], rk.binomial(tensor.shape(x)))
            print "RK", rk.value
            f['rk'] = 9873456
            print "RK", rk.value
        
            rvals = [f([1,2,3,4,6, 7, 8]) for i in xrange(5)]
            print rvals
            for i in xrange(5-1):
                for j in xrange(i+1, 5):
                    assert not N.all(rvals[i] == rvals[j])

    def test_B(self):
        """Test that random numbers change from call to call!
        
        Also, make sure that the seeding strategy doesn't change without failing a test.
        
        Random numbers can't be too random or experiments aren't repeatable.  Email theano-dev
        before updating the `rvals` in this test.
        """
        class B(RModule):
            def __init__(self):
                super(B, self).__init__()
                
                self.x = compile.Member(tensor.dvector())
                self.r = self.random.uniform(tensor.shape(self.x))
                
                self.f = compile.Method([self.x], self.r)
        class E(RModule):
            def __init__(self):
                super(E, self).__init__()
                self.b = B()
                self.f = compile.Method([self.b.x], self.b.r)

        b = E()
        m = b.make()
        
        m.seed(1000)
    #print m.f(N.ones(5))
    #print m.f(N.ones(5))
    #print m.f(N.ones(5))
        rvals = ["0.74802375876 0.872308123517 0.294830748897 0.803123780003 0.6321109955",
                 "0.00168744844365 0.278638315678 0.725436793755 0.7788480779 0.629885140994",
                 "0.545561221664 0.0992011009108 0.847112593242 0.188015424144 0.158046201298",
                 "0.054382248842 0.563459168529 0.192757276954 0.360455221883 0.174805216702",
                 "0.961942907777 0.49657319422 0.0316111492826 0.0915054717012 0.195877184515"]

        for i in xrange(5):
            s = " ".join([str(n) for n in m.f(N.ones(5))])
            print s
            assert s == rvals[i]

if __name__ == '__main__':
    from theano.tests import main
    main("test_rmodule")
