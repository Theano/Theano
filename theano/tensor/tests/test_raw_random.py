## TODO: REDO THESE TESTS

import unittest
import numpy as N

from theano.tensor.raw_random import *

from theano import tensor

from theano import compile, gof

def test_state_propagation():
    x = tensor.vector()
    rk = RandomKit('rk', 1000)
    f = compile.function([x, (rk, [gof.Container(r = gof.generic, storage = [123], name='bla')])], rk.binomial(tensor.shape(x)), mode='FAST_COMPILE')
    f['rk'] = 9873456
    
    rvals = [f([1,2,3,4,6, 7, 8]) for i in xrange(5)]
    print rvals
    for i in xrange(5-1):
        for j in xrange(i+1, 5):
            assert not N.all(rvals[i] == rvals[j])

def test_B():
    """Test that random numbers change from call to call!
    
    Also, make sure that the seeding strategy doesn't change without failing a test.

    Random numbers can't be too random or experiments aren't repeatable.  Email theano-dev
    before updating the `rvals` in this test.
    """
    class B(RModule):
        def __init__(self):
            super(B, self).__init__(self)

            self.x = compile.Member(tensor.dvector())
            self.r = self.random.uniform(tensor.shape(self.x))

            self.f = compile.Method([self.x], self.r)
    class E(RModule):
        def __init__(self):
            super(E, self).__init__(self)
            self.b = B()
            self.f = compile.Method([self.b.x], self.b.r)

    b = E()
    m = b.make(mode='FAST_COMPILE')

    m.seed(1000)
    #print m.f(N.ones(5))
    #print m.f(N.ones(5))
    #print m.f(N.ones(5))
    rvals = ["0.0655889727823 0.566937256035 0.486897708861 0.939594224804 0.731948448071",
        "0.407174827663 0.450046718267 0.454825370073 0.874814293401 0.828759935744",
        "0.573194634066 0.746015418896 0.864696705461 0.8405810785 0.540268740918",
        "0.924477905238 0.96687901023 0.306490321744 0.654349923901 0.789402591813",
        "0.513182053208 0.0426565286449 0.0723651478047 0.454308519009 0.86151064181"]


    for i in xrange(5):
        s = " ".join([str(n) for n in m.f(N.ones(5))])
        assert s == rvals[i]


if __name__ == '__main__':
    unittest.main()

