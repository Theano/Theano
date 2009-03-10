__docformat__ = "restructuredtext en"

import sys
import unittest
import numpy as N

from theano.tensor.deprecated.rmodule import *

from theano import tensor
from theano import compile, gof


if 0:
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
