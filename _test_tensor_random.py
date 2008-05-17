import unittest

from tensor_random import *

import compile

def Uniform(s, n):
    return NumpyGenerator(s, n, numpy.random.RandomState.uniform)

class T_Random(unittest.TestCase):
    def test0(self):

        rng = Uniform(12345, 2)

        r0 = rng((2,3))
        r1 = rng((2,3))

        f0 = compile.function([], [r0])
        f1 = compile.function([], [r1])

        v0 = f0()
        self.failUnless(v0.shape == (2,3))
        self.failUnless(str(v0[0,0]).startswith('0.929616'))
        self.failUnless(str(v0[1,2]).startswith('0.595544'))
        v1 = f1()
        self.failUnless(numpy.all(v0 == v1))
        v1 = f1()
        self.failUnless(numpy.all(v0 != v1))

    def test1(self):
        rng = RandomState(12345)

        f0 = compile.function([], [rng.gen('uniform', (3,))])
        f1 = compile.function([], [rng.gen('uniform', (3,))])

        v0, v1 = f0(), f1()

        self.failUnless(v0.shape == (3,))
        self.failUnless(numpy.all(v0 != v1))

    def test2(self):
        x = tensor.ivector()

        f0 = compile.function([x], [Uniform(123, 1)(x)])
        f1 = compile.function([x], [Uniform(123, 1)(x)])

        v0, v1 = f0([3]), f1([7])

        self.failUnless(v0.shape == (3,))
        self.failUnless(numpy.all(v0 == v1[:3]))

    def test3(self):
        rng = RandomState(12345)
        template = tensor.fmatrix()
        f0 = compile.function([template], [rng.gen_like('uniform', template)])

        v0 = f0(numpy.zeros((2,3)))
        self.failUnless(str(v0[1,2]).startswith('0.595544'))

    def test4(self):
        rng = RandomState(123455)
        template = tensor.fmatrix()
        f0 = compile.function([template],
                [rng.gen_like(('beta',{'a':0.5,'b':0.65}), template)])

        v0 = f0(numpy.zeros((2,3)))
        self.failUnless(v0.shape == (2,3))
        self.failUnless(str(v0[0,0]).startswith('0.013259'))
        self.failUnless(str(v0[1,2]).startswith('0.753368'))

    def test5(self):
        """Test that two NumpyGenerators with the same dist compare equal"""

        rng0 = RandomState(123456)
        rng1 = RandomState(123456)

        d0 = rng0.gen(('beta',{'a':0.5,'b':0.65}), (2,3,4))
        d1 = rng1.gen(('beta',{'a':0.5,'b':0.65}), (2,3,4))

        self.failUnless(d0.owner.op == d1.owner.op)
        self.failUnless(hash(d0.owner.op) == hash(d1.owner.op))

    def test6(self):

        x = tensor.vector()
        u = uniform(9999,x,0.,10.)
        fu = compile.function([x],[u])
        res1 = fu(numpy.zeros((3)))
        res2 = fu(numpy.zeros((3)))
        self.failUnless(str(res1[0]).startswith('8.23389'))
        self.failUnless(str(res2[0]).startswith('5.45926'))

        b = binomial(121212,x,1,0.8)
        fb = compile.function([x],[b])
        res1 = fb(numpy.zeros((10)))
        res2 = fb(numpy.zeros((10)))
        self.failUnless(list(res1) == [1,0,1,1,1,1,1,1,1,1])
        self.failUnless(list(res2) == [1,1,0,1,1,1,0,0,1,1])

if __name__ == '__main__':
    unittest.main()

