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

        f0 = compile.function([], [rng.uniform((3,))])
        f1 = compile.function([], [rng.uniform((3,))])

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
        f0 = compile.function([template], [rng.uniform_like(template)])

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

if __name__ == '__main__':
    unittest.main()

