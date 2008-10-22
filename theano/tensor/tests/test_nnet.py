
import unittest
import theano
from theano import tensor as T
from theano import gof
import test_basic as TT
import numpy

from theano.tensor.nnet import *


class T_sigmoid(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test_elemwise(self):
        TT.verify_grad(self, sigmoid, [numpy.random.rand(3,4)])

class T_softplus(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test_elemwise(self):
        TT.verify_grad(self, softplus, [numpy.random.rand(3,4)])

class T_Softmax(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test0(self):
        def f(a):
            return softmax(a)[:,0]
        TT.verify_grad(self, f, [numpy.random.rand(3,4)])
    def test1(self):
        def f(a):
            return softmax(a)[:,1]
        TT.verify_grad(self, f, [numpy.random.rand(3,4)])
    def test2(self):
        def f(a):
            return softmax(a)[:,2]
        TT.verify_grad(self, f, [numpy.random.rand(3,4)])
    def test3(self):
        def f(a):
            return softmax(a)[:,3]
        TT.verify_grad(self, f, [numpy.random.rand(3,4)])


class T_SoftmaxWithBias(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test0(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,0]
        TT.verify_grad(self, f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test1(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,1]
        TT.verify_grad(self, f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test2(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,2]
        TT.verify_grad(self, f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test3(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,3]
        TT.verify_grad(self, f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])

class T_CrossentropySoftmax1Hot(unittest.TestCase):
    def setUp(self):
        numpy.random.seed(9999)
    def test0(self):
        y_idx = [0,1,3]
        def f(a, b):
            return crossentropy_softmax_1hot_with_bias(a, b, y_idx)[0]
        TT.verify_grad(self, f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test1(self):
        y_idx = [0,1,3]
        def f(a):
            return crossentropy_softmax_1hot(a, y_idx)[0]
        TT.verify_grad(self, f, [numpy.random.rand(3,4)])

class T_prepend(unittest.TestCase):
    def test0(self):
        """basic functionality"""
        x=tensor.matrix('x')
        y=Prepend_scalar_constant_to_each_row(4.)(x)
        f=theano.function([x],[y])
        m=numpy.random.rand(3,5)
        my = f(m)
        self.failUnless(my.shape == (3, 6), my.shape)
        self.failUnless(numpy.all( my[:,0] == 4.0))


class T_prepend(unittest.TestCase):
    def test0(self):
        """basic functionality"""
        x=tensor.matrix('x')
        y=Prepend_scalar_to_each_row()(5.,x)
        f=theano.function([x],y)
        m=numpy.ones((3,5),dtype="float32")
        my = f(m)
        self.failUnless(str(my.dtype) == 'float64')
        self.failUnless(my.shape == (3, 6))
        self.failUnless(numpy.all(my[:,0] == 5.0))

class T_solve(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(666)

    def test0(self):
        A=self.rng.randn(5,5)
        b=numpy.array(range(5),dtype=float)
        x=numpy.linalg.solve(A,b)
        Ax = numpy.dot(A,x)
        are = T.numeric_grad.abs_rel_err(Ax, b)
        self.failUnless(numpy.all(are < 1.0e-5), (are, Ax, b))
        #print A,b
        #print numpy.dot(A,x)


if __name__ == '__main__':
    unittest.main()
