
import unittest
import theano
from theano import tensor as T
from theano import gof
import test_basic as TT
import numpy
from theano.tests import unittest_tools as utt

from theano.tensor.nnet import *


class T_sigmoid(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    def test_elemwise(self):
        utt.verify_grad(sigmoid, [numpy.random.rand(3,4)])

class T_softplus(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    def test_elemwise(self):
        utt.verify_grad(softplus, [numpy.random.rand(3,4)])

class T_Softmax(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    def test0(self):
        def f(a):
            return softmax(a)[:,0]
        utt.verify_grad(f, [numpy.random.rand(3,4)])
    def test1(self):
        def f(a):
            return softmax(a)[:,1]
        utt.verify_grad(f, [numpy.random.rand(3,4)])
    def test2(self):
        def f(a):
            return softmax(a)[:,2]
        utt.verify_grad(f, [numpy.random.rand(3,4)])
    def test3(self):
        def f(a):
            return softmax(a)[:,3]
        utt.verify_grad(f, [numpy.random.rand(3,4)])


class T_SoftmaxWithBias(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    def test0(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,0]
        utt.verify_grad(f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test1(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,1]
        utt.verify_grad(f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test2(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,2]
        utt.verify_grad(f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test3(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:,3]
        utt.verify_grad(f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])

class T_CrossentropySoftmax1Hot(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
    def test0(self):
        y_idx = [0,1,3]
        def f(a, b):
            return crossentropy_softmax_1hot_with_bias(a, b, y_idx)[0]
        utt.verify_grad(f, [numpy.random.rand(3,4),
            numpy.random.rand(4)])
    def test1(self):
        y_idx = [0,1,3]
        def f(a):
            return crossentropy_softmax_1hot(a, y_idx)[0]
        utt.verify_grad(f, [numpy.random.rand(3,4)])

class T_prepend(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
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
        self.rng = numpy.random.RandomState(utt.fetch_seed(666))

    def test0(self):
        A=self.rng.randn(5,5)
        b=numpy.array(range(5),dtype=float)
        x=numpy.linalg.solve(A,b)
        Ax = numpy.dot(A,x)
        are = T.numeric_grad.abs_rel_err(Ax, b)
        self.failUnless(numpy.all(are < 1.0e-5), (are, Ax, b))
        #print A,b
        #print numpy.dot(A,x)


class T_CrossentropyCategorical1Hot(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test_grad(self):
        x = tensor.matrix('x')
        one_of_n = tensor.lvector('one_of_n')

        op = crossentropy_categorical_1hot

        xe = op(x, one_of_n)

        f = theano.function([x, one_of_n], xe)

        xe_val = f(numpy.asarray([[.4, .6, .0], [.1, .8, .1]]), [0,1])

        assert numpy.allclose(xe_val, -numpy.log([.4, .8]))

        def oplike(x):
            return op(x, [0,1])

        tensor.verify_grad(oplike, [numpy.asarray([[.4, .6, .0], [.1, .8, .1]])],
                rng=numpy.random)


    def test_softmax_optimizations(self):
        x = tensor.matrix('x')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot

        xe = op(x, one_of_n)

        env = gof.Env(
                [x, one_of_n],
                [op(softmax(x), one_of_n)])
        assert env.outputs[0].owner.op == op

        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(env)

        assert env.outputs[0].owner.op == crossentropy_softmax_argmax_1hot_with_bias

    def test_softmax_optimizations_w_bias(self):
        x = tensor.matrix('x')
        b = tensor.vector('b')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot

        xe = op(x, one_of_n)

        env = gof.Env(
                [x, b, one_of_n],
                [op(softmax(x+b), one_of_n)])
        assert env.outputs[0].owner.op == op

        print 'BEFORE'
        for node in env.toposort():
            print node.op
        print '----'

        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(env)

        assert len(env.toposort()) == 1

        assert env.outputs[0].owner.op == crossentropy_softmax_argmax_1hot_with_bias


    def test_softmax_grad_optimizations(self):
        x = tensor.matrix('x')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot

        xe = op(softmax(x), one_of_n)

        sum_xe = tensor.sum(xe)
        g_x = tensor.grad(sum_xe, x)
        env = gof.Env(
                [x, one_of_n],
                [g_x])

        print 'BEFORE'
        for node in env.toposort():
            print node.op, node.inputs
        print '----'
        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(env)

        print 'AFTER'
        for node in env.toposort():
            print node.op, node.inputs

        # the function has 9 ops because the dimshuffle and elemwise{second} aren't getting
        # cleaned up as well as we'd like.
        has_cx1hot = False
        has_cx1hotdx = False
        has_softmax = False
        has_softmaxdx = False
        for node in env.toposort():
            if node.op == crossentropy_softmax_argmax_1hot_with_bias:
                has_cx1hot = True
            if node.op == crossentropy_softmax_1hot_with_bias_dx :
                has_cx1hotdx = True
            if node.op == softmax:
                has_softmax = True
            if node.op == softmax_grad:
                has_softmaxdx = True

        assert has_cx1hot
        assert has_cx1hotdx
        assert not has_softmax
        assert not has_softmaxdx

def test_argmax_pushdown():
    x = tensor.dmatrix()

    env = gof.Env(
            [x],
            [tensor.max(softmax(tensor.exp(tensor.tanh(sigmoid(x)))))])

    theano.compile.mode.optdb.query(
            theano.compile.mode.OPT_FAST_RUN).optimize(env)

    #print 'AFTER'
    #for node in env.toposort():
        #print node.op
    assert len(env.toposort()) == 1
    assert env.toposort()[0].op == tensor._max_and_argmax

def test_argmax_pushdown_bias():
    x = tensor.dmatrix()
    b = tensor.dvector()

    env = gof.Env(
            [x,b],
            [tensor.max(softmax_with_bias(x, b))])

    theano.compile.mode.optdb.query(
            theano.compile.mode.OPT_FAST_RUN).optimize(env)

    #print 'AFTER'
    #for node in env.toposort():
        #print node.op
    assert len(env.toposort()) == 3

if __name__ == '__main__':
    unittest.main()
