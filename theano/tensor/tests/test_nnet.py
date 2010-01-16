
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

        assert str(env.outputs[0].owner.op) == 'OutputGuard'
        assert env.outputs[0].owner.inputs[0].owner.op == crossentropy_softmax_argmax_1hot_with_bias

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

        assert len(env.toposort()) == 2

        assert str(env.outputs[0].owner.op) == 'OutputGuard'
        assert env.outputs[0].owner.inputs[0].owner.op == crossentropy_softmax_argmax_1hot_with_bias


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
    assert len(env.toposort()) == 2 # an output_guard is second
    assert env.toposort()[0].op == tensor._max_and_argmax

def test_argmax_pushdown_bias():
    x = tensor.dmatrix()
    b = tensor.dvector()

    env = gof.Env(
            [x,b],
            [tensor.max(softmax_with_bias(x, b))])

    theano.compile.mode.optdb.query(
            theano.compile.mode.OPT_FAST_RUN).optimize(env)

    print 'AFTER'
    for node in env.toposort():
        print node.op
    assert len(env.toposort()) == 4
    assert isinstance(env.toposort()[0].op, tensor.DimShuffle)
    assert isinstance(env.toposort()[1].op, tensor.Elemwise)
    assert isinstance(env.toposort()[2].op, tensor.MaxAndArgmax)
    assert str(env.toposort()[3].op) == 'OutputGuard'

def test_asymptotic_32():
    """
    This test makes sure that our functions behave sensibly when huge values are present
    """
    for dtype in 'float32', 'float64':
        if dtype == 'float32':
            x = tensor.fmatrix()
            x2 = tensor.fvector()
        else:
            x = tensor.dmatrix()
            x2 = tensor.dvector()
        y = tensor.lvector()

        c = categorical_crossentropy(softmax(x+x2), y)
        f = theano.function([x,y,x2], [c.sum(), tensor.grad(c, x)])
        if 0:
            for i, n in enumerate( f.maker.env.toposort()):
                print i, n

        xval = numpy.zeros((5, 5), dtype=dtype)
        x2val = numpy.zeros(5, dtype=xval.dtype)
        for i in xrange(100):

            cval, gxval =  f(xval, numpy.arange(5), x2val)
            xval -= 100.3 * gxval
            #print cval, gxval
        assert cval == 0 # no problem going to zero error

        #what about when x gets really big?

        xval = numpy.zeros((5, 5), dtype=dtype)
        x2val = numpy.zeros(5, dtype=xval.dtype)
        for i in xrange(100):

            cval, gxval =  f(xval, numpy.arange(5), x2val)
            xval += 100000.3 * gxval
            #print cval, gxval

        assert cval > 61750000
        assert gxval[0,0] == -1.0
        assert gxval[0,1] == 0.25


def test_get_rid_of_advanced_indexing_version_of_xent():
    verbose = 0
    if 0: mode = 'DEBUG_MODE'
    else: mode = 'FAST_RUN'

    rng = numpy.random.RandomState(utt.fetch_seed())

    x_val = rng.randn(3,5)
    b_val = rng.randn(5)
    y_val = numpy.asarray([2,4,1])

    x = T.dmatrix('x')
    b = T.dvector('b')
    y = T.lvector('y')

    def print_graph(func):
        for i, node in enumerate(func.maker.env.toposort()):
            print i, node
        # Last node should be the output
        print i, pprint(node.outputs[0])

    ## Basic case
    expressions = [
            T.sum(-T.log(softmax(x)[T.arange(y.shape[0]), y])),
            -T.sum(T.log(softmax(x)[T.arange(y.shape[0]), y])),
            -T.sum(T.log(softmax(x))[T.arange(y.shape[0]), y]),
            T.sum(-T.log(softmax(x))[T.arange(y.shape[0]), y])]

    for expr in expressions:
        # Verify the optimizer worked on the expressions
        f = theano.function([x,y], expr, mode=mode)
        if verbose: print_graph(f)
        assert len(f.maker.env.toposort()) == 4
        f(x_val, y_val)

        # Also verify the gradient wrt x
        g = theano.function([x,y], T.grad(expr, x), mode=mode)
        if verbose: print_graph(g)
        assert len(g.maker.env.toposort()) == 4
        g(x_val, y_val)


    ## Test that a biased softmax is optimized correctly
    bias_expressions = [
            T.sum(-T.log(softmax(x+b)[T.arange(y.shape[0]), y])),
            -T.sum(T.log(softmax(b+x)[T.arange(y.shape[0]), y])),
            -T.sum(T.log(softmax(x+b))[T.arange(y.shape[0]), y]),
            T.sum(-T.log(softmax(b+x))[T.arange(y.shape[0]), y])]

    for expr in bias_expressions:
        f = theano.function([x,b,y], expr, mode=mode)
        if verbose: print_graph(f)
        assert len(f.maker.env.toposort()) == 2 # [big_op, sum]
        f(x_val, b_val, y_val)

        g = theano.function([x,b,y], T.grad(expr, x), mode=mode)
        if verbose: print_graph(g)
        assert len(g.maker.env.toposort()) == 4
        g(x_val, b_val, y_val)

    ## Test that using "mean" instead of sum works, too
    mean_expressions = [
            T.mean(-T.log(softmax(x)[T.arange(y.shape[0]), y])),
            -T.mean(T.log(softmax(x)[T.arange(y.shape[0]), y])),
            -T.mean(T.log(softmax(x))[T.arange(y.shape[0]), y]),
            T.mean(-T.log(softmax(x))[T.arange(y.shape[0]), y])]

    for expr in mean_expressions:
        f = theano.function([x,y], expr, mode=mode)
        if verbose: print_graph(f)
        assert len(f.maker.env.toposort()) == 7
        f(x_val, y_val)

        g = theano.function([x,y], T.grad(expr, x), mode=mode)
        if verbose: print_graph(g)
        assert len(g.maker.env.toposort()) == 8
        g(x_val, y_val)

    mean_bias_expressions = [
            T.mean(-T.log(softmax(x+b)[T.arange(y.shape[0]), y])),
            -T.mean(T.log(softmax(b+x)[T.arange(y.shape[0]), y])),
            -T.mean(T.log(softmax(x+b))[T.arange(y.shape[0]), y]),
            T.mean(-T.log(softmax(b+x))[T.arange(y.shape[0]), y])]

    for expr in mean_bias_expressions:
        f = theano.function([x,b,y], expr, mode=mode)
        if verbose: print_graph(f)
        assert len(f.maker.env.toposort()) == 5

        g = theano.function([x,b,y], T.grad(expr, x), mode=mode)
        if verbose: print_graph(g)
        assert len(g.maker.env.toposort()) == 8
        g(x_val, b_val, y_val)




    #   hint - call the argmax push-down optimization first too
if __name__ == '__main__':
    unittest.main()
