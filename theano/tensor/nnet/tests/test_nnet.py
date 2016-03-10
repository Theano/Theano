from __future__ import print_function
import unittest

import numpy
from nose.plugins.skip import SkipTest
from six.moves import xrange

import theano
from theano import config
from theano import tensor as T
from theano import tensor
from theano import gof
from theano.tests import unittest_tools as utt
from theano import printing
from theano.tensor.nnet import (categorical_crossentropy,
                                crossentropy_categorical_1hot,
                                crossentropy_softmax_1hot,
                                crossentropy_softmax_1hot_with_bias,
                                crossentropy_softmax_1hot_with_bias_dx,
                                crossentropy_softmax_argmax_1hot_with_bias,
                                CrossentropySoftmax1HotWithBiasDx,
                                CrossentropySoftmaxArgmax1HotWithBias,
                                CrossentropyCategorical1Hot,
                                CrossentropyCategorical1HotGrad,
                                sigmoid, softplus, Softmax, softmax,
                                softmax_op, softmax_graph, SoftmaxWithBias,
                                softmax_with_bias, LogSoftmax, logsoftmax_op,
                                softmax_grad, SoftmaxGrad,
                                Prepend_scalar_constant_to_each_row,
                                Prepend_scalar_to_each_row,
                                relu,
                                h_softmax,
                                elu)
from theano.tensor import matrix, vector, lvector, scalar


class T_sigmoid(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(sigmoid, [numpy.random.rand(3, 4)])


class T_softplus(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def test_elemwise(self):
        utt.verify_grad(softplus, [numpy.random.rand(3, 4)])


class T_Softmax(utt.InferShapeTester):

    def test0(self):
        def f(a):
            return softmax_op(a)[:, 0]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test1(self):
        def f(a):
            return softmax_op(a)[:, 1]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test2(self):
        def f(a):
            return softmax_op(a)[:, 2]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test3(self):
        def f(a):
            return softmax_op(a)[:, 3]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test_infer_shape(self):
        admat = matrix()
        admat_val = numpy.random.rand(3, 4).astype(config.floatX)
        self._compile_and_check([admat], [Softmax()(admat)],
                                [admat_val], Softmax)

    def test_vector(self):
        x = T.vector()
        f = theano.function([x], softmax_op(x))

        xv = numpy.random.randn(6).astype(config.floatX)
        assert numpy.allclose(f(xv), numpy.exp(xv) / numpy.exp(xv).sum())

    def test_vector_grad(self):
        def f(a):
            return softmax_op(a)
        utt.verify_grad(f, [numpy.random.rand(4)])


class T_SoftmaxWithBias(utt.InferShapeTester):

    def test0(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:, 0]
        utt.verify_grad(f, [numpy.random.rand(3, 4),
                        numpy.random.rand(4)])

    def test1(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:, 1]
        utt.verify_grad(f, [numpy.random.rand(3, 4),
                        numpy.random.rand(4)])

    def test2(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:, 2]
        utt.verify_grad(f, [numpy.random.rand(3, 4),
                        numpy.random.rand(4)])

    def test3(self):
        def f(a, b):
            return softmax_with_bias(a, b)[:, 3]
        utt.verify_grad(f, [numpy.random.rand(3, 4),
                        numpy.random.rand(4)])

    def test_broadcast(self):
        # test that we don't raise an error during optimization for no good
        # reason as softmax_with_bias don't support correctly some/all
        # broadcasted inputs pattern
        initial_W = numpy.asarray([[0.1, 0.1, 0.1],
                                   [0.1, 0.1, 0.1],
                                   [0.1, 0.1, 0.1]],
                                  dtype=theano.config.floatX)
        W = theano.shared(value=initial_W, name='W')
        vbias = theano.shared(value=0.1, name='vbias')  # 0.01
        hid = T.vector('hid')
        f = theano.function([hid],
                            T.nnet.softmax_op(T.dot(hid, W.T) + vbias))
        ops = [node.op for node in f.maker.fgraph.toposort()]
        assert softmax_with_bias not in ops
        assert softmax_op in ops

        f([0, 1, 0])
        # print f.maker.fgraph.toposort()

    def test_softmax_with_bias_trace(self):
        a = theano.shared(
            numpy.random.randn(3).astype(config.floatX))
        b = theano.shared(numpy.float32(numpy.random.randn()))
        sm = T.nnet.softmax(a + b)
        f = theano.function([], sm)
        self.assertTrue(hasattr(f.maker.fgraph.outputs[0].tag, 'trace'))
        print('f.maker.fgraph.outputs[0]: {0}'.format(f.maker.fgraph.outputs[0], ))

    def test_infer_shape(self):
        admat = matrix()
        advec = vector()
        admat_val = numpy.random.rand(3, 4).astype(config.floatX)
        advec_val = numpy.random.rand(4).astype(config.floatX)
        self._compile_and_check([admat, advec],
                                [SoftmaxWithBias()(admat, advec)],
                                [admat_val, advec_val], SoftmaxWithBias)


class T_LogSoftmax(utt.InferShapeTester):

    def test0(self):
        def f(a):
            return logsoftmax_op(a)[:, 0]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test1(self):
        def f(a):
            return logsoftmax_op(a)[:, 1]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test2(self):
        def f(a):
            return logsoftmax_op(a)[:, 2]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test3(self):
        def f(a):
            return logsoftmax_op(a)[:, 3]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test_matrix(self):
        def f(a):
            return logsoftmax_op(a)
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test_vector(self):
        x = T.vector()
        f = theano.function([x], logsoftmax_op(x))

        xv = numpy.random.randn(6).astype(config.floatX)
        assert numpy.allclose(f(xv),
                              numpy.log(numpy.exp(xv) / numpy.exp(xv).sum()))

    def test_vector_grad(self):
        def f(a):
            return logsoftmax_op(a)
        utt.verify_grad(f, [numpy.random.rand(4)])

    def test_allclose(self):
        m = theano.config.mode
        m = theano.compile.get_mode(m)
        m.check_isfinite = False
        x, y = tensor.matrices('xy')
        # regular softmax and crossentropy
        sm = tensor.nnet.softmax(x)
        cm = tensor.nnet.categorical_crossentropy(sm, y)

        # numerically stable log-softmax with crossentropy
        logsm = tensor.nnet.logsoftmax(x)
        sm2 = tensor.exp(logsm)  # just used to show equivalence with sm
        cm2 = -tensor.sum(y * logsm, axis=1)
        grad = tensor.grad(cm2.mean(), x)

        # create some inputs into a softmax that are large and labels
        a = numpy.exp(10 * numpy.random.rand(5, 10).astype(theano.config.floatX))
        # create some one-hot coded labels
        b = numpy.eye(5, 10).astype(theano.config.floatX)

        # show equivalence of softmax and exponentiated numerically stable
        # log-softmax
        f1 = theano.function([x], [sm, sm2])
        sm_, sm2_ = f1(a)
        utt.assert_allclose(sm_, sm2_)

        # now show that the two versions result in the same crossentropy cost
        # this indicates that the forward function does provide some numerical
        # stability
        f2 = theano.function([x, y], [cm, cm2], mode=m)
        cm_, cm2_ = f2(a, b)
        utt.assert_allclose(cm_, cm2_)

        # now, show that in the standard softmax case the gradients blow up
        # while in the log-softmax case they don't
        f3 = theano.function([x, y], [grad])
        grad_ = f3(a, b)
        assert numpy.all(numpy.isnan(grad_) == False)

    def test_isclose(self):
        def f(a):
            return logsoftmax_op(a)

    def test_local_softmax_optimization(self):
        """Test the Logsoftmax substitution

        Check that Log(Softmax(x)) is substituted with Logsoftmax(x). Note that
        only the forward pass is checked (i.e., doesn't check the gradient)
        """
        x, y = tensor.matrices('xy')
        sm = tensor.nnet.softmax(x)
        logsm = tensor.log(sm)
        f = theano.function([x], logsm)
        self.assertTrue(hasattr(f.maker.fgraph.outputs[0].tag, 'trace'))
        assert isinstance(f.maker.fgraph.outputs[0].owner.op,
                          theano.tensor.nnet.nnet.LogSoftmax)

    def test_local_softmax_grad_optimization_and_big_input(self):
        """Test the Logsoftmax's grad substitution.

        Check that Log(Softmax(x))'s grad is substituted with Logsoftmax(x)'s
        grad and that the new operation does not explode for big inputs.
        Note that only the grad is checked.
        """
        m = theano.config.mode
        m = theano.compile.get_mode(m)
        m.check_isfinite = False
        # some inputs that are large to make the gradient explode in the non
        # optimized case
        a = numpy.exp(10 * numpy.random.rand(5, 10).astype(theano.config.floatX))

        def myfunc(x):
            sm = tensor.nnet.softmax(x)
            logsm = tensor.log(sm)
            return logsm
        # We set step to 0.1 because for big values we need a big epsilon
        utt.verify_grad(myfunc, [a], eps=0.1, mode=m)
        f = theano.function([], myfunc(a))
        self.assertTrue(hasattr(f.maker.fgraph.outputs[0].tag, 'trace'))


class T_SoftmaxGrad(utt.InferShapeTester):

    def test_infer_shape(self):
        admat = matrix()
        bdmat = matrix()
        admat_val = numpy.random.rand(3, 4).astype(config.floatX)
        bdmat_val = numpy.random.rand(3, 4).astype(config.floatX)
        self._compile_and_check([admat, bdmat], [SoftmaxGrad()(admat, bdmat)],
                                [admat_val, bdmat_val], SoftmaxGrad)


class T_CrossentropySoftmax1Hot(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def test0(self):
        y_idx = [0, 1, 3]

        def f(a, b):
            return crossentropy_softmax_1hot_with_bias(a, b, y_idx)[0]
        utt.verify_grad(f, [numpy.random.rand(3, 4),
                            numpy.random.rand(4)])

    def test1(self):
        y_idx = [0, 1, 3]

        def f(a):
            return crossentropy_softmax_1hot(a, y_idx)[0]
        utt.verify_grad(f, [numpy.random.rand(3, 4)])

    def test_vector(self):
        y_idx = [3]

        def f(a):
            return crossentropy_softmax_1hot(T.shape_padleft(a), y_idx)[0]
        utt.verify_grad(f, [numpy.random.rand(4)])

    def test_vectors(self):
        y_idx = [3]

        def f(a, b):
            return crossentropy_softmax_1hot(T.shape_padleft(a) + b, y_idx)[0]
        utt.verify_grad(f, [numpy.random.rand(4), numpy.random.rand(4)])


class T_CrossentropySoftmax1HotWithBiasDx(utt.InferShapeTester):

    def test0(self):
        def ff(class_dtype):
            def f(sm):
                # Class indices
                y = numpy.random.randint(low=0, high=5, size=10).astype(class_dtype)
                return theano.tensor.nnet.crossentropy_softmax_1hot_with_bias_dx(
                    numpy.random.rand(10),  # Gradient w.r.t. NLL.
                    sm,                     # Softmax output.
                    y)
            return f
        # Build a random softmax output whose rows sum to 1.
        softmax_output = numpy.random.rand(10, 5)
        softmax_output /= softmax_output.sum(axis=1).reshape(10, 1)
        for dtype in ['uint8', 'int8', 'uint64', 'int64']:
            utt.verify_grad(ff(dtype), [softmax_output])

    def test1(self):
        rng = numpy.random.RandomState(utt.fetch_seed())
        softmax_output = rng.rand(10, 5)
        softmax_output /= softmax_output.sum(axis=1).reshape(10, 1)

        def f(dy):
            return (theano.tensor.nnet.crossentropy_softmax_1hot_with_bias_dx(
                dy,
                softmax_output,
                rng.randint(low=0, high=5, size=10)))
        utt.verify_grad(f, [rng.rand(10)])

    def test_infer_shape(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = numpy.random.RandomState(utt.fetch_seed())
        admat_val = rng.rand(10, 5).astype(config.floatX)
        admat_val /= admat_val.sum(axis=1).reshape(10, 1)
        advec_val = rng.rand(10).astype(config.floatX)
        alvec_val = rng.randint(low=0, high=5, size=10)
        self._compile_and_check([advec, admat, alvec],
                    [CrossentropySoftmax1HotWithBiasDx()(advec, admat, alvec)],
                    [advec_val, admat_val, alvec_val],
                    CrossentropySoftmax1HotWithBiasDx)

    def test_neg_idx(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = numpy.random.RandomState(utt.fetch_seed())
        admat_val = rng.rand(10, 5).astype(config.floatX)
        admat_val /= admat_val.sum(axis=1).reshape(10, 1)
        advec_val = rng.rand(10).astype(config.floatX)
        alvec_val = rng.randint(low=0, high=5, size=10)
        alvec_val[1] = -1
        out = CrossentropySoftmax1HotWithBiasDx()(advec, admat, alvec)
        f = theano.function([advec, admat, alvec], out)
        self.assertRaises(ValueError, f, advec_val, admat_val, alvec_val)


class T_CrossentropySoftmaxArgmax1HotWithBias(utt.InferShapeTester):

    def setUp(self):
        super(T_CrossentropySoftmaxArgmax1HotWithBias, self).setUp()
        self.op = theano.tensor.nnet.crossentropy_softmax_argmax_1hot_with_bias

    def test0(self):
        n_classes = 5
        n_samples = 3

        # First test gradient when getting a gradient on the NLL output.
        def grad_on_nll_dtype(dtype):
            def grad_on_nll(x, b):
                y_idx = numpy.random.randint(low=0, high=n_classes, size=n_samples).astype(dtype)
                return self.op(x, b, y_idx=y_idx)[0]
            return grad_on_nll
        for dtype in ['uint8', 'int8', 'uint64', 'int64']:
            utt.verify_grad(grad_on_nll_dtype(dtype),
                            [numpy.random.rand(n_samples, n_classes),
                             numpy.random.rand(n_classes)])

        # Then test gradient when getting a gradient on the softmax output.
        def grad_on_softmax(x, b):
            return self.op(x, b, y_idx=numpy.random.randint(
                low=0, high=n_classes, size=n_samples))[1]
        utt.verify_grad(grad_on_softmax,
                [numpy.random.rand(n_samples, n_classes),
                    numpy.random.rand(n_classes)])

    def test_infer_shape(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = numpy.random.RandomState(utt.fetch_seed())
        admat_val = rng.rand(3, 5).astype(config.floatX)
        advec_val = rng.rand(5).astype(config.floatX)
        alvec_val = rng.randint(low=0, high=5, size=3)
        self._compile_and_check([admat, advec, alvec],
                CrossentropySoftmaxArgmax1HotWithBias()(admat, advec, alvec),
                [admat_val, advec_val, alvec_val],
                CrossentropySoftmaxArgmax1HotWithBias)

    def test_neg_idx(self):
        admat = matrix()
        advec = vector()
        alvec = lvector()
        rng = numpy.random.RandomState(utt.fetch_seed())
        admat_val = rng.rand(3, 5).astype(config.floatX)
        advec_val = rng.rand(5).astype(config.floatX)
        alvec_val = rng.randint(low=0, high=5, size=3)
        alvec_val[1] = -1
        out = CrossentropySoftmaxArgmax1HotWithBias()(admat, advec, alvec)
        f = theano.function([admat, advec, alvec], out)
        self.assertRaises(ValueError, f, admat_val, advec_val, alvec_val)


class T_prepend(utt.InferShapeTester):

    def test0(self):
        x = tensor.matrix('x')
        y = Prepend_scalar_constant_to_each_row(4.)(x)
        f = theano.function([x], y)
        m = numpy.random.rand(3, 5).astype(config.floatX)
        my = f(m)
        self.assertTrue(my.shape == (3, 6), my.shape)
        self.assertTrue(numpy.all(my[:, 0] == 4.0))

    def test1(self):
        "basic functionality"
        x = tensor.matrix('x')
        y = Prepend_scalar_to_each_row()(5., x)
        f = theano.function([x], y)
        m = numpy.ones((3, 5), dtype="float32")
        my = f(m)
        self.assertTrue(my.shape == (3, 6))
        self.assertTrue(numpy.all(my[:, 0] == 5.0))

    def test_infer_shape(self):
        admat = matrix()
        adscal = scalar()
        rng = numpy.random.RandomState(utt.fetch_seed())
        admat_val = rng.rand(3, 5).astype(config.floatX)
        adscal_val = numpy.asarray(rng.rand(), dtype=config.floatX).item()
        self._compile_and_check([admat],
                   [Prepend_scalar_constant_to_each_row(adscal_val)(admat)],
                    [admat_val],
                    Prepend_scalar_constant_to_each_row)

        self._compile_and_check([adscal, admat],
                   [Prepend_scalar_to_each_row()(adscal, admat)],
                    [adscal_val, admat_val],
                    Prepend_scalar_to_each_row)


class T_CrossentropyCategorical1HotGrad(utt.InferShapeTester):

    def test_infer_shape(self):
        advec = vector()
        admat = matrix()
        alvec = lvector()
        rng = numpy.random.RandomState(utt.fetch_seed())
        advec_val = rng.rand(3).astype(config.floatX)
        admat_val = rng.rand(3, 2).astype(config.floatX)
        alvec_val = [0, 1, 0]
        self._compile_and_check([advec, admat, alvec],
                    [CrossentropyCategorical1HotGrad()(advec, admat, alvec)],
                    [advec_val, admat_val, alvec_val],
                    CrossentropyCategorical1HotGrad)


class T_CrossentropyCategorical1Hot(utt.InferShapeTester):

    def test_grad(self):
        x = tensor.matrix('x')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot
        xe = op(x, one_of_n)
        f = theano.function([x, one_of_n], xe)
        x_val = numpy.asarray([[.4, .6, .0], [.1, .8, .1]],
                dtype=config.floatX)
        xe_val = f(x_val, [0, 1])
        assert numpy.allclose(xe_val, -numpy.log([.4, .8]))

        def oplike(x):
            return op(x, [0, 1])

        tensor.verify_grad(oplike, [x_val], rng=numpy.random)

    def test_infer_shape(self):
        admat = matrix()
        alvec = lvector()
        rng = numpy.random.RandomState(utt.fetch_seed())
        admat_val = rng.rand(3, 2).astype(config.floatX)
        alvec_val = [0, 1, 0]
        self._compile_and_check([admat, alvec],
                    [CrossentropyCategorical1Hot()(admat, alvec)],
                    [admat_val, alvec_val],
                    CrossentropyCategorical1Hot)

    def test_softmax_optimizations(self):
        x = tensor.matrix('x')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot
        xe = op(x, one_of_n)

        fgraph = gof.FunctionGraph(
                [x, one_of_n],
                [op(softmax_op(x), one_of_n)])
        assert fgraph.outputs[0].owner.op == op

        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)

        assert str(fgraph.outputs[0].owner.op) == 'OutputGuard'
        assert (fgraph.outputs[0].owner.inputs[0].owner.op ==
                crossentropy_softmax_argmax_1hot_with_bias)

    def test_softmax_optimizations_vector(self):
        x = tensor.vector('x')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot
        fgraph = gof.FunctionGraph(
                [x, one_of_n],
                [op(softmax_op(x), one_of_n)])
        assert fgraph.outputs[0].owner.op == op

        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)
        assert str(fgraph.outputs[0].owner.op) == 'OutputGuard'
        assert (fgraph.outputs[0].owner.inputs[0].owner.op ==
                crossentropy_softmax_argmax_1hot_with_bias)

    def test_softmax_optimizations_w_bias(self):
        x = tensor.matrix('x')
        b = tensor.vector('b')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot
        xe = op(x, one_of_n)

        fgraph = gof.FunctionGraph(
                [x, b, one_of_n],
                [op(softmax_op(x + b), one_of_n)])
        assert fgraph.outputs[0].owner.op == op

        # print 'BEFORE'
        # for node in fgraph.toposort():
        #    print node.op
        # print printing.pprint(node.outputs[0])
        # print '----'

        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)

        # print 'AFTER'
        # for node in fgraph.toposort():
        #    print node.op
        # print printing.pprint(node.outputs[0])
        # print '===='
        assert len(fgraph.toposort()) == 2

        assert str(fgraph.outputs[0].owner.op) == 'OutputGuard'
        assert (fgraph.outputs[0].owner.inputs[0].owner.op ==
                crossentropy_softmax_argmax_1hot_with_bias)

    def test_softmax_optimizations_w_bias2(self):
        x = tensor.matrix('x')
        b = tensor.vector('b')
        c = tensor.vector('c')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot

        fgraph = gof.FunctionGraph(
                [x, b, c, one_of_n],
                [op(softmax_op(T.add(x, b, c)), one_of_n)])
        assert fgraph.outputs[0].owner.op == op

        # print 'BEFORE'
        # for node in fgraph.toposort():
        #    print node.op
        # print '----'

        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)

        # print 'AFTER'
        # for node in fgraph.toposort():
        #    print node.op
        # print '===='
        assert len(fgraph.toposort()) == 3

        assert str(fgraph.outputs[0].owner.op) == 'OutputGuard'
        assert (fgraph.outputs[0].owner.inputs[0].owner.op ==
                crossentropy_softmax_argmax_1hot_with_bias)

    def test_softmax_optimizations_w_bias_vector(self):
        x = tensor.vector('x')
        b = tensor.vector('b')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot
        fgraph = gof.FunctionGraph(
                [x, b, one_of_n],
                [op(softmax_op(x + b), one_of_n)])
        assert fgraph.outputs[0].owner.op == op
        # print 'BEFORE'
        # for node in fgraph.toposort():
        #    print node.op
        # print printing.pprint(node.outputs[0])
        # print '----'

        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)
        # print 'AFTER'
        # for node in fgraph.toposort():
        #    print node.op
        # print '===='
        assert len(fgraph.toposort()) == 3
        assert str(fgraph.outputs[0].owner.op) == 'OutputGuard'
        assert (fgraph.outputs[0].owner.inputs[0].owner.op ==
                crossentropy_softmax_argmax_1hot_with_bias)

    def test_softmax_grad_optimizations(self):
        x = tensor.matrix('x')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot
        xe = op(softmax_op(x), one_of_n)
        sum_xe = tensor.sum(xe)
        g_x = tensor.grad(sum_xe, x)
        fgraph = gof.FunctionGraph(
                [x, one_of_n],
                [g_x])
        self.assertTrue(hasattr(fgraph.outputs[0].tag, 'trace'))

        # print 'BEFORE'
        # for node in fgraph.toposort():
        #    print node.op, node.inputs
        # print '----'
        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)

        # print 'AFTER'
        # for node in fgraph.toposort():
        #    print node.op, node.inputs

        has_cx1hot = False
        has_cx1hotdx = False
        has_softmax = False
        has_softmaxdx = False
        for node in fgraph.toposort():
            if node.op == crossentropy_softmax_argmax_1hot_with_bias:
                has_cx1hot = True
            if node.op == crossentropy_softmax_1hot_with_bias_dx:
                has_cx1hotdx = True
            if node.op == softmax_op:
                has_softmax = True
            if node.op == softmax_grad:
                has_softmaxdx = True
        assert not has_cx1hot
        assert has_cx1hotdx
        assert has_softmax
        assert not has_softmaxdx

    def test_softmax_grad_optimizations_vector(self):
        x = tensor.vector('x')
        one_of_n = tensor.lvector('one_of_n')
        op = crossentropy_categorical_1hot
        xe = op(softmax_op(x), one_of_n)
        sum_xe = tensor.sum(xe)
        g_x = tensor.grad(sum_xe, x)
        fgraph = gof.FunctionGraph(
                [x, one_of_n],
                [g_x])

        # print 'BEFORE'
        # for node in fgraph.toposort():
        #    print node.op, node.inputs
        # print '----'
        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)

        # print 'AFTER'
        # for node in fgraph.toposort():
        #    print node.op, node.inputs

        has_cx1hot = False
        has_cx1hotdx = False
        has_softmax = False
        has_softmaxdx = False
        for node in fgraph.toposort():
            if node.op == crossentropy_softmax_argmax_1hot_with_bias:
                has_cx1hot = True
            if node.op == crossentropy_softmax_1hot_with_bias_dx:
                has_cx1hotdx = True
            if node.op == softmax_op:
                has_softmax = True
            if node.op == softmax_grad:
                has_softmaxdx = True
        assert not has_cx1hot
        assert has_cx1hotdx
        assert has_softmax
        assert not has_softmaxdx

    def test_get_rid_of_advanced_indexing_version_of_xent(self):
        verbose = 0
        # TODO: add the optimization in FAST_COMPILE?
        # In the mean time, run it as 'FAST_RUN' instead
        mode = theano.compile.mode.get_default_mode()
        if mode == theano.compile.mode.get_mode('FAST_COMPILE'):
            mode = 'FAST_RUN'
        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.randn(3, 5).astype(config.floatX)
        b_val = rng.randn(5).astype(config.floatX)
        y_val = numpy.asarray([2, 4, 1])
        x = T.matrix('x')
        b = T.vector('b')
        y = T.lvector('y')

        # Basic case
        expressions = [
                T.sum(-T.log(softmax(x)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(x)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(x))[T.arange(y.shape[0]), y]),
                T.sum(-T.log(softmax(x))[T.arange(y.shape[0]), y])
                ]
        for expr in expressions:
            # Verify the optimizer worked on the expressions
            f = theano.function([x, y], expr, mode=mode)
            self.assertTrue(hasattr(f.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                assert len(ops) == 4
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
                f(x_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise

            # Also verify the gradient wrt x
            g = theano.function([x, y], T.grad(expr, x), mode=mode)
            self.assertTrue(hasattr(g.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) == 2
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_op in ops
                assert softmax_grad not in ops
                g(x_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

        # Test that a biased softmax is optimized correctly
        bias_expressions = [
                T.sum(-T.log(softmax(x + b)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(b + x)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(x + b))[T.arange(y.shape[0]), y]),
                T.sum(-T.log(softmax(b + x))[T.arange(y.shape[0]), y])]

        for expr in bias_expressions:
            f = theano.function([x, b, y], expr, mode=mode)
            self.assertTrue(hasattr(f.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                assert len(ops) == 2  # [big_op, sum]
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                f(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise
            g = theano.function([x, b, y], T.grad(expr, x), mode=mode)
            self.assertTrue(hasattr(g.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) == 2
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_with_bias in ops
                assert softmax_grad not in ops
                g(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

        # Test that using "mean" instead of sum works, too
        mean_expressions = [
                T.mean(-T.log(softmax(x)[T.arange(y.shape[0]), y])),
                -T.mean(T.log(softmax(x)[T.arange(y.shape[0]), y])),
                -T.mean(T.log(softmax(x))[T.arange(y.shape[0]), y]),
                T.mean(-T.log(softmax(x))[T.arange(y.shape[0]), y])]

        for expr in mean_expressions:
            f = theano.function([x, y], expr, mode=mode)
            self.assertTrue(hasattr(f.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                assert len(ops) == 6
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
                f(x_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise

            g = theano.function([x, y], T.grad(expr, x), mode=mode)
            self.assertTrue(hasattr(g.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) == 5
                #there's an extra dimshuffle in there
                # but I can't think of a good rule to get rid of it
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_op in ops
                assert softmax_grad not in ops
                g(x_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

        mean_bias_expressions = [
                T.mean(-T.log(softmax(x + b)[T.arange(y.shape[0]), y])),
                -T.mean(T.log(softmax(b + x)[T.arange(y.shape[0]), y])),
                -T.mean(T.log(softmax(x + b))[T.arange(y.shape[0]), y]),
                T.mean(-T.log(softmax(b + x))[T.arange(y.shape[0]), y])]

        for expr in mean_bias_expressions:
            f = theano.function([x, b, y], expr, mode=mode)
            self.assertTrue(hasattr(f.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                assert len(ops) == 4
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
            except Exception:
                theano.printing.debugprint(f)
                raise
            g = theano.function([x, b, y], T.grad(expr, x), mode=mode)
            self.assertTrue(hasattr(g.maker.fgraph.outputs[0].tag, 'trace'))
            if verbose:
                theano.printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) == 5
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_with_bias in ops
                assert softmax_grad not in ops
                g(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

    def test_xent_thing_int32(self):
        verbose = 0
        mode = theano.compile.mode.get_default_mode()
        if mode == theano.compile.mode.get_mode('FAST_COMPILE'):
            mode = 'FAST_RUN'
        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.randn(3, 5).astype(config.floatX)
        y_val = numpy.asarray([2, 4, 1], dtype='int64')
        x = T.matrix('x')
        y = T.lvector('y')
        yi = T.cast(y, 'int32')
        expressions = [
                T.sum(-T.log(softmax(x)[T.arange(yi.shape[0]), yi])),
                -T.sum(T.log(softmax(x)[T.arange(yi.shape[0]), yi])),
                -T.sum(T.log(softmax(x))[T.arange(yi.shape[0]), yi]),
                T.sum(-T.log(softmax(x))[T.arange(yi.shape[0]), yi])
                ]

        for expr in expressions:
            # Verify the optimizer worked on the expressions
            f = theano.function([x, y], expr, mode=mode)
            if verbose:
                theano.printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                assert len(ops) == 5
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
                f(x_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise

            # Also verify the gradient wrt x
            g = theano.function([x, y], T.grad(expr, x), mode=mode)
            if verbose:
                theano.printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) == 3
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_op in ops
                assert softmax_grad not in ops
                g(x_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

    def test_optimize_xent_vector(self):
        verbose = 0
        mode = theano.compile.mode.get_default_mode()
        if mode == theano.compile.mode.get_mode('FAST_COMPILE'):
            mode = 'FAST_RUN'
        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.randn(5).astype(config.floatX)
        y_val = numpy.asarray([2])

        x = T.vector('x')
        y = T.lvector('y')

        # Test that a biased softmax is optimized correctly
        bias_expressions = [
                T.sum(-T.log(softmax(x)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(x)[T.arange(y.shape[0]), y]))]

        for expr in bias_expressions:
            f = theano.function([x, y], expr, mode=mode)
            if verbose:
                printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                assert len(ops) == 5
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
                f(x_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise
            g = theano.function([x, y], T.grad(expr, x), mode=mode)
            if verbose:
                printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) == 4
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_op in ops
                assert softmax_grad not in ops
                g(x_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

    def test_optimize_xent_vector2(self):
        verbose = 0
        mode = theano.compile.mode.get_default_mode()
        if mode == theano.compile.mode.get_mode('FAST_COMPILE'):
            mode = 'FAST_RUN'
        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.randn(5).astype(config.floatX)
        b_val = rng.randn(5).astype(config.floatX)
        y_val = numpy.asarray([2])

        x = T.vector('x')
        b = T.vector('b')
        y = T.lvector('y')

        # Test that a biased softmax is optimized correctly
        bias_expressions = [
                T.sum(-T.log(softmax(x + b)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(b + x)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(x + b))[T.arange(y.shape[0]), y]),
                T.sum(-T.log(softmax(b + x))[T.arange(y.shape[0]), y])]

        for expr in bias_expressions:
            f = theano.function([x, b, y], expr, mode=mode)
            if verbose:
                printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                # [big_op, sum, dim_shuffle]
                assert len(ops) == 3
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
                f(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise

            backup = config.warn.sum_div_dimshuffle_bug
            config.warn.sum_div_dimshuffle_bug = False
            try:
                g = theano.function([x, b, y], T.grad(expr, x), mode=mode)
            finally:
                config.warn.sum_div_dimshuffle_bug = backup

            if verbose:
                printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) <= 6
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_with_bias in ops
                assert softmax_grad not in ops
                g(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

    def test_optimize_xent_vector3(self):
        # Same as test_optimize_xent_vector2, but y is the result of
        # a "flatten", and it used to make the constant-folding
        # of arange(y.shape[0]) happen before the xent optimization
        verbose = 0
        mode = theano.compile.mode.get_default_mode()
        if mode == theano.compile.mode.get_mode('FAST_COMPILE'):
            mode = 'FAST_RUN'
        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.randn(5).astype(config.floatX)
        b_val = rng.randn(5).astype(config.floatX)
        y_val = numpy.asarray([2])

        x = T.vector('x')
        b = T.vector('b')
        y_ = T.lvector('y_')
        y = y_.flatten()

        # Test that a biased softmax is optimized correctly
        bias_expressions = [
                T.sum(-T.log(softmax(x + b)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(b + x)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(x + b))[T.arange(y.shape[0]), y]),
                T.sum(-T.log(softmax(b + x))[T.arange(y.shape[0]), y])]

        for expr in bias_expressions:
            f = theano.function([x, b, y_], expr, mode=mode)
            if verbose:
                printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                # [big_op, sum, dim_shuffle, flatten]
                assert len(ops) <= 4
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
                f(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise

            backup = config.warn.sum_div_dimshuffle_bug
            config.warn.sum_div_dimshuffle_bug = False
            try:
                g = theano.function([x, b, y], T.grad(expr, x), mode=mode)
            finally:
                config.warn.sum_div_dimshuffle_bug = backup

            if verbose:
                printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) <= 6
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_with_bias in ops
                assert softmax_grad not in ops
                g(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

    def test_optimize_xent_vector4(self):
        # Same as test_optimize_xent_vector2, but y is the result of
        # a "specify_shape" that indicates its length is 1, so the
        # constant-folding of arange(y.shape[0]) happen before the xent
        # optimization
        verbose = 0
        mode = theano.compile.mode.get_default_mode()
        if mode == theano.compile.mode.get_mode('FAST_COMPILE'):
            mode = 'FAST_RUN'
        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.randn(5).astype(config.floatX)
        b_val = rng.randn(5).astype(config.floatX)
        y_val = numpy.asarray([2])

        x = T.vector('x')
        b = T.vector('b')
        y_ = T.lvector('y_')
        y = T.specify_shape(y_, (1,))

        # Test that a biased softmax is optimized correctly
        bias_expressions = [
                T.sum(-T.log(softmax(x + b)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(b + x)[T.arange(y.shape[0]), y])),
                -T.sum(T.log(softmax(x + b))[T.arange(y.shape[0]), y]),
                T.sum(-T.log(softmax(b + x))[T.arange(y.shape[0]), y])]

        for expr in bias_expressions:
            f = theano.function([x, b, y_], expr, mode=mode)
            if verbose:
                printing.debugprint(f)
            try:
                ops = [node.op for node in f.maker.fgraph.toposort()]
                # [big_op, sum, dim_shuffle, specify_shape]
                assert len(ops) <= 4
                assert crossentropy_softmax_argmax_1hot_with_bias in ops
                assert not [1 for o in ops
                            if isinstance(o, T.AdvancedSubtensor)]
                f(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(f)
                raise

            backup = config.warn.sum_div_dimshuffle_bug
            config.warn.sum_div_dimshuffle_bug = False
            try:
                g = theano.function([x, b, y], T.grad(expr, x), mode=mode)
            finally:
                config.warn.sum_div_dimshuffle_bug = backup

            if verbose:
                printing.debugprint(g)
            try:
                ops = [node.op for node in g.maker.fgraph.toposort()]
                assert len(ops) <= 6
                assert crossentropy_softmax_1hot_with_bias_dx in ops
                assert softmax_with_bias in ops
                assert softmax_grad not in ops
                g(x_val, b_val, y_val)
            except Exception:
                theano.printing.debugprint(g)
                raise

    def test_crossentropy_softmax_1hot_with_bias_dxcale_cost(self):
        # TODO: add the optimization in FAST_COMPILE?
        # In the mean time, run it as 'FAST_RUN' instead
        mode = theano.compile.mode.get_default_mode()
        if mode == theano.compile.mode.get_mode('FAST_COMPILE'):
            mode = 'FAST_RUN'
        rng = numpy.random.RandomState(utt.fetch_seed())
        x_val = rng.randn(3, 5).astype(config.floatX)
        y_val = numpy.asarray([2, 4, 1])
        x = T.matrix('x')
        y = T.lvector('y')
        a = T.scalar('a')

        def validate_fn_graph(func):
            # The graph of the function should not have softmax anymore
            has_cx1hot = False
            has_softmax = False
            for node in func.maker.fgraph.toposort():
                if node.op == crossentropy_softmax_argmax_1hot_with_bias:
                    has_cx1hot = True
                if node.op == softmax_op:
                    has_softmax = True

            assert has_cx1hot
            assert not has_softmax

        def validate_grad_graph(func):
            # The graph of the gradient should not have softmaxgrad anymore
            has_cx1hotdx = False
            has_softmax = False
            has_softmaxdx = False
            for node in func.maker.fgraph.toposort():
                if node.op == crossentropy_softmax_1hot_with_bias_dx:
                    has_cx1hotdx = True
                if node.op == softmax_op:
                    has_softmax = True
                if node.op == softmax_grad:
                    has_softmaxdx = True

            assert has_cx1hotdx
            assert has_softmax
            assert not has_softmaxdx

        # Cases to test
        expressions = [
                a * T.sum(-T.log(softmax(x)[T.arange(y.shape[0]), y])),
                -a * T.sum(T.log(softmax(x)[T.arange(y.shape[0]), y])),
                a * (-T.sum(T.log(softmax(x)[T.arange(y.shape[0]), y]))),
                a * T.sum(T.log(softmax(x)[T.arange(y.shape[0]), y])),

                a * T.sum(-T.log(softmax(x))[T.arange(y.shape[0]), y]),
                -a * T.sum(T.log(softmax(x))[T.arange(y.shape[0]), y]),
                a * (-T.sum(T.log(softmax(x))[T.arange(y.shape[0]), y])),
                a * T.sum(T.log(softmax(x))[T.arange(y.shape[0]), y]),

                a * T.mean(-T.log(softmax(x)[T.arange(y.shape[0]), y])),
                -a * T.mean(T.log(softmax(x)[T.arange(y.shape[0]), y])),
                a * (-T.mean(T.log(softmax(x)[T.arange(y.shape[0]), y]))),
                a * T.mean(T.log(softmax(x)[T.arange(y.shape[0]), y])),

                a * T.mean(-T.log(softmax(x))[T.arange(y.shape[0]), y]),
                -a * T.mean(T.log(softmax(x))[T.arange(y.shape[0]), y]),
                a * (-T.mean(T.log(softmax(x))[T.arange(y.shape[0]), y])),
                a * T.mean(T.log(softmax(x))[T.arange(y.shape[0]), y]),
                ]

        for expr in expressions:
            # Verify the optimizer worked on the expressions
            f = theano.function([x, y, a], expr, mode=mode)
            try:
                assert 5 <= len(f.maker.fgraph.toposort()) <= 10
                validate_fn_graph(f)
                f(x_val, y_val, 0.1)
            except Exception:
                theano.printing.debugprint(f)
                raise

            # Verify the gradient wrt x
            g = theano.function([x, y, a], T.grad(expr, x), mode=mode)
            try:
                assert 3 <= len(g.maker.fgraph.toposort()) <= 6
                validate_grad_graph(g)
                g(x_val, y_val, 0.1)
            except Exception:
                theano.printing.debugprint(g)
                raise

            # Verify the gradient when providing output gradient
            h = theano.function([x, y, a],
                    T.grad(expr, x, known_grads={expr: a * x.sum()}), mode=mode)
            try:
                assert 6 <= len(h.maker.fgraph.toposort()) <= 8
                validate_grad_graph(h)
                h(x_val, y_val, 0.1)
            except Exception:
                theano.printing.debugprint(h)
                raise


def test_argmax_pushdown():
    x = tensor.matrix()
    for sm in [softmax_graph, softmax_op]:
        # test that the max_and_argmax is pushed down if the max is not used
        out = tensor.max_and_argmax(
                sm(tensor.exp(tensor.tanh(sigmoid(x)))),
                axis=-1)[1]
        fgraph = gof.FunctionGraph(
                [x],
                [out])
        theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)

        # print 'AFTER'
        # for node in fgraph.toposort():
            # print node.op
        assert len(fgraph.toposort()) == 2  # an output_guard is second
        assert fgraph.toposort()[0].op == tensor.basic._max_and_argmax
        assert str(fgraph.toposort()[1].op) == 'OutputGuard'
        x = tensor.matrix()
        # test that the max_and_argmax is not pushed down if the max is used
        out = tensor.max_and_argmax(
                sm(tensor.exp(tensor.tanh(sigmoid(x)))),
                axis=-1)[0]
        fgraph = gof.FunctionGraph(
                [x],
                [out])
        assert hasattr(fgraph.outputs[0].tag, 'trace')

        backup = config.warn.argmax_pushdown_bug
        config.warn.argmax_pushdown_bug = False
        try:
            theano.compile.mode.optdb.query(
                theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)
        finally:
            config.warn.argmax_pushdown_bug = backup

        # print 'AFTER'
        # for node in fgraph.toposort():
            # print node.op
        assert len(fgraph.toposort()) == 4  # an output_guard is second
        assert isinstance(fgraph.toposort()[0].op, tensor.Elemwise)
        assert isinstance(fgraph.toposort()[1].op, Softmax)
        assert isinstance(fgraph.toposort()[2].op, tensor.CAReduce)
        assert isinstance(fgraph.toposort()[2].op.scalar_op, theano.scalar.Maximum)
        assert str(fgraph.toposort()[3].op) == 'OutputGuard'


def test_argmax_pushdown_bias():
    x = tensor.matrix()
    b = tensor.vector()

    out = tensor.argmax(softmax_with_bias(x, b), axis=-1)
    fgraph = gof.FunctionGraph(
            [x, b],
            [out])
    f = theano.function([x, b], out)
    assert hasattr(f.maker.fgraph.outputs[0].tag, 'trace')

    theano.compile.mode.optdb.query(
            theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)

    # print 'AFTER'
    # for node in fgraph.toposort():
    #    print node.op
    assert len(fgraph.toposort()) == 4
    assert isinstance(fgraph.toposort()[0].op, tensor.DimShuffle)
    assert isinstance(fgraph.toposort()[1].op, tensor.Elemwise)
    assert isinstance(fgraph.toposort()[2].op, tensor.MaxAndArgmax)
    assert str(fgraph.toposort()[3].op) == 'OutputGuard'

    x = tensor.matrix()
    b = tensor.vector()
    out = tensor.max_and_argmax(softmax_with_bias(x, b), axis=-1)[0]
    fgraph = gof.FunctionGraph(
            [x, b],
            [out])
    f = theano.function([x, b], out)
    assert hasattr(f.maker.fgraph.outputs[0].tag, 'trace')

    backup = config.warn.argmax_pushdown_bug
    config.warn.argmax_pushdown_bug = False
    try:
        theano.compile.mode.optdb.query(
            theano.compile.mode.OPT_FAST_RUN).optimize(fgraph)
    finally:
        config.warn.argmax_pushdown_bug = backup

    # print 'AFTER'
    # for node in fgraph.toposort():
    #    print node.op
    assert len(fgraph.toposort()) == 3
    assert isinstance(fgraph.toposort()[0].op, SoftmaxWithBias)
    assert isinstance(fgraph.toposort()[1].op, tensor.CAReduce)
    assert isinstance(fgraph.toposort()[1].op.scalar_op, theano.scalar.Maximum)
    assert str(fgraph.toposort()[2].op) == 'OutputGuard'


def test_asymptotic_32():
    """
    This test makes sure that our functions behave sensibly when
    huge values are present
    """

    # TODO: consider adding the optimization of crossentropy into the current
    # mode for the purpose of running this test

    for dtype in 'float32', 'float64':
        if dtype == 'float32':
            x = tensor.fmatrix()
            x2 = tensor.fvector()
        else:
            x = tensor.dmatrix()
            x2 = tensor.dvector()
        y = tensor.lvector()

        c = categorical_crossentropy(softmax(x + x2), y)
        f = theano.function([x, y, x2], [c.sum(),
                            tensor.grad(c.sum(), x)], mode='FAST_RUN')
        if 0:
            for i, n in enumerate(f.maker.fgraph.toposort()):
                print(i, n)

        xval = numpy.zeros((5, 5), dtype=dtype).astype(dtype)
        x2val = numpy.zeros(5, dtype=xval.dtype).astype(dtype)
        for i in xrange(100):
            cval, gxval = f(xval, numpy.arange(5), x2val)
            xval -= 100.3 * gxval
            # print cval, gxval
        assert cval == 0  # no problem going to zero error

        # what about when x gets really big?

        xval = numpy.zeros((5, 5), dtype=dtype)
        x2val = numpy.zeros(5, dtype=xval.dtype)
        for i in xrange(100):

            cval, gxval = f(xval, numpy.arange(5), x2val)
            xval += 100000.3 * gxval
            # print cval, gxval

        assert cval > 61750000
        assert gxval[0, 0] == -1.0
        assert gxval[0, 1] == 0.25


class Test_softmax_opt:
    # Test that expressions of softmax in terms of exponentiated things
    # divided by row sums are replaced by softmax expressions.
    #
    # Softmax_grad isn't that interesting as an Op, but it has the signature
    # we look for when trying to insert CrossEntropySoftmax... grad.  So, for
    # now, we add softmax_grad to graphs. In the future, we may modify the
    # CrossEntropySoftmax...grad to look for the more basic pattern.
    #

    def setUp(self):
        utt.seed_rng()
        self.rng = numpy.random.RandomState(utt.fetch_seed())
        self.mode = theano.compile.mode.get_default_mode()
        self.mode = self.mode.including('canonicalize')

    def test_basic(self):
        c = T.matrix()
        p_y = T.exp(c) / T.exp(c).sum(axis=1).dimshuffle(0, 'x')

        # test that function contains softmax and no div.
        f = theano.function([c], p_y, mode=self.mode)

        assert hasattr(f.maker.fgraph.outputs[0].tag, 'trace')

        f_ops = [n.op for n in f.maker.fgraph.toposort()]
        # print '--- f ='
        # printing.debugprint(f)
        # print '==='
        assert len(f_ops) == 1
        assert softmax_op in f_ops
        f(self.rng.rand(3, 4).astype(config.floatX))

    def test_basic_keepdims(self):
        c = T.matrix()
        p_y = T.exp(c) / T.exp(c).sum(axis=1, keepdims=True)

        # test that function contains softmax and no div.
        f = theano.function([c], p_y, mode=self.mode)

        assert hasattr(f.maker.fgraph.outputs[0].tag, 'trace')

        f_ops = [n.op for n in f.maker.fgraph.toposort()]
        # print '--- f ='
        # printing.debugprint(f)
        # print '==='
        assert len(f_ops) == 1
        assert softmax_op in f_ops
        f(self.rng.rand(3, 4).astype(config.floatX))

    def test_grad(self):
        c = T.matrix()
        p_y = T.exp(c) / T.exp(c).sum(axis=1).dimshuffle(0, 'x')

        # test that function contains softmax and softmaxgrad
        w = T.matrix()
        backup = config.warn.sum_div_dimshuffle_bug
        config.warn.sum_div_dimshuffle_bug = False
        try:
            g = theano.function([c, w], T.grad((p_y * w).sum(), c))
            hasattr(g.maker.fgraph.outputs[0].tag, 'trace')
        finally:
            config.warn.sum_div_dimshuffle_bug = backup
        g_ops = [n.op for n in g.maker.fgraph.toposort()]
        # print '--- g ='
        # printing.debugprint(g)
        # print '==='

        raise SkipTest('Optimization not enabled for the moment')
        assert len(g_ops) == 2
        assert softmax_op in g_ops
        assert softmax_grad in g_ops
        g(self.rng.rand(3, 4), self.rng.uniform(.5, 1, (3, 4)))

    def test_transpose_basic(self):
        # this should be a transposed softmax
        c = T.matrix()
        p_y = T.exp(c) / T.exp(c).sum(axis=0)

        # test that function contains softmax and no div.
        f = theano.function([c], p_y)
        # printing.debugprint(f)

        # test that function contains softmax and no div.
        backup = config.warn.sum_div_dimshuffle_bug
        config.warn.sum_div_dimshuffle_bug = False
        try:
            g = theano.function([c], T.grad(p_y.sum(), c))
            hasattr(g.maker.fgraph.outputs[0].tag, 'trace')
        finally:
            config.warn.sum_div_dimshuffle_bug = backup
        # printing.debugprint(g)
        raise SkipTest('Optimization not enabled for the moment')

    def test_1d_basic(self):
        # this should be a softmax, but of a one-row matrix
        c = T.vector()
        p_y = T.exp(c) / T.exp(c).sum()

        # test that function contains softmax and no div.
        f = theano.function([c], p_y)
        hasattr(f.maker.fgraph.outputs[0].tag, 'trace')
        # printing.debugprint(f)

        # test that function contains softmax and no div.
        backup = config.warn.sum_div_dimshuffle_bug
        config.warn.sum_div_dimshuffle_bug = False
        try:
            g = theano.function([c], T.grad(p_y.sum(), c))
            hasattr(g.maker.fgraph.outputs[0].tag, 'trace')
        finally:
            config.warn.sum_div_dimshuffle_bug = backup
        # printing.debugprint(g)
        raise SkipTest('Optimization not enabled for the moment')

    # REPEAT 3 CASES in presence of log(softmax) with the advanced indexing
    # etc.


def test_softmax_graph():
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.shared(rng.normal(size=(3, 4)))

    def f(inputs):
        y = softmax_graph(x)
        return theano.grad(None, x, known_grads={y: inputs})

    utt.verify_grad(f, [rng.rand(3, 4)])


def test_grad_softmax_grad():
    rng = numpy.random.RandomState(utt.fetch_seed())
    x = theano.shared(rng.normal(size=(3, 4)))

    def f(inputs):
        y = softmax_op(x)
        return theano.grad(None, x, known_grads={y: inputs})
    utt.verify_grad(f, [rng.rand(3, 4)])


def test_stabilize_log_softmax():
    mode = theano.compile.mode.get_default_mode()
    mode = mode.including('local_log_softmax', 'specialize')

    x = matrix()
    y = softmax(x)
    z = theano.tensor.log(y)

    f = theano.function([x], z, mode=mode)
    assert hasattr(f.maker.fgraph.outputs[0].tag, 'trace')

    # check that the softmax has been optimized out
    for node in f.maker.fgraph.toposort():
        assert not isinstance(node.op, y.owner.op.__class__)

    # call the function so debug mode can verify the optimized
    # version matches the unoptimized version
    rng = numpy.random.RandomState([2012, 8, 22])
    f(numpy.cast[config.floatX](rng.randn(2, 3)))


def test_relu():
    x = matrix('x')
    seed = theano.tests.unittest_tools.fetch_seed()
    rng = numpy.random.RandomState(seed)
    X = rng.randn(20, 30).astype(config.floatX)

    # test the base case, without custom alpha value
    y = relu(x).eval({x: X})
    assert numpy.allclose(y, numpy.maximum(X, 0))

    # test for different constant alpha values (also outside of [0, 1])
    for alpha in 0, 0.3, 1, 2, -0.3, -1, -2:
        y = relu(x, alpha).eval({x: X})
        assert numpy.allclose(y, numpy.where(X > 0, X, alpha * X))

    # test for variable alpha (scalar, vector and matrix)
    for alpha in scalar(), vector(), matrix():
        # create value for alpha (correct ndim and broadcastable against X)
        A = numpy.array(rng.randn(*X.shape[::-1][:alpha.ndim][::-1]),
                        dtype=config.floatX)
        y = relu(x, alpha).eval({x: X, alpha: A})
        assert numpy.allclose(y, numpy.where(X > 0, X, A * X), rtol=3e-5)


def test_h_softmax():
    """
    Tests the output dimensions of the h_softmax when a target is provided or
    not.
    """

    #############
    # Config
    #############

    input_size = 4
    batch_size = 2
    h_softmax_level1_size = 5
    h_softmax_level2_size = 3
    output_size = h_softmax_level1_size * h_softmax_level2_size

    #############
    # Initialize shared variables
    #############

    floatX = theano.config.floatX
    shared = theano.shared

    # First level of h_softmax
    W1 = numpy.asarray(numpy.random.normal(
        size=(input_size, h_softmax_level1_size)), dtype=floatX)
    W1 = shared(W1)
    b1 = shared(numpy.asarray(numpy.zeros((h_softmax_level1_size,)),
                              dtype=floatX))

    # Second level of h_softmax
    W2 = numpy.asarray(numpy.random.normal(
        size=(h_softmax_level1_size, input_size, h_softmax_level2_size)),
        dtype=floatX)
    W2 = shared(W2)
    b2 = shared(
        numpy.asarray(numpy.zeros((h_softmax_level1_size,
                                   h_softmax_level2_size)), dtype=floatX))

    #############
    # Build graph
    #############
    x = tensor.matrix('x')
    y = tensor.ivector('y')

    # This only computes the output corresponding to the target
    y_hat_tg = h_softmax(x, batch_size, output_size, h_softmax_level1_size,
                         h_softmax_level2_size, W1, b1, W2, b2, y)

    # This computes all the outputs
    y_hat_all = h_softmax(x, batch_size, output_size, h_softmax_level1_size,
                          h_softmax_level2_size, W1, b1, W2, b2)

    #############
    # Compile functions
    #############
    fun_output_tg = theano.function([x, y], y_hat_tg)
    fun_output = theano.function([x], y_hat_all)

    #############
    # Test
    #############
    x_mat = numpy.random.normal(size=(batch_size, input_size)).astype(floatX)
    y_mat = numpy.random.randint(0, output_size, batch_size).astype('int32')
    tg_output = fun_output_tg(x_mat, y_mat)
    all_outputs = fun_output(x_mat)

    assert(tg_output.shape == (batch_size,))
    assert(all_outputs.shape == (batch_size, output_size))

    # Verifies that the outputs computed by fun_output_tg are the same as those
    # computed by fun_output.
    utt.assert_allclose(
            all_outputs[numpy.arange(0, batch_size), y_mat], tg_output)


def test_elu():
    x = matrix('x')
    seed = theano.tests.unittest_tools.fetch_seed()
    rng = numpy.random.RandomState(seed)
    X = rng.randn(20, 30).astype(config.floatX)

    # test the base case, without custom alpha value
    y = elu(x).eval({x: X})
    utt.assert_allclose(y, numpy.where(X > 0, X, numpy.exp(X) - 1))

    # test for different constant alpha values
    for alpha in 1.5, 2, -1, -1.5, -2:
        y = elu(x, alpha).eval({x: X})
        utt.assert_allclose(y, numpy.where(X > 0, X, alpha * (numpy.exp(X) - 1)))
