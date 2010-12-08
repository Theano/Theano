import unittest

import numpy

import theano.tensor.inplace
from theano import tensor as T
from theano import config
from theano.tests import unittest_tools as utt
from theano.tensor.nnet import sigmoid, sigmoid_inplace, softplus, tensor
from theano.tensor.nnet.sigm import register_local_1msigmoid


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

class T_sigmoid_opts(unittest.TestCase):
    def test_exp_over_1_plus_exp(self):
        m = theano.config.mode
        if m == 'FAST_COMPILE':
            m = 'FAST_RUN'
        m = theano.compile.mode.get_mode(m)
        m = m.excluding('local_elemwise_fusion')

        x = T.dvector()

        # tests exp_over_1_plus_exp
        f = theano.function([x], T.exp(x)/(1+T.exp(x)), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [sigmoid]

        # tests inv_1_plus_exp
        f = theano.function([x], T.fill(x,1.0) / (1+T.exp(-x)), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [sigmoid]

        # tests inv_1_plus_exp with neg
        f = theano.function([x], T.fill(x,-1.0) / (1+T.exp(-x)), mode=m)
        assert [node.op for node in f.maker.env.toposort()] == [sigmoid,
                theano.tensor.inplace.neg_inplace]

        # tests double inv_1_plus_exp with neg
        # (-1)(exp(x)) / (1+exp(x))(1+exp(-x))
        # = (-1)/(1+exp(-x)) * exp(x)/(1+exp(x))
        # = - (sigm(x) * sigm(x))
        f = theano.function([x], (T.fill(x,-1.0)*T.exp(x)) / ((1+T.exp(x))*(1+T.exp(-x))), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [sigmoid,
                T.mul, theano.tensor.inplace.neg_inplace]

    def test_1msigmoid(self):
        if not register_local_1msigmoid:
            return

        m = theano.config.mode
        if m == 'FAST_COMPILE':
            m = 'FAST_RUN'

        x = T.fmatrix()

        # tests exp_over_1_plus_exp
        f = theano.function([x], 1 - T.exp(x)/(1+T.exp(x)), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [tensor.neg, sigmoid_inplace]

        # tests inv_1_plus_exp
        f = theano.function([x], 1 - T.fill(x,1.0) / (1+T.exp(-x)), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [tensor.neg,
                sigmoid_inplace]


class T_softplus_opts(unittest.TestCase):
    def setUp(self):
        if theano.config.mode == 'FAST_COMPILE':
            m = theano.compile.mode.get_mode('FAST_RUN').excluding('local_elemwise_fusion')
        else:
            m = theano.compile.mode.get_default_mode().excluding('local_elemwise_fusion')
        self.m = m
        utt.seed_rng()
    def test_logsigm_to_softplus(self):
        x = T.vector()

        out = T.log(sigmoid(x))
        f = theano.function([x],out,mode=self.m)
        topo = f.maker.env.toposort()
        print topo
        assert len(topo)==3
        assert isinstance(topo[0].op.scalar_op, theano.scalar.Neg)
        assert isinstance(topo[1].op.scalar_op, theano.tensor.nnet.sigm.ScalarSoftplus)
        assert isinstance(topo[2].op.scalar_op, theano.scalar.Neg)
        f(numpy.random.rand(54).astype(config.floatX))

    def test_log1msigm_to_softplus(self):
        x = T.vector()

        out = T.log(1-sigmoid(x))
        f = theano.function([x],out,mode=self.m)
        topo = f.maker.env.toposort()
        assert len(topo)==2
        assert isinstance(topo[0].op.scalar_op, theano.tensor.nnet.sigm.ScalarSoftplus)
        assert isinstance(topo[1].op.scalar_op, theano.scalar.Neg)
        f(numpy.random.rand(54).astype(config.floatX))
        
    def test_log1pexp_to_softplus(self):
        m = theano.config.mode
        if m == 'FAST_COMPILE':
            m = 'FAST_RUN'

        x = T.vector()
    
        out = T.log(1+T.exp(x))
        f = theano.function([x],out,mode=self.m)
        topo = f.maker.env.toposort()
        assert len(topo)==1
        assert isinstance(topo[0].op.scalar_op,theano.tensor.nnet.sigm.ScalarSoftplus)
        f(numpy.random.rand(54).astype(config.floatX))
