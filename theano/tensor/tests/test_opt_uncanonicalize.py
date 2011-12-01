import unittest

import numpy

import theano
from theano import function, config
from theano import scalar
import theano.tensor as tensor
#from theano.tensor import matrix,max_and_argmax,MaaxAndArgmax,neg
from theano.tensor.elemwise import CAReduce, Elemwise
from theano.tests import unittest_tools as utt


class T_max_and_argmax(unittest.TestCase):
    def test_optimization(self):
        #If we use only the max output, we should replace this op with
        #a faster one.
        mode = theano.compile.mode.get_default_mode().including(
            'canonicalize', 'fast_run')

        for axis in [0, 1, -1]:
            data = numpy.asarray(numpy.random.rand(2, 3), dtype=config.floatX)
            n = tensor.matrix()

            f = function([n], tensor.max_and_argmax(n, axis)[0], mode=mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)

            f = function([n], tensor.max_and_argmax(n, axis), mode=mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, tensor.MaxAndArgmax)


class T_min_max(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        self.mode = theano.compile.mode.get_default_mode().including(
            'canonicalize', 'fast_run')

    def test_optimization_max(self):
        data = numpy.asarray(numpy.random.rand(2, 3), dtype=config.floatX)
        n = tensor.matrix()

        for axis in [0, 1, -1]:
            f = function([n], tensor.max(n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)
            f(data)

            f = function([n], tensor.max(-n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, Elemwise)
            assert isinstance(topo[0].op.scalar_op, scalar.Neg)
            assert isinstance(topo[1].op, CAReduce)
            f(data)

            f = function([n], -tensor.max(n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, CAReduce)
            assert isinstance(topo[1].op, Elemwise)
            assert isinstance(topo[1].op.scalar_op, scalar.Neg)
            f(data)

            f = function([n], -tensor.max(-n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)  # min
            f(data)

    def test_optimization_min(self):
        data = numpy.asarray(numpy.random.rand(2, 3), dtype=config.floatX)
        n = tensor.matrix()

        for axis in [0, 1, -1]:
            f = function([n], tensor.min(n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)
            f(data)

            #test variant with neg to make sure we optimize correctly
            f = function([n], tensor.min(-n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, CAReduce)  # max
            assert isinstance(topo[1].op, Elemwise)
            assert isinstance(topo[1].op.scalar_op, scalar.Neg)
            f(data)

            f = function([n], -tensor.min(n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, Elemwise)
            assert isinstance(topo[0].op.scalar_op, scalar.Neg)
            assert isinstance(topo[1].op, CAReduce)  # max
            f(data)

            f = function([n], -tensor.min(-n, axis), mode=self.mode)
            topo = f.maker.env.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)  # max
            f(data)
