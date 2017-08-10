from __future__ import absolute_import, print_function, division
import unittest

import numpy as np

import theano
from theano import function, config
from theano import scalar
from theano.gof import FunctionGraph
from theano.gof.opt import out2in
from theano.tensor.opt_uncanonicalize import (
    local_alloc_dimshuffle,
    local_reshape_dimshuffle,
    local_dimshuffle_alloc,
    local_dimshuffle_subtensor,
    )
import theano.tensor as tensor
#from theano.tensor import matrix,max_and_argmax,MaaxAndArgmax,neg
from theano.tensor.elemwise import CAReduce, Elemwise, DimShuffle
from theano.tests import unittest_tools as utt


class T_max_and_argmax(unittest.TestCase):
    def test_optimization(self):
        # If we use only the max output, we should replace this op with
        # a faster one.
        mode = theano.compile.mode.get_default_mode().including(
            'canonicalize', 'fast_run')

        for axis in [0, 1, -1]:
            data = np.asarray(np.random.rand(2, 3), dtype=config.floatX)
            n = tensor.matrix()

            f = function([n], tensor.max_and_argmax(n, axis)[0], mode=mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)

            f = function([n], tensor.max_and_argmax(n, axis), mode=mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, tensor.MaxAndArgmax)


class T_min_max(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        self.mode = theano.compile.mode.get_default_mode().including(
            'canonicalize', 'fast_run')

    def test_optimization_max(self):
        data = np.asarray(np.random.rand(2, 3), dtype=config.floatX)
        n = tensor.matrix()

        for axis in [0, 1, -1]:
            f = function([n], tensor.max(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)
            f(data)

            f = function([n], tensor.max(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, Elemwise)
            assert isinstance(topo[0].op.scalar_op, scalar.Neg)
            assert isinstance(topo[1].op, CAReduce)
            f(data)

            f = function([n], -tensor.max(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, CAReduce)
            assert isinstance(topo[1].op, Elemwise)
            assert isinstance(topo[1].op.scalar_op, scalar.Neg)
            f(data)

            f = function([n], -tensor.max(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)  # min
            f(data)

    def test_optimization_min(self):
        data = np.asarray(np.random.rand(2, 3), dtype=config.floatX)
        n = tensor.matrix()

        for axis in [0, 1, -1]:
            f = function([n], tensor.min(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)
            f(data)

            # test variant with neg to make sure we optimize correctly
            f = function([n], tensor.min(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, CAReduce)  # max
            assert isinstance(topo[1].op, Elemwise)
            assert isinstance(topo[1].op.scalar_op, scalar.Neg)
            f(data)

            f = function([n], -tensor.min(n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, Elemwise)
            assert isinstance(topo[0].op.scalar_op, scalar.Neg)
            assert isinstance(topo[1].op, CAReduce)  # max
            f(data)

            f = function([n], -tensor.min(-n, axis), mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, CAReduce)  # max
            f(data)


def test_local_alloc_dimshuffle():

    alloc_dimshuffle = out2in(local_alloc_dimshuffle)

    x = tensor.vector('x')
    m = tensor.iscalar('m')

    y = x.dimshuffle('x', 0)
    out = tensor.alloc(y, m, 1, x.shape[0])

    g = FunctionGraph([x, m], [out])
    alloc_dimshuffle(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])


def test_local_reshape_dimshuffle():

    reshape_dimshuffle = out2in(local_reshape_dimshuffle)

    x = tensor.matrix('x')

    y = x.dimshuffle('x', 0, 'x', 1)
    out = tensor.reshape(y, (1, x.shape[0] * x.shape[1], 1))

    g = FunctionGraph([x], [out])
    reshape_dimshuffle(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])


def test_local_dimshuffle_alloc():

    reshape_dimshuffle = out2in(local_dimshuffle_alloc)

    x = tensor.vector('x')

    out = tensor.alloc(x, 3, 2).dimshuffle('x', 'x', 0, 1)

    g = FunctionGraph([x], [out])
    reshape_dimshuffle(g)

    l = theano.gof.PerformLinker()
    l.accept(g)
    f = l.make_function()

    assert f([3, 4]).ndim == 4

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])


def test_local_dimshuffle_subtensor():

    dimshuffle_subtensor = out2in(local_dimshuffle_subtensor)

    x = tensor.dtensor4('x')
    x = tensor.patternbroadcast(x, (False, True, False, False))
    i = tensor.iscalar('i')

    out = x[:, :, 10:30, ::i].dimshuffle(0, 2, 3)

    g = FunctionGraph([x, i], [out])
    dimshuffle_subtensor(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])

    # Test dimshuffle remove dimensions the subtensor don't "see".
    x = tensor.tensor(broadcastable=(False, True, False), dtype='float64')
    out = x[i].dimshuffle(1)

    g = FunctionGraph([x, i], [out])
    dimshuffle_subtensor(g)

    topo = g.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])

    # Test dimshuffle remove dimensions the subtensor don't "see" but
    # have in between dimensions.
    x = tensor.tensor(broadcastable=(False, True, False, True),
                      dtype='float64')
    out = x[i].dimshuffle(1)

    f = theano.function([x, i], out)

    topo = f.maker.fgraph.toposort()
    assert any([not isinstance(x, DimShuffle) for x in topo])
    assert f(np.random.rand(5, 1, 4, 1), 2).shape == (4,)

    # Test a corner case that had Theano return a bug.
    x = tensor.dtensor4('x')
    x = tensor.patternbroadcast(x, (False, True, False, False))

    assert x[:,:, 0:3, ::-1].dimshuffle(0,2,3).eval({x: np.ones((5, 1, 6, 7))}).shape == (5, 3, 7)
