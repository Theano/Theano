from __future__ import absolute_import, print_function, division
from theano.tensor.xlogx import xlogx, xlogy0

import theano
from theano.tensor import as_tensor_variable
from . import test_basic as TT

import numpy.random
from theano.tests import unittest_tools as utt


class Test_XlogX():
    def setup_method(self):
        utt.seed_rng()

    def test0(self):
        x = as_tensor_variable([1, 0])
        y = xlogx(x)
        f = theano.function([], [y])
        assert numpy.all(f() == numpy.asarray([0, 0.]))

    def test1(self):
        # class Dummy(object):
        #     def make_node(self, a):
        #         return [xlogx(a)[:,2]]
        utt.verify_grad(xlogx, [numpy.random.rand(3, 4)])


class Test_XlogY0():
    def setup_method(self):
        utt.seed_rng()

    def test2(self):
        utt.verify_grad(xlogy0, [numpy.random.rand(3, 4), numpy.random.rand(3, 4)])

    def test3(self):
        x = as_tensor_variable([1, 0])
        y = as_tensor_variable([1, 0])
        z = xlogy0(x, y)
        f = theano.function([], z)
        assert numpy.all(f() == numpy.asarray([0, 0.]))

