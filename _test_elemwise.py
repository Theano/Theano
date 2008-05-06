
import time
import unittest

from gof import Result, Op, Env
import gof

from scalar import *

import tensor
from elemwise import *



class _test_DimShuffle(unittest.TestCase):

    def with_linker(self, linker):
        for xsh, shuffle, zsh in [((2, 3), (1, 'x', 0), (3, 1, 2)),
                                  ((1, 2, 3), (1, 2), (2, 3)),
                                  ((1, 2, 1, 3), (1, 3), (2, 3)),
                                  ((2, 3, 4), (2, 1, 0), (4, 3, 2)),
                                  ((2, 3, 4), ('x', 2, 1, 0, 'x'), (1, 4, 3, 2, 1)),
                                  ((1, 4, 3, 2, 1), (3, 2, 1), (2, 3, 4)),
                                  ((1, 1, 4), (1, 2), (1, 4))]:
            ib = [(entry == 1) for entry in xsh]
            x = Tensor('float64', ib)('x')
            e = DimShuffle(ib, shuffle)(x)
            f = linker(Env([x], [e])).make_function()
            assert f(numpy.ones(xsh)).shape == zsh

    def test_perform(self):
        self.with_linker(gof.PerformLinker)


class _test_Broadcast(unittest.TestCase):

    def with_linker(self, linker):
        for xsh, ysh in [((3, 5), (3, 5)),
                         ((3, 5), (1, 5)),
                         ((3, 5), (3, 1)),
                         ((1, 5), (5, 1)),
                         ((1, 1), (1, 1)),
                         ((2, 3, 4, 5), (2, 3, 4, 5)),
                         ((2, 3, 4, 5), (1, 3, 1, 5)),
                         ((2, 3, 4, 5), (1, 1, 1, 1)),
                         ((), ())]:
            x = Tensor('float64', [(entry == 1) for entry in xsh])('x')
            y = Tensor('float64', [(entry == 1) for entry in ysh])('y')
            e = Elemwise(add)(x, y)
            f = linker(Env([x, y], [e])).make_function()
            xv = numpy.asarray(numpy.random.rand(*xsh))
            yv = numpy.asarray(numpy.random.rand(*ysh))
            zv = xv + yv

            self.failUnless((f(xv, yv) == zv).all())

    def with_linker_inplace(self, linker):
        for xsh, ysh in [((5, 5), (5, 5)),
                         ((5, 5), (1, 5)),
                         ((5, 5), (5, 1)),
                         ((1, 1), (1, 1)),
                         ((2, 3, 4, 5), (2, 3, 4, 5)),
                         ((2, 3, 4, 5), (1, 3, 1, 5)),
                         ((2, 3, 4, 5), (1, 1, 1, 1)),
                         ((), ())]:
            x = Tensor('float64', [(entry == 1) for entry in xsh])('x')
            y = Tensor('float64', [(entry == 1) for entry in ysh])('y')
            e = Elemwise(Add(transfer_type(0)), {0:0})(x, y)
            f = linker(Env([x, y], [e])).make_function()
            xv = numpy.asarray(numpy.random.rand(*xsh))
            yv = numpy.asarray(numpy.random.rand(*ysh))
            zv = xv + yv

            f(xv, yv)

            self.failUnless((xv == zv).all())

    def test_perform(self):
        self.with_linker(gof.PerformLinker)

    def test_c(self):
        self.with_linker(gof.CLinker)

    def test_perform_inplace(self):
        self.with_linker_inplace(gof.PerformLinker)

    def test_c_inplace(self):
        self.with_linker_inplace(gof.CLinker)

    def test_fill(self):
        x = Tensor('float64', [0, 0])('x')
        y = Tensor('float64', [1, 1])('y')
        e = Elemwise(Second(transfer_type(0)), {0:0})(x, y)
        f = gof.CLinker(Env([x, y], [e])).make_function()
        xv = numpy.ones((5, 5))
        yv = numpy.random.rand(1, 1)
        f(xv, yv)
        assert (xv == yv).all()

    def test_weird_strides(self):
        x = Tensor('float64', [0, 0, 0, 0, 0])('x')
        y = Tensor('float64', [0, 0, 0, 0, 0])('y')
        e = Elemwise(add)(x, y)
        f = gof.CLinker(Env([x, y], [e])).make_function()
        xv = numpy.random.rand(2, 2, 2, 2, 2)
        yv = numpy.random.rand(2, 2, 2, 2, 2).transpose(4, 0, 3, 1, 2)
        zv = xv + yv
        assert (f(xv, yv) == zv).all()

    def test_same_inputs(self):
        x = Tensor('float64', [0, 0])('x')
        e = Elemwise(add)(x, x)
        f = gof.CLinker(Env([x], [e])).make_function()
        xv = numpy.random.rand(2, 2)
        zv = xv + xv
        assert (f(xv) == zv).all()


class _test_CAReduce(unittest.TestCase):

    def with_linker(self, linker):
        for xsh, tosum in [((5, 6), None),
                           ((5, 6), (0, 1)),
                           ((5, 6), (0, )),
                           ((5, 6), (1, )),
                           ((5, 6), ()),
                           ((2, 3, 4, 5), (0, 1, 3)),
                           ((), ())]:
            x = Tensor('float64', [(entry == 1) for entry in xsh])('x')
            e = CAReduce(add, axis = tosum)(x)
            if tosum is None: tosum = range(len(xsh))
            f = linker(Env([x], [e])).make_function()
            xv = numpy.asarray(numpy.random.rand(*xsh))
            zv = xv
            for axis in reversed(sorted(tosum)):
                zv = numpy.add.reduce(zv, axis)
            self.failUnless((numpy.abs(f(xv) - zv) < 1e-10).all())

    def test_perform(self):
        self.with_linker(gof.PerformLinker)

    def test_c(self):
        self.with_linker(gof.CLinker)
        

if __name__ == '__main__':
    unittest.main()
