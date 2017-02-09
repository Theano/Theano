from __future__ import absolute_import, print_function, division
from copy import copy
import unittest
import math

import numpy
from nose.plugins.skip import SkipTest
from nose.tools import raises
from six.moves import xrange
import six.moves.cPickle as pickle

import theano
from theano.compat import imap
from theano import gof, scalar, config

from theano import tensor
from theano.tensor import TensorType, as_tensor_variable
from theano.compile.mode import get_default_mode
from theano.tensor.elemwise import (CAReduce, Elemwise, DimShuffle,
                                    Prod, ProdWithoutZeros)
from theano.tests import unittest_tools
from theano.tests.unittest_tools import attr


def FunctionGraph(i, o):
    e = gof.FunctionGraph(i, o)
    return e


class test_DimShuffle(unittest_tools.InferShapeTester):
    op = DimShuffle
    type = TensorType
    dtype = theano.config.floatX

    def with_linker(self, linker):
        for xsh, shuffle, zsh in [((2, 3), (1, 'x', 0), (3, 1, 2)),
                                  ((1, 2, 3), (1, 2), (2, 3)),
                                  ((1, 2, 1, 3), (1, 3), (2, 3)),
                                  ((2, 3, 4), (2, 1, 0), (4, 3, 2)),
                                  ((2, 3, 4), ('x', 2, 1, 0, 'x'),
                                   (1, 4, 3, 2, 1)),
                                  ((1, 4, 3, 2, 1), (3, 2, 1), (2, 3, 4)),
                                  ((1, 1, 4), (1, 2), (1, 4)),
                                  ((1, 1, 1), (), ()),
                                  ((1,), ('x', 'x'), (1, 1))]:
            ib = [(entry == 1) for entry in xsh]
            x = self.type(self.dtype, ib)('x')
            e = self.op(ib, shuffle)(x)
            f = copy(linker).accept(FunctionGraph([x], [e])).make_function()
            assert f(numpy.ones(xsh, dtype=self.dtype)).shape == zsh
            # test that DimShuffle.infer_shape work correctly
            x = self.type(self.dtype, ib)('x')
            e = self.op(ib, shuffle)(x)
            f = copy(linker).accept(FunctionGraph([x],
                                                  [e.shape])).make_function()
            assert all(f(numpy.ones(xsh, dtype=self.dtype))) == all(zsh)

        # Test when we drop a axis that is not broadcastable
        ib = [False, True, False]
        x = self.type(self.dtype, ib)('x')
        self.assertRaises(ValueError, self.op, ib, shuffle)

        # Test when we drop a axis that don't have shape 1
        ib = [True, True, False]
        x = self.type(self.dtype, ib)('x')
        e = self.op(ib, (1, 2))(x)
        f = copy(linker).accept(FunctionGraph([x], [e.shape])).make_function()
        self.assertRaises(TypeError, f, numpy.ones((2, 1, 4)))

        # Test that we can't take a dimensions multiple time
        xsh, shuffle, zsh = ((1, 1, 4), (0, 1, 2, 0), (1, 4))
        ib = [False, True, False]
        x = self.type(self.dtype, ib)('x')
        self.assertRaises(ValueError, DimShuffle, ib, shuffle)

    def test_perform(self):
        self.with_linker(gof.PerformLinker())

    def test_c_or_py(self):
        # Shape op don't have C code.
        # But This will test DimShuffle c code
        self.with_linker(gof.OpWiseCLinker())

    def test_infer_shape(self):

        for xsh, shuffle in [((2, 3), (1, 'x', 0)),
                             ((1, 2, 3), (1, 2)),
                             ((1, 2, 1, 3), (1, 3)),
                             ((2, 3, 4), (2, 1, 0)),
                             ((2, 3, 4), ('x', 2, 1, 0, 'x')),
                             ((1, 4, 3, 2, 1), (3, 2, 1)),
                             ((1, 1, 4), (1, 2)),
                             ((1, 1, 1), ()),
                             ((1,), ('x', 'x'))]:
            ib = [(entry == 1) for entry in xsh]
            adtens = self.type(self.dtype, ib)('x')
            adtens_val = numpy.ones(xsh, dtype=self.dtype)
            self._compile_and_check([adtens],
                                    [self.op(ib, shuffle)(adtens)],
                                    [adtens_val], self.op,
                                    warn=False)

    def test_too_big_rank(self):
        x = self.type(self.dtype, broadcastable=())()
        y = x.dimshuffle(('x',) * (numpy.MAXDIMS + 1))
        self.assertRaises(ValueError, y.eval, {x: 0})


class test_reduce_axes(unittest.TestCase):

    def test_sum_axes(self):
        axes = [None, 0, 1, [0, 1], numpy.array(1),
                [numpy.array(0), numpy.array(1)]]
        for a in axes:
            x = tensor.matrix()
            x.sum(a)

    def test_mean_axes(self):
        axes = [None, 0, 1, [0, 1], numpy.array(1),
                [numpy.array(0), numpy.array(1)]]
        for a in axes:
            x = tensor.matrix()
            x.mean(a)

    def test_max_axes(self):
        axes = [None, 0, 1, [0, 1], numpy.array(1),
                [numpy.array(0), numpy.array(1)]]
        for a in axes:
            x = tensor.matrix()
            x.max(a)

    def test_min_axes(self):
        axes = [None, 0, 1, [0, 1], numpy.array(1),
                [numpy.array(0), numpy.array(1)]]
        for a in axes:
            x = tensor.matrix()
            x.min(a)

    def test_argmax_axes(self):
        axes = [None, 0, 1, [0, 1], numpy.array(1),
                [numpy.array(0), numpy.array(1)]]
        for a in axes:
            x = tensor.matrix()
            x.argmax(a)

    def test_var_axes(self):
        axes = [None, 0, 1, [0, 1], numpy.array(1),
                [numpy.array(0), numpy.array(1)]]
        for a in axes:
            x = tensor.matrix()
            x.var(a)


class test_Broadcast(unittest.TestCase):
    # this is to allow other types to reuse this class to test their ops
    type = TensorType
    op = Elemwise

    ctype = TensorType
    cop = Elemwise

    openmp_minsize = 2 * config.openmp_elemwise_minsize
    openmp_minsize_sqrt = int(math.ceil(math.sqrt(openmp_minsize)))

    # The order is important if you change them.
    linkers = [gof.PerformLinker, gof.CLinker]

    def rand_val(self, shp):
        return numpy.asarray(numpy.random.rand(*shp),
                             dtype=theano.config.floatX)

    def rand_cval(self, shp):
        return numpy.asarray(numpy.random.rand(*shp),
                             dtype=theano.config.floatX)

    def setUp(self):
        unittest_tools.seed_rng()

    def with_linker(self, linker, op, type, rand_val):
        for xsh, ysh in [((3, 5), (3, 5)),
                         ((3, 5), (1, 5)),
                         ((3, 5), (3, 1)),
                         ((1, 5), (5, 1)),
                         ((1, 1), (1, 1)),
                         ((self.openmp_minsize,), (self.openmp_minsize,)),
                         ((self.openmp_minsize_sqrt,
                           self.openmp_minsize_sqrt),
                          (self.openmp_minsize_sqrt,
                           self.openmp_minsize_sqrt)),
                         ((2, 3, 4, 5), (2, 3, 4, 5)),
                         ((2, 3, 4, 5), (1, 3, 1, 5)),
                         ((2, 3, 4, 5), (1, 1, 1, 1)),
                         ((), ())]:
            x = type(theano.config.floatX,
                     [(entry == 1) for entry in xsh])('x')
            y = type(theano.config.floatX,
                     [(entry == 1) for entry in ysh])('y')
            e = op(scalar.add)(x, y)
            f = copy(linker).accept(FunctionGraph([x, y], [e])).make_function()
            xv = rand_val(xsh)
            yv = rand_val(ysh)
            zv = xv + yv

            unittest_tools.assert_allclose(f(xv, yv), zv)

            # test Elemwise.infer_shape
            # the Shape op don't implement c_code!
            if isinstance(linker, gof.PerformLinker):
                x = type(theano.config.floatX,
                         [(entry == 1) for entry in xsh])('x')
                y = type(theano.config.floatX,
                         [(entry == 1) for entry in ysh])('y')
                e = op(scalar.add)(x, y)
                f = copy(linker).accept(FunctionGraph(
                    [x, y], [e.shape])).make_function()
                assert tuple(f(xv, yv)) == tuple(zv.shape)

    def with_linker_inplace(self, linker, op, type, rand_val):
        for xsh, ysh in [((5, 5), (5, 5)),
                         ((5, 5), (1, 5)),
                         ((5, 5), (5, 1)),
                         ((1, 1), (1, 1)),
                         ((2, 3, 4, 5), (2, 3, 4, 5)),
                         ((2, 3, 4, 5), (1, 3, 1, 5)),
                         ((2, 3, 4, 5), (1, 1, 1, 1)),
                         ((), ())]:
            x = type(theano.config.floatX,
                     [(entry == 1) for entry in xsh])('x')
            y = type(theano.config.floatX,
                     [(entry == 1) for entry in ysh])('y')
            e = op(scalar.Add(scalar.transfer_type(0)), {0: 0})(x, y)
            f = copy(linker).accept(FunctionGraph([x, y], [e])).make_function()
            xv = rand_val(xsh)
            yv = rand_val(ysh)
            zv = xv + yv

            f(xv, yv)

            self.assertTrue((xv == zv).all())
            # test Elemwise.infer_shape
            # the Shape op don't implement c_code!
            if isinstance(linker, gof.PerformLinker):
                x = type(theano.config.floatX,
                         [(entry == 1) for entry in xsh])('x')
                y = type(theano.config.floatX,
                         [(entry == 1) for entry in ysh])('y')
                e = op(scalar.Add(scalar.transfer_type(0)), {0: 0})(x, y)
                f = copy(linker).accept(FunctionGraph(
                    [x, y], [e.shape])).make_function()
                xv = rand_val(xsh)
                yv = rand_val(ysh)
                zv = xv + yv

                f(xv, yv)

                assert xv.shape == zv.shape

    def test_perform(self):
        self.with_linker(gof.PerformLinker(), self.op, self.type,
                         self.rand_val)

    def test_c(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        self.with_linker(gof.CLinker(), self.cop, self.ctype, self.rand_cval)

    def test_perform_inplace(self):
        self.with_linker_inplace(gof.PerformLinker(), self.op, self.type,
                                 self.rand_val)

    def test_c_inplace(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        self.with_linker_inplace(gof.CLinker(), self.cop, self.ctype,
                                 self.rand_cval)

    def test_fill(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        for linker, op, t, rval in zip(self.linkers, [self.op, self.cop],
                                       [self.type, self.ctype],
                                       [self.rand_val, self.rand_cval]):
            x = t(theano.config.floatX, [0, 0])('x')
            y = t(theano.config.floatX, [1, 1])('y')
            e = op(scalar.Second(scalar.transfer_type(0)), {0: 0})(x, y)
            f = linker().accept(FunctionGraph([x, y], [e])).make_function()
            xv = rval((5, 5))
            yv = rval((1, 1))
            f(xv, yv)
            assert (xv == yv).all()

    def test_fill_var(self):
        x = tensor.matrix()
        x.fill(3)

    def test_fill_grad(self):
        # Fix bug reported at
        # https://groups.google.com/d/topic/theano-users/nQshB8gUA6k/discussion
        x = TensorType(config.floatX, [0, 1, 0])('x')
        y = TensorType(config.floatX, [0, 1, 0])('y')
        e = tensor.second(x, y)
        theano.grad(e.sum(), y)

    def test_weird_strides(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        for linker, op, t, rval in zip(self.linkers, [self.op, self.cop],
                                       [self.type, self.ctype],
                                       [self.rand_val, self.rand_cval]):
            x = t(theano.config.floatX, [0, 0, 0, 0, 0])('x')
            y = t(theano.config.floatX, [0, 0, 0, 0, 0])('y')
            e = op(scalar.add)(x, y)
            f = linker().accept(FunctionGraph([x, y], [e])).make_function()
            xv = rval((2, 2, 2, 2, 2))
            yv = rval((2, 2, 2, 2, 2)).transpose(4, 0, 3, 1, 2)
            zv = xv + yv
            assert (f(xv, yv) == zv).all()

    def test_same_inputs(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        for linker, op, t, rval in zip(self.linkers, [self.op, self.cop],
                                       [self.type, self.ctype],
                                       [self.rand_val, self.rand_cval]):
            x = t(theano.config.floatX, [0, 0])('x')
            e = op(scalar.add)(x, x)
            f = linker().accept(FunctionGraph([x], [e])).make_function()
            xv = rval((2, 2))
            zv = xv + xv
            assert (f(xv) == zv).all()


def reduce_bitwise_and(x, axis=-1, dtype='int8'):
    identity = numpy.array((-1,), dtype=dtype)[0]

    shape_without_axis = tuple([s for i, s in enumerate(x.shape) if i != axis])
    if 0 in shape_without_axis:
        return numpy.empty(shape=shape_without_axis, dtype=x.dtype)

    def custom_reduce(a):
        out = identity
        for i in range(a.size):
            out = numpy.bitwise_and(a[i], out)
        return out

    return numpy.apply_along_axis(custom_reduce, axis, x)


class test_CAReduce(unittest_tools.InferShapeTester):
    op = CAReduce
    cases = [((5, 6), None),
             ((5, 6), (0, 1)),
             ((5, 6), (0, )),
             ((5, 6), (1, )),
             ((5, 6), (-1, )),
             ((5, 6), (-2, )),
             ((5, 6), ()),
             ((2, 3, 4, 5), (0, 1, 3)),
             ((2, 3, 4, 5), (-2, -3)),
             ((5, 0), None),
             ((5, 0), (0, )),
             ((5, 0), (1, )),
             ((5, 0), ()),
             ((), None),
             ((), ())]
    type = TensorType

    def with_linker(self, linker, scalar_op=scalar.add, dtype="floatX",
                    pre_scalar_op=None,
                    test_nan=False, tensor_op=None):
        for xsh, tosum in self.cases:
            if dtype == "floatX":
                dtype = theano.config.floatX
            x = self.type(dtype, [(entry == 1) for entry in xsh])('x')
            d = {}
            if pre_scalar_op is not None:
                d = {"pre_scalar_op": pre_scalar_op}
            if tensor_op is None:
                e = as_tensor_variable(self.op(scalar_op, axis=tosum, **d)(x))
            else:
                e = as_tensor_variable(tensor_op(x, axis=tosum, **d))

            if tosum is None:
                tosum = list(range(len(xsh)))

            f = copy(linker).accept(FunctionGraph([x], [e])).make_function()
            xv = numpy.asarray(numpy.random.rand(*xsh))

            if dtype not in tensor.discrete_dtypes:
                xv = numpy.asarray(xv, dtype=dtype)
            else:
                xv = numpy.asarray(xv < 0.5, dtype=dtype)

            if test_nan and xv.size > 0:
                if len(xsh) > 0:
                    xv = xv.flatten()
                    xv[0] = numpy.nan
                    xv = xv.reshape(*xsh)
                else:
                    xv = numpy.asarray(numpy.nan, dtype=dtype)
            zv = xv
            if pre_scalar_op is not None:
                zv = Elemwise(scalar_op=pre_scalar_op)(x).eval({x: xv})
            numpy_raised = False
            if len(tosum) > 1 and any([a < 0 for a in tosum]):
                # In that case, we need to use the good order of axis
                # in the reduction.
                axis2 = []
                for a in tosum:
                    if a < 0:
                        axis2.append(a + len(xsh))
                    else:
                        axis2.append(a)
                assert len(axis2) == len(tosum)
                tosum = tuple(axis2)
            if tensor_op == tensor.all:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.all(zv, axis)
                if len(tosum) == 0:
                    zv = zv != 0
            elif tensor_op == tensor.any:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.any(zv, axis)
                if len(tosum) == 0:
                    zv = zv != 0
            elif scalar_op == scalar.add:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.add.reduce(zv, axis)
            elif scalar_op == scalar.mul:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.multiply.reduce(zv, axis)
            elif scalar_op == scalar.maximum:
                try:
                    for axis in reversed(sorted(tosum)):
                        zv = numpy.maximum.reduce(zv, axis)
                except ValueError:
                    numpy_raised = True
            elif scalar_op == scalar.minimum:
                try:
                    for axis in reversed(sorted(tosum)):
                        zv = numpy.minimum.reduce(zv, axis)
                except ValueError:
                    numpy_raised = True
            elif scalar_op == scalar.or_:
                for axis in reversed(sorted(tosum)):
                    zv = numpy.bitwise_or.reduce(zv, axis)
            elif scalar_op == scalar.and_:
                for axis in reversed(sorted(tosum)):
                    zv = reduce_bitwise_and(zv, axis, dtype=dtype)
            elif scalar_op == scalar.xor:
                # There is no identity value for the xor function
                # So we can't support shape of dimensions 0.
                if numpy.prod(zv.shape) == 0:
                    continue
                for axis in reversed(sorted(tosum)):
                    zv = numpy.bitwise_xor.reduce(zv, axis)
            else:
                raise Exception(
                    "Test for CAReduce with scalar_op %s not implemented" %
                    str(scalar_op))
            if scalar_op in [scalar.maximum, scalar.minimum] and numpy_raised:
                try:
                    out = f(xv)
                    assert out.dtype == dtype
                except ValueError:
                    pass
                else:
                    self.fail()
            else:
                if test_nan:
                    try:
                        self.assertTrue(
                            self.type.values_eq(f(xv), zv),
                            (f(xv), zv))
                    except NotImplementedError:
                        # GpuCAReduce don't implement all cases when size is 0
                        assert xv.size == 0
                else:
                    try:
                        f_xv = f(xv)
                        self.assertTrue((f_xv.shape == zv.shape), (f_xv, zv))
                        self.assertTrue(numpy.allclose(f_xv, zv),
                                        (f_xv, zv, xsh, tosum))
                    except NotImplementedError:
                        # GpuCAReduce don't implement all cases when size is 0
                        assert xv.size == 0

            x = self.type(dtype, [(entry == 1) for entry in xsh])('x')
            if tensor_op is None:
                e = self.op(scalar_op, axis=tosum)(x)
            else:
                e = tensor_op(x, axis=tosum)
            if tosum is None:
                tosum = list(range(len(xsh)))
            f = copy(linker).accept(FunctionGraph([x],
                                                  [e.shape])).make_function()
            if not(scalar_op in [scalar.maximum, scalar.minimum] and
                   ((xsh == () or numpy.prod(xsh) == 0))):
                try:
                    assert all(f(xv) == zv.shape)
                except NotImplementedError:
                    # GpuCAReduce don't implement all cases when size is 0
                    assert xv.size == 0

    def test_perform(self):
        for dtype in ["floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_linker(gof.PerformLinker(), scalar.add, dtype=dtype)
            self.with_linker(gof.PerformLinker(), scalar.mul, dtype=dtype)
            self.with_linker(gof.PerformLinker(), scalar.maximum, dtype=dtype)
            self.with_linker(gof.PerformLinker(), scalar.minimum, dtype=dtype)
            self.with_linker(gof.PerformLinker(), scalar.and_, dtype=dtype,
                             tensor_op=tensor.all)
            self.with_linker(gof.PerformLinker(), scalar.or_, dtype=dtype,
                             tensor_op=tensor.any)
        for dtype in ["int8", "uint8"]:
            self.with_linker(gof.PerformLinker(), scalar.or_, dtype=dtype)
            self.with_linker(gof.PerformLinker(), scalar.and_, dtype=dtype)
            self.with_linker(gof.PerformLinker(), scalar.xor, dtype=dtype)

    def test_perform_nan(self):
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_linker(gof.PerformLinker(), scalar.add, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), scalar.mul, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), scalar.maximum, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), scalar.minimum, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.PerformLinker(), scalar.or_, dtype=dtype,
                             test_nan=True, tensor_op=tensor.any)
            self.with_linker(gof.PerformLinker(), scalar.and_, dtype=dtype,
                             test_nan=True, tensor_op=tensor.all)

    @attr('slow')
    def test_c(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")

        for dtype in ["floatX", "complex64", "complex128", "int8", "uint8"]:
            self.with_linker(gof.CLinker(), scalar.add, dtype=dtype)
            self.with_linker(gof.CLinker(), scalar.mul, dtype=dtype)
        for dtype in ["floatX", "int8", "uint8"]:
            self.with_linker(gof.CLinker(), scalar.minimum, dtype=dtype)
            self.with_linker(gof.CLinker(), scalar.maximum, dtype=dtype)
            self.with_linker(gof.CLinker(), scalar.and_, dtype=dtype,
                             tensor_op=tensor.all)
            self.with_linker(gof.CLinker(), scalar.or_, dtype=dtype,
                             tensor_op=tensor.any)
        for dtype in ["int8", "uint8"]:
            self.with_linker(gof.CLinker(), scalar.or_, dtype=dtype)
            self.with_linker(gof.CLinker(), scalar.and_, dtype=dtype)
            self.with_linker(gof.CLinker(), scalar.xor, dtype=dtype)

    @attr('slow')
    def test_c_nan(self):
        if not theano.config.cxx:
            raise SkipTest("G++ not available, so we need to skip this test.")
        for dtype in ["floatX", "complex64", "complex128"]:
            self.with_linker(gof.CLinker(), scalar.add, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.CLinker(), scalar.mul, dtype=dtype,
                             test_nan=True)
        for dtype in ["floatX"]:
            self.with_linker(gof.CLinker(), scalar.minimum, dtype=dtype,
                             test_nan=True)
            self.with_linker(gof.CLinker(), scalar.maximum, dtype=dtype,
                             test_nan=True)

    def test_infer_shape(self, dtype=None, pre_scalar_op=None):
        if dtype is None:
            dtype = theano.config.floatX
        for xsh, tosum in self.cases:
            x = self.type(dtype, [(entry == 1) for entry in xsh])('x')
            if pre_scalar_op is not None:
                x = pre_scalar_op(x)
            if tosum is None:
                tosum = list(range(len(xsh)))
            xv = numpy.asarray(numpy.random.rand(*xsh), dtype=dtype)
            d = {}
            if pre_scalar_op is not None:
                xv = x.eval({x.owner.inputs[0]: xv})
                d = {pre_scalar_op: pre_scalar_op}
            self._compile_and_check([x],
                                    [self.op(scalar.add, axis=tosum, *d)(x)],
                                    [xv], self.op,
                                    ["local_cut_useless_reduce"],
                                    warn=0 not in xsh)


class test_Prod(unittest.TestCase):
    def setUp(self):
        unittest_tools.seed_rng()

        # we want to allow nans in the matrices, so we disable this
        # DEBUG_MODE check
        mode = theano.compile.mode.get_default_mode()
        mode = copy(mode)
        mode.check_isfinite = False

        self.mode = mode

    @attr('slow')
    def test_verify_grad(self):

        # including zeros, as the case with zeros is important
        # (and special cases: 1 zero in the row, more than 1 zero in the row)
        x_val = numpy.asarray([[.1, .2, .3], [.4, .5, .6], [.7, .8, .9]],
                              dtype='float32')
        # now with verify_grad
        unittest_tools.verify_grad(Prod(axis=1), [x_val], mode=self.mode)

        # second time, with some added complexity
        # verify_grad takes the sum of the matrices anyway
        def fn(x2):
            return theano.tensor.sqr(Prod(axis=1)(x2))

        unittest_tools.verify_grad(fn, [x_val], mode=self.mode)

    def test_verify_grad_with_zeros(self):
        # including zeros, as the case with zeros is important
        # (and special cases: 1 zero in the row, more than 1 zero in the row)
        x_val = numpy.asarray([[1., 2., 3.], [0., 5., 6.], [0., 0., 9.]],
                              dtype='float32')
        x = theano.tensor.dmatrix()

        # sanity check
        p = Prod(axis=1)(x)

        # Uncomment this for debugging if needed
        # x2 = theano.tensor.dmatrix()
        # p2 = Prod(axis=1)(x2)
        # fn = theano.function([x, x2], [p - p2], mode=self.mode)
        # print("hand computed diff for each row")
        # x2_val = numpy.asarray([[1., 2., 3.003], [0.003, 5., 6], [
        #     0., 0., 9.01]])
        # print(fn(x_val, x2_val))
        # fn2 = theano.function([x], [theano.tensor.grad(p.sum(), x)],
        #                       mode=self.mode)
        # print("real grad")
        # print(fn2(x_val))
        fn3 = theano.function([x], [p], mode=self.mode)
        assert numpy.allclose(fn3(x_val), [6., 0., 0.])

        # now with verify_grad
        unittest_tools.verify_grad(Prod(axis=1), [x_val], mode=self.mode)

        # second time, with some added complexity
        # verify_grad takes the sum of the matrices anyway
        # def fn5(x5):
        #    return theano.tensor.sqr(Prod(axis=1)(x5))

        # x4 = theano.tensor.dmatrix()
        # p4 = theano.tensor.sqr(Prod(axis=1)(x4))
        # fn4 = theano.function([x4], p4)
        # print("with sqr")
        # print(fn4(x_val))
        # print(fn4(x2_val))

        # unittest_tools.verify_grad(fn5, [x_val])

    @attr('slow')
    def test_prod_no_zeros_in_input(self):
        x = theano.tensor.dmatrix()
        x_val = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')
        pwz = Prod(axis=1, no_zeros_in_input=True)(x)
        fn = theano.function([x], pwz, mode=self.mode)

        assert numpy.allclose(fn(x_val), [6, 120, 504])

        pwz = Prod(no_zeros_in_input=True)(x)
        g = theano.grad(pwz, x)
        gg = theano.grad(g.sum(), x)
        fn = theano.function([x], g, mode=self.mode)
        assert numpy.allclose(fn(x_val),
                              [[362880., 181440., 120960.],
                               [90720., 72576., 60480.],
                               [51840., 45360., 40320.]])
        fn = theano.function([x], gg, mode=self.mode)
        assert numpy.allclose(fn(x_val),
                              [[663696., 422568., 301872.],
                               [233964., 190800., 161016.],
                               [139248., 122652., 109584.]])
        unittest_tools.verify_grad(Prod(axis=1, no_zeros_in_input=True),
                                   [x_val],
                                   mode=self.mode)
        unittest_tools.verify_grad(Prod(no_zeros_in_input=True), [x_val],
                                   mode=self.mode)

        def second_deriv(x):
            return theano.grad(Prod(no_zeros_in_input=True)(x), x)
        unittest_tools.verify_grad(second_deriv, [x_val],
                                   mode=self.mode)

    def test_prod_without_zeros(self):
        x = theano.tensor.dmatrix()
        x_val = numpy.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]], dtype='float32')
        pwz = ProdWithoutZeros(axis=1)(x)
        fn = theano.function([x], pwz, mode=self.mode)
        assert numpy.allclose(fn(x_val), [6, 30, 9])

        pwz_a0 = ProdWithoutZeros(axis=0)(x)
        fn_a0 = theano.function([x], pwz_a0, mode=self.mode)
        assert numpy.allclose(fn_a0(x_val), [1, 10, 162])

    @raises(theano.gradient.NullTypeGradError)
    def test_prod_without_zeros_grad(self):
        x = theano.tensor.dmatrix()
        pwz_a1 = ProdWithoutZeros(axis=0)(x)
        pwz_grad = theano.grad(theano.tensor.sum(pwz_a1), x)
        theano.function([x], pwz_grad, mode=self.mode)

    @attr('slow')
    def test_other_grad_tests(self):
        x = theano.tensor.dmatrix()
        x_val1 = numpy.array([[1, 2, 3], [0, 5, 6], [0, 0, 9]],
                             dtype='float32')
        x_val2 = numpy.array([[1, 2, 0], [0, 5, 6], [7, 8, 9], [9, 10, 0]],
                             dtype='float32')
        rng = rng = numpy.random.RandomState(43)

        p = Prod(axis=1)
        grad_p = theano.tensor.grad(p(x).sum(), x)
        grad_fn = theano.function([x], grad_p, mode=self.mode)
        assert numpy.allclose(
            grad_fn(x_val1),
            [[6., 3., 2.], [30., 0., 0.], [0., 0., 0.]])
        assert numpy.allclose(
            grad_fn(x_val2),
            [[0., 0., 2.], [30., 0., 0.], [72., 63., 56.], [0., 0., 90.]])

        p_axis0 = Prod(axis=0)
        grad_p_axis0 = theano.tensor.grad(p_axis0(x).sum(), x)
        grad_fn_axis0 = theano.function([x], grad_p_axis0, mode=self.mode)
        assert numpy.allclose(
            grad_fn_axis0(x_val2),
            [[0., 400., 0.], [63., 160., 0.], [0., 100., 0.], [0., 80., 0.]])

        tensor.verify_grad(p, [x_val1], rng=rng, mode=self.mode)

    def test_mul_without_zeros_zeros(self):
        a = numpy.zeros((3, 3))

        x = theano.tensor.dmatrix()

        mul1 = ProdWithoutZeros(axis=0)(x)

        fn_debug = theano.function([x], mul1, mode=self.mode)

        fn_debug(a)

    def test_pickle_bug(self):
        # Regression test for bug fixed in 24d4fd291054.
        o = Prod()
        s = pickle.dumps(o, protocol=-1)
        o = pickle.loads(s)
        pickle.dumps(o)


class test_IsInf_IsNan(unittest.TestCase):

    def setUp(self):
        self.test_vals = [numpy.array(x, dtype=config.floatX) for x in [
            0,
            1,
            numpy.nan,
            numpy.inf,
            -numpy.inf,
            [numpy.nan, numpy.inf, -numpy.inf, 0, 1, -1],
            ]]
        self.scalar = tensor.scalar()
        self.vector = tensor.vector()
        self.mode = get_default_mode()
        if isinstance(self.mode, theano.compile.debugmode.DebugMode):
            # Disable the check preventing usage of NaN / Inf values.
            self.mode = copy(self.mode)
            self.mode.check_isfinite = False

    def run_isfunc(self, isfunc):
        for input in (self.scalar, self.vector):
            theano_isfunc = theano.function([input],
                                            getattr(tensor, isfunc)(input),
                                            mode=self.mode)
            numpy_isfunc = getattr(numpy, isfunc)
            for x in self.test_vals:
                if ((x.ndim == 0 and input is not self.scalar) or
                        (x.ndim == 1 and input is not self.vector)):
                    # We only test with the appropriate input type.
                    continue
                t_out = theano_isfunc(x)
                n_out = numpy_isfunc(x)
                assert (t_out == n_out).all(), (t_out, n_out)

    def test_isinf(self):
        return self.run_isfunc('isinf')

    def test_isnan(self):
        return self.run_isfunc('isnan')


class T_reduce_dtype(unittest.TestCase):
    mode = theano.compile.get_default_mode().excluding(
        'local_cut_useless_reduce')
    op = CAReduce
    axes = [None, 0, 1, [], [0], [1], [0, 1]]
    methods = ['sum', 'prod']
    dtypes = list(imap(str, theano.scalar.all_types))

    # Test the default dtype of a method().
    def test_reduce_default_dtype(self):
        # We try multiple axis combinations even though axis should not matter.
        for method in self.methods:
            for idx, dtype in enumerate(self.dtypes):
                axis = self.axes[idx % len(self.axes)]
                x = tensor.matrix(dtype=dtype)
                s = getattr(x, method)(axis=axis)
                assert s.dtype == dict(
                    bool='int64',
                    int8='int64',
                    int16='int64',
                    int32='int64',
                    uint8='uint64',
                    uint16='uint64',
                    uint32='uint64',
                ).get(dtype, dtype)
                f = theano.function([x], s, mode=self.mode)
                topo = f.maker.fgraph.toposort()
                assert [n for n in topo if isinstance(n.op, self.op)], (topo,
                                                                        dtype)
                data = numpy.random.rand(3, 4) * 10
                data = data.astype(dtype)
                f(data)

    def test_reduce_default_acc_dtype(self):
        # Test the default acc_dtype of a reduce().

        # We try multiple axis combinations even though axis should not matter.
        for method in self.methods:
            for idx, dtype in enumerate(self.dtypes):
                axis = self.axes[idx % len(self.axes)]
                x = tensor.matrix(dtype=dtype)
                s = getattr(x, method)(axis=axis)
                assert s.owner.op.acc_dtype == dict(
                    bool='int64',
                    int8='int64',
                    int16='int64',
                    int32='int64',
                    uint8='uint64',
                    uint16='uint64',
                    uint32='uint64',
                    float16='float32',
                    float32='float64',
                    complex64='complex128',
                ).get(dtype, dtype)
                f = theano.function([x], s, mode=self.mode)
                topo = f.maker.fgraph.toposort()
                assert [n for n in topo if isinstance(n.op, self.op)], (topo,
                                                                        dtype)
                data = numpy.random.rand(3, 4) * 10
                data = data.astype(dtype)
                f(data)

    @attr('slow')
    def test_reduce_custom_dtype(self):
        # Test the ability to provide your own output dtype for a reduce.

        # We try multiple axis combinations even though axis should not matter.
        idx = 0
        for method in self.methods:
            for input_dtype in self.dtypes:
                x = tensor.matrix(dtype=input_dtype)
                for output_dtype in self.dtypes:
                    # Only tests case where both input and output are complex.
                    icomplex = input_dtype.startswith('complex')
                    ocomplex = output_dtype.startswith('complex')
                    if icomplex != ocomplex:
                        continue

                    axis = self.axes[idx % len(self.axes)]
                    var = getattr(x, method)(dtype=output_dtype, axis=axis)
                    assert var.dtype == output_dtype

                    f = theano.function([x], var, mode=self.mode)
                    topo = f.maker.fgraph.toposort()
                    assert [n for n in topo if isinstance(n.op, self.op)], \
                        (topo, output_dtype)
                    data = numpy.random.rand(3, 4) * 10
                    data = data.astype(input_dtype)
                    if output_dtype == 'float16' and method == 'prod':
                        # We will likely get something infinite,
                        # and DebugMode will complain.
                        data = data[0:1]
                    f(data)
                    if "complex" in input_dtype:
                        continue
                    # Check that we can take the gradient
                    tensor.grad(var.sum(), x,
                                disconnected_inputs='ignore')
                    idx += 1

    def test_reduce_custom_acc_dtype(self):
        # Test the ability to provide your own accumulator dtype for a reduce.

        # We try multiple axis combinations even though axis should not matter.
        idx = 0
        for method in self.methods:
            for input_dtype in self.dtypes:
                x = tensor.matrix(dtype=input_dtype)
                for acc_dtype in self.dtypes:
                    # If the accumulator is a complex, the gradient of the reduce will
                    # cast the complex to the input dtype. We can't call the normal
                    # cast on a complex to a not complex as this is ambiguous.
                    if (not input_dtype.startswith('complex') and
                            acc_dtype.startswith('complex')):
                        continue

                    axis = self.axes[idx % len(self.axes)]
                    # If output_dtype would force a downcast, we expect a TypeError
                    # We always allow int/uint inputs with float/complex outputs.
                    upcasted_dtype = scalar.upcast(input_dtype, acc_dtype)
                    if (acc_dtype == upcasted_dtype or
                        (input_dtype in tensor.discrete_dtypes and
                            acc_dtype in tensor.continuous_dtypes)):
                        var = getattr(x, method)(acc_dtype=acc_dtype,
                                                 axis=axis)
                        assert var.owner.op.acc_dtype == acc_dtype

                        if "complex" in input_dtype:
                            continue
                    # Check that we can take the gradient
                        tensor.grad(var.sum(), x,
                                    disconnected_inputs='ignore')
                    else:
                        self.assertRaises(TypeError,
                                          getattr(x, method),
                                          acc_dtype=acc_dtype, axis=axis)

                    idx += 1

    def test_reduce_precision(self):
        # Check that the default accumulator precision is sufficient
        for method in self.methods:
            x = theano.shared(numpy.asarray([1e8, 1, -1e8],
                                            dtype='float32'))
            s = getattr(x, method)()
            f = theano.function([], s, mode=self.mode)
            topo = f.maker.fgraph.toposort()
            assert [n for n in topo if isinstance(n.op, self.op)], topo
            s_val = f()
            # Use extra precision in NumPy to compute the good answer.
            ret = getattr(numpy.asarray([1e8, 1, -1e8], dtype='float64'),
                          method)()
            assert numpy.allclose(s_val, ret), (s_val, ret)


class T_mean_dtype(unittest.TestCase):
    def test_mean_default_dtype(self):
        # Test the default dtype of a mean().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        for idx, dtype in enumerate(imap(str, theano.scalar.all_types)):
            axis = axes[idx % len(axes)]
            x = tensor.matrix(dtype=dtype)
            m = x.mean(axis=axis)
            if dtype in tensor.discrete_dtypes:
                assert m.dtype == 'float64'
            else:
                assert m.dtype == dtype, (m, m.dtype, dtype)
            f = theano.function([x], m)
            data = numpy.random.rand(3, 4) * 10
            data = data.astype(dtype)
            f(data)

    @attr('slow')
    def test_mean_custom_dtype(self):
        # Test the ability to provide your own output dtype for a mean.

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        idx = 0
        for input_dtype in imap(str, theano.scalar.all_types):
            x = tensor.matrix(dtype=input_dtype)
            for sum_dtype in imap(str, theano.scalar.all_types):
                axis = axes[idx % len(axes)]
                # If the inner sum cannot be created, it will raise a
                # TypeError.
                try:
                    mean_var = x.mean(dtype=sum_dtype, axis=axis)
                except TypeError:
                    pass
                else:
                    # Executed if no TypeError was raised
                    if sum_dtype in tensor.discrete_dtypes:
                        assert mean_var.dtype == 'float64', (
                            (mean_var.dtype, sum_dtype))
                    else:
                        assert mean_var.dtype == sum_dtype, (
                            (mean_var.dtype, sum_dtype))
                    if (('complex' in input_dtype or
                         'complex' in sum_dtype) and
                            input_dtype != sum_dtype):
                        continue
                    f = theano.function([x], mean_var)
                    data = numpy.random.rand(3, 4) * 10
                    data = data.astype(input_dtype)
                    f(data)
                    # Check that we can take the gradient, when implemented
                    if "complex" in mean_var.dtype:
                        continue
                    try:
                        tensor.grad(mean_var.sum(), x,
                                    disconnected_inputs='ignore')
                    except NotImplementedError:
                        # TrueDiv does not seem to have a gradient when
                        # the numerator is complex.
                        if mean_var.dtype in tensor.complex_dtypes:
                            pass
                        else:
                            raise

                idx += 1

    def test_mean_precision(self):
        # Check that the default accumulator precision is sufficient
        x = theano.shared(numpy.asarray([1e8, 1, -1e8], dtype='float32'))
        m = x.mean()
        f = theano.function([], m)
        m_val = f()
        assert numpy.allclose(m_val, 1. / 3)


class T_prod_without_zeros_dtype(unittest.TestCase):
    def test_prod_without_zeros_default_dtype(self):
        # Test the default dtype of a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        for idx, dtype in enumerate(imap(str, theano.scalar.all_types)):
            axis = axes[idx % len(axes)]
            x = ProdWithoutZeros(axis=axis)(tensor.matrix(dtype=dtype))
            assert x.dtype == dict(
                bool='int64',
                int8='int64',
                int16='int64',
                int32='int64',
                uint8='uint64',
                uint16='uint64',
                uint32='uint64',
            ).get(dtype, dtype)

    def test_prod_without_zeros_default_acc_dtype(self):
        # Test the default dtype of a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        for idx, dtype in enumerate(imap(str, theano.scalar.all_types)):
            axis = axes[idx % len(axes)]
            x = tensor.matrix(dtype=dtype)
            p = ProdWithoutZeros(axis=axis)(x)
            assert p.owner.op.acc_dtype == dict(
                bool='int64',
                int8='int64',
                int16='int64',
                int32='int64',
                uint8='uint64',
                uint16='uint64',
                uint32='uint64',
                float16='float32',
                float32='float64',
                complex64='complex128'
                ).get(dtype, dtype)

            if 'complex' in dtype:
                continue
            f = theano.function([x], p)
            data = numpy.random.rand(2, 3) * 3
            data = data.astype(dtype)
            f(data)

    @attr('slow')
    def test_prod_without_zeros_custom_dtype(self):
        # Test ability to provide your own output dtype for a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        idx = 0
        for input_dtype in imap(str, theano.scalar.all_types):
            x = tensor.matrix(dtype=input_dtype)
            for output_dtype in imap(str, theano.scalar.all_types):
                axis = axes[idx % len(axes)]
                prod_woz_var = ProdWithoutZeros(
                    axis=axis, dtype=output_dtype)(x)
                assert prod_woz_var.dtype == output_dtype
                idx += 1
                if ('complex' in output_dtype or
                        'complex' in input_dtype):
                    continue
                f = theano.function([x], prod_woz_var)
                data = numpy.random.rand(2, 3) * 3
                data = data.astype(input_dtype)
                f(data)

    @attr('slow')
    def test_prod_without_zeros_custom_acc_dtype(self):
        # Test ability to provide your own acc_dtype for a ProdWithoutZeros().

        # We try multiple axis combinations even though axis should not matter.
        axes = [None, 0, 1, [], [0], [1], [0, 1]]
        idx = 0
        for input_dtype in imap(str, theano.scalar.all_types):
            x = tensor.matrix(dtype=input_dtype)
            for acc_dtype in imap(str, theano.scalar.all_types):
                axis = axes[idx % len(axes)]
                # If acc_dtype would force a downcast, we expect a TypeError
                # We always allow int/uint inputs with float/complex outputs.
                upcasted_dtype = scalar.upcast(input_dtype, acc_dtype)
                if (acc_dtype == upcasted_dtype or
                        (input_dtype in tensor.discrete_dtypes and
                            acc_dtype in tensor.continuous_dtypes)):
                    prod_woz_var = ProdWithoutZeros(
                        axis=axis, acc_dtype=acc_dtype)(x)
                    assert prod_woz_var.owner.op.acc_dtype == acc_dtype

                    if (acc_dtype.startswith('complex') and
                            input_dtype != acc_dtype):
                        continue
                    f = theano.function([x], prod_woz_var)
                    data = numpy.random.rand(2, 3) * 3
                    data = data.astype(input_dtype)
                    f(data)
                else:
                    self.assertRaises(
                        TypeError,
                        ProdWithoutZeros(axis=axis, acc_dtype=acc_dtype),
                        x)

                idx += 1


class TestBitOpReduceGrad(unittest.TestCase):
    def setUp(self):
        self.rng = numpy.random.RandomState(unittest_tools.fetch_seed())

    def test_all_grad(self):
        x = tensor.bmatrix('x')
        x_all = x.all()
        gx = theano.grad(x_all, x)
        f = theano.function([x], gx)
        x_random = self.rng.binomial(n=1, p=0.5, size=(5, 7)).astype('int8')
        for x_val in (x_random,
                      numpy.zeros_like(x_random),
                      numpy.ones_like(x_random)):
            gx_val = f(x_val)
            assert gx_val.shape == x_val.shape
            assert numpy.all(gx_val == 0)

    def test_any_grad(self):
        x = tensor.bmatrix('x')
        x_all = x.any()
        gx = theano.grad(x_all, x)
        f = theano.function([x], gx)
        x_random = self.rng.binomial(n=1, p=0.5, size=(5, 7)).astype('int8')
        for x_val in (x_random,
                      numpy.zeros_like(x_random),
                      numpy.ones_like(x_random)):
            gx_val = f(x_val)
            assert gx_val.shape == x_val.shape
            assert numpy.all(gx_val == 0)


class TestElemwise(unittest_tools.InferShapeTester):
    def test_elemwise_grad_bool(self):
        x = theano.tensor.scalar(dtype='bool')
        y = theano.tensor.bscalar()
        z = x * y
        dx, dy = theano.grad(z, [x, y])

    def test_infer_shape(self):

        for s_left, s_right in [
                ((5, 6), (5, 6)),
                ((5, 6), (5, 1)),
                ((5, 6), (1, 6)),
                ((5, 1), (5, 6)),
                ((1, 6), (5, 6)),
                ((2, 3, 4, 5), (2, 3, 4, 5)),
                ((2, 3, 4, 5), (2, 3, 1, 5)),
                ((2, 3, 4, 5), (1, 3, 4, 5)),
                ((2, 1, 4, 5), (2, 3, 4, 5)),
                ((2, 3, 4, 1), (2, 3, 4, 5))]:
            dtype = theano.config.floatX
            t_left = TensorType(dtype, [(entry == 1) for entry in s_left])()
            t_right = TensorType(dtype, [(entry == 1) for entry in s_right])()
            t_left_val = numpy.zeros(s_left, dtype=dtype)
            t_right_val = numpy.zeros(s_right, dtype=dtype)
            self._compile_and_check(
                [t_left, t_right],
                [Elemwise(scalar.add)(t_left, t_right)],
                [t_left_val, t_right_val], Elemwise)

    def test_input_dimensions_overflow(self):
        # Elemwise.perform used to compute the product
        # of input shapes to check if there was a zero in them,
        # it overflowed in this case.
        a, b, c, d, e, f = tensor.vectors('abcdef')
        s = a + b + c + d + e + f
        g = theano.function([a, b, c, d, e, f], s,
                            mode=theano.compile.Mode(linker='py'))
        g(*[numpy.zeros(2 ** 11, config.floatX) for i in xrange(6)])


def test_gt_grad():
    """A user test that failed.

    Something about it made Elemwise.grad return something that was
    too complicated for get_scalar_constant_value to recognize as being 0, so
    gradient.grad reported that it was not a valid gradient of an
    integer.

    """
    floatX = config.floatX
    T = theano.tensor

    input_ = T.vector(dtype=floatX)
    random_values = numpy.random.RandomState(1234).uniform(
        low=-1, high=1, size=(2, 2))
    W_values = numpy.asarray(random_values, dtype=floatX)
    W = theano.shared(value=W_values, name='weights')
    correct_score = T.dot(input_, W)
    wrong_input = T.vector(dtype=floatX)
    wrong_score = theano.clone(correct_score, {input_: wrong_input})
    # Hinge loss

    scores = T.ones_like(correct_score) - correct_score + wrong_score
    cost = (scores * (scores > 0)).sum()
    T.grad(cost, input_)

"""
if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite([test_Prod('test_mul_without_zeros_zeros')])
    #suite.addTest(test_Prod('test_verify_grad_with_zeros'))
    #suite.addTest(test_Prod('test_prod_without_zeros'))
    #suite.addTest(test_Prod('test_other_grad_tests'))
    unittest.TextTestRunner().run(suite)
"""


def test_clip_grad():

    # test the gradient of clip
    def func(x, y, z):
        return theano.tensor.clip(x, y, z)
    # use an x value less than y, an x value between y and z, and an x value
    # greater than z
    unittest_tools.verify_grad(func,
                               [numpy.asarray([-1., 0.5, 2.]), 0., 1.])


def test_grad_useless_sum():
    """Test absence of useless sum.

    When an operation (such as T.mul) is done on a broadcastable vector and
    a matrix, the gradient in backward path is computed for the broadcasted
    vector. So a sum reverts the broadcasted vector to a vector. In the case
    of operations on two broadcastable vectors, the sum should not be generated.

    This test checks whether there is a useless sum in the gradient
    computations.
    """
    mode = theano.compile.get_default_mode().including('canonicalize')
    mode.check_isfinite = False
    x = TensorType(theano.config.floatX, (True,))('x')
    l = tensor.log(1.0 - tensor.nnet.sigmoid(x))[0]
    g = tensor.grad(l, x)
    nodes = theano.gof.graph.ops([x], [g])

    f = theano.function([x], g, mode=mode)
    test_values = [-100, -1, 0, 1, 100]
    outputs = []
    old_values_eq_approx = staticmethod(TensorType.values_eq_approx)
    TensorType.values_eq_approx = staticmethod(
        tensor.type.values_eq_approx_remove_nan)
    try:
        for test_value in test_values:
            outputs.append(f(numpy.array([test_value]).astype('float32')))
    finally:
        TensorType.values_eq_approx = old_values_eq_approx

    assert not any([isinstance(node.op, theano.tensor.elemwise.Sum) for node in nodes])
    assert numpy.allclose(outputs, [[-3.72007598e-44],
                                    [-0.26894142],
                                    [-0.5],
                                    [-0.73105858],
                                    [-1.]])


def test_elemwise_grad_broadcast():
    # This crashed in the past.

    x = tensor.tensor(dtype='float32',
                      broadcastable=(True, False, False, False))
    y = tensor.tensor(dtype='float32',
                      broadcastable=(True, True, False, False))

    theano.grad(theano.tensor.tanh(x).sum(), x)
    theano.grad(theano.tensor.tanh(x + y).sum(), y)
    theano.grad(theano.tensor.tanh(x + y).sum(), [x, y])


def test_clip_grad_int():

    # test that integers don't crash clip gradient
    x = tensor.iscalar()
    y = tensor.iscalar()
    z = tensor.iscalar()
    c = tensor.clip(x, y, z)
    tensor.grad(c, [x, y, z])


def test_not_implemented_elemwise_grad():
    """
    Regression test for unimplemented gradient in an Elemwise Op.
    """

    class TestOp(scalar.ScalarOp):

        def __init__(self):
            self.output_types_preference = scalar.upgrade_to_float

        def impl(self, n, x):
            return x * n

        def grad(self, inputs, gout):
            (n, x) = inputs
            (gz,) = gout
            dy_dx = n
            return [theano.gradient.grad_not_implemented(self, 0, n),
                    gz * dy_dx]

    test_op = tensor.Elemwise(TestOp())
    x = tensor.scalar()
    # The call to `grad` used to crash.
    tensor.grad(test_op(2, x), x)
    # Verify that trying to use the not implemented gradient fails.
    try:
        tensor.grad(test_op(x, 2), x)
        assert False
    except theano.gradient.NullTypeGradError:
        pass


if __name__ == '__main__':

    t = TestElemwise('setUp')
    t.setUp()
    t.test_infer_shape()
