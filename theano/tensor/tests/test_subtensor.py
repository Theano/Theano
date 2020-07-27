from __future__ import absolute_import, division, print_function

import logging
import sys
import unittest

import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_array_equal
from six import StringIO
from six.moves import xrange

import theano
import theano.scalar as scal
import theano.tensor as tensor
from theano import config, gof
from theano.compat import PY3, izip
from theano.compile import DeepCopyOp
from theano.tensor import (_shared, cscalar, ctensor3, dmatrix,
                           dscalar, dtensor4, dvector, fmatrix, fscalar,
                           fvector, ftensor4, iscalar, lmatrix, lrow, lvector,
                           matrix, vector)
from theano.tensor.basic import DimShuffle
from theano.tensor.subtensor import (AdvancedIncSubtensor,
                                     AdvancedIncSubtensor1, AdvancedSubtensor,
                                     IncSubtensor,
                                     Subtensor, advanced_inc_subtensor,
                                     advanced_inc_subtensor1,
                                     advanced_set_subtensor,
                                     advanced_set_subtensor1,
                                     get_canonical_form_slice, inc_subtensor,
                                     set_subtensor)

from theano.tensor.tests.test_basic import inplace_func, rand, randint_ranged
from theano.tests import unittest_tools as utt
from theano.tests.unittest_tools import attr
from theano import change_flags

if PY3:
    def L(i):
        return i
else:
    def L(i):
        return long(i)  # noqa for Python 3


class T_subtensor(unittest.TestCase, utt.TestOptimizationMixin):
    """
    This is build in a way that allow to reuse it to test the equivalent gpu op.
    """
    def __init__(self, name, shared=tensor._shared,
                 sub=tensor.Subtensor,
                 inc_sub=tensor.IncSubtensor,
                 adv_sub1=tensor.AdvancedSubtensor1,
                 adv_incsub1=tensor.AdvancedIncSubtensor1,
                 adv_sub=tensor.AdvancedSubtensor,
                 adv_bool_sub=tensor.AdvancedBooleanSubtensor,
                 adv_bool_inc_sub=tensor.AdvancedBooleanIncSubtensor,
                 mode=None,
                 dtype=theano.config.floatX,
                 type=tensor.TensorType,
                 ignore_topo=DeepCopyOp,
                 dimshuffle=DimShuffle):
        self.shared = shared
        self.sub = sub
        self.inc_sub = inc_sub
        self.adv_sub1 = adv_sub1
        self.adv_incsub1 = adv_incsub1
        self.adv_sub = adv_sub
        self.adv_bool_sub = adv_bool_sub
        self.adv_bool_inc_sub = adv_bool_inc_sub
        self.dimshuffle = dimshuffle
        if mode is None:
            mode = theano.compile.mode.get_default_mode()
            mode = mode.including("local_useless_subtensor")
        self.mode = mode
        self.dtype = dtype
        self.type = type
        self.ignore_topo = ignore_topo
        self.fast_compile = theano.config.mode == 'FAST_COMPILE'
        self.ops = (sub, inc_sub, adv_sub1, adv_incsub1,
                    adv_bool_sub, adv_bool_inc_sub)
        super(T_subtensor, self).__init__(name)

    def function(self, inputs, outputs, accept_inplace=False,
                 op=None, mode=None, N=1, N_fast=None):
        """
        wrapper around theano.function that also check the output

        :param N: the number of op expected in the toposort
                  if tuple of length 2, (expected if fast_compile,
                                         if not fast_compile)
        """
        if self.fast_compile and N_fast is not None:
            N = N_fast
        if mode is None:
            mode = self.mode
        if op is None:
            op = self.sub

        f = theano.function(inputs, outputs, mode=mode,
                            accept_inplace=accept_inplace)
        self.assertFunctionContainsClassN(f, op, N)
        return f

    def setUp(self):
        Subtensor.debug = False
        utt.seed_rng()

    def eval_output_and_check(self, t, op_type=None, mode=None, length=1):
        if op_type is None:
            op_type = self.sub
        if mode is None:
            mode = self.mode
        f = inplace_func([], t, mode=mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
                                                         self.ignore_topo)]
        assert_equal(len(topo_), length)
        if length == 1:
            assert isinstance(topo_[0].op, op_type)
        tval = f()
        return tval

    def test0_err_invalid(self):
        # it is impossible to retrieve a view of a 0-d tensor
        n = self.shared(np.ones((), dtype=self.dtype))
        self.assertRaises(IndexError, n.__getitem__, 0)

    @change_flags(compute_test_value='off')
    def test1_err_bounds(self):
        n = self.shared(np.ones(3, dtype=self.dtype))
        t = n[7]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        # Silence expected error messages
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                self.eval_output_and_check(t)
            except IndexError:
                return
            self.fail()
        finally:
            _logger.setLevel(oldlevel)

    def test1_err_subslice(self):
        n = self.shared(np.ones(3, dtype=self.dtype))
        try:
            n[slice(0, slice(1, 2, None), None)]
        except Exception:
            # Relax constraint on the type of Exception,
            # since this might be handled by AvancedSubtensor
            # if e[0] != Subtensor.e_indextype:
            #    raise
            return
        self.fail()

    def test1_ok_range_finite(self):
        n = self.shared(np.arange(3, dtype=self.dtype))
        t = n[0:2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [0, 1]).all())

    def test2_ok_range_finite(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((3, 4)))
        # Also check negative index
        for idx in [(slice(0, 2), 3), ((slice(0, 2), -1)), (slice(0, 2), -4)]:
            t = n[idx]  # l]#0:2,3]
            self.assertTrue(isinstance(t.owner.op, Subtensor))
            tval = self.eval_output_and_check(t)
            self.assertTrue(tval.shape == (2,))
            self.assertTrue(np.allclose(tval, n.get_value()[idx]))

    def test1_0_dims(self):
        n = self.shared(np.ones((), dtype=self.dtype))
        t = self.sub([])(n)
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        self.eval_output_and_check(
            t, mode=self.mode.excluding("local_useless_subtensor"))

    def test1_err_invalid(self):
        n = self.shared(np.ones(1, dtype=self.dtype))
        self.assertRaises(IndexError, n.__getitem__, (0, 0))

    def test1_ok_elem(self):
        n = self.shared(np.ones(1, dtype=self.dtype) * 5)
        t = n[0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(tval == 5.0)

    def test1_ok_range_infinite(self):
        n = self.shared(np.arange(3, dtype=self.dtype))
        t = n[1:]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [1.0, 2.0]).all())

    def test1_ok_strided(self):
        n = self.shared(np.arange(5, dtype=self.dtype))
        t = n[1::2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [1.0, 3.0]).all())

        t = n[0:-1:2]  # 0 to 1 from the end stepping by 2
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [0.0, 2.0]).all())

    @change_flags(compute_test_value='off')
    def test2_err_bounds0(self):
        n = self.shared(np.ones((2, 3), dtype=self.dtype) * 5)
        for idx in [(0, 4), (0, -4)]:
            t = n[idx]
            self.assertTrue(isinstance(t.owner.op, Subtensor))
            # Silence expected warnings
            _logger = logging.getLogger('theano.gof.opt')
            oldlevel = _logger.level
            _logger.setLevel(logging.CRITICAL)
            try:
                self.assertRaises(IndexError,
                                  self.eval_output_and_check, [t])
            finally:
                _logger.setLevel(oldlevel)

    @change_flags(compute_test_value='off')
    def test2_err_bounds1(self):
        n = self.shared((np.ones((2, 3), dtype=self.dtype) * 5))
        t = n[4:5, 3]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            self.assertRaises(IndexError,
                              self.eval_output_and_check, [t])
        finally:
            sys.stderr = old_stderr

    def test2_ok_elem(self):
        n = self.shared(np.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[0, 2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(np.all(tval == 2))

    def test2_ok_row(self):
        n = self.shared(np.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[1]
        self.assertFalse(any(n.type.broadcastable))
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (3,))
        self.assertTrue(np.all(tval == [3, 4, 5]))

    def test2_ok_col(self):
        n = self.shared(np.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[:, 0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        self.assertFalse(any(n.type.broadcastable))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(np.all(tval == [0, 3]))

    def test2_ok_rows_finite(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[1:3, 0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(np.all(tval == [3, 6]))

    def test2_ok_cols_infinite(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[1, 2:]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (1,))
        self.assertTrue(np.all(tval == 5))

    def test2_ok_strided(self):
        n = self.shared(np.arange(20, dtype=self.dtype).reshape((4, 5)))
        t = n[1:4:2, 1:5:2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2, 2))
        self.assertTrue(np.all(tval == [[6, 8], [16, 18]]))

    def test3_ok_mat(self):
        n = self.shared(np.arange(24, dtype=self.dtype).reshape((2, 3, 4)))
        t = n[0, 0, 0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(np.all(tval == 0))

    def test_long(self):
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[L(1):L(4):L(2), L(1)]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(np.all(tval == [4, 10]))

    def test_long_too_big(self):
        # Currently, we cast Python longs to int64 when used for indexing.
        # This test checks that using a long that does not fit raises an error.
        n = self.shared(np.arange(12, dtype=self.dtype).reshape((4, 3)))
        self.assertRaises(Exception, lambda: n[:L(2 ** 63)])

    def test_list_slice(self):
        x = theano.tensor.arange(100).reshape((5, 5, 4))
        res = x[[slice(1, -1)] * x.ndim].eval()
        x = np.arange(100).reshape((5, 5, 4))
        np.allclose(res, x[[slice(1, -1)] * x.ndim])

    def test_slice_symbol(self):
        x = self.shared(np.random.rand(5, 4).astype(self.dtype))
        y = self.shared(np.random.rand(1, 2, 3).astype(self.dtype))
        o = x[:y.shape[0], None, :]
        f = theano.function([], o, mode=self.mode)
        ret = f()
        assert ret.shape == (1, 1, 4)

    def test_ellipsis(self):
        numpy_n = np.arange(24, dtype=self.dtype).reshape((2, 3, 4))
        n = self.shared(numpy_n)
        test_cases = [
            (0, Subtensor, self.sub, np.index_exp[...]),
            (1, Subtensor, self.sub, np.index_exp[..., 1]),
            (1, Subtensor, self.sub, np.index_exp[1, ...]),
            (1, Subtensor, self.sub, np.index_exp[..., 1, 2, 3]),
            (1, Subtensor, self.sub, np.index_exp[1, ..., 2, 3]),
            (1, Subtensor, self.sub, np.index_exp[1, 2, 3, ...]),
            (3, DimShuffle, self.dimshuffle,
             np.index_exp[..., [0, 2, 3]]),
            (1, DimShuffle, self.dimshuffle,
             np.index_exp[np.newaxis, ...]),
            (1, AdvancedSubtensor, self.adv_sub,
             np.index_exp[..., np.newaxis, [1, 2]])]

        for length, op_type, op_type_opt, slice_ in test_cases:
            numpy_tval = numpy_n[slice_]
            t = n[slice_]
            self.assertTrue(isinstance(t.owner.op, op_type))
            tval = self.eval_output_and_check(t,
                                              op_type=op_type_opt,
                                              length=length)
            assert_equal(tval.shape, numpy_tval.shape)
            assert_array_equal(tval, numpy_tval)

    def test_boolean(self):
        def numpy_inc_subtensor(x, idx, a):
            x = x.copy()
            x[idx] += a
            return x

        numpy_n = np.arange(6, dtype=self.dtype).reshape((2, 3))
        n = self.shared(numpy_n)

        # indexing with a mask for some dimensions
        mask = np.array([True, False])
        val = self.eval_output_and_check(n[mask], op_type=self.adv_bool_sub)
        assert_array_equal(numpy_n[mask], val)
        val = self.eval_output_and_check(inc_subtensor(n[mask], 1),
                                         op_type=self.adv_bool_inc_sub)
        assert_array_equal(numpy_inc_subtensor(numpy_n, mask, 1), val)
        assert_array_equal(numpy_inc_subtensor(numpy_n, mask, numpy_n[mask]),
                           inc_subtensor(n[mask], n[mask]).eval())

        # test gradient
        utt.verify_grad(lambda m: m[mask], [numpy_n])
        utt.verify_grad(lambda m: inc_subtensor(m[mask], 1), [numpy_n])

        # indexing with a comparison (should translate to a boolean mask)
        assert_array_equal(numpy_n[numpy_n > 2], n[n > 2].eval())
        assert_array_equal(numpy_n[[0], numpy_n[0] > 2], n[[0], n[0] > 2].eval())
        assert_array_equal(numpy_n[[1], numpy_n[0] > 2], n[[1], n[0] > 2].eval())

        # indexing with a mask for the second dimension
        mask = np.array([True, False, True])
        assert_array_equal(numpy_n[0, mask], n[0, mask].eval())
        assert_array_equal(numpy_n[:, mask], n[:, mask].eval())
        assert_array_equal(numpy_n[:, mask], n[:, self.shared(mask)].eval())
        assert_array_equal(numpy_n[1:, mask], n[1:, mask].eval())
        assert_array_equal(numpy_n[:1, mask], n[:1, mask].eval())
        assert_array_equal(numpy_n[1:, mask, np.newaxis], n[1:, mask, np.newaxis].eval())
        assert_array_equal(numpy_n[np.newaxis, 1:, mask], n[np.newaxis, 1:, mask].eval())
        assert_array_equal(numpy_inc_subtensor(numpy_n, [0, mask], 1),
                           inc_subtensor(n[(0,) + mask.nonzero()], 1).eval())
        assert_array_equal(numpy_inc_subtensor(numpy_n, [0, mask], 1),
                           inc_subtensor(n[0, mask], 1).eval())
        assert_array_equal(numpy_inc_subtensor(numpy_n, [slice(None), mask], 1),
                           inc_subtensor(n[:, mask], 1).eval())

        # indexing with a boolean ndarray
        mask = np.array([[True, False, True], [False, False, True]])
        assert_array_equal(numpy_n[mask], n[mask].eval())
        assert_array_equal(numpy_n[mask], n[self.shared(mask)].eval())
        assert_array_equal(numpy_inc_subtensor(numpy_n, mask, 1),
                           inc_subtensor(n[mask], 1).eval())

        # indexing with ellipsis
        numpy_n4 = np.arange(48, dtype=self.dtype).reshape((2, 3, 4, 2))
        n4 = self.shared(numpy_n4)
        assert_array_equal(numpy_n4[numpy_n > 2, ...], n4[n > 2, ...].eval())
        assert_array_equal(numpy_n4[numpy_n > 2, ..., 1], n4[n > 2, ..., 1].eval())
        assert_array_equal(numpy_n4[numpy_n > 2, ..., 0, 1], n4[n > 2, ..., 0, 1].eval())
        assert_array_equal(numpy_inc_subtensor(numpy_n4, [numpy_n > 2, Ellipsis], 1),
                           inc_subtensor(n4[n > 2, ...], 1).eval())
        assert_array_equal(numpy_inc_subtensor(numpy_n4, [numpy_n > 2, Ellipsis, 1], 1),
                           inc_subtensor(n4[n > 2, ..., 1], 1).eval())
        assert_array_equal(numpy_inc_subtensor(numpy_n4, [numpy_n > 2, Ellipsis, 0, 1], 1),
                           inc_subtensor(n4[n > 2, ..., 0, 1], 1).eval())

        with change_flags(compute_test_value='off'):
            # the boolean mask should have the correct shape
            # - too large, padded with True
            mask = np.array([True, False, True])
            self.assertRaises(IndexError, n[mask].eval)
            self.assertRaises(IndexError, n[mask, ...].eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask], 1).eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask, ...], 1).eval)
            mask = np.array([[True, False, False, True], [False, True, False, True]])
            self.assertRaises(IndexError, n[mask].eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask], 1).eval)
            # - too large, padded with False (this works in NumPy < 0.13.0)
            mask = np.array([True, False, False])
            self.assertRaises(IndexError, n[mask].eval)
            self.assertRaises(IndexError, n[mask, ...].eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask], 1).eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask, ...], 1).eval)
            mask = np.array([[True, False, False, False], [False, True, False, False]])
            self.assertRaises(IndexError, n[mask].eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask], 1).eval)
            # - mask too small (this works in NumPy < 0.13.0)
            mask = np.array([True])
            self.assertRaises(IndexError, n[mask].eval)
            self.assertRaises(IndexError, n[mask, ...].eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask], 1).eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask, ...], 1).eval)
            mask = np.array([[True], [True]])
            self.assertRaises(IndexError, n[mask].eval)
            self.assertRaises(IndexError, inc_subtensor(n[mask], 1).eval)
            # - too many dimensions
            mask = np.array([[[True, False, False],
                              [False, True, False]]])
            self.assertRaises(IndexError, n.__getitem__, mask)
            self.assertRaises(IndexError, n.__getitem__, mask)

            # special cases: Python bools and bools nested in Python arrays are not supported
            self.assertRaises(TypeError, n.__getitem__, (True,))
            self.assertRaises(TypeError, n.__getitem__, (False,))
            self.assertRaises(TypeError, n.__getitem__, (True, False))
            self.assertRaises(TypeError, n.__getitem__, ([True, False]))
            self.assertRaises(TypeError, n.__getitem__, ([0, 1], [0, False]))
            self.assertRaises(TypeError, n.__getitem__, ([0, 1], [0, theano.shared(True)]))

    def test_newaxis(self):
        # newaxis support comes from logic in the __getitem__ of TensorType
        # Variables, which currently inserts dimshuffle to get the right number
        # of dimensions, and adjusts the slice tuple accordingly.
        #
        # So testing is done via square-bracket notation rather than direct
        # interaction with the Subtensor Op (which has no support of its own for
        # newaxis).

        newaxis = np.newaxis

        n = self.shared(np.arange(24, dtype=self.dtype).reshape((2, 3, 4)))
        assert n.ndim == 3

        n4 = n[newaxis, :, :, :]
        assert n4.broadcastable == (True, False, False, False), n4

        n4 = n[:, newaxis, :, :]
        assert n4.broadcastable == (False, True, False, False), n4

        n4 = n[:, :, newaxis, :]
        assert n4.broadcastable == (False, False, True, False), n4

        n4 = n[:, :, :, newaxis]
        assert n4.broadcastable == (False, False, False, True), n4

        n3 = n.flatten()[newaxis, :, newaxis]
        assert n3.broadcastable == (True, False, True), n3

        s = cscalar()
        s1 = s[newaxis]
        assert s1.broadcastable == (True,), s1

        vs1, vn3, vn4 = theano.function([s], [s1, n3, n4], mode=self.mode)(-2.0)

        assert np.all(vs1 == [-2.0])
        assert np.all(vn3 ==
                      np.arange(24)[newaxis, :, newaxis])
        assert np.all(vn4 ==
                      np.arange(24).reshape((2, 3, 4))[:, :, :, newaxis])

    def test_grad_1d(self):
        subi = 0
        data = np.asarray(rand(2, 3), dtype=self.dtype)
        n = self.shared(data)
        z = scal.constant(subi).astype('int32')
        t = n[z:, z]
        gn = theano.tensor.grad(theano.tensor.sum(theano.tensor.exp(t)), n)

        f = inplace_func([], gn, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
                                                         self.ignore_topo)]
        if not self.fast_compile:
            assert len(topo_) == 6
        assert np.sum([isinstance(node.op, self.inc_sub)
                       for node in topo_]) == 1
        assert np.sum([isinstance(node.op, self.sub)
                       for node in topo_]) == 1
        gval = f()

        good = np.zeros_like(data)
        good[subi:, subi] = np.exp(data[subi:, subi])
        self.assertTrue(np.allclose(gval, good), (gval, good))

    def test_grad_2d_inc_set_subtensor(self):
        for n_shape, m_shape in [
            [(2, 3), (2, 2)],
            [(3, 2), (2, 2)],
            [(3, 2), (1, 2)],
            [(3, 2), (2,)],
        ]:
            for op in [inc_subtensor, set_subtensor]:
                subi = 2
                data = np.asarray(rand(*n_shape), dtype=self.dtype)
                n = self.shared(data)
                z = scal.constant(subi)
                m = matrix('m', dtype=self.dtype)
                mv = np.asarray(rand(*m_shape), dtype=self.dtype)

                t = op(n[:z, :z], m)
                gn, gm = theano.tensor.grad(theano.tensor.sum(t), [n, m])
                utt.verify_grad(lambda m: op(n[:z, :z], m), [mv], mode=self.mode)
                utt.verify_grad(lambda nn: op(nn[:z, :z], mv), [data], mode=self.mode)

    def test_grad_0d(self):
        data = np.asarray(rand(2, 3), dtype=self.dtype)
        n = self.shared(data)
        t = n[1, 0]
        gn = theano.tensor.grad(theano.tensor.sum(theano.tensor.exp(t)), n)
        f = self.function([], gn)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo
                 if not isinstance(node.op, self.ignore_topo)]
        if not self.fast_compile:
            assert_equal(len(topo_), 6)
        assert np.sum([isinstance(node.op, self.inc_sub)
                       for node in topo_]) == 1
        assert np.sum([isinstance(node.op, self.sub)
                       for node in topo_]) == 1

        gval = f()
        good = np.zeros_like(data)
        good[1, 0] = np.exp(data[1, 0])
        self.assertTrue(np.allclose(gval, good), (gval, good))

    def test_ok_list(self):
        for data, idx in [(rand(4), [1, 0]),
                          (rand(4, 5), [2, 3, -1]),
                          (rand(4, 2, 3), [0, 3]),
                          (rand(4, 2, 3), [3, 3, 1, 1, 2, 2, 0, 0]),
                          (rand(4, 2, 3), [3, 3, 1, 1, 2, 2, 0, 0,
                                           -1, -2, -3, -4]),
                          # Test 4 dims as gpu code use another algo
                          # in that case This new algo is not as much
                          # optimized for that case.
                          (rand(4, 4, 2, 3),
                           [3, 3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
                          # Test with TensorConstant index.
                          (rand(4, 2, 3),
                           theano.tensor.constant([3, 3, 1, 1, 2, 2, 0, 0])),
                          ]:
            data = np.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]

            # We test again AdvancedSubtensor1 as we transfer data to the cpu.
            self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

            val = self.eval_output_and_check(t, op_type=self.adv_sub1)
            if isinstance(idx, list):
                good = data[idx]
            else:
                good = data[idx.data]
            self.assertTrue(val.ndim == data.ndim)
            self.assertTrue(np.allclose(val, good), (val, good))

            # Test reuse of output memory
            if type(self.adv_sub1) == tensor.AdvancedSubtensor1:
                op = self.adv_sub1()
                # When idx is a TensorConstant.
                if hasattr(idx, "data"):
                    idx = idx.data
                test_out = [[None]]
                op.perform(None, [data, idx], test_out)
                out1 = test_out[0][0]
                op.perform(None, [data, idx], test_out)
                out2 = test_out[0][0]
                assert out1 is out2

            # test the grad
            gn = theano.grad(t.sum(), n)
            g = self.function([], gn, op=self.adv_incsub1)
            utt.verify_grad(lambda m: m[[1, 3]],
                            [np.random.rand(5, 5).astype(self.dtype)],
                            mode=self.mode)
            g()
            utt.verify_grad(lambda m: m[idx],
                            [data], mode=self.mode)

    def test_noncontiguous_idx(self):
        data = rand(4, 2, 3)
        idx = [2, 2, 0, 0, 1, 1]
        n = self.shared(data)
        t = n[self.shared(np.asarray(idx).astype('int64'))[::2]]
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))
        val = self.eval_output_and_check(t, op_type=self.adv_sub1, length=2)
        utt.assert_allclose(data[idx[::2]], val)

    def test_err_invalid_list(self):
        n = self.shared(np.asarray(5, dtype=self.dtype))
        self.assertRaises(IndexError, n.__getitem__, [0, 0])

    def test_err_invalid_2list_dtype(self):
        n = self.shared(np.ones((3, 3), dtype=self.dtype) * 5)
        self.assertRaises(TypeError, n.__getitem__, ([0., 0], [1, 1]))

    def test_err_bound_list(self):
        n = self.shared(np.ones((2, 3), dtype=self.dtype) * 5)
        l = lvector()
        t = n[l]
        # We test again AdvancedSubtensor1 as we transfer data to the cpu.
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

        f = self.function([l], t, op=self.adv_sub1)

        # the grad
        g = self.function([l],
                          inc_subtensor(t, np.asarray([[1.]], self.dtype)),
                          op=self.adv_incsub1)

        for shp in [[0, 4], [0, -3], [-10]]:
            self.assertRaises(IndexError, f, shp)
            self.assertRaises(IndexError, g, shp)

    def test_adv_sub1_broadcast(self):
        v = np.arange(3, dtype=self.dtype).reshape((1, 3))
        n = self.shared(v * 5, broadcastable=(True, False))
        idx = tensor.lvector()
        t = n[idx]
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

        f = self.function([idx], t, op=self.adv_sub1)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
                                                         self.ignore_topo)]
        assert len(topo_) == 1
        self.assertTrue(isinstance(topo_[0].op, self.adv_sub1))
        f_0 = f([0])
        self.assertTrue(f_0.shape == (1, 3))
        self.assertTrue(np.allclose(f_0, v * 5))
        f_00 = f([0, 0])
        self.assertTrue(f_00.shape == (2, 3))
        self.assertTrue(np.allclose(f_00, v * 5))
        self.assertRaises(IndexError, f, [0, 1])

        # Test the gradient
        c = t.sum()
        gn = theano.grad(c, n)
        g = self.function([idx], gn, op=self.adv_incsub1)
        g_0 = g([0])
        self.assertTrue(g_0.shape == (1, 3))
        self.assertTrue(np.allclose(g_0, 1))
        g_00 = g([0, 0])
        self.assertTrue(g_00.shape == (1, 3))
        self.assertTrue(np.allclose(g_00, 2))

        utt.verify_grad(lambda m: m[[1, 3]],
                        [np.random.rand(5, 5).astype(self.dtype)],
                        mode=self.mode)

        def fun(x, y):
            return advanced_inc_subtensor1(x, y, [1, 3])
        utt.verify_grad(fun, [np.random.rand(5, 5).astype(self.dtype),
                              np.random.rand(2, 5).astype(self.dtype)],
                        mode=self.mode)

        def fun(x, y):
            return advanced_set_subtensor1(x, y, [1, 3])
        utt.verify_grad(fun, [np.random.rand(5, 5).astype(self.dtype),
                              np.random.rand(2, 5).astype(self.dtype)],
                        mode=self.mode)

        # test set_subtensor broadcast
        self.dtype = 'float32'

        x = tensor.tensor4('x', dtype=self.dtype)
        indexes = theano.shared(np.int32([1, 2, 3, 4]))
        W = self.shared(np.random.random(
            (10, 10, 3, 3)).astype(self.dtype))

        h = x + W
        h = tensor.set_subtensor(h[indexes], h[indexes])
        g = tensor.grad(h.sum(), W)
        N = 2
        if theano.config.mode == "FAST_COMPILE" and self.adv_incsub1 is tensor.AdvancedIncSubtensor1:
            N = 3
        f = self.function([x], g, op=self.adv_incsub1, N=N)

        f(np.random.random((10, 10, 3, 3)).astype(self.dtype))

    def test_adv_sub1_idx_broadcast(self):
        # The idx can be a broadcastable vector.
        ones = np.ones((4, 3), dtype=self.dtype)
        n = self.shared(ones * 5)
        idx = tensor.TensorType(dtype='int64', broadcastable=(True,))()
        assert idx.type.broadcastable == (True,)
        t = n[idx]
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

        f = self.function([idx], t, op=self.adv_sub1)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
                                                         self.ignore_topo)]
        assert len(topo_) == 1
        self.assertTrue(isinstance(topo_[0].op, self.adv_sub1))
        f_0 = f([0])
        self.assertTrue(f_0.shape == (1, 3))
        self.assertTrue(np.allclose(f_0, 5))

        # Test the gradient
        c = t.sum()
        gn = theano.grad(c, n)
        g = self.function([idx], gn, op=self.adv_incsub1)
        g_0 = g([0])
        self.assertTrue(g_0.shape == (4, 3))
        self.assertTrue(np.allclose(g_0[0], 1))
        self.assertTrue(np.allclose(g_0[1:], 0))

    @attr('slow')
    def test_shape_i_const(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = self.mode.including("fast_run")
        data = self.shared(np.array(np.arange(5), dtype=self.dtype))
        for start in [None] + [-8, -5, -1, 0, 1, 5, 8]:
            outs = []
            shapes = []
            for stop in [None] + [-8, -5, -1, 0, 1, 5, 8]:
                for step in [None] + [-3, -1, 2]:
                    outs += [data[start:stop:step].shape]
                    shapes += [data.get_value(
                        borrow=True)[start:stop:step].shape]
            f = self.function([], outs, mode=mode_opt,
                              op=self.ops, N=0)
            t_shapes = f()
            for t_shape, shape in zip(t_shapes, shapes):
                assert np.all(t_shape == shape)
            assert tensor.Subtensor not in [x.op
                                            for x in f.maker.fgraph.toposort()]

    def test_shape_i_scalar(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = self.mode.including("fast_run")

        v_data = np.array(np.arange(5), dtype=self.dtype)
        t_data = self.shared(v_data)
        start = tensor.iscalar('b')
        stop = tensor.iscalar('e')
        step = tensor.iscalar('s')
        f = self.function([start, stop, step],
                          t_data[start:stop:step].shape,
                          mode=mode_opt,
                          op=self.ops,
                          N=0)
        assert tensor.Subtensor not in [x.op
                                        for x in f.maker.fgraph.toposort()]
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                for step in [-3, -1, 2, 5]:
                    assert np.all(f(start, stop, step) ==
                                  v_data[start:stop:step].shape)

    def test_slice_canonical_form_0(self):
        start = tensor.iscalar('b')
        stop = tensor.iscalar('e')
        step = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = get_canonical_form_slice(slice(start, stop, step), length)
        f = self.function([start, stop, step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1])], N=0, op=self.ops)

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                for step in [-6, -3, -1, 2, 5]:
                    out = f(start, stop, step, length)
                    t_out = a[out[0]:out[1]:out[2]][::out[3]]
                    v_out = a[start:stop:step]
                    assert np.all(t_out == v_out)
                    assert np.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_1(self):
        stop = tensor.iscalar('e')
        step = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = get_canonical_form_slice(slice(None, stop, step), length)
        f = self.function([stop, step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1])], N=0, op=self.ops)

        length = 5
        a = np.arange(length)
        for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for step in [-6, -3, -1, 2, 5]:
                out = f(stop, step, length)
                t_out = a[out[0]:out[1]:out[2]][::out[3]]
                v_out = a[:stop:step]
                assert np.all(t_out == v_out)
                assert np.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_2(self):
        start = tensor.iscalar('b')
        step = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = get_canonical_form_slice(slice(start, None, step), length)
        f = self.function([start, step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1])], N=0, op=self.ops)

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for step in [-6, -3, -1, 2, 5]:
                out = f(start, step, length)
                t_out = a[out[0]:out[1]:out[2]][::out[3]]
                v_out = a[start:None:step]
                assert np.all(t_out == v_out)
                assert np.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_3(self):
        start = tensor.iscalar('b')
        stop = tensor.iscalar('e')
        length = tensor.iscalar('l')
        cnf = get_canonical_form_slice(slice(start, stop, None), length)
        f = self.function([start, stop, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1])], N=0, op=self.ops)

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                out = f(start, stop, length)
                t_out = a[out[0]:out[1]:out[2]][::out[3]]
                v_out = a[start:stop:None]
                assert np.all(t_out == v_out)
                assert np.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_4(self):
        step = tensor.iscalar('s')
        length = tensor.iscalar('l')
        cnf = get_canonical_form_slice(slice(None, None, step), length)
        f = self.function([step, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1])], N=0, op=self.ops)

        length = 5
        a = np.arange(length)
        for step in [-6, -3, -1, 2, 5]:
            out = f(step, length)
            t_out = a[out[0]:out[1]:out[2]][::out[3]]
            v_out = a[None:None:step]
            assert np.all(t_out == v_out)
            assert np.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_5(self):
        start = tensor.iscalar('b')
        length = tensor.iscalar('l')
        cnf = get_canonical_form_slice(slice(start, None, None), length)
        f = self.function([start, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1])], N=0, op=self.ops)

        length = 5
        a = np.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            out = f(start, length)
            t_out = a[out[0]:out[1]:out[2]][::out[3]]
            v_out = a[start:None:None]
            assert np.all(t_out == v_out)
            assert np.all(t_out.shape == v_out.shape)

    def test_slice_canonical_form_6(self):
        stop = tensor.iscalar('e')
        length = tensor.iscalar('l')
        cnf = get_canonical_form_slice(slice(None, stop, None), length)
        f = self.function([stop, length], [
            tensor.as_tensor_variable(cnf[0].start),
            tensor.as_tensor_variable(cnf[0].stop),
            tensor.as_tensor_variable(cnf[0].step),
            tensor.as_tensor_variable(cnf[1])], N=0, op=self.ops)

        length = 5
        a = np.arange(length)
        for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            out = f(stop, length)
            t_out = a[out[0]:out[1]:out[2]][::out[3]]
            v_out = a[None:stop:None]
            assert np.all(t_out == v_out)
            assert np.all(t_out.shape == v_out.shape)

    def grad_list_(self, idxs, data):
        n = self.shared(data)

        for idx in idxs:
            # Should stay on the cpu.
            idx_ = _shared(np.asarray(idx))
            t = n[idx_]
            gn = theano.tensor.grad(theano.tensor.sum(theano.tensor.exp(t)), n)
            f = self.function([], [gn, gn.shape], op=self.adv_incsub1)
            topo = f.maker.fgraph.toposort()
            if not self.fast_compile:
                assert any([isinstance(node.op, self.adv_incsub1) and
                            node.op.inplace for node in topo])
            else:
                assert any([isinstance(node.op, self.adv_incsub1)
                            for node in topo])
            assert any([isinstance(node.op, self.adv_sub1) for node in topo])
            gval, gshape = f()
            good = np.zeros_like(data)
            # don't work when the same index is used many time
            # good[idx] += np.exp(data[idx])
            for i in idx:
                good[i] += np.exp(data[i])
            self.assertTrue(gval.ndim == data.ndim)
            self.assertTrue(np.allclose(gval, good), (gval, good))
            self.assertTrue(np.allclose(gshape, data.shape))

            def fct(t):
                return theano.tensor.sum(t[idx_])
            utt.verify_grad(fct, [data], mode=self.mode)

            # Test the grad of the grad (e.i. AdvancedIncSubtensor1.grad)
            def fct2(t):
                return theano.tensor.grad(theano.tensor.sum(t[idx_]), t)
            utt.verify_grad(fct2, [data], mode=self.mode)

            # Test shape of AdvancedIncSubtensor1 and AdvancedSubtensor1
            if not self.fast_compile:
                ops = (self.adv_incsub1, self.adv_sub1)
            else:
                ops = self.ops
            if idx is idxs[0]:
                f = self.function([], [gn.shape, n[idx_].shape],
                                  op=ops,
                                  N=0, N_fast=2)
                f()

    def test_wrong_exception_regression(self):
        a = fscalar()
        b = fscalar()
        c = vector()
        try:
            c[a:b]
        except NotImplementedError:
            self.fail()
        except TypeError:
            pass
        try:
            c[a:]
        except NotImplementedError:
            self.fail()
        except TypeError:
            pass
        try:
            c[:b]
        except NotImplementedError:
            self.fail()
        except TypeError:
            pass

    @attr('slow')
    def test_grad_list(self):
        data = rand(4)
        data = np.asarray(data, dtype=self.dtype)
        idxs = [[i] for i in range(data.shape[0])]
        for i in range(data.shape[0]):
            for j in range(0, data.shape[0], 2):
                idxs.append([i, j, (i + 1) % data.shape[0]])
        self.grad_list_(idxs, data)

        data = rand(4, 3)
        data = np.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

        data = rand(4, 3, 2)
        data = np.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

    def test_shape_list(self):
        # TODO for all type of subtensor shape
        for data, idx in [(rand(4), [1, 0]),
                          (rand(4, 2), [2, 3]),
                          (rand(4, 2, 3), [0, 3]),
                          (rand(4, 2, 3), [3, 3, 1, 2, 2, ]),
                          ]:
            data = np.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]
            f = self.function([], t.shape, op=self.ops, N=0, N_fast=1)
            val = f()
            self.assertTrue(np.allclose(val, data[idx].shape))

    def test_grad_advanced_inc_subtensor(self):
        def inc_slice(*s):
            def just_numeric_args(a, b):
                cost = (a[s] + b).sum()
                cost_wrt_a = theano.tensor.grad(cost, a)
                cost_wrt_b = theano.tensor.grad(cost, b)
                grads = cost_wrt_a.sum() + cost_wrt_b.sum()
                return grads
            return just_numeric_args

        # vector
        utt.verify_grad(
            inc_slice(slice(2, 4, None)),
            (np.asarray([0, 1, 2, 3, 4, 5.]), np.asarray([9, 9.]),),
            mode=self.mode)

        # matrix
        utt.verify_grad(
            inc_slice(slice(1, 2, None), slice(None, None, None)),
            (np.asarray([[0, 1], [2, 3], [4, 5.]]),
             np.asarray([[9, 9.]]),),
            mode=self.mode)

        # single element
        utt.verify_grad(
            inc_slice(2, 1),
            (np.asarray([[0, 1], [2, 3], [4, 5.]]), np.asarray(9.),),
            mode=self.mode)

    def test_inc_and_set_subtensor(self):
        # Test increment and set with broadcast

        X = self.shared(np.ones((9, 9)).astype(self.dtype))
        y = set_subtensor(X[1::, 1::], 0)
        f = self.function([], [y],
                          op=self.inc_sub,
                          N=1)
        out = f()

        res = np.ones((9, 9))
        res[1::, 1::] = 0
        assert np.allclose(out, res)

    def test_advanced1_inc_and_set(self):
        # Test advanced increment and set.

        rng = np.random.RandomState(seed=utt.fetch_seed())
        all_inputs_var = []
        all_inputs_num = []
        all_outputs_var = []
        all_outputs_num = []
        all_params = []
        for set_instead_of_inc in (False, True):
            for inplace in (False, True):
                for data_shape in ((10,), (4, 5), (1, 2, 3), (4, 5, 6, 7)):
                    data_n_dims = len(data_shape)
                    data_size = np.product(data_shape)
                    # Corresponding numeric variable.
                    data_num_init = np.arange(data_size, dtype=self.dtype)
                    data_num_init = data_num_init.reshape(data_shape)
                    inc_shapes = [data_shape[i:]
                                  for i in xrange(0, len(data_shape) + 1)]
                    # Test broadcasting of y.
                    inc_shapes += [(1,) + inc_shapes[-1][1:]]
                    for inc_shape in inc_shapes:
                        inc_n_dims = len(inc_shape)
                        # We copy the numeric value to be 100% sure there is no
                        # risk of accidentally sharing it.
                        data_num = data_num_init.copy()
                        # Symbolic variable to be incremented.
                        # We create a new one every time in order not to
                        # have duplicated variables in the function's inputs
                        data_var = self.type(
                            broadcastable=[False] * data_n_dims,
                            dtype=self.dtype)()
                        # Symbolic variable with rows to be incremented.
                        idx_var = theano.tensor.vector(dtype='int64')
                        n_to_inc = rng.randint(data_shape[0])
                        if (n_to_inc == 1 and
                                len(inc_shape) > 0 and
                                inc_shape[0] == 1 and
                                data_shape[0] > 1):
                            n_to_inc = 2
                        # Corresponding numeric variable.
                        # If set_instead_of_inc, we want to avoid repeating
                        # indices, as the order is not guaranteed.
                        idx_num = rng.choice(np.arange(data_shape[0]),
                                             n_to_inc,
                                             replace=(not set_instead_of_inc))
                        idx_num = idx_num.astype('int64')
                        # Symbolic variable with increment value.
                        inc_var = self.type(
                            broadcastable=[False] * inc_n_dims,
                            dtype=self.dtype)()
                        # Trick for the case where `inc_shape` is the same as
                        # `data_shape`: what we actually want is the first
                        # shape element to be equal to the number of rows to
                        # increment.
                        if len(inc_shape) == len(data_shape) and (
                                len(inc_shapes) == 0 or inc_shape[0] != 1):
                            inc_shape = (n_to_inc,) + inc_shape[1:]
                        # The param dtype is needed when inc_shape is empty.
                        # By default, it would return a float and rng.uniform
                        # with NumPy 1.10 will raise a Deprecation warning.
                        inc_size = np.product(inc_shape, dtype='int')
                        # Corresponding numeric variable.
                        inc_num = rng.uniform(size=inc_size).astype(self.dtype)
                        inc_num = inc_num.reshape(inc_shape)
                        # Result of the incrementation.
                        # (i) Theano
                        if set_instead_of_inc:
                            op = set_subtensor
                        else:
                            op = inc_subtensor
                        output = op(data_var[idx_var], inc_var,
                                    inplace=inplace)
                        # (ii) Numpy (note that Numpy increments only once
                        # duplicated indices, so we cannot directly use +=).
                        data_copy = data_num.copy()
                        for j, idx in enumerate(idx_num):
                            if len(inc_shape) == len(data_shape):
                                if inc_shape[0] == 1:
                                    # Allow broadcasting of y[0]
                                    inc_num0 = inc_num[0]
                                    if set_instead_of_inc:
                                        data_copy[idx] = inc_num0
                                    else:
                                        data_copy[idx] += inc_num0
                                else:
                                    # Special case where there is no broadcasting.
                                    if set_instead_of_inc:
                                        data_copy[idx] = inc_num[j]
                                    else:
                                        data_copy[idx] += inc_num[j]
                            else:
                                if set_instead_of_inc:
                                    data_copy[idx] = inc_num
                                else:
                                    data_copy[idx] += inc_num
                        data_var = theano.In(data_var, mutable=True)

                        # Remember data for the Theano function (see below).
                        all_inputs_var += [data_var, idx_var, inc_var]
                        all_inputs_num += [data_num, idx_num, inc_num]
                        all_outputs_var.append(output)
                        all_outputs_num.append(data_copy)
                        all_params.append((set_instead_of_inc, inplace, data_shape, inc_shape))
                        if False:  # Enable for debugging purpose.
                            f = self.function([data_var, idx_var, inc_var],
                                              output, accept_inplace=inplace,
                                              op=self.adv_incsub1)
                            if inplace:
                                # Ensure calling `f` will not alter `data_num`.
                                data_num = data_num.copy()
                            f_out = f(data_num.copy(), idx_num, inc_num)
                            assert np.allclose(f_out, data_copy)
                            if not inplace:
                                # Sanity check: `data_num` should be intact.
                                assert (data_num == data_num_init).all()

        # Actual test (we compile a single Theano function to make it faster).
        orig_warn = theano.config.warn.gpu_set_subtensor1
        try:
            theano.config.warn.gpu_set_subtensor1 = False
            f = self.function(all_inputs_var, all_outputs_var,
                              accept_inplace=True,
                              op=self.adv_incsub1,
                              N=len(all_outputs_var))
        finally:
            theano.config.warn.gpu_set_subtensor1 = orig_warn

        f_outs = f(*all_inputs_num)
        assert len(f_outs) == len(all_outputs_num)
        for params, f_out, output_num in izip(all_params, f_outs, all_outputs_num):
            # NB: if this assert fails, it will probably be easier to debug if
            # you enable the debug code above.
            assert np.allclose(f_out, output_num), (params, f_out, output_num)

    def test_adv_constant_arg(self):
        # Test case provided (and bug detected, gh-607) by John Salvatier
        m = matrix('m')
        gv = np.array([0, 1, 3])
        g = theano.tensor.constant(gv)
        i = theano.tensor.lvector('i')

        # s1 used to fail
        s1 = m[gv, i]
        s2 = m[g, i]

        assert gof.graph.is_same_graph(s1, s2)

    def test_adv1_inc_sub_notlastdim(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most) works.
        m = matrix('m')
        i = lvector('i')

        m1 = set_subtensor(m[:, i], 0)
        m2 = inc_subtensor(m[:, i], 1)
        f = theano.function([m, i], [m1, m2], mode=self.mode)

        m_val = rand(3, 5)
        i_val = randint_ranged(min=0, max=4, shape=(4,))
        m1_ref = m_val.copy()
        m2_ref = m_val.copy()

        m1_val, m2_val = f(m_val, i_val)
        for idx in i_val:
            m1_ref[:, idx] = 0
            m2_ref[:, idx] += 1

        assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
        assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)

    def test_adv1_inc_sub_notlastdim_2didx(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most) works,
        # if the index is a matrix.
        m = matrix('m')
        i = lmatrix('i')

        m1 = set_subtensor(m[:, i], 0)
        m2 = inc_subtensor(m[:, i], 1)

        f = theano.function([m, i], [m1, m2], mode=self.mode)

        m_val = rand(5, 7)
        i_val = randint_ranged(min=0, max=6, shape=(4, 2))
        m1_ref = m_val.copy()
        m2_ref = m_val.copy()

        m1_val, m2_val = f(m_val, i_val)
        for idx in i_val.ravel():
            m1_ref[:, idx] = 0
            m2_ref[:, idx] += 1

        assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
        assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)

    def test_adv1_inc_sub_notlastdim_1_2dval_broadcast(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most),
        # and incrementing/setting with broadcast
        m = matrix('m')

        # Test for both vector and matrix as index
        sym_i = (lvector('i'), lmatrix('i'))
        shape_i = ((4,), (4, 2))
        shape_val = ((3, 1), (3, 1, 1))

        # Disable the warning emitted for that case
        orig_warn = config.warn.inc_set_subtensor1
        try:
            config.warn.inc_set_subtensor1 = False

            for i, shp_i, shp_v in zip(sym_i, shape_i, shape_val):
                sub_m = m[:, i]
                m1 = set_subtensor(sub_m, np.zeros(shp_v))
                m2 = inc_subtensor(sub_m, np.ones(shp_v))
                f = theano.function([m, i], [m1, m2], mode=self.mode)

                m_val = rand(3, 5)
                i_val = randint_ranged(min=0, max=4, shape=shp_i)
                m1_ref = m_val.copy()
                m2_ref = m_val.copy()

                m1_val, m2_val = f(m_val, i_val)
                for idx in i_val.ravel():
                    m1_ref[:, idx] = 0
                    m2_ref[:, idx] += 1

                assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
                assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)
        finally:
            config.warn.inc_set_subtensor1 = orig_warn

    def test_adv1_inc_sub_notlastdim_1_2dval_no_broadcast(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most),
        # and incrementing/setting without broadcast
        m = matrix('m')

        # Test for both vector and matrix as index
        sym_i = (lvector('i'), lmatrix('i'))
        shape_i = ((4,), (4, 2))
        shape_val = ((3, 4), (3, 4, 2))

        # Disable the warning emitted for that case
        orig_warn = config.warn.inc_set_subtensor1

        try:
            config.warn.inc_set_subtensor1 = False
            for i, shp_i, shp_v in zip(sym_i, shape_i, shape_val):
                sub_m = m[:, i]
                m1 = set_subtensor(sub_m, np.zeros(shp_v))
                m2 = inc_subtensor(sub_m, np.ones(shp_v))
                f = theano.function([m, i], [m1, m2], mode=self.mode)

                m_val = rand(3, 5)
                i_val = randint_ranged(min=0, max=4, shape=shp_i)
                m1_ref = m_val.copy()
                m2_ref = m_val.copy()

                m1_val, m2_val = f(m_val, i_val)
                # We have to explicitly loop over all individual indices,
                # not as a list or array, numpy only increments the indexed
                # elements once even if the indices are repeated.
                for idx in i_val.ravel():
                    m1_ref[:, idx] = 0
                    m2_ref[:, idx] += 1

                assert np.allclose(m1_val, m1_ref), (m1_val, m1_ref)
                assert np.allclose(m2_val, m2_ref), (m2_val, m2_ref)
        finally:
            config.warn.inc_set_subtensor1 = orig_warn

    def test_take(self):
        a = tensor.matrix()
        f = theano.function(
            [a], a.take(0, axis=-1),
            allow_input_downcast=True, mode=self.mode)
        f(np.random.normal(0, 1, (30, 4)))


class TestIncSubtensor1(unittest.TestCase):
    # test inc_subtensor
    # also tests set_subtensor

    def setUp(self):
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

        self.s = tensor.iscalar()
        self.v = tensor.fvector()
        self.m = tensor.dmatrix()
        self.t = tensor.ctensor3()

        self.adv1q = tensor.lvector()  # advanced 1d query

    def test_cant_adv_idx_into_scalar(self):
        self.assertRaises(IndexError, lambda: self.s[self.adv1q])

    def test_index_into_vec_w_vec(self):
        a = self.v[self.adv1q]
        assert a.type == self.v.type

    def test_1d_set_adv_selection(self):
        a = set_subtensor(self.v[self.adv1q], self.v[self.adv1q])

        assert a.type == self.v.type

        # TODO: compile a function and verify that the subtensor is removed
        #      completely, because the whole expression is redundant.

        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1, 2])
        assert np.allclose(aval, [.4, 0.9, 0.1])

    def test_1d_inc_adv_selection(self):
        a = inc_subtensor(self.v[self.adv1q], self.v[self.adv1q])

        assert a.type == self.v.type
        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1, 2])
        assert np.allclose(aval, [.4, 1.8, 0.2])

    def test_1d_inc_adv_selection_w_broadcasting(self):
        a = inc_subtensor(self.v[self.adv1q], 3.0)

        assert a.type == self.v.type
        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1, 2])
        assert np.allclose(aval, [.4, 3.9, 3.1])

    def test_assigning_matrix_to_vector_selection(self):
        self.assertRaises(TypeError,
                          lambda: inc_subtensor(self.v[self.adv1q], fmatrix()))

    def test_matrix_idx(self):
        idx = tensor.lmatrix()
        a = self.m[idx]
        a2 = inc_subtensor(a, a)
        f = theano.function([self.m, idx], a2)

        mval = self.rng.random_sample((4, 10))
        idxval = np.array([[1, 2], [3, 2]])
        a2val = f(mval, idxval)

        utt.assert_allclose(a2val[0], mval[0])
        utt.assert_allclose(a2val[1], mval[1] * 2)
        utt.assert_allclose(a2val[2], mval[2] * 3)
        utt.assert_allclose(a2val[3], mval[3] * 2)

    def test_inc_bcastableidx(self):
        idx = tensor.constant([0])
        c_inc = tensor.col()
        m_inc = tensor.matrix()
        out1 = inc_subtensor(self.m[:, idx], c_inc)
        out2 = inc_subtensor(self.m[:, idx], m_inc)

        f = theano.function([self.m, c_inc, m_inc], [out1, out2])
        mval = self.rng.random_sample((10, 5))
        incval = self.rng.random_sample((10, 1)).astype(config.floatX)

        out1val, out2val = f(mval, incval, incval)
        utt.assert_allclose(out1val, out2val)


class TestAdvancedSubtensor(unittest.TestCase):
    # test inc_subtensor
    # also tests set_subtensor
    def __init__(self, name,
                 shared=tensor._shared,
                 sub=tensor.AdvancedSubtensor,
                 inc_sub=tensor.AdvancedIncSubtensor,
                 mode=None,
                 dtype=theano.config.floatX,
                 ignore_topo=DeepCopyOp):
        self.shared = shared
        self.sub = sub
        self.inc_sub = inc_sub
        if mode is None:
            mode = theano.compile.mode.get_default_mode()
        self.mode = mode
        self.dtype = dtype
        self.ignore_topo = ignore_topo
        super(TestAdvancedSubtensor, self).__init__(name)

    def setUp(self):
        self.s = iscalar()
        self.v = fvector()
        self.m = dmatrix()
        self.t = ctensor3()
        self.ft4 = ftensor4()

        self.ix1 = lvector()  # advanced 1d query
        self.ix12 = lvector()
        self.ix2 = lmatrix()
        self.ixr = lrow()

    def test_advinc_subtensor(self):
        x_shp = (20, 15, 10, 5)

        def check(idx, y_val, x_val, true):
            x = self.shared(x_val, name='x')
            y = tensor.tensor(dtype='float32',
                              broadcastable=(False,) * len(y_val.shape),
                              name='y')
            sym_idx = [tensor.as_tensor_variable(ix) for ix in idx]
            expr = tensor.advanced_inc_subtensor(x, y, *sym_idx)
            f = theano.function([y], expr, mode=self.mode)
            rval = f(y_val)
            assert np.allclose(rval, true)

        idxs_y_shp_pairs = [
            ((0, [1, 3, 5], 1), (3, 5)),
            (([1, 2, 4, 8],), (4, 15, 10, 5)),
            (([0, 1, 2], 0, [0, 1, 2]), (3, 3, 5)),
            (([[0, 1], [2, 3]], [[0, 1], [2, 3]]), (2, 2, 10, 5)),
        ]

        for idx, y_shps in idxs_y_shp_pairs:
            for i in range(len(y_shps) - 1):
                y_shp = y_shps[i:]
                x_val = np.arange(np.prod(x_shp), dtype='float32').reshape(x_shp) + 1
                y_val = np.arange(np.prod(y_shp), dtype='float32').reshape(y_shp) + 1
                rep = x_val.copy()
                try:
                    rep[idx] += y_val
                except ValueError:
                    continue
                check(idx, y_val, x_val, rep)
            x_val = np.arange(np.prod(x_shp), dtype='float32').reshape(x_shp) + 1
            y_val = np.array(1).astype(np.float32)
            rep = x_val.copy()
            rep[idx] += y_val
            check(idx, y_val, x_val, rep)

    def eval_output_and_check(self, t):
        f = inplace_func([], t, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo
                 if not isinstance(node.op, self.ignore_topo)]
        assert len(topo_) == 1
        assert isinstance(topo_[0].op, self.sub)
        tval = f()
        return tval

    def test_cant_adv_idx_into_scalar(self):
        self.assertRaises(IndexError, lambda: self.s[self.ix1])

    def test_index_into_vec_w_vec(self):
        a = self.v[self.ix1]
        assert a.type == self.v.type, (a.type, self.v.type)

    def test_index_into_vec_w_matrix(self):
        a = self.v[self.ix2]
        assert a.dtype == self.v.dtype, (a.dtype, self.v.dtype)
        assert a.broadcastable == self.ix2.broadcastable, (
            a.broadcastable, self.ix2.broadcastable)

    def test_index_into_mat_w_row(self):
        a = self.m[self.ixr]
        assert a.dtype == self.m.dtype, (a.dtype, self.m.dtype)
        assert a.broadcastable == (True, False, False)

    def test_index_w_int_and_vec(self):
        # like test_ok_list, but with a single index on the first one
        # data has to have at least 2 dimensions
        for data, idx in [(rand(4, 5), [2, 3]),
                          (rand(2, 4, 3), [0, 3]),
                          (rand(2, 4, 3), [3, 3, 1, 1, 2, 2, 0, 0]),
                          (rand(2, 4, 3), [3, 3, 1, 1, 2, 2, 0, 0,
                                           -1, -2, -3, -4]),
                          # Test 4 dims as gpu code use another algo
                          # in that case This new algo is not as much
                          # optimized for that case.
                          (rand(4, 4, 2, 3),
                           [3, 3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
                          # Test with TensorConstant index.
                          (rand(2, 4, 3),
                           theano.tensor.constant([3, 3, 1, 1, 2, 2, 0, 0])),
                          ]:
            data = np.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[0, idx]

            self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor))

            val = self.eval_output_and_check(t)
            if isinstance(idx, list):
                good = data[0, idx]
            else:
                good = data[0, idx.data]
            self.assertTrue(val.ndim == data.ndim - 1)
            self.assertTrue(np.allclose(val, good), (val, good))

    def test_inc_adv_subtensor_w_matrix(self):
        subt = self.v[self.ix2]
        a = inc_subtensor(subt, subt)

        assert a.type == self.v.type, (a.type, self.v.type)
        f = theano.function([self.v, self.ix2], a, allow_input_downcast=True,
                            mode=self.mode)
        aval = f([.4, .9, .1], [[1, 2],
                                [1, 2]])
        assert np.allclose(aval, [.4, .9 * 3, .1 * 3])

    def test_adv_subtensor_w_int_and_matrix(self):
        subt = self.ft4[0, :, self.ix2, :]
        f = theano.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype('float32')
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[0, :, ix2v, :]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_none_and_matrix(self):
        subt = self.ft4[:, None, :, self.ix2, :]
        f = theano.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype('float32')
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, None, :, ix2v, :]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_slice_and_matrix(self):
        subt = self.ft4[:, 0:1, self.ix2, :]
        f = theano.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype('float32')
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, 0:1, ix2v, :]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_matrix_and_int(self):
        subt = self.ft4[:, :, self.ix2, 0]
        f = theano.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype('float32')
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, :, ix2v, 0]
        utt.assert_allclose(rval, aval)

    def test_adv_subtensor_w_matrix_and_none(self):
        subt = self.ft4[:, :, self.ix2, None, :]
        f = theano.function([self.ft4, self.ix2], subt, mode=self.mode)
        ft4v = np.random.random((2, 3, 4, 5)).astype('float32')
        ix2v = np.asarray([[0, 1], [1, 0]])
        aval = f(ft4v, ix2v)
        rval = ft4v[:, :, ix2v, None, :]
        utt.assert_allclose(rval, aval)

    def test_inc_adv_subtensor_w_2vec(self):
        subt = self.m[self.ix1, self.ix12]
        a = inc_subtensor(subt, subt)

        typ = tensor.TensorType(self.m.type.dtype, self.ix2.type.broadcastable)
        assert a.type == typ, (a.type, typ)
        f = theano.function([self.m, self.ix1, self.ix12], a,
                            allow_input_downcast=True,
                            mode=self.mode)
        aval = f([[.4, .9, .1],
                  [5, 6, 7],
                  [.5, .3, .15]],
                 [1, 2, 1],
                 [0, 1, 0])
        assert np.allclose(aval,
                           [[.4, .9, .1],
                            [5 * 3, 6, 7],
                            [.5, .3 * 2, .15]]), aval

    def test_inc_adv_subtensor_with_broadcasting(self):
        inc = dscalar()
        a = inc_subtensor(self.m[self.ix1, self.ix12], inc)
        g_inc = tensor.grad(a.sum(), inc)

        assert a.type == self.m.type, (a.type, self.m.type)
        f = theano.function([self.m, self.ix1, self.ix12, inc], [a, g_inc],
                            allow_input_downcast=True,
                            mode=self.mode)
        aval, gval = f([[.4, .9, .1],
                        [5, 6, 7],
                        [.5, .3, .15]],
                       [1, 2, 1],
                       [0, 1, 0],
                       2.1)
        assert np.allclose(aval,
                           [[.4, .9, .1],
                            [5 + 2.1 * 2, 6, 7],
                            [.5, .3 + 2.1, .15]]), aval
        assert np.allclose(gval, 3.0), gval

    def test_inc_adv_subtensor1_with_broadcasting(self):
        inc = dscalar()
        a = inc_subtensor(self.m[self.ix1], inc)
        g_inc = tensor.grad(a.sum(), inc)

        assert a.type == self.m.type, (a.type, self.m.type)
        f = theano.function([self.m, self.ix1, inc], [a, g_inc],
                            allow_input_downcast=True,
                            mode=self.mode)
        aval, gval = f([[.4, .9, .1],
                        [5, 6, 7],
                        [.5, .3, .15]],
                       [0, 1, 0],
                       2.1)
        assert np.allclose(aval,
                           [[.4 + 2.1 * 2, .9 + 2.1 * 2, .1 + 2.1 * 2],
                            [5 + 2.1, 6 + 2.1, 7 + 2.1],
                            [.5, .3, .15]]), aval
        assert np.allclose(gval, 9.0), gval

    def test_inc_adv_subtensor_with_index_broadcasting(self):
        a = inc_subtensor(self.m[self.ix1, self.ix2], 2.1)

        assert a.type == self.m.type, (a.type, self.m.type)
        f = theano.function([self.m, self.ix1, self.ix2], a,
                            allow_input_downcast=True,
                            mode=self.mode)
        aval = f([[.4, .9, .1],
                  [5, 6, 7],
                  [.5, .3, .15]],
                 [0, 2, 0],
                 [[0, 1, 0],
                  [2, 2, 2]])
        assert np.allclose(aval,
                           [[.4 + 2 * 2.1, .9, .1 + 2 * 2.1],
                            [5, 6, 7],
                            [.5, .3 + 2.1, .15 + 2.1]]), aval

    def test_advanced_indexing(self):
        # tests advanced indexing in Theano for 2D and 3D tensors
        rng = np.random.RandomState(utt.fetch_seed())
        a = rng.uniform(size=(3, 3))
        b = theano.shared(a)
        i = tensor.iscalar()
        j = tensor.iscalar()
        z = b[[i, j], :]
        f1 = theano.function([i, j], z, mode=self.mode)
        cmd = f1(0, 1) == a[[0, 1], :]
        self.assertTrue(cmd.all())

        aa = rng.uniform(size=(4, 2, 3))
        bb = theano.shared(aa)
        k = tensor.iscalar()
        z = bb[[i, j, k], :, i:k]
        f2 = theano.function([i, j, k], z, mode=self.mode)
        cmd = f2(0, 1, 2) == aa[[0, 1, 2], :, 0:2]
        self.assertTrue(cmd.all())

    def test_adv_sub_3d(self):
        # Reported in https://github.com/Theano/Theano/issues/5674
        X = tensor.tensor3("X")

        xx = np.zeros((3, 2, 2), config.floatX)
        for i in range(3):
            for j in range(2):
                for k in range(2):
                    xx[i, j, k] = 100 * i + 10 * j + k

        b_idx = np.zeros((2, 2), 'int32')
        b_idx[0, 1] = 1
        b_idx[1, 1] = 2

        r_idx = np.arange(xx.shape[1])[:, np.newaxis]
        c_idx = np.arange(xx.shape[2])[np.newaxis, :]

        f = theano.function([X], X[b_idx, r_idx, c_idx], mode=self.mode)
        out = f(xx)
        utt.assert_allclose(out, xx[b_idx, r_idx, c_idx])

    def test_adv_sub_slice(self):
        # Reported in https://github.com/Theano/Theano/issues/5898
        var = self.shared(np.zeros([3, 3], dtype=config.floatX))
        slc = tensor.slicetype()
        f = theano.function([slc], var[slc], mode=self.mode)
        s = slice(1, 3)
        f(s)

    def test_adv_grouped(self):
        # Reported in https://github.com/Theano/Theano/issues/6152
        rng = np.random.RandomState(utt.fetch_seed())
        var_v = rng.rand(3, 63, 4).astype(config.floatX)
        var = self.shared(var_v)
        idx1_v = rng.randint(0, 61, size=(5, 4)).astype('int32')
        idx1 = self.shared(idx1_v)
        idx2 = tensor.arange(4)
        out = var[:, idx1, idx2]
        f = theano.function([], out, mode=self.mode)
        out_v = f()
        assert out_v.shape == (3, 5, 4)

        out_np = var_v[:, idx1_v, np.arange(4)]
        utt.assert_allclose(out_v, out_np)

    def test_grad(self):
        ones = np.ones((1, 3), dtype=self.dtype)
        n = self.shared(ones * 5, broadcastable=(True, False))
        idx = tensor.lvector()
        idx2 = tensor.lvector()
        t = n[idx, idx2]
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor))

        utt.verify_grad(lambda m: m[[1, 3], [2, 4]],
                        [np.random.rand(5, 5).astype(self.dtype)],
                        mode=self.mode)

        def fun(x, y):
            return advanced_inc_subtensor(x, y, [1, 3], [2, 4])
        utt.verify_grad(fun, [np.random.rand(5, 5).astype(self.dtype),
                              np.random.rand(2).astype(self.dtype)],
                        mode=self.mode)

        def fun(x, y):
            return advanced_set_subtensor(x, y, [1, 3], [2, 4])
        utt.verify_grad(fun, [np.random.rand(5, 5).astype(self.dtype),
                              np.random.rand(2).astype(self.dtype)],
                        mode=self.mode)


class TestInferShape(utt.InferShapeTester):
    @attr('slow')
    def test_infer_shape(self):
        # IncSubtensor
        admat = dmatrix()
        bdmat = dmatrix()
        advec = dvector()
        adscal = dscalar()
        admat_val = rand(5, 4)
        self._compile_and_check([admat, bdmat],
                                [inc_subtensor(admat[2:4], bdmat)],
                                [admat_val, [[1, 2, 3, 4]]], IncSubtensor)

        self._compile_and_check([admat, advec],
                                [inc_subtensor(admat[2], advec)],
                                [admat_val, [1, 2, 3, 4]], IncSubtensor)

        self._compile_and_check([admat, adscal],
                                [inc_subtensor(admat[2, 3], adscal)],
                                [admat_val, 1], IncSubtensor)

        self._compile_and_check([admat, adscal],
                                [inc_subtensor(admat[1:3, 2], adscal)],
                                [admat_val, 1], IncSubtensor)

        self._compile_and_check([admat, bdmat],
                                [set_subtensor(admat[2:4], bdmat)],
                                [admat_val, [[1, 2, 3, 4]]], IncSubtensor)

        self._compile_and_check([admat, advec],
                                [set_subtensor(admat[2], advec)],
                                [admat_val, [1, 2, 3, 4]], IncSubtensor)

        self._compile_and_check([admat, adscal],
                                [set_subtensor(admat[2, 3], adscal)],
                                [admat_val, 1], IncSubtensor)

        self._compile_and_check([admat, adscal],
                                [set_subtensor(admat[1:3, 2], adscal)],
                                [admat_val, 1], IncSubtensor)

        adtens4 = dtensor4()
        bdtens4 = dtensor4()
        adtens4_val = rand(3, 4, 2, 5)
        self._compile_and_check([adtens4, bdtens4],
                                [inc_subtensor(adtens4[::, 2:4, ::, ::], bdtens4)],
                                [adtens4_val, [[[[1, 2, 3, 4, 5]]]]], IncSubtensor,
                                warn=False)
        self._compile_and_check([adtens4, bdmat],
                                [inc_subtensor(adtens4[2, 2:4, 1, ::], bdmat)],
                                [adtens4_val, [[1, 2, 3, 4, 5]]], IncSubtensor)

        self._compile_and_check([adtens4, advec],
                                [inc_subtensor(adtens4[0, 1, ::, 4], advec)],
                                [adtens4_val, [1, 2]], IncSubtensor)

        self._compile_and_check([adtens4, adscal],
                                [inc_subtensor(adtens4[1:3, 1, ::, 2:4], adscal)],
                                [adtens4_val, 1], IncSubtensor)

        self._compile_and_check([adtens4, bdtens4],
                                [set_subtensor(adtens4[::, 2:4, ::, ::], bdtens4)],
                                [adtens4_val, [[[[1, 2, 3, 4, 5]]]]], IncSubtensor,
                                warn=False)

        self._compile_and_check([adtens4, bdmat],
                                [set_subtensor(adtens4[2, 2:4, 1, ::], bdmat)],
                                [adtens4_val, [[1, 2, 3, 4, 5]]], IncSubtensor)

        self._compile_and_check([adtens4, advec],
                                [set_subtensor(adtens4[0, 1, ::, 4], advec)],
                                [adtens4_val, [1, 2]], IncSubtensor)

        self._compile_and_check([adtens4, adscal],
                                [set_subtensor(adtens4[1:3, 1, ::, 2:4], adscal)],
                                [adtens4_val, 1], IncSubtensor)

        # AdvancedIncSubtensor1
        admat = dmatrix()
        bdmat = dmatrix()
        advec = dvector()
        adscal = dscalar()
        admat_val = rand(5, 4)
        aivec_val = [2, 3]
        self._compile_and_check([admat, bdmat],
                                [set_subtensor(admat[aivec_val], bdmat)],
                                [admat_val, [[1, 2, 3, 4]]], AdvancedIncSubtensor1)

        aivec_val = [1, 3, 2]
        self._compile_and_check([admat, advec],
                                [set_subtensor(admat[aivec_val], advec)],
                                [admat_val, [1, 2, 3, 4]], AdvancedIncSubtensor1)

        aivec_val = [0, 3, 0]
        self._compile_and_check([admat, adscal],
                                [set_subtensor(admat[aivec_val], adscal)],
                                [admat_val, 1], AdvancedIncSubtensor1)

        bdtens4 = dtensor4()
        adtens4_val = rand(4, 3, 2, 5)
        aivec_val = [2, 3]
        self._compile_and_check([adtens4, bdtens4],
                                [set_subtensor(adtens4[aivec_val], bdtens4)],
                                [adtens4_val, [[[[1, 2, 3, 4, 5]]]]],
                                AdvancedIncSubtensor1,
                                warn=False)

        aivec_val = [1, 3, 2]
        self._compile_and_check([adtens4, advec],
                                [set_subtensor(adtens4[aivec_val], advec)],
                                [adtens4_val, [1, 2, 3, 4, 5]],
                                AdvancedIncSubtensor1)

        aivec_val = [0, 3, 0]
        self._compile_and_check([adtens4, adscal],
                                [set_subtensor(adtens4[aivec_val], adscal)],
                                [adtens4_val, 1],
                                AdvancedIncSubtensor1)

        aivec_val = [2, 3]
        self._compile_and_check([admat, bdmat],
                                [inc_subtensor(admat[aivec_val], bdmat)],
                                [admat_val, [[1, 2, 3, 4], [5, 6, 7, 8]]],
                                AdvancedIncSubtensor1)

        aivec_val = [1, 3, 2]
        self._compile_and_check([admat, advec],
                                [inc_subtensor(admat[aivec_val], advec)],
                                [admat_val, [1, 2, 3, 4]], AdvancedIncSubtensor1)

        aivec_val = [0, 3, 0]
        self._compile_and_check([admat, adscal],
                                [inc_subtensor(admat[aivec_val], adscal)],
                                [admat_val, 1], AdvancedIncSubtensor1)

        bdtens4 = dtensor4()
        adtens4_val = rand(4, 3, 2, 5)
        aivec_val = [2, 3]
        self._compile_and_check([adtens4, bdtens4],
                                [inc_subtensor(adtens4[aivec_val], bdtens4)],
                                [adtens4_val, [[[[1, 2, 3, 4, 5]]],
                                               [[[6, 7, 8, 9, 10]]]]],
                                AdvancedIncSubtensor1,
                                warn=False)

        aivec_val = [1, 2, 1]
        self._compile_and_check([adtens4, advec],
                                [inc_subtensor(adtens4[aivec_val], advec)],
                                [adtens4_val, [1, 2, 3, 4, 5]],
                                AdvancedIncSubtensor1)

        aivec_val = [0, 3, 0]
        self._compile_and_check([adtens4, adscal],
                                [inc_subtensor(adtens4[aivec_val], adscal)],
                                [adtens4_val, 2],
                                AdvancedIncSubtensor1)

        # AdvancedIncSubtensor
        aivec_val = [1, 3, 2]
        bivec_val = [0, 3, 3]
        advec_val = [23, 24, 25]
        self._compile_and_check([admat, advec],
                                [set_subtensor(admat[aivec_val, bivec_val], advec)],
                                [admat_val, advec_val], AdvancedIncSubtensor)

    def test_adv_sub(self):
        admat = dmatrix()
        aivec = lvector()
        bivec = lvector()

        admat_val = rand(5, 4)
        aivec_val = [1, 3, 2]
        bivec_val = [0, 3, 3]
        self._compile_and_check([admat, aivec, bivec],
                                [admat[aivec, bivec]],
                                [admat_val, aivec_val, bivec_val], AdvancedSubtensor)
        # Test case that aren't implemented, but make sure they do not crash.
        self._compile_and_check([admat, aivec],
                                [admat[aivec, 1:3]],
                                [admat_val, aivec_val], AdvancedSubtensor,
                                check_topo=False)
        self._compile_and_check([admat, aivec],
                                [admat[1:3, aivec]],
                                [admat_val, aivec_val], AdvancedSubtensor,
                                check_topo=False)

    def test_boolean(self):
        n = dmatrix()
        n_val = np.arange(6).reshape((2, 3))

        # infer_shape is not implemented, but it should not crash
        self._compile_and_check([n],
                                [n[n[:, 0] > 2, n[0, :] > 2]],
                                [n_val], tensor.AdvancedBooleanSubtensor,
                                check_topo=False)
        self._compile_and_check([n],
                                [n[n[:, 0] > 2]],
                                [n_val], tensor.AdvancedBooleanSubtensor,
                                check_topo=False)
