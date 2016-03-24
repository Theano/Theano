from __future__ import absolute_import, print_function, division
import logging
import sys
import unittest

from nose.plugins.skip import SkipTest
import numpy
from six import StringIO
from six.moves import xrange

import theano
from theano.compat import exc_message, izip, PY3
from theano.compile import DeepCopyOp
from theano import config
from theano import gof
import theano.scalar as scal
import theano.tensor as tensor
from theano.tests import unittest_tools as utt
from theano.tensor.subtensor import (inc_subtensor, set_subtensor,
                                     advanced_inc_subtensor1,
                                     advanced_set_subtensor1,
                                     advanced_inc_subtensor,
                                     advanced_set_subtensor,
                                     Subtensor, IncSubtensor,
                                     AdvancedSubtensor1, AdvancedSubtensor,
                                     advanced_subtensor1, inplace_increment,
                                     AdvancedIncSubtensor1,
                                     AdvancedIncSubtensor,
                                     get_canonical_form_slice)
from theano.tensor import (as_tensor_variable, _shared,
                           NotScalarConstantError,
                           fscalar, iscalar, dscalar, cscalar,
                           vector, dvector, fvector, lvector, lrow,
                           fmatrix, dmatrix, lmatrix, matrix,
                           ctensor3, dtensor4)
from theano.tensor.tests.test_basic import rand, randint_ranged, inplace_func
from theano.tests.unittest_tools import attr

if PY3:
    def L(i):
        return i
else:
    def L(i):
        return long(i)


class T_subtensor(unittest.TestCase, utt.TestOptimizationMixin):
    """
    This is build in a way that allow to reuse it to test the
    equivalent gpu op.
    """
    def __init__(self, name, shared=tensor._shared,
                 sub=tensor.Subtensor,
                 inc_sub=tensor.IncSubtensor,
                 adv_sub1=tensor.AdvancedSubtensor1,
                 adv_incsub1=tensor.AdvancedIncSubtensor1,
                 mode=None,
                 dtype=theano.config.floatX,
                 type=tensor.TensorType,
                 ignore_topo=DeepCopyOp):
        self.shared = shared
        self.sub = sub
        self.inc_sub = inc_sub
        self.adv_sub1 = adv_sub1
        self.adv_incsub1 = adv_incsub1
        if mode is None:
            mode = theano.compile.mode.get_default_mode()
        self.mode = mode
        self.dtype = dtype
        self.type = type
        self.ignore_topo = ignore_topo
        self.fast_compile = theano.config.mode == 'FAST_COMPILE'
        self.ops = (sub, inc_sub, adv_sub1, adv_incsub1)
        return super(T_subtensor, self).__init__(name)

    def function(self, inputs, outputs, accept_inplace=False,
                 op=None, mode=None, N=1, N_fast=None):
        """ wrapper around theano.function that also check the output

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

    def eval_output_and_check(self, t, list=False, mode=None):
        if mode is None:
            mode = self.mode
        f = inplace_func([], t, mode=mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
                                                         self.ignore_topo)]
        assert len(topo_) == 1
        if not list:
            assert isinstance(topo_[0].op, self.sub)
        else:
            assert isinstance(topo_[0].op, self.adv_sub1)
        tval = f()
        return tval

    def test0_err_invalid(self):
        # it is impossible to retrieve a view of a 0-d tensor
        n = self.shared(numpy.ones((), dtype=self.dtype))
        try:
            t = n[0]
        except ValueError as e:
            self.assertTrue(hasattr(e, 'subtensor_invalid'))
            return
        self.fail()

    def test1_err_bounds(self):
        n = self.shared(numpy.ones(3, dtype=self.dtype))
        ctv_backup = config.compute_test_value
        config.compute_test_value = 'off'
        try:
            t = n[7]
        finally:
            config.compute_test_value = ctv_backup
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        # Silence expected error messages
        _logger = logging.getLogger('theano.gof.opt')
        oldlevel = _logger.level
        _logger.setLevel(logging.CRITICAL)
        try:
            try:
                self.eval_output_and_check(t)
            except IndexError as e:
                return
            self.fail()
        finally:
            _logger.setLevel(oldlevel)

    def test1_err_subslice(self):
        n = self.shared(numpy.ones(3, dtype=self.dtype))
        try:
            t = n[slice(0, slice(1, 2, None), None)]
        except Exception as e:
            # Relax constraint on the type of Exception,
            # since this might be handled by AvancedSubtensor
            # if e[0] != Subtensor.e_indextype:
            #    raise
            return
        self.fail()

    def test1_ok_range_finite(self):
        n = self.shared(numpy.arange(3, dtype=self.dtype))
        t = n[0:2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [0, 1]).all())

    def test2_ok_range_finite(self):
        n = self.shared(numpy.arange(12, dtype=self.dtype).reshape((3, 4)))
        # Also check negative index
        for idx in [(slice(0, 2), 3), ((slice(0, 2), -1)), (slice(0, 2), -4)]:
            t = n[idx]  # l]#0:2,3]
            self.assertTrue(isinstance(t.owner.op, Subtensor))
            tval = self.eval_output_and_check(t)
            self.assertTrue(tval.shape == (2,))
            self.assertTrue(numpy.allclose(tval, n.get_value()[idx]))

    def test1_0_dims(self):
        n = self.shared(numpy.ones((), dtype=self.dtype))
        t = self.sub([])(n)
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        self.eval_output_and_check(
            t, mode=self.mode.excluding("local_useless_subtensor"))

    def test1_err_invalid(self):
        n = self.shared(numpy.ones(1, dtype=self.dtype))
        try:
            t = n[0, 0]
        except ValueError as e:
            self.assertTrue(hasattr(e, 'subtensor_invalid'))
            return
        self.fail()

    def test1_ok_elem(self):
        n = self.shared(numpy.ones(1, dtype=self.dtype) * 5)
        t = n[0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(tval == 5.0)

    def test1_ok_range_infinite(self):
        n = self.shared(numpy.arange(3, dtype=self.dtype))
        t = n[1:]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [1.0, 2.0]).all())

    def test1_ok_strided(self):
        n = self.shared(numpy.arange(5, dtype=self.dtype))
        t = n[1::2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [1.0, 3.0]).all())

        t = n[0:-1:2]  # 0 to 1 from the end stepping by 2
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue((tval == [0.0, 2.0]).all())

    def test2_err_bounds0(self):
        n = self.shared(numpy.ones((2, 3), dtype=self.dtype) * 5)
        ctv_backup = config.compute_test_value
        config.compute_test_value = 'off'
        try:
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
        finally:
            config.compute_test_value = ctv_backup

    def test2_err_bounds1(self):
        n = self.shared((numpy.ones((2, 3), dtype=self.dtype) * 5))
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
        n = self.shared(numpy.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[0, 2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(numpy.all(tval == 2))

    def test2_ok_row(self):
        n = self.shared(numpy.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[1]
        self.assertFalse(any(n.type.broadcastable))
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (3,))
        self.assertTrue(numpy.all(tval == [3, 4, 5]))

    def test2_ok_col(self):
        n = self.shared(numpy.arange(6, dtype=self.dtype).reshape((2, 3)))
        t = n[:, 0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        self.assertFalse(any(n.type.broadcastable))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(numpy.all(tval == [0, 3]))

    def test2_ok_rows_finite(self):
        n = self.shared(numpy.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[1:3, 0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(numpy.all(tval == [3, 6]))

    def test2_ok_cols_infinite(self):
        n = self.shared(numpy.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[1, 2:]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (1,))
        self.assertTrue(numpy.all(tval == 5))

    def test2_ok_strided(self):
        n = self.shared(numpy.arange(20, dtype=self.dtype).reshape((4, 5)))
        t = n[1:4:2, 1:5:2]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2, 2))
        self.assertTrue(numpy.all(tval == [[6, 8], [16, 18]]))

    def test3_ok_mat(self):
        n = self.shared(numpy.arange(24, dtype=self.dtype).reshape((2, 3, 4)))
        t = n[0, 0, 0]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == ())
        self.assertTrue(numpy.all(tval == 0))

    def test_long(self):
        n = self.shared(numpy.arange(12, dtype=self.dtype).reshape((4, 3)))
        t = n[L(1):L(4):L(2), L(1)]
        self.assertTrue(isinstance(t.owner.op, Subtensor))
        tval = self.eval_output_and_check(t)
        self.assertTrue(tval.shape == (2,))
        self.assertTrue(numpy.all(tval == [4, 10]))

    def test_long_too_big(self):
        # Currently, we cast Python longs to int64 when used for indexing.
        # This test checks that using a long that does not fit raises an error.
        n = self.shared(numpy.arange(12, dtype=self.dtype).reshape((4, 3)))
        self.assertRaises(Exception, lambda: n[:L(2 ** 63)])

    def test_list_slice(self):
        x = theano.tensor.arange(100).reshape((5, 5, 4))
        res = x[[slice(1, -1)] * x.ndim].eval()
        x = numpy.arange(100).reshape((5, 5, 4))
        numpy.allclose(res, x[[slice(1, -1)] * x.ndim])

    def test_newaxis(self):
        """
        newaxis support comes from logic in the __getitem__ of TensorType
        Variables, which currently inserts dimshuffle to get the right number
        of dimensions, and adjusts the slice tuple accordingly.

        So testing is done via square-bracket notation rather than direct
        interaction with the Subtensor Op (which has no support of its own for
        newaxis).
        """
        newaxis = numpy.newaxis

        n = self.shared(numpy.arange(24, dtype=self.dtype).reshape((2, 3, 4)))
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

        vs1, vn3, vn4 = theano.function([s], [s1, n3, n4])(-2.0)

        assert numpy.all(vs1 == [-2.0])
        assert numpy.all(vn3
                == numpy.arange(24)[newaxis, :, newaxis])
        assert numpy.all(vn4
                == numpy.arange(24).reshape((2, 3, 4))[:, :, :, newaxis])

    def test_grad_1d(self):
        subi = 0
        data = numpy.asarray(rand(2, 3), dtype=self.dtype)
        n = self.shared(data)
        z = scal.constant(subi)
        t = n[z:, z]
        gn = theano.tensor.grad(theano.tensor.sum(theano.tensor.exp(t)), n)

        f = inplace_func([], gn, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
                                                         self.ignore_topo)]
        if not self.fast_compile:
            assert len(topo_) == 6
        assert numpy.sum([isinstance(node.op, self.inc_sub)
                          for node in topo_]) == 1
        assert numpy.sum([isinstance(node.op, self.sub)
                          for node in topo_]) == 1
        gval = f()

        good = numpy.zeros_like(data)
        good[subi:, subi] = numpy.exp(data[subi:, subi])
        self.assertTrue(numpy.allclose(gval, good), (gval, good))

    def test_grad_2d_inc_set_subtensor(self):
        for n_shape, m_shape in [
            [(2, 3), (2, 2)],
            [(3, 2), (2, 2)],
            [(3, 2), (1, 2)],
            [(3, 2), (2,)],
        ]:
            for op in [inc_subtensor, set_subtensor]:
                subi = 2
                data = numpy.asarray(rand(*n_shape), dtype=self.dtype)
                n = self.shared(data)
                z = scal.constant(subi)
                m = matrix('m', dtype=self.dtype)
                mv = numpy.asarray(rand(*m_shape), dtype=self.dtype)

                t = op(n[:z, :z], m)
                gn, gm = theano.tensor.grad(theano.tensor.sum(t), [n, m])
                utt.verify_grad(lambda m: op(n[:z, :z], m), [mv])
                utt.verify_grad(lambda nn: op(nn[:z, :z], mv), [data])

    def test_grad_0d(self):
        data = numpy.asarray(rand(2, 3), dtype=self.dtype)
        n = self.shared(data)
        t = n[1, 0]
        gn = theano.tensor.grad(theano.tensor.sum(theano.tensor.exp(t)), n)
        f = self.function([], gn)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
             self.ignore_topo)]
        if not self.fast_compile:
            assert len(topo_) == 6
        assert numpy.sum([isinstance(node.op, self.inc_sub)
             for node in topo_]) == 1
        assert numpy.sum([isinstance(node.op, self.sub)
             for node in topo_]) == 1

        gval = f()
        good = numpy.zeros_like(data)
        good[1, 0] = numpy.exp(data[1, 0])
        self.assertTrue(numpy.allclose(gval, good), (gval, good))

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
                          (rand(4, 4, 2, 3), [3,
                               3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
                          # Test with TensorConstant index.
                          (rand(4, 2, 3),
                           theano.tensor.constant([3, 3, 1, 1, 2, 2, 0, 0])),
                          ]:
            data = numpy.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]

            # We test again AdvancedSubtensor1 as we transfer data to the cpu.
            self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

            val = self.eval_output_and_check(t, list=True)
            if isinstance(idx, list):
                good = data[idx]
            else:
                good = data[idx.data]
            self.assertTrue(val.ndim == data.ndim)
            self.assertTrue(numpy.allclose(val, good), (val, good))

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
                            [numpy.random.rand(5, 5).astype(self.dtype)])
            g_0 = g()
            utt.verify_grad(lambda m: m[idx],
                            [data])

    def test_err_invalid_list(self):
        n = self.shared(numpy.asarray(5, dtype=self.dtype))
        self.assertRaises(TypeError, n.__getitem__, [0, 0])

    def test_err_invalid_2list_dtype(self):
        n = self.shared(numpy.ones((3, 3), dtype=self.dtype) * 5)
        self.assertRaises(TypeError, n.__getitem__, ([0., 0], [1, 1]))

    def test_err_bound_list(self):
        n = self.shared(numpy.ones((2, 3), dtype=self.dtype) * 5)
        l = lvector()
        t = n[l]
        # We test again AdvancedSubtensor1 as we transfer data to the cpu.
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor1))

        f = self.function([l], t, op=self.adv_sub1)

        # the grad
        g = self.function([l],
                          inc_subtensor(t, numpy.asarray([[1.]], self.dtype)),
                          op=self.adv_incsub1)

        for shp in [[0, 4], [0, -3], [-10]]:
            self.assertRaises(IndexError, f, shp)
            self.assertRaises(IndexError, g, shp)

    def test_adv_sub1_broadcast(self):
        v = numpy.arange(3, dtype=self.dtype).reshape((1, 3))
        n = self.shared(v*5, broadcastable=(True, False))
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
        self.assertTrue(numpy.allclose(f_0, v*5))
        f_00 = f([0, 0])
        self.assertTrue(f_00.shape == (2, 3))
        self.assertTrue(numpy.allclose(f_00, v*5))
        self.assertRaises(IndexError, f, [0, 1])

        # Test the gradient
        c = t.sum()
        gn = theano.grad(c, n)
        g = self.function([idx], gn, op=self.adv_incsub1)
        g_0 = g([0])
        self.assertTrue(g_0.shape == (1, 3))
        self.assertTrue(numpy.allclose(g_0, 1))
        g_00 = g([0, 0])
        self.assertTrue(g_00.shape == (1, 3))
        self.assertTrue(numpy.allclose(g_00, 2))

        utt.verify_grad(lambda m: m[[1, 3]],
                        [numpy.random.rand(5, 5).astype(self.dtype)])

        def fun(x, y):
            return advanced_inc_subtensor1(x, y, [1, 3])
        utt.verify_grad(fun, [numpy.random.rand(5, 5).astype(self.dtype),
                              numpy.random.rand(2, 5).astype(self.dtype)])

        def fun(x, y):
            return advanced_set_subtensor1(x, y, [1, 3])
        utt.verify_grad(fun, [numpy.random.rand(5, 5).astype(self.dtype),
                              numpy.random.rand(2, 5).astype(self.dtype)])

    def test_adv_sub1_idx_broadcast(self):
        # The idx can be a broadcastable vector.
        ones = numpy.ones((4, 3), dtype=self.dtype)
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
        self.assertTrue(numpy.allclose(f_0, 5))

        # Test the gradient
        c = t.sum()
        gn = theano.grad(c, n)
        g = self.function([idx], gn, op=self.adv_incsub1)
        g_0 = g([0])
        self.assertTrue(g_0.shape == (4, 3))
        self.assertTrue(numpy.allclose(g_0[0], 1))
        self.assertTrue(numpy.allclose(g_0[1:], 0))

    @attr('slow')
    def test_shape_i_const(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = self.mode.including("fast_run")
        data = self.shared(numpy.array(numpy.arange(5), dtype=self.dtype))
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
                assert numpy.all(t_shape == shape)
            assert tensor.Subtensor not in [x.op for x in
                                           f.maker.fgraph.toposort()]

    def test_shape_i_scalar(self):
        # Each axis is treated independently by shape_i/shape operators

        mode_opt = self.mode.including("fast_run")

        v_data = numpy.array(numpy.arange(5), dtype=self.dtype)
        t_data = self.shared(v_data)
        start = tensor.iscalar('b')
        stop = tensor.iscalar('e')
        step = tensor.iscalar('s')
        f = self.function([start, stop, step],
                          t_data[start:stop:step].shape,
                          mode=mode_opt,
                          op=self.ops,
                          N=0)
        assert tensor.Subtensor not in [x.op for x in f.maker.
            fgraph.toposort()]
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                for step in [-3, -1, 2, 5]:
                    assert numpy.all(f(start, stop, step) ==
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
        a = numpy.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in  [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                for step in [-6, -3, -1, 2, 5]:
                    out = f(start, stop, step, length)
                    t_out = a[out[0]:out[1]:out[2]][::out[3]]
                    v_out = a[start:stop:step]
                    assert numpy.all(t_out == v_out)
                    assert numpy.all(t_out.shape == v_out.shape)

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
        a = numpy.arange(length)
        for stop in  [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for step in [-6, -3, -1, 2, 5]:
                out = f(stop, step, length)
                t_out = a[out[0]:out[1]:out[2]][::out[3]]
                v_out = a[:stop:step]
                assert numpy.all(t_out == v_out)
                assert numpy.all(t_out.shape == v_out.shape)

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
        a = numpy.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for step in [-6, -3, -1, 2, 5]:
                out = f(start, step, length)
                t_out = a[out[0]:out[1]:out[2]][::out[3]]
                v_out = a[start:None:step]
                assert numpy.all(t_out == v_out)
                assert numpy.all(t_out.shape == v_out.shape)

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
        a = numpy.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            for stop in  [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
                out = f(start, stop, length)
                t_out = a[out[0]:out[1]:out[2]][::out[3]]
                v_out = a[start:stop:None]
                assert numpy.all(t_out == v_out)
                assert numpy.all(t_out.shape == v_out.shape)

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
        a = numpy.arange(length)
        for step in [-6, -3, -1, 2, 5]:
            out = f(step, length)
            t_out = a[out[0]:out[1]:out[2]][::out[3]]
            v_out = a[None:None:step]
            assert numpy.all(t_out == v_out)
            assert numpy.all(t_out.shape == v_out.shape)

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
        a = numpy.arange(length)
        for start in [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            out = f(start, length)
            t_out = a[out[0]:out[1]:out[2]][::out[3]]
            v_out = a[start:None:None]
            assert numpy.all(t_out == v_out)
            assert numpy.all(t_out.shape == v_out.shape)

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
        a = numpy.arange(length)
        for stop in  [-8, -5, -4, -1, 0, 1, 4, 5, 8]:
            out = f(stop, length)
            t_out = a[out[0]:out[1]:out[2]][::out[3]]
            v_out = a[None:stop:None]
            assert numpy.all(t_out == v_out)
            assert numpy.all(t_out.shape == v_out.shape)

    def grad_list_(self, idxs, data):
        n = self.shared(data)

        for idx in idxs:
            # Should stay on the cpu.
            idx_ = _shared(numpy.asarray(idx))
            t = n[idx_]
            gn = theano.tensor.grad(theano.tensor.sum(theano.tensor.exp(t)), n)
            f = self.function([], [gn, gn.shape], op=self.adv_incsub1)
            topo = f.maker.fgraph.toposort()
            if not self.fast_compile:
                assert any([isinstance(node.op, self.
                    adv_incsub1) and node.op.inplace for node in topo])
            else:
                assert any([isinstance(node.op, self.
                    adv_incsub1) for node in topo])
            assert any([isinstance(node.op, self.adv_sub1) for node in topo])
            gval, gshape = f()
            good = numpy.zeros_like(data)
            # don't work when the same index is used many time
            # good[idx] += numpy.exp(data[idx])
            for i in idx:
                good[i] += numpy.exp(data[i])
            self.assertTrue(gval.ndim == data.ndim)
            self.assertTrue(numpy.allclose(gval, good), (gval, good))
            self.assertTrue(numpy.allclose(gshape, data.shape))

            def fct(t):
                return theano.tensor.sum(t[idx_])
            utt.verify_grad(fct, [data])

            # Test the grad of the grad (e.i. AdvancedIncSubtensor1.grad)
            def fct2(t):
                return theano.tensor.grad(theano.tensor.sum(t[idx_]), t)
            utt.verify_grad(fct2, [data])

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
        data = numpy.asarray(data, dtype=self.dtype)
        idxs = [[i] for i in range(data.shape[0])]
        for i in range(data.shape[0]):
            for j in range(0, data.shape[0], 2):
                idxs.append([i, j, (i + 1) % data.shape[0]])
        self.grad_list_(idxs, data)

        data = rand(4, 3)
        data = numpy.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

        data = rand(4, 3, 2)
        data = numpy.asarray(data, dtype=self.dtype)
        self.grad_list_(idxs, data)

    def test_shape_list(self):
        # TODO for all type of subtensor shape
        for data, idx in [(rand(4), [1, 0]),
                          (rand(4, 2), [2, 3]),
                          (rand(4, 2, 3), [0, 3]),
                          (rand(4, 2, 3), [3, 3, 1, 2, 2, ]),
                          ]:
            data = numpy.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[idx]
            f = self.function([], t.shape, op=self.ops, N=0, N_fast=1)
            val = f()
            self.assertTrue(numpy.allclose(val, data[idx].shape))

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
            (numpy.asarray([0, 1, 2, 3, 4, 5.]), numpy.asarray([9, 9.]),))

        # matrix
        utt.verify_grad(
            inc_slice(slice(1, 2, None), slice(None, None, None)),
            (numpy.asarray([[0, 1], [2, 3], [4, 5.]]),
             numpy.asarray([[9, 9.]]),))

        # single element
        utt.verify_grad(
            inc_slice(2, 1),
            (numpy.asarray([[0, 1], [2, 3], [4, 5.]]), numpy.asarray(9.),))

    def test_inc_and_set_subtensor(self):
        """
        Test increment and set with broadcast
        """

        X = self.shared(numpy.ones((9, 9)).astype(self.dtype))
        y = set_subtensor(X[1::, 1::],  0)
        f = self.function([], [y],
                          op=self.inc_sub,
                          N=1)
        out = f()

        res = numpy.ones((9, 9))
        res[1::, 1::] = 0
        assert numpy.allclose(out, res)

    def test_advanced1_inc_and_set(self):
        """
        Test advanced increment and set.
        """
        rng = numpy.random.RandomState(seed=utt.fetch_seed())
        all_inputs_var = []
        all_inputs_num = []
        all_outputs_var = []
        all_outputs_num = []
        for set_instead_of_inc in (False, True):
            for inplace in (False, True):
                for data_shape in ((10,), (4, 5), (1, 2, 3), (4, 5, 6, 7)):
                    data_n_dims = len(data_shape)
                    data_size = numpy.product(data_shape)
                    # Corresponding numeric variable.
                    data_num_init = numpy.arange(data_size, dtype=self.dtype)
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
                        idx_num = rng.randint(0, data_shape[0], n_to_inc)
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
                        inc_size = numpy.product(inc_shape, dtype='int')
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
                        if False:  # Enable for debugging purpose.
                            f = self.function([data_var, idx_var, inc_var],
                                              output, accept_inplace=inplace,
                                              op=self.adv_incsub1)
                            if inplace:
                                # Ensure calling `f` will not alter `data_num`.
                                data_num = data_num.copy()
                            f_out = f(data_num.copy(), idx_num, inc_num)
                            assert numpy.allclose(f_out, data_copy)
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
        for f_out, output_num in izip(f_outs, all_outputs_num):
            # NB: if this assert fails, it will probably be easier to debug if
            # you enable the debug code above.
            assert numpy.allclose(f_out, output_num)

    def test_adv_constant_arg(self):
        # Test case provided (and bug detected, gh-607) by John Salvatier
        m = matrix('m')
        gv = numpy.array([0, 1, 3])
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
        f = theano.function([m, i], [m1, m2])

        m_val = rand(3, 5)
        i_val = randint_ranged(min=0, max=4, shape=(4,))
        m1_ref = m_val.copy()
        m2_ref = m_val.copy()

        m1_val, m2_val = f(m_val, i_val)
        for idx in i_val:
            m1_ref[:, idx] = 0
            m2_ref[:, idx] += 1

        assert numpy.allclose(m1_val, m1_ref), (m1_val, m1_ref)
        assert numpy.allclose(m2_val, m2_ref), (m2_val, m2_ref)

    def test_adv1_inc_sub_notlastdim_2didx(self):
        # Test that taking 1-dimensional advanced indexing
        # over a dimension that's not the first (outer-most) works,
        # if the index is a matrix.
        m = matrix('m')
        i = lmatrix('i')

        m1 = set_subtensor(m[:, i], 0)
        m2 = inc_subtensor(m[:, i], 1)

        f = theano.function([m, i], [m1, m2])

        m_val = rand(5, 7)
        i_val = randint_ranged(min=0, max=6, shape=(4, 2))
        m1_ref = m_val.copy()
        m2_ref = m_val.copy()

        m1_val, m2_val = f(m_val, i_val)
        for idx in i_val.ravel():
            m1_ref[:, idx] = 0
            m2_ref[:, idx] += 1

        assert numpy.allclose(m1_val, m1_ref), (m1_val, m1_ref)
        assert numpy.allclose(m2_val, m2_ref), (m2_val, m2_ref)

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
                m1 = set_subtensor(sub_m, numpy.zeros(shp_v))
                m2 = inc_subtensor(sub_m, numpy.ones(shp_v))
                f = theano.function([m, i], [m1, m2])

                m_val = rand(3, 5)
                i_val = randint_ranged(min=0, max=4, shape=shp_i)
                m1_ref = m_val.copy()
                m2_ref = m_val.copy()

                m1_val, m2_val = f(m_val, i_val)
                for idx in i_val.ravel():
                    m1_ref[:, idx] = 0
                    m2_ref[:, idx] += 1

                assert numpy.allclose(m1_val, m1_ref), (m1_val, m1_ref)
                assert numpy.allclose(m2_val, m2_ref), (m2_val, m2_ref)
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
                m1 = set_subtensor(sub_m, numpy.zeros(shp_v))
                m2 = inc_subtensor(sub_m, numpy.ones(shp_v))
                f = theano.function([m, i], [m1, m2])

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

                assert numpy.allclose(m1_val, m1_ref), (m1_val, m1_ref)
                assert numpy.allclose(m2_val, m2_ref), (m2_val, m2_ref)
        finally:
            config.warn.inc_set_subtensor1 = orig_warn


class TestIncSubtensor1(unittest.TestCase):
    # test inc_subtensor
    # also tests set_subtensor

    def setUp(self):
        self.s = tensor.iscalar()
        self.v = tensor.fvector()
        self.m = tensor.dmatrix()
        self.t = tensor.ctensor3()

        self.adv1q = tensor.lvector()  # advanced 1d query

    def test_cant_adv_idx_into_scalar(self):
        self.assertRaises(TypeError, lambda: self.s[self.adv1q])

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
        assert numpy.allclose(aval, [.4, 0.9, 0.1])

    def test_1d_inc_adv_selection(self):
        a = inc_subtensor(self.v[self.adv1q], self.v[self.adv1q])

        assert a.type == self.v.type
        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1, 2])
        assert numpy.allclose(aval, [.4, 1.8, 0.2])

    def test_1d_inc_adv_selection_w_broadcasting(self):
        a = inc_subtensor(self.v[self.adv1q], 3.0)

        assert a.type == self.v.type
        f = theano.function([self.v, self.adv1q], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [1, 2])
        assert numpy.allclose(aval, [.4, 3.9, 3.1])

    def test_assigning_matrix_to_vector_selection(self):
        self.assertRaises(TypeError,
                          lambda: inc_subtensor(self.v[self.adv1q], fmatrix()))


inplace_increment_missing = SkipTest(
    "inc_subtensor with advanced indexing not enabled. "
    "Installing NumPy 1.8 or the latest development version "
    "should make that feature available.")


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

        self.ix1 = lvector()  # advanced 1d query
        self.ix12 = lvector()
        self.ix2 = lmatrix()
        self.ixr = lrow()

    def eval_output_and_check(self, t):
        f = inplace_func([], t, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        topo_ = [node for node in topo if not isinstance(node.op,
             self.ignore_topo)]
        assert len(topo_) == 1
        assert isinstance(topo_[0].op, self.sub)
        tval = f()
        return tval

    def test_cant_adv_idx_into_scalar(self):
        self.assertRaises(TypeError, lambda: self.s[self.ix1])

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
                          (rand(4, 4, 2, 3), [3,
                               3, 1, 1, 2, 2, 0, 0, -1, -2, -3, -4]),
                          # Test with TensorConstant index.
                          (rand(2, 4, 3),
                           theano.tensor.constant([3, 3, 1, 1, 2, 2, 0, 0])),
                          ]:
            data = numpy.asarray(data, dtype=self.dtype)
            n = self.shared(data)
            t = n[0, idx]

            self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor))

            val = self.eval_output_and_check(t)
            if isinstance(idx, list):
                good = data[0, idx]
            else:
                good = data[0, idx.data]
            self.assertTrue(val.ndim == data.ndim - 1)
            self.assertTrue(numpy.allclose(val, good), (val, good))

    def test_inc_adv_subtensor_w_matrix(self):
        subt = self.v[self.ix2]
        a = inc_subtensor(subt, subt)

        assert a.type == self.v.type, (a.type, self.v.type)
        f = theano.function([self.v, self.ix2], a, allow_input_downcast=True)
        aval = f([.4, .9, .1], [[1, 2],
                                [1, 2]])
        assert numpy.allclose(aval, [.4, .9 * 3, .1 * 3])

    def test_inc_adv_subtensor_w_2vec(self):
        if inplace_increment is None:
            raise inplace_increment_missing

        subt = self.m[self.ix1, self.ix12]
        a = inc_subtensor(subt, subt)

        typ = tensor.TensorType(self.m.type.dtype, self.ix2.type.broadcastable)
        assert a.type == typ, (a.type, typ)
        f = theano.function([self.m, self.ix1, self.ix12], a,
                            allow_input_downcast=True)
        aval = f([[.4, .9, .1],
                  [5, 6, 7],
                  [.5, .3, .15]],
                 [1, 2, 1],
                 [0, 1, 0])
        assert numpy.allclose(aval,
                [[.4, .9, .1],
                  [5 * 3, 6, 7],
                  [.5, .3 * 2, .15]]), aval

    def test_inc_adv_subtensor_with_broadcasting(self):
        if inplace_increment is None:
            raise inplace_increment_missing

        inc = dscalar()
        a = inc_subtensor(self.m[self.ix1, self.ix12], inc)
        g_inc = tensor.grad(a.sum(), inc)

        assert a.type == self.m.type, (a.type, self.m.type)
        f = theano.function([self.m, self.ix1, self.ix12, inc], [a, g_inc],
                            allow_input_downcast=True)
        aval, gval = f([[.4, .9, .1],
                        [5, 6, 7],
                        [.5, .3, .15]],
                       [1, 2, 1],
                       [0, 1, 0],
                       2.1)
        assert numpy.allclose(aval,
                [[.4, .9, .1],
                  [5 + 2.1 * 2, 6, 7],
                  [.5, .3 + 2.1, .15]]), aval
        assert numpy.allclose(gval, 3.0), gval

    def test_inc_adv_subtensor1_with_broadcasting(self):
        if inplace_increment is None:
            raise inplace_increment_missing

        inc = dscalar()
        a = inc_subtensor(self.m[self.ix1], inc)
        g_inc = tensor.grad(a.sum(), inc)

        assert a.type == self.m.type, (a.type, self.m.type)
        f = theano.function([self.m, self.ix1, inc], [a, g_inc],
                            allow_input_downcast=True)
        aval, gval = f([[.4, .9, .1],
                        [5, 6, 7],
                        [.5, .3, .15]],
                       [0, 1, 0],
                       2.1)
        assert numpy.allclose(aval,
                [[.4 + 2.1 * 2, .9  + 2.1 * 2, .1 + 2.1 * 2],
                  [5 + 2.1, 6 + 2.1, 7 + 2.1],
                  [.5, .3, .15]]), aval
        assert numpy.allclose(gval, 9.0), gval

    def test_inc_adv_subtensor_with_index_broadcasting(self):
        if inplace_increment is None:
            raise inplace_increment_missing

        a = inc_subtensor(self.m[self.ix1, self.ix2], 2.1)

        assert a.type == self.m.type, (a.type, self.m.type)
        f = theano.function([self.m, self.ix1, self.ix2], a,
                            allow_input_downcast=True)
        aval = f([[.4, .9, .1],
                  [5, 6, 7],
                  [.5, .3, .15]],
                 [0, 2, 0],
                 [[0, 1, 0],
                  [2, 2, 2]])
        assert numpy.allclose(aval,
                [[.4 + 2 * 2.1, .9, .1 + 2 * 2.1],
                  [5, 6, 7],
                  [.5, .3 + 2.1, .15 + 2.1]]), aval

    def test_advanced_indexing(self):
        # tests advanced indexing in Theano for 2D and 3D tensors
        rng = numpy.random.RandomState(utt.seed_rng())
        a = rng.uniform(size=(3, 3))
        b = theano.shared(a)
        i = tensor.iscalar()
        j = tensor.iscalar()
        z = b[[i, j], :]
        f1 = theano.function([i, j], z)
        cmd = f1(0, 1) == a[[0, 1], :]
        self.assertTrue(cmd.all())

        aa = rng.uniform(size=(4, 2, 3))
        bb = theano.shared(aa)
        k = tensor.iscalar()
        z = bb[[i, j, k], :, i:k]
        f2 = theano.function([i, j, k], z)
        cmd = f2(0, 1, 2) == aa[[0, 1, 2], :, 0:2]
        self.assertTrue(cmd.all())

    def test_grad(self):
        ones = numpy.ones((1, 3), dtype=self.dtype)
        n = self.shared(ones * 5, broadcastable=(True, False))
        idx = tensor.lvector()
        idx2 = tensor.lvector()
        t = n[idx, idx2]
        self.assertTrue(isinstance(t.owner.op, tensor.AdvancedSubtensor))

        utt.verify_grad(lambda m: m[[1, 3], [2, 4]],
                        [numpy.random.rand(5, 5).astype(self.dtype)])

        def fun(x, y):
            return advanced_inc_subtensor(x, y, [1, 3], [2, 4])
        utt.verify_grad(fun, [numpy.random.rand(5, 5).astype(self.dtype),
                              numpy.random.rand(2).astype(self.dtype)])

        def fun(x, y):
            return advanced_set_subtensor(x, y, [1, 3], [2, 4])
        utt.verify_grad(fun, [numpy.random.rand(5, 5).astype(self.dtype),
                              numpy.random.rand(2).astype(self.dtype)])


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
