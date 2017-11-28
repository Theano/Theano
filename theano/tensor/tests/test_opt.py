from __future__ import absolute_import, print_function, division

import copy
import logging
import time
import unittest

import numpy as np
from six.moves import xrange
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises, assert_true

import theano
import theano.scalar as scal
from six import StringIO
from theano import compile
from theano.compile import deep_copy_op, DeepCopyOp
from theano.compile import get_mode
from theano import config
from theano import function
from theano import gof
from theano import pprint
from theano import shared
from theano.gof import FunctionGraph
import theano.tensor.opt as opt
from theano.tensor.opt import (
    local_add_specialize,
    local_dimshuffle_lift,
    local_useless_dimshuffle_in_reshape,
    local_useless_alloc,
    local_merge_alloc,
    local_greedy_distributor,
    local_useless_reshape,
    local_reshape_to_dimshuffle,
    mul_canonizer,
    Shape_i,
    Assert,
    MakeVector,
    make_vector,
    local_canonicalize_alloc
    )
from theano import tensor
from theano import tensor as T
from theano.tensor import scalar, iscalar, lscalar, fscalar, dscalar
from theano.tensor import vector, lvector, fvector, dvector
from theano.tensor import matrix, fmatrix, dmatrix, tensor3
from theano.tensor import vectors, matrices, fmatrices, dmatrices
from theano.tensor import (
    AdvancedSubtensor,
    AdvancedSubtensor1,
    as_tensor_variable,
    IncSubtensor,
    AdvancedIncSubtensor,
    AdvancedIncSubtensor1,
    inplace,
    Join,
    join,
    Subtensor,
    TensorType,
    tile
    )
from theano.tensor.elemwise import DimShuffle
from theano.tensor.type import values_eq_approx_remove_nan
from theano.tests import unittest_tools as utt
from theano.gof.opt import check_stack_trace, out2in
from theano import change_flags
from nose.plugins.attrib import attr

mode_opt = theano.config.mode
if mode_opt == 'FAST_COMPILE':
    mode_opt = 'FAST_RUN'
mode_opt = theano.compile.mode.get_mode(mode_opt)

dimshuffle_lift = out2in(local_dimshuffle_lift)

_optimizer_stabilize = gof.Query(include=['fast_run'])
_optimizer_stabilize.position_cutoff = 1.51
_optimizer_stabilize = compile.optdb.query(_optimizer_stabilize)

_optimizer_specialize = gof.Query(include=['fast_run'])
_optimizer_specialize.position_cutoff = 2.01
_optimizer_specialize = compile.optdb.query(_optimizer_specialize)

_optimizer_fast_run = gof.Query(include=['fast_run'])
_optimizer_fast_run = compile.optdb.query(_optimizer_fast_run)


def ds(x, y):
    return DimShuffle(x.type.broadcastable, y)(x)


def optimize(g, level='fast_run'):
    if level == 'fast_run':
        _optimizer_fast_run.optimize(g)
    elif level == 'specialize':
        _optimizer_specialize.optimize(g)
    elif level == 'stabilize':
        _optimizer_stabilize.optimize(g)
    else:
        raise ValueError(level)
    return g


def inputs(xbc=(0, 0), ybc=(0, 0), zbc=(0, 0)):
    x = TensorType(broadcastable=xbc, dtype='float64')('x')
    y = TensorType(broadcastable=ybc, dtype='float64')('y')
    z = TensorType(broadcastable=zbc, dtype='float64')('z')
    return x, y, z


class test_dimshuffle_lift(unittest.TestCase):
    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = FunctionGraph([x], [e])
        self.assertTrue(str(g) == "[InplaceDimShuffle{1,0}(InplaceDimShuffle{1,0}(x))]")
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[x]")
        # no need to check_stack_trace as graph is supposed to be empty

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 'x', 0)), (2, 0, 'x', 1))
        g = FunctionGraph([x], [e])
        self.assertTrue(str(g) == "[InplaceDimShuffle{2,0,x,1}(InplaceDimShuffle{1,x,0}(x))]",
                        str(g))
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[InplaceDimShuffle{0,1,x,x}(x)]", str(g))
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(g, ops_to_check='all'))

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, 'x', 1)), (2, 0, 'x', 1)), (1, 0))
        g = FunctionGraph([x], [e])
        self.assertTrue(str(g) == ("[InplaceDimShuffle{1,0}(InplaceDimShuffle{2,0,x,1}"
                                   "(InplaceDimShuffle{0,x,1}(x)))]"),
                        str(g))
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[x]", str(g))
        # no need to check_stack_trace as graph is supposed to be empty

    def test_lift(self):
        x, y, z = inputs([False] * 1, [False] * 2, [False] * 3)
        e = x + y + z
        g = FunctionGraph([x, y, z], [e])

        # It does not really matter if the DimShuffles are inplace
        # or not.
        init_str_g_inplace = (
            "[Elemwise{add,no_inplace}(InplaceDimShuffle{x,0,1}"
            "(Elemwise{add,no_inplace}(InplaceDimShuffle{x,0}(x), y)), z)]")
        init_str_g_noinplace = (
            "[Elemwise{add,no_inplace}(DimShuffle{x,0,1}"
            "(Elemwise{add,no_inplace}(DimShuffle{x,0}(x), y)), z)]")
        self.assertTrue(str(g) in (init_str_g_inplace, init_str_g_noinplace),
                        str(g))

        opt_str_g_inplace = (
            "[Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle{x,0,1}(y)), z)]")
        opt_str_g_noinplace = (
            "[Elemwise{add,no_inplace}(Elemwise{add,no_inplace}"
            "(DimShuffle{x,x,0}(x), DimShuffle{x,0,1}(y)), z)]")
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) in (opt_str_g_inplace, opt_str_g_noinplace),
                        str(g))
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(g, ops_to_check='all'))

    def test_recursive_lift(self):
        v = T.vector(dtype="float64")
        m = T.matrix(dtype="float64")
        out = ((v + 42) * (m + 84)).T
        g = FunctionGraph([v, m], [out])
        init_str_g = ("[InplaceDimShuffle{1,0}(Elemwise{mul,no_inplace}"
                      "(InplaceDimShuffle{x,0}(Elemwise{add,no_inplace}"
                      "(<TensorType(float64, vector)>, "
                      "InplaceDimShuffle{x}(TensorConstant{42}))), "
                      "Elemwise{add,no_inplace}"
                      "(<TensorType(float64, matrix)>, "
                      "InplaceDimShuffle{x,x}(TensorConstant{84}))))]")
        self.assertTrue(str(g) == init_str_g)
        new_out = local_dimshuffle_lift.transform(g.outputs[0].owner)[0]
        new_g = FunctionGraph(g.inputs, [new_out])
        opt_str_g = ("[Elemwise{mul,no_inplace}(Elemwise{add,no_inplace}"
                     "(InplaceDimShuffle{0,x}(<TensorType(float64, vector)>), "
                     "InplaceDimShuffle{x,x}(TensorConstant{42})), "
                     "Elemwise{add,no_inplace}(InplaceDimShuffle{1,0}"
                     "(<TensorType(float64, matrix)>), "
                     "InplaceDimShuffle{x,x}(TensorConstant{84})))]")
        self.assertTrue(str(new_g) == opt_str_g)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(new_g, ops_to_check='all'))

    def test_useless_dimshuffle(self):
        x, _, _ = inputs()
        e = ds(x, (0, 1))
        g = FunctionGraph([x], [e])
        self.assertTrue(str(g) == "[InplaceDimShuffle{0,1}(x)]")
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[x]")
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(hasattr(g.outputs[0].tag, 'trace'))

    def test_dimshuffle_on_broadcastable(self):
        x, y, z = inputs([False, True], [True, False, True], [False, False, True])
        u = tensor.constant(1)
        ds_x = ds(x, (0, 'x'))   # useless
        ds_y = ds(y, (2, 1, 0))  # useless
        ds_z = ds(z, (2, 1, 0))  # useful
        ds_u = ds(u, ('x'))  # useful
        g = FunctionGraph([x, y, z, u], [ds_x, ds_y, ds_z, ds_u])
        self.assertTrue(str(g) == "[InplaceDimShuffle{0,x}(x), InplaceDimShuffle{2,1,0}(y), InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1})]")
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[x, y, InplaceDimShuffle{2,1,0}(z), InplaceDimShuffle{x}(TensorConstant{1})]")
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(hasattr(g.outputs[0].tag, 'trace'))


def test_local_useless_dimshuffle_in_reshape():
    vector = TensorType(broadcastable=(False,), dtype='float64')('vector')
    mat = TensorType(broadcastable=(False, False), dtype='float64')('mat')
    row = TensorType(broadcastable=(True, False), dtype='float64')('row')
    col = TensorType(broadcastable=(False, True), dtype='float64')('col')

    reshape_dimshuffle_vector = tensor.reshape(vector.dimshuffle('x', 0), vector.shape)
    reshape_dimshuffle_mat = tensor.reshape(mat.dimshuffle('x', 0, 'x', 1), mat.shape)
    reshape_dimshuffle_row = tensor.reshape(row.dimshuffle(1, 'x'), row.shape)
    reshape_dimshuffle_col = tensor.reshape(col.dimshuffle(0), col.shape)

    g = FunctionGraph([vector, mat, row, col],
                      [reshape_dimshuffle_vector, reshape_dimshuffle_mat,
                       reshape_dimshuffle_row, reshape_dimshuffle_col])

    print(str(g))
    assert_true(str(g) == "[Reshape{1}(InplaceDimShuffle{x,0}(vector), Shape(vector)), "
                          "Reshape{2}(InplaceDimShuffle{x,0,x,1}(mat), Shape(mat)), "
                          "Reshape{2}(InplaceDimShuffle{1,x}(row), Shape(row)), "
                          "Reshape{2}(InplaceDimShuffle{0}(col), Shape(col))]")
    useless_dimshuffle_in_reshape = out2in(local_useless_dimshuffle_in_reshape)
    useless_dimshuffle_in_reshape.optimize(g)
    assert_true(str(g) == "[Reshape{1}(vector, Shape(vector)), "
                          "Reshape{2}(mat, Shape(mat)), "
                          "Reshape{2}(row, Shape(row)), "
                          "Reshape{2}(col, Shape(col))]")

    # Check stacktrace was copied over correctly after opt was applied
    assert_true(check_stack_trace(g, ops_to_check='all'))

    # Check that the optimization does not get applied when the order
    # of dimensions has changed.
    reshape_dimshuffle_mat2 = tensor.reshape(mat.dimshuffle('x', 1, 'x', 0), mat.shape)
    h = FunctionGraph([mat], [reshape_dimshuffle_mat2])
    str_h = str(h)
    useless_dimshuffle_in_reshape.optimize(h)
    assert_true(str(h) == str_h)


def test_add_canonizer_problem0():
    n_segments = 10
    label = lscalar('label')
    segment_labels = label + theano._asarray([0] * n_segments, dtype='int64')

    r = segment_labels * 5
    f = function([label], r)
    f(3)


class test_greedy_distribute(unittest.TestCase):
    def test_main(self):
        a, b, c, d, x, y, z = matrices('abcdxyz')

        # 1. ((a/x + b/y) * x * y) --> a*y + b*x
        e = (a / z + b / x) * x * z
        g = FunctionGraph([a, b, c, d, x, y, z], [e])
        # print pprint(g.outputs[0])
        mul_canonizer.optimize(g)
        gof.TopoOptimizer(gof.LocalOptGroup(local_greedy_distributor),
                          order='out_to_in').optimize(g)
        # print pprint(g.outputs[0])
        assert str(pprint(g.outputs[0])) == "((a * x) + (b * z))"

        # 2. ((a/x + b) * x) --> a + b*x
        e = (a / x + b) * x
        g = FunctionGraph([a, b, x], [e])
        # print pprint(g.outputs[0])
        mul_canonizer.optimize(g)
        gof.TopoOptimizer(gof.LocalOptGroup(local_greedy_distributor),
                          order='out_to_in').optimize(g)
        # print pprint(g.outputs[0])
        assert str(pprint(g.outputs[0])) == "(a + (b * x))"

    def test_kording_bug(self):
        x, y = vectors('xy')
        eps = scalar('eps')
        s = scalar('s')

        # r = theano.tensor.mul(theano.tensor.fill(x, 2.*a), x/a , (y+z) , a)
        # r = theano.tensor.mul((x/a+y) , a, z)
        r = tensor.mul(s - 1,
                       eps + x / s,
                       eps + y / s,
                       s)

        f = function([s, eps, x, y], r ** 2)

        s_val = np.asarray(4, dtype=config.floatX)
        eps_val = np.asarray(1.e-6, dtype=config.floatX)
        x_val = np.asarray([1.5, 2], dtype=config.floatX)
        y_val = np.asarray([2.3, 3.1], dtype=config.floatX)

        r0 = f(s_val, eps_val, x_val, y_val)
        r1 = f(s_val, eps_val, x_val, y_val)
        r2 = f(s_val, eps_val, x_val, y_val)

        assert np.all(r0 == r1)
        assert np.all(r0 == r2)


class test_canonize(unittest.TestCase):
    def test_muldiv(self):
        x, y, z = matrices('xyz')
        a, b, c, d = matrices('abcd')
        # e = (2.0 * x) / (2.0 * y)
        # e = (2.0 * x) / (4.0 * y)
        # e = x / (y / z)
        # e = (x * y) / x
        # e = (x / y) * (y / z) * (z / x)
        # e = (a / b) * (b / c) * (c / d)
        # e = (a * b) / (b * c) / (c * d)
        # e = 2 * x / 2
        # e = x / y / x
        # e = (x / x) * (y / y)
        e = (-1 * x) / y / (-2 * z)
        g = FunctionGraph([x, y, z, a, b, c, d], [e])
        print(pprint(g.outputs[0]))
        mul_canonizer.optimize(g)
        print(pprint(g.outputs[0]))

    def test_elemwise_multiple_inputs_optimisation(self):
        # verify that the Canonizer merge sequential Elemwise({mul,add}) part 1
        #
        # This part are that case that is done, but don't include case
        # that are not implemented but are supposed to be.
        #
        # Test with and without DimShuffle

        shp = (5, 5)
        fx, fy, fz = fmatrices('xyz')
        dx, dy, dz = dmatrices('xyz')
        # fv = fvector('r').dimshuffle('x', 0)
        # dv = dvector('s').dimshuffle('x', 0)
        fxv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fyv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fzv = theano._asarray(np.random.rand(*shp), dtype='float32')
        # fvv = theano._asarray(np.random.rand(shp[0]), dtype='float32').reshape(1, shp[0])
        # dxv = theano._asarray(np.random.rand(*shp), dtype='float64')
        # dyv = theano._asarray(np.random.rand(*shp), dtype='float64')
        # dzv = theano._asarray(np.random.rand(*shp), dtype='float64')
        # dvv = theano._asarray(np.random.rand(shp[0]), dtype='float64').reshape(1, shp[0])
        cases = [
            (fx + fy, (fx, fy), (fxv, fyv), 1, 'float32'),
            (fx * fy, (fx, fy), (fxv, fyv), 1, 'float32'),
            # (fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            # (dx+dy+dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
            # (fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            # (dx*dy*dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
            # (fx*fy*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            # (dx*dy*(dx+dy+dz),(dx,dy,dz),(dxv,dyv,dzv),2,'float64'),
            # (fx*fy*(fx+fy+dz),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),  # check mixed type add
            # (dz*fy*(fx+fy),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),  # check mixed type mul
            # check with dimshuffle of constant
            (fx + fy + fz + 2, (fx, fy, fz), (fxv, fyv, fzv), 1,
                {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            (fx * fy * fz * 2, (fx, fy, fz), (fxv, fyv, fzv), 1,
                {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            # (2+fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            # (2*fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2 + fx + fy + fz + 2, (fx, fy, fz), (fxv, fyv, fzv), 1,
                {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            (2 * fx * fy * fz * 2, (fx, fy, fz), (fxv, fyv, fzv), 1,
                {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            # (fx*fy*2*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            # (fx*fy*(2+fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            (fx * fy * 2 * (fx + fy + fz + 2), (fx, fy, fz), (fxv, fyv, fzv), 2,
                {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),

            # check with broadcast of row
            # (fx+fy+fz+fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fx*fy*fz*fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fv+fx+fy+fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fv*fx*fy*fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            # (fx*fy*fv*(fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            # (fx*fy*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            # (fx*fy*fv*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            # (dx+dy+dz+dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dx*dy*dz*dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dv+dx+dy+dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dv*dx*dy*dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            # (dx*dy*dv*(dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            # (dx*dy*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            # (dx*dy*dv*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            ]  # [10:11]
        # print cases

        # We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode = compile.mode.get_default_mode()
        opt = gof.Query(["canonicalize"])
        opt = opt.excluding('local_elemwise_fusion')
        mode = mode.__class__(linker=mode.linker, optimizer=opt)
        for id, [g, sym_inputs, val_inputs,
                 nb_elemwise, out_dtype] in enumerate(cases):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = compile.function(list(sym_inputs), g,
                                 # we need the optimisation enabled, debug do this.
                                 mode=mode)

            out = f(*val_inputs)
            assert(len(f.maker.fgraph.toposort()) == nb_elemwise)
            assert(out_dtype == out.dtype)

    def test_elemwise_multiple_inputs_optimisation2(self):
        # verify that the Canonizer merge sequential Elemwise({mul,add}) part 2.
        # This part are that case that should have been done, but that are not implemented.
        # Test with and without DimShuffle

        raise SkipTest("Current implementation of Canonizer does not "
                       "implement all cases. Skip the corresponding test.")

        shp = (5, 5)
        fx, fy, fz = fmatrices('xyz')
        dx, dy, dz = dmatrices('xyz')
        fv = fvector('r').dimshuffle('x', 0)
        dv = dvector('s').dimshuffle('x', 0)
        fxv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fyv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fzv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fvv = theano._asarray(np.random.rand(shp[0]), dtype='float32').reshape(1, shp[0])
        dxv = theano._asarray(np.random.rand(*shp), dtype='float64')
        dyv = theano._asarray(np.random.rand(*shp), dtype='float64')
        dzv = theano._asarray(np.random.rand(*shp), dtype='float64')
        dvv = theano._asarray(np.random.rand(shp[0]), dtype='float64').reshape(1, shp[0])
        cases = [
            (fx + fy, (fx, fy), (fxv, fyv), 1, 'float32'),
            (fx * fy, (fx, fy), (fxv, fyv), 1, 'float32'),
            (fx + fy + fz, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (dx + dy + dz, (dx, dy, dz), (dxv, dyv, dzv), 1, 'float64'),
            (fx * fy * fz, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (dx * dy * dz, (dx, dy, dz), (dxv, dyv, dzv), 1, 'float64'),
            (fx * fy * (fx + fy + fz), (fx, fy, fz), (fxv, fyv, fzv), 2, 'float32'),
            (dx * dy * (dx + dy + dz), (dx, dy, dz), (dxv, dyv, dzv), 2, 'float64'),
            (fx * fy * (fx + fy + dz), (fx, fy, dz), (dxv, dyv, dzv), 2, 'float64'),  # check mixed type add
            (dz * fy * (fx + fy), (fx, fy, dz), (dxv, dyv, dzv), 2, 'float64'),  # check mixed type mul
            # check with dimshuffle of constant
            (fx + fy + fz + 2, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (fx * fy * fz * 2, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (2 + fx + fy + fz, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (2 * fx * fy * fz, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (2 + fx + fy + fz + 2, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (2 * fx * fy * fz * 2, (fx, fy, fz), (fxv, fyv, fzv), 1, 'float32'),
            (fx * fy * 2 * (fx + fy + fz), (fx, fy, fz), (fxv, fyv, fzv), 2, 'float32'),
            (fx * fy * (2 + fx + fy + fz), (fx, fy, fz), (fxv, fyv, fzv), 2, 'float32'),
            (fx * fy * 2 * (fx + fy + fz + 2), (fx, fy, fz), (fxv, fyv, fzv), 2, 'float32'),

            # check with broadcast of row
            (fx + fy + fz + fv, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, 'float32'),
            (fx * fy * fz * fv, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, 'float32'),
            (fv + fx + fy + fz, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, 'float32'),
            (fv * fx * fy * fz, (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 1, 'float32'),
            (fx * fy * fv * (fx + fy + fz), (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 2, 'float32'),
            (fx * fy * (fv + fx + fy + fz), (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 2, 'float32'),
            (fx * fy * fv * (fv + fx + fy + fz), (fx, fy, fz, fv), (fxv, fyv, fzv, fvv), 2, 'float32'),
            (dx + dy + dz + dv, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, 'float64'),
            (dx * dy * dz * dv, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, 'float64'),
            (dv + dx + dy + dz, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, 'float64'),
            (dv * dx * dy * dz, (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 1, 'float64'),
            (dx * dy * dv * (dx + dy + dz), (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 2, 'float64'),
            (dx * dy * (dv + dx + dy + dz), (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 2, 'float64'),
            (dx * dy * dv * (dv + dx + dy + dz), (dx, dy, dz, dv), (dxv, dyv, dzv, dvv), 2, 'float64'),
            ]  # [10:11]
        # print cases

        # We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode = compile.mode.get_default_mode()
        mode._optimizer = gof.Query(["canonicalize"])
        mode._optimizer = mode._optimizer.excluding('local_elemwise_fusion')
        for id, [g, sym_inputs, val_inputs, nb_elemwise, out_dtype] in enumerate(cases):
            f = compile.function(list(sym_inputs), g,
                                 # we need the optimisation enabled, debug do this.
                                 mode=mode)

            out = f(*val_inputs)
            assert(len(f.maker.fgraph.toposort()) == nb_elemwise)
            assert(out_dtype == out.dtype)

    @attr('slow')
    def test_multiple_case(self):
        # test those case take from the comment in Canonizer
        # x / x -> 1
        # (x * y) / x -> y
        # x / y / x -> 1 / y
        # x / y / z -> x / (y * z)
        # x / (y / z) -> (x * z) / y
        # (a / b) * (b / c) * (c / d) -> a / d
        # (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
        # 2 * x / 2 -> x
        # with and without DimShuffle
        # TODO: with DimShuffle

        shp = (3, 3)
        fx, fy, fz, fw = fmatrices('xyzw')
        dx, dy, dz, dw = dmatrices('xyzw')
        fv = fvector('r').dimshuffle('x', 0)
        dv = dvector('s').dimshuffle('x', 0)
        fxv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fyv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fzv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fwv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fvv = theano._asarray(np.random.rand(shp[0]), dtype='float32').reshape(1, shp[0])
        dxv = theano._asarray(np.random.rand(*shp), dtype='float64')
        dyv = theano._asarray(np.random.rand(*shp), dtype='float64')
        dzv = theano._asarray(np.random.rand(*shp), dtype='float64')
        dwv = theano._asarray(np.random.rand(*shp), dtype='float64')
        dvv = theano._asarray(np.random.rand(shp[0]), dtype='float64').reshape(1, shp[0])

        # We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode = compile.mode.get_default_mode()

        opt = gof.Query(["canonicalize"])
        opt = opt.including('ShapeOpt', 'local_fill_to_alloc')
        opt = opt.excluding(
            'local_elemwise_fusion')
        mode = mode.__class__(linker=mode.linker, optimizer=opt)
        # test x / x -> 1
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate([
                (fx / fx, [fx], [fxv], 'float32'),
                (dx / dx, [dx], [dxv], 'float64'),
                (fv / fv, [fv], [fvv], 'float32'),
                (dv / dv, [dv], [dvv], 'float64')]):
            f = compile.function(list(sym_inputs), g,
                                 mode=mode)
            out = f(*val_inputs)
            assert (out == np.ones(shp, dtype=out_dtype)).all()
            topo = f.maker.fgraph.toposort()
            if sym_inputs[0].broadcastable[0]:
                assert len(topo) == 2
                assert isinstance(topo[0].op, Shape_i)
                assert isinstance(topo[1].op, tensor.Alloc)
            else:
                assert len(topo) == 3
                assert isinstance(topo[0].op, Shape_i)
                assert isinstance(topo[1].op, Shape_i)
                assert isinstance(topo[2].op, tensor.Alloc)
            assert(out_dtype == out.dtype)

        # test (x * y) / x -> y
        for id, (g, sym_inputs, val_inputs, nb_elemwise, out_dtype) in enumerate([
                ((dx * dy) / dx, [dx, dy], [dxv, dyv], 0, 'float64'),
                ((fx * fy) / fx, [fx, fy], [fxv, fyv], 0, 'float32'),
                ((dv * dy) / dv, [dv, dy], [dvv, dyv], 0, 'float64'),
                ((fv * fy) / fv, [fv, fy], [fvv, fyv], 0, 'float32'),
                # must broadcast as there is a dimshuffle in the computation
                ((dx * dv) / dx, [dx, dv], [dxv, dvv], 1, 'float64'),
                # topo: [Elemwise{second,no_inplace}(x, <TensorType(float64, row)>)]
                ((fx * fv) / fx, [fx, fv], [fxv, fvv], 1, 'float32')
                # topo: [Elemwise{second,no_inplace}(x, <TensorType(float32, row)>)]
                ]):
            f = compile.function(list(sym_inputs), g,
                                 mode=mode)
            out = f(*val_inputs)
            assert(out_dtype == out.dtype)
            utt.assert_allclose(out, val_inputs[1])
            topo = f.maker.fgraph.toposort()
            if topo and not(len(topo) == 1 and topo[0].op == deep_copy_op):
                for node in topo[:-1]:
                    assert isinstance(node.op, Shape_i)
                assert isinstance(topo[-1].op, tensor.Alloc)

        # test x / y / x -> 1 / y
        for id, (g, sym_inputs, val_inputs, nb_elemwise, out_dtype) in enumerate([
                ((dx / dy) / dx, [dx, dy], [dxv, dyv], 1, 'float64'),
                ((fx / fy) / fx, [fx, fy], [fxv, fyv], 1, 'float32'),
                ((dv / dy) / dv, [dv, dy], [dvv, dyv], 1, 'float64'),
                ((fv / fy) / fv, [fv, fy], [fvv, fyv], 1, 'float32'),
                # must broadcast as their is a dimshuffle in the computation
                ((dx / dv) / dx, [dx, dv], [dxv, dvv], 1, 'float64'),
                # topo: [Shape_i, Shape_i, Elemwise{inv,no_inplace}(<TensorType(float64, row)>), Alloc]
                ((fx / fv) / fx, [fx, fv], [fxv, fvv], 1, 'float32'),
                # topo: [Shape_i, Shape_i, Elemwise{inv,no_inplace}(<TensorType(float32, row)>), Alloc]
                ]):
            f = compile.function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, (1 / val_inputs[1]))
            topo = f.maker.fgraph.toposort()
            elem = [t for t in topo if isinstance(t.op, T.Elemwise)]
            assert len(elem) == nb_elemwise
            assert isinstance(elem[0].op, (T.Elemwise, ))
            assert isinstance(elem[0].op.scalar_op, (
                theano.scalar.basic.Inv, theano.scalar.basic.TrueDiv))
            assert(out_dtype == out.dtype)

        # test (a / b) * (b / c) * (c / d) -> a / d
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate([
                ((dx / dy) * (dy / dz) * (dz / dw), [dx, dy, dz, dw], [dxv, dyv, dzv, dwv], 'float64'),
                ((fx / fy) * (fy / fz) * (fz / fw), [fx, fy, fz, fw], [fxv, fyv, fzv, fwv], 'float32'),
                ((dv / dy) * (dy / dz) * (dz / dw), [dv, dy, dz, dw], [dvv, dyv, dzv, dwv], 'float64'),
                ((fv / fy) * (fy / fz) * (fz / fw), [fv, fy, fz, fw], [fvv, fyv, fzv, fwv], 'float32'),
                ((dx / dv) * (dv / dz) * (dz / dw), [dx, dv, dz, dw], [dxv, dvv, dzv, dwv], 'float64'),
                ((fx / fv) * (fv / fz) * (fz / fw), [fx, fv, fz, fw], [fxv, fvv, fzv, fwv], 'float32'),
                ((dx / dy) * (dy / dv) * (dv / dw), [dx, dy, dv, dw], [dxv, dyv, dvv, dwv], 'float64'),
                ((fx / fy) * (fy / fv) * (fv / fw), [fx, fy, fv, fw], [fxv, fyv, fvv, fwv], 'float32'),
                ((dx / dy) * (dy / dz) * (dz / dv), [dx, dy, dz, dv], [dxv, dyv, dzv, dvv], 'float64'),
                ((fx / fy) * (fy / fz) * (fz / fv), [fx, fy, fz, fv], [fxv, fyv, fzv, fvv], 'float32'),
                ]):
            f = compile.function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, (val_inputs[0] / val_inputs[3]))
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, (T.Elemwise, ))
            assert isinstance(topo[0].op.scalar_op, theano.scalar.basic.TrueDiv)
            assert len(topo[0].inputs) == 2
            assert(out_dtype == out.dtype)

        # test (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate([
                (((2.0 * dx) / (4.0 * dy)), [dx, dy], [dxv, dyv], 'float64'),
                (((2.0 * fx) / (4.0 * fy)), [fx, fy], [fxv, fyv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                (((2.0 * dv) / (4.0 * dy)), [dv, dy], [dvv, dyv], 'float64'),
                (((2.0 * fv) / (4.0 * fy)), [fv, fy], [fvv, fyv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                (((2.0 * dx) / (4.0 * dv)), [dx, dv], [dxv, dvv], 'float64'),
                (((2.0 * fx) / (4.0 * fv)), [fx, fv], [fxv, fvv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ]):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = compile.function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, (0.5 * val_inputs[0] / val_inputs[1]))
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, (T.Elemwise, ))
            assert isinstance(topo[0].op.scalar_op, theano.scalar.basic.Mul)
            assert len(topo[0].inputs) == 2
            assert isinstance(topo[1].op, (T.Elemwise, ))
            assert isinstance(topo[1].op.scalar_op, theano.scalar.basic.TrueDiv)
            assert len(topo[1].inputs) == 2
            assert(out_dtype == out.dtype)

        # test 2 * x / 2 -> x
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate([
                ((2 * dx) / 2, [dx], [dxv], 'float64'),
                ((2 * fx) / 2, [fx], [fxv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ((2 * dv) / 2, [dv], [dvv], 'float64'),
                ((2 * fv) / 2, [fv], [fvv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ]):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = compile.function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, val_inputs[0])
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            topo[0].op == deep_copy_op
            assert(out_dtype == out.dtype)

        # test x / abs(x) -> sign(x)
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate([
                (dx / abs(dx), [dx], [0.5 - dxv], 'float64'),
                (fx / abs(fx), [fx], [0.5 - fxv], 'float32'),
                (dx / abs(dx), [dx], [0.1 * dxv], 'float64'),
                (fx / abs(fx), [fx], [0.1 * fxv], 'float32'),
                (dv / abs(dv), [dv], [0.5 - dvv], 'float64'),
                (fv / abs(fv), [fv], [0.5 - fvv], 'float32'),
                ]):
            f = compile.function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            assert np.all(np.isfinite(out))
            utt.assert_allclose(out, np.sign(val_inputs[0]))
            assert(out_dtype == out.dtype)
            assert len(f.maker.fgraph.toposort()) == 1

        # test (2*x) / (3*abs(x)) -> sign(x)
        for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate([
                ((2 * dx) / (3 * abs(dx)), [dx], [0.5 - dxv], 'float64'),
                ((2 * fx) / (3 * abs(fx)), [fx], [0.5 - fxv],
                    {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ((2 * dx) / (3 * abs(dx)), [dx], [0.1 * dxv], 'float64'),
                ((2 * fx) / (3 * abs(fx)), [fx], [0.1 * fxv],
                    {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ((2 * dv) / (3 * abs(dv)), [dv], [0.5 - dvv], 'float64'),
                ((2 * fv) / (3 * abs(fv)), [fv], [0.5 - fvv],
                    {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ]):

            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            f = compile.function(list(sym_inputs), g,
                                 mode=mode)
            topo = f.maker.fgraph.toposort()
            out = f(*val_inputs)
            assert np.all(np.isfinite(out))
            utt.assert_allclose(out, np.sign(val_inputs[0]) * 2 / 3)
            assert(out_dtype == out.dtype)

    def test_abs_mul_div(self):
        # test that if we have
        # 4 * x / abs(2*x) it get simplifier during canonicalisation.

        x = T.dscalar()
        # a = T.abs_(x)

        if theano.config.mode == 'FAST_COMPILE':
            mode = theano.compile.mode.get_mode('FAST_RUN').excluding(
                "local_elemwise_fusion")
        else:
            mode = theano.compile.mode.get_default_mode().excluding(
                "local_elemwise_fusion")

        f = theano.function([x], [(4 * x) / abs(2 * x)], mode=mode)
        print(f.maker.fgraph.toposort())
        print()
        f(.1)
        f(-1)
        # some stabilization optimization make the output be finite instead of nan
        # debug_mode will raise an error when he see nan
        if not isinstance(mode, theano.compile.debugmode.DebugMode):
            assert np.isfinite(f(0))

        assert len(f.maker.fgraph.toposort()) == 2
        assert f.maker.fgraph.toposort()[0].op == T.sgn

        f = theano.function([x], [(4 * x) / abs(x / 2)], mode=mode)
        print(f.maker.fgraph.toposort())
        print()
        f(.1)
        f(-1)
        # some stabilization optimization make the output be finite instead of nan
        # debug_mode will raise an error when he see nan
        if not isinstance(mode, theano.compile.debugmode.DebugMode):
            assert np.isfinite(f(0))

        assert len(f.maker.fgraph.toposort()) == 2
        assert f.maker.fgraph.toposort()[0].op == T.sgn

    def test_multiple_case_that_fail(self):
        raise SkipTest("Current implementation of Canonizer does not "
                       "implement all cases. Skip the corresponding test.")

        shp = (4, 4)
        fx, fy, fz = fmatrices('xyz')
        dx, dy, dz = dmatrices('xyz')
        fxv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fyv = theano._asarray(np.random.rand(*shp), dtype='float32')
        fzv = theano._asarray(np.random.rand(*shp), dtype='float32')
        dxv = theano._asarray(np.random.rand(*shp), dtype='float32')
        dyv = theano._asarray(np.random.rand(*shp), dtype='float32')
        dzv = theano._asarray(np.random.rand(*shp), dtype='float32')
        # fvv = theano._asarray(np.random.rand(shp[0]), dtype='float32').reshape(1, shp[0])
        # We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode = compile.mode.get_default_mode()

        opt = gof.Query(["canonicalize"])
        opt = opt.excluding('local_elemwise_fusion')
        mode = mode.__class__(linker=mode.linker, optimizer=opt)
        # test fail!
        # test x / y / z -> x / (y * z)
        for (g, sym_inputs, val_inputs, out_dtype) in [
                ((dx / dy) / dz, [dx, dy, dz], [dxv, dyv, dzv], 'float64'),
                ((fx / fy) / fz, [fx, fy, fz], [fxv, fyv, fzv], 'float32')
                ]:
            f = compile.function(list(sym_inputs), g, mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, val_inputs[0] / val_inputs[1] / val_inputs[2])
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, (T.Elemwise, ))
            assert isinstance(topo[0].op.scalar_op, theano.scalar.basic.Inv)
            assert len(topo[0].inputs) == 1
            assert(out_dtype == out.dtype)

        # test x / (y / z) -> (x * z) / y
        for (g, sym_inputs, val_inputs, out_dtype) in [
                (dx / (dy / dz), [dx, dy, dz], [dxv, dyv, dzv], 'float64'),
                (fx / (fy / fz), [fx, fy, fz], [fxv, fyv, fzv], 'float32')
                ]:
            f = compile.function(list(sym_inputs), g,
                                 mode=mode)
            out = f(*val_inputs)
            utt.assert_allclose(out, val_inputs[0] / (val_inputs[1] / val_inputs[2]))
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 2
            assert isinstance(topo[0].op, (T.Elemwise, ))
            assert isinstance(topo[0].op.scalar_op, theano.scalar.basic.Inv)
            assert len(topo[0].inputs) == 1
            assert(out_dtype == out.dtype)

    def test_dont_merge_if_multiple_client(self):
        # test those case take from the comment in Canonizer
        raise SkipTest("Not implemented")

    def test_canonicalize_nan(self):
        # Regression test for bug in canonicalization of NaN values.
        # This bug caused an infinite loop which was caught by the equilibrium
        # optimizer, resulting in an error log message.

        sio = StringIO()
        handler = logging.StreamHandler(sio)
        handler.setLevel(logging.ERROR)
        logging.getLogger('theano.gof.opt').addHandler(handler)
        try:
            x = vector()
            theano.function([x], x + np.nan)
        finally:
            logging.getLogger('theano.gof.opt').removeHandler(handler)
        # Ideally this test would only catch the maxed out equilibrium
        # optimizer error message, but to be safe in case this message
        # is modified in the future, we assert that there is no error
        # at all.
        assert not sio.getvalue()


def test_local_merge_abs():
    x, y, z = T.matrices('xyz')
    x_val = np.random.rand(5, 5).astype(config.floatX)
    y_val = np.random.rand(5, 5).astype(config.floatX)
    z_val = np.random.rand(5, 5).astype(config.floatX)
    mode = theano.config.mode
    if mode == "FAST_COMPILE":
        mode = "FAST_RUN"
    mode = theano.compile.mode.get_mode(mode).excluding(
        "local_elemwise_fusion")

    f = theano.function([y, z], (abs(y * z * -2)), mode=mode)
    f(y_val, z_val)
    assert isinstance(f.maker.fgraph.toposort()[1].op.scalar_op, scal.Abs)
    assert len(f.maker.fgraph.toposort()) == 2

    f = theano.function([x, y], abs(x / y), mode=mode)
    f(x_val, y_val)
    assert isinstance(f.maker.fgraph.toposort()[1].op.scalar_op, scal.Abs)
    assert len(f.maker.fgraph.toposort()) == 2


def test_merge_abs_bugfix():
    # Test crash in optimization reported by Jeremiah Lowin at
    # https://groups.google.com/d/topic/theano-users/TaXfqXP2Mj0/discussion
    input = T.matrix()
    # normalize on cols
    step1 = input / input.sum(0)
    # normalize on rows
    step2 = step1 / step1.sum(1)
    # get l1 norm
    l1_norm = T.abs_(step2).sum()
    theano.function([input], T.grad(l1_norm, input))


def test_mixeddiv():
    # Test that int division is preserved
    i = iscalar()
    d = dscalar()
    assert 0 == function([i, d], d * (i // (i + 1)))(3, 1.0)


def test_const_type_in_mul_canonizer():
    input = dmatrix()
    w = dmatrix()
    visb = dvector()
    hidb = dvector()
    betas = dvector()
    a = dvector()

    def sigm(x):
        return 1. / (1 + tensor.exp(-x))

    hid = sigm((tensor.dot(w, input) + hidb) * betas)

    vis_gauss1 = (tensor.dot(w.T, hid) + visb) * betas / (2 * a * a)
    vis_gauss2 = (tensor.dot(w.T, hid) + visb) * betas / (2. * a * a)

    f1 = function([input, w, visb, hidb, betas, a], vis_gauss1)
    f2 = function([input, w, visb, hidb, betas, a], vis_gauss2)

    ival = np.random.rand(5, 5)
    wval = np.random.rand(5, 5)
    visbval = np.random.rand(5)
    hidbval = np.random.rand(5)
    betaval = np.random.rand(5)
    aval = np.random.rand(5)

    utt.assert_allclose(
        f2(ival, wval, visbval, hidbval, betaval, aval),
        f1(ival, wval, visbval, hidbval, betaval, aval))


def test_cast_in_mul_canonizer():
    x, y = tensor.vectors('xy')
    m = tensor.minimum(x, y)
    o = m.sum()
    go = tensor.fill(o, 1)
    e = tensor.eq(go, x)
    o1 = (1 - e) * go
    o2 = e * go
    mode = theano.compile.get_default_mode().excluding('fusion').including('fast_run')
    f = theano.function([x, y], [o1, o2], mode=mode)
    theano.printing.debugprint(f, print_type=True)
    nodes = f.maker.fgraph.apply_nodes
    assert len([n for n in nodes if isinstance(getattr(n.op, 'scalar_op', None),
                                               theano.scalar.Identity)]) == 0
    assert len([n for n in nodes if isinstance(getattr(n.op, 'scalar_op'),
                                               theano.scalar.Cast)]) == 1
    f([1], [1])


class test_fusion(unittest.TestCase):
    mode = copy.copy(compile.mode.get_default_mode())
    _shared = staticmethod(shared)
    topo_exclude = ()

    def do(self, mode, shared_fn, shp, nb_repeat=1, assert_len_topo=True, slice=None):
        """
        param shared_fn: if None, will use compile.function
        verify that the elemwise fusion work
        Test with and without DimShuffle
        """
        # TODO: disable the canonizer?
        def my_init(shp, dtype='float64', num=0):
            ret = np.zeros(shp, dtype=dtype) + num
            return ret
        fw, fx, fy, fz = [theano.tensor.tensor(dtype='float32',
                                               broadcastable=[False] * len(shp),
                                               name=n) for n in 'wxyz']
        dw, dx, dy, dz = [theano.tensor.tensor(dtype='float64',
                                               broadcastable=[False] * len(shp),
                                               name=n) for n in 'wxyz']
        ix, iy, iz = [theano.tensor.tensor(dtype='int32',
                                           broadcastable=[False] * len(shp),
                                           name=n) for n in 'xyz']
        fv = fvector('v')
        fs = fscalar('s')

        fwv = my_init(shp, 'float32', 1)
        fxv = my_init(shp, 'float32', 2)
        fyv = my_init(shp, 'float32', 3)
        fzv = my_init(shp, 'float32', 4)
        fvv = theano._asarray(np.random.rand(shp[0]), dtype='float32')
        fsv = np.asarray(np.random.rand(), dtype='float32')
        dwv = my_init(shp, 'float64', 5)
        ixv = theano._asarray(my_init(shp, num=60), dtype='int32')
        iyv = theano._asarray(my_init(shp, num=70), dtype='int32')
        izv = theano._asarray(my_init(shp, num=70), dtype='int32')
        fwx = fw + fx
        ftanx = theano.tensor.tan(fx)
        cases = [
            (fx + fy + fz, (fx, fy, fz), (fxv, fyv, fzv), 1, fxv +
                fyv + fzv, 'float32'),  # 0
            (fx * fy * fz, (fx, fy, fz), (fxv, fyv, fzv), 1, fxv *
                fyv * fzv, 'float32'),  # 1
            (fx + fy * fz, (fx, fy, fz), (fxv, fyv, fzv), 1, fxv +
                fyv * fzv, 'float32'),  # 2
            (fx * fy + fz, (fx, fy, fz), (fxv, fyv, fzv), 1, fxv *
                fyv + fzv, 'float32'),  # 3
            (fw + fx + fy + fz, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv + fyv + fzv, 'float32'),
            ((fw + fx) + (fy + fz), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv + fyv + fzv, 'float32'),  # 5
            (((fw + fx) + fy) + fz, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv + fyv + fzv, 'float32'),
            ((fw + (fx + fy)) + fz, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv + fyv + fzv, 'float32'),
            ((fw + (fx + fy) + fz), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv + fyv + fzv, 'float32'),
            (fw + (fx + (fy + fz)), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv + fyv + fzv, 'float32'),
            ((fw + fx) + (fy + fz), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv + fyv + fzv, 'float32'),  # 10
            (fw * fx * fy * fz, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv * fxv * fyv * fzv, 'float32'),
            (fw + fx * fy * fz, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fwv + fxv * fyv * fzv, 'float32'),
            (fx + fy * fz * fx, (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv + fyv * fzv * fxv, 'float32'),
            (fx * fy + fz + fy, (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv * fyv + fzv + fyv, 'float32'),
            (fx * fy * fz * fw + fx + fy + fz + fw, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv), 1,
                fxv * fyv * fzv * fwv + fxv + fyv + fzv + fwv, 'float32'),  # 15
            # test with constant
            ((fw + fx) + (fy + fz) + 2., (fw, fx, fy, fz), (fwv, fxv, fyv, fzv),
                1, fwv + fxv + fyv + fzv + 2, 'float32'),
            (((fw + fx) + 2. + fy) + fz, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv),
                1, fwv + fxv + fyv + fzv + 2, 'float32'),
            ((fw + (fx + 2. + fy)) + fz, (fw, fx, fy, fz), (fwv, fxv, fyv, fzv),
                1, fwv + fxv + fyv + fzv + 2, 'float32'),
            ((fw + (fx + fy) + 2 + fz), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv),
                1, fwv + fxv + fyv + fzv + 2, 'float32'),
            (fw + (fx + (fy + fz) + 2.), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv),
                1, fwv + fxv + fyv + fzv + 2, 'float32'),  # 20
            (2 + (fw + fx) + (fy + fz), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv),
                1, fwv + fxv + fyv + fzv + 2, 'float32'),
            # mix float32 and float64
            (2 + (dw + fx) + (fy + fz), (dw, fx, fy, fz), (dwv, fxv, fyv, fzv),
                1, dwv + fxv + fyv + fzv + 2, 'float64'),
            (2 + (fw + dw) + (fy + fz), (fw, dw, fy, fz), (fwv, dwv, fyv, fzv),
                1, fwv + dwv + fyv + fzv + 2, 'float64'),
            (2 + (fw + fx) + (dw + fz), (fw, fx, dw, fz), (fwv, fxv, dwv, fzv),
                1, fwv + fxv + dwv + fzv + 2, 'float64'),
            (2 + (fw + fx) + (fy + dw), (fw, fx, fy, dw), (fwv, fxv, fyv, dwv),
                1, fwv + fxv + fyv + dwv + 2, 'float64'),  # 25
            # test when their is other op then elemwise.
            ((fwx.sum()) + (fwx) + (fy + fz), (fw, fx, fy, fz), (fwv, fxv, fyv, fzv),
                4, (fwv + fxv).sum() + fwv + fxv + fyv + fzv, 'float32'),
            # test other elemwise op
            (fx + fy + tensor.cos(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv + fyv + np.cos(fzv), 'float32'),
            (fx + fy + tensor.cosh(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv + fyv + np.cosh(fzv), 'float32'),
            (fx + fy + abs(fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv + fyv +
                np.absolute(fzv), 'float32'),
            (ix + iy + abs(iz), (ix, iy, iz), (ixv, iyv, izv), 1, ixv + iyv +
                np.absolute(izv), 'int32'),  # 30
            (fx + fy + theano.tensor.log(fz), (fx, fy, fz), (fxv, fyv, fzv),
                1, fxv + fyv + np.log(fzv), 'float32'),
            (fx + fy + theano.tensor.log2(fz), (fx, fy, fz), (fxv, fyv, fzv),
                1, fxv + fyv + np.log2(fzv), 'float32'),
            (fx + fy + theano.tensor.log10(fz), (fx, fy, fz), (fxv, fyv, fzv),
                1, fxv + fyv + np.log10(fzv), 'float32'),
            (fx + fy ** fz, (fx, fy, fz), (fxv, fyv, fzv), 1, fxv + fyv ** fzv,
                'float32'),  # pow
            (fx + fy + theano.tensor.exp(fz), (fx, fy, fz), (fxv, fyv, fzv),
                1, fxv + fyv + np.exp(fzv), 'float32'),  # 35
            (fx - fy - fz, (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - fyv - fzv, 'float32'),
            (fx - (fy / fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - (fyv / fzv), 'float32'),
            (fx - theano.tensor.true_div(fy, 2), (fx, fy), (fxv, fyv),
                1, fxv - (fyv / 2), 'float32'),
            (fx - theano.tensor.true_div(fy, fz), (fx, fy, fz), (fxv, fyv, fzv),
                1, fxv - (fyv / fzv), 'float32'),
            (fx - theano.tensor.int_div(ix * 100, iy * 1000), (fx, ix, iy), (fxv, ixv, iyv),
                1, fxv - ((ixv * 100) // (iyv * 1000)), {'custom': 'float64', 'numpy + floatX': config.floatX, 'numpy': 'float64'}),  # 40
            (fx - (fy / 2), (fx, fy), (fxv, fyv), 1, fxv - (fyv / 2), 'float32'),
            (fx - (fy % fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - (fyv % fzv), 'float32'),
            (fx - (fy > fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - (fyv > fzv), 'float32'),
            (fx - (fy >= fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - (fyv >= fzv), 'float32'),
            (fx - (fy < fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - (fyv < fzv), 'float32'),  # 45
            (fx - (fy <= fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - (fyv <= fzv), 'float32'),
            (fx - T.eq(fy, fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - (fyv == fzv), 'float32'),
            (fx - T.neq(fy, fz), (fx, fy, fz), (fxv, fyv, fzv), 1, fxv - (
                fyv != fzv), 'float32'),
            (fx - fy + tensor.tan(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + np.tan(fzv), 'float32'),
            (fx - fy + tensor.tanh(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + np.tanh(fzv), 'float32'),  # 50
            (fx - fy + tensor.sin(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + np.sin(fzv), 'float32'),
            (fx - fy + tensor.sinh(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + np.sinh(fzv), 'float32'),
            (fx - fy + theano.tensor.sqr(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + (fzv * fzv), 'float32'),
            (fx - fy + theano.tensor.sqrt(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + np.sqrt(fzv), 'float32'),
            (fx - fy + theano.tensor.inv(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + (1 / fzv), 'float32'),  # 55
            (fx - fy + theano.tensor.neg(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + (-fzv), 'float32'),
            (fx - fy + theano.tensor.round(fz), (fx, fy, fz), (fxv, fyv, fzv), 1,
                fxv - fyv + np.round(fzv), 'float32'),
            (ix - iy + theano.tensor.iround(fz), (ix, iy, fz), (ixv, iyv, fzv), 1,
                ixv - iyv + np.round(fzv), 'int64'),
            # Bit op
            (fx - theano.tensor.or_(iy, iz), (fx, iy, iz), (fxv, iyv, izv), 1,
                fxv - (iyv | izv), {'custom': 'float64', 'numpy + floatX': config.floatX, 'numpy': 'float64'}),
            (fx - theano.tensor.xor(iy, iz), (fx, iy, iz), (fxv, iyv, izv), 1,
                fxv - (iyv ^ izv), {'custom': 'float64', 'numpy + floatX': config.floatX, 'numpy': 'float64'}),  # 60
            (fx - theano.tensor.and_(iy, iz), (fx, iy, iz), (fxv, iyv, izv), 1,
                fxv - (iyv & izv), {'custom': 'float64', 'numpy + floatX': config.floatX, 'numpy': 'float64'}),
            (fx - theano.tensor.invert(iy), (fx, iy), (fxv, iyv), 1,
                fxv - (~iyv), {'custom': 'float64', 'numpy + floatX': config.floatX, 'numpy': 'float64'}),

            (fx - theano.tensor.cast(fy, dtype='float64'), (fx, fy), (fxv, fyv), 1,
                fxv - np.asarray(fyv, 'float64'), 'float64'),
            (theano.tensor.pow(fx * fy + fz, fx * fy), (fx, fy, fz), (fxv, fyv, fzv), 1,
                np.power(fxv * fyv + fzv, fxv * fyv), 'float32'),
            (fv + fy ** fz, (fv, fy, fz), (fvv, fyv, fzv), 2, fvv + fyv ** fzv, 'float32'),  # fused with a dimshuffle #65
            (fv - fy + tensor.tanh(fz), (fv, fy, fz), (fvv, fyv, fzv), 2,
                fvv - fyv + np.tanh(fzv), 'float32'),  # fused with a dimshuffle

            # Cases where the same input is reused many times.
            (theano.tensor.mul(fx, fx, fx, fx), (fx,), (fxv,), 1, fxv *
                fxv * fxv * fxv, 'float32'),
            (theano.tensor.mul(fx, ftanx, ftanx), (fx,), (fxv,), 1,
                fxv * np.tan(fxv) * np.tan(fxv), 'float32'),
            (theano.tensor.mul(fx, ftanx, ftanx, fx), (fx,), (fxv,),
                1, fxv * np.tan(fxv) * np.tan(fxv) * fxv, 'float32'),
            (theano.tensor.mul(ftanx, ftanx, fx + fy), (fx, fy), (fxv, fyv),
                1, np.tan(fxv) * np.tan(fxv) * (fxv + fyv), 'float32'),  # 70

            # Cases with different broadcast pattern. They should not
            # be merged as this would duplicate computation
            # The graph should have 2 elemwise and 1 dimshuffle
            (fx * theano.tensor.sin(fs), (fx, fs), (fxv, fsv), 3,
                fxv * np.sin(fsv), 'float32'),
            ]
        if slice:
            cases = cases[slice]
        times = np.zeros(len(cases))
        fail1 = []
        fail2 = []
        fail3 = []
        fail4 = []
        for id, [g, sym_inputs, val_inputs,
                 nb_elemwise, answer, out_dtype] in enumerate(cases):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            print("new cases", id)

            if shared_fn is None:
                f = compile.function(list(sym_inputs), g, mode=mode)
                for x in xrange(nb_repeat):
                    out = f(*val_inputs)
                t1 = time.time()
            else:
                out = shared_fn(np.zeros(shp, dtype=out_dtype), 'out')
                assert out.dtype == g.dtype
                f = function(sym_inputs, [], updates=[(out, g)], mode=mode)
                t0 = time.time()
                for x in xrange(nb_repeat):
                    f(*val_inputs)
                t1 = time.time()
                out = out.get_value()

            times[id] = t1 - t0
            atol = 1e-8
            if out_dtype == 'float32':
                atol = 1e-6
            if not np.allclose(out, answer * nb_repeat, atol=atol):
                fail1.append(id)
                print(val_inputs)
                print(out)
                print(answer * nb_repeat)
            topo = f.maker.fgraph.toposort()
            topo_ = [n for n in topo
                     if not isinstance(n.op, self.topo_exclude)]
            if assert_len_topo:
                if not len(topo_) == nb_elemwise:
                    fail3.append((id, topo_, nb_elemwise))
                if nb_elemwise == 1:
                    # if no variable appears multiple times in the
                    # input of g,
                    # check that the number of input to the Composite
                    # Elemwise is ok
                    if len(set(g.owner.inputs)) == len(g.owner.inputs):
                        expected_len_sym_inputs = np.sum(
                            [not isinstance(x, theano.gof.Constant)
                             for x in topo_[0].inputs])
                        assert expected_len_sym_inputs == len(sym_inputs)

            if not out_dtype == out.dtype:
                fail4.append((id, out_dtype, out.dtype))

        failed = len(fail1 + fail2 + fail3 + fail4)
        print("Executed", len(cases), "cases", "failed", failed)
        if failed > 0:
            raise Exception("Failed %d cases" % failed, fail1,
                            fail2, fail3, fail4)

        return times

    def test_elemwise_fusion(self):
        shp = (5, 5)
        mode = copy.copy(self.mode)
        # we need the optimisation enabled and the canonicalize.
        # the canonicalize is needed to merge multiplication/addition by constant.
        mode._optimizer = mode._optimizer.including(
            'local_elemwise_fusion', 'composite_elemwise_fusion',
            'canonicalize')
        self.do(mode, self._shared, shp)

    @attr('slow')
    def test_elemwise_fusion_4d(self):
        shp = (3, 3, 3, 3)
        mode = copy.copy(self.mode)
        # we need the optimisation enabled and the canonicalize.
        # the canonicalize is needed to merge multiplication/addition by constant.
        mode._optimizer = mode._optimizer.including(
            'local_elemwise_fusion', 'composite_elemwise_fusion',
            'canonicalize')
        self.do(mode, self._shared, shp, slice=slice(0, 1))

    def test_fusion_35inputs(self):
        # Make sure a fused graph with more than 35 inputs does not segfault
        # or error.
        inpts = vectors(['i%i' % i for i in xrange(35)])
        # Make an elemwise graph looking like:
        # sin(i34 + sin(i33 + sin(... i1 + sin(i0) ...)))
        out = tensor.sin(inpts[0])
        for idx in xrange(1, 35):
            out = tensor.sin(inpts[idx] + out)

        f = function(inpts, out, mode=self.mode)
        # Test it on some dummy values
        f(*[list(range(i, 4 + i)) for i in xrange(35)])

    def test_pickle_big_fusion(self):
        # In the past, pickle of Composite generated in that case
        # crashed with max recusion limit. So we where not able to
        # generate C code in that case.

        if not theano.config.cxx:
            raise SkipTest("no c compiler, so can't use big elemwise!")
        factors = []
        sd = tensor.dscalar()
        means = tensor.dvector()

        cst_05 = theano.tensor.constant(.5)
        cst_m05 = theano.tensor.constant(-.5)
        cst_2 = theano.tensor.constant(2)
        cst_m2 = theano.tensor.constant(-2)
        ones = theano.tensor.constant(np.ones(10))
        n = 85
        if theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            n = 10

        for i in xrange(n):
            f = (cst_m05 * sd ** cst_m2 * (ones - means[i]) ** cst_2 +
                 cst_05 * tensor.log(cst_05 * (sd ** cst_m2) / np.pi))
            factors.append(tensor.sum(f))

        logp = tensor.add(*factors)

        vars = [sd, means]
        dlogp = function(vars, [theano.grad(logp, v) for v in vars])
        dlogp(2, np.random.rand(n))

    def speed_fusion(self, s=None):
        """
        param type s: a slice object
        param s: a slice to apply to the case to execute. If None, exec all case.
        """

        shp = (3000, 3000)
        shp = (1000, 1000)
        nb_repeat = 50
        # linker=gof.CLinker
        # linker=gof.OpWiseCLinker

        mode1 = copy.copy(self.mode)
        mode1._optimizer = mode1._optimizer.including('local_elemwise_fusion')
        # TODO:clinker is much faster... but use to much memory
        # Possible cause: as their is do deletion of intermediate value when we don't keep the fct.
        # More plausible cause: we keep a link to the output data?
        # Follow up. Clinker do the same... second cause?
        mode2 = copy.copy(self.mode)
        mode2._optimizer = mode2._optimizer.excluding('local_elemwise_fusion')
        print("test with linker", str(mode1.linker))
        times1 = self.do(mode1, self._shared, shp, nb_repeat=nb_repeat,
                         assert_len_topo=False, slice=s)
        times2 = self.do(mode2, self._shared, shp, nb_repeat=nb_repeat,
                         assert_len_topo=False, slice=s)
        print("times1 with local_elemwise_fusion")
        print(times1, times1.min(), times1.max(), times1.sum())
        print("times2 without local_elemwise_fusion")
        print(times2, times2.min(), times2.max(), times2.sum())
        d = times2 / times1

        print("times2/times1")
        print(d)
        print("min", d.min(), "argmin", d.argmin(), "max", d.max(),
              "mean", d.mean(), "std", d.std())

    def test_fusion_inplace(self):
        mode = copy.copy(self.mode)
        # we need the optimisation enabled and the canonicalize.
        # the canonicalize is needed to merge multiplication/addition by constant.
        mode._optimizer = mode._optimizer.including(
            'local_elemwise_fusion', 'composite_elemwise_fusion',
            'canonicalize', 'inplace')

        x, y, z = dmatrices('xyz')
        f = theano.function([x, y, z], tensor.dot(x, y) + x + y + z, mode=mode)
        topo = [n for n in f.maker.fgraph.toposort()
                if not isinstance(n.op, self.topo_exclude)]
        assert len(topo) == 2
        assert topo[-1].op.inplace_pattern
        f(np.random.random((5, 5)), np.random.random((5, 5)),
            np.random.random((5, 5)))

    def speed_log_exp(self):
        s = slice(31, 36)
        print("time", self.do(self.mode, self._shared, shp=(1000, 1000),
                              assert_len_topo=False, slice=s, nb_repeat=100))


class TimesN(theano.scalar.basic.UnaryScalarOp):
    """
    Used in test TestCompositeCodegen

    Must be outside of the class, otherwise, the c cache code can't
    pickle this class and this cause stuff printing during test.
    """
    def __eq__(self, other):
        return super(TimesN, self).__eq__(other) and self.n == other.n

    def __hash__(self):
        return super(TimesN, self).__hash__() ^ hash(self.n)

    def __init__(self, n, *args, **kwargs):
        self.n = n
        theano.scalar.basic.UnaryScalarOp.__init__(self, *args, **kwargs)

    def impl(self, x):
        return x * self.n

    def c_support_code_apply(self, node, nodename):
        n = str(self.n)
        return """
        float %(nodename)s_timesn(float x) { return x * %(n)s; }
        """ % locals()

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = %(name)s_timesn(%(x)s);" % locals()


class TestCompositeCodegen(unittest.TestCase):
    """
    Test The Composite Ops code generation in a case where there is multiple
    scalar ops with support code.
    """
    def setUp(self):
        upgrade_to_float = theano.scalar.basic.upgrade_to_float

        self.scal_times_2 = TimesN(2, upgrade_to_float, name='times_2')
        self.times_2 = theano.tensor.elemwise.Elemwise(
            self.scal_times_2,
            name='times_2')

        self.scal_times_3 = TimesN(3, upgrade_to_float, name='times_3')
        self.times_3 = theano.tensor.elemwise.Elemwise(
            self.scal_times_3,
            name='times_3')

        self.x = fvector()

    def test_nested_composite(self):
        y = self.times_2(self.x)
        z = self.times_3(y)
        f = function([self.x], z)
        if config.mode != "FAST_COMPILE":
            assert len(f.maker.fgraph.toposort()) == 1
        fval = f([1, 2, 3])
        assert np.all(fval == [6, 12, 18])

    def test_local_useless_composite(self):
        x = theano.scalar.float32()
        c = theano.scalar.Composite([x], [x + 1, x - 1])
        X = theano.tensor.matrix()
        o = theano.tensor.Elemwise(scalar_op=c)(X)
        mode = theano.compile.mode.get_default_mode().including(
            'local_useless_composite')

        f = theano.function([X], o[0], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert len(topo[0].outputs) == 1
        utt.assert_allclose(f([[1.]]), [[2.]])

        f = theano.function([X], o[1], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert len(topo[0].outputs) == 1
        utt.assert_allclose(f([[1.]]), [[0.]])


@utt.assertFailure_fast
def test_log1p():
    m = theano.config.mode
    if m == 'FAST_COMPILE':
        m = 'FAST_RUN'
    m = compile.mode.get_mode(m)
    m = m.excluding('fusion')
    # check some basic cases
    x = dvector()
    f = function([x], T.log(1 + (x)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.log1p]
    f = function([x], T.log(1 + (-x)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [
        T.neg, inplace.log1p_inplace]
    f = function([x], -T.log(1 + (-x)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [
        T.neg, inplace.log1p_inplace, inplace.neg_inplace]

    # check trickier cases (and use different dtype)
    y = fmatrix()
    f = function([x, y], T.log(tensor.fill(y, 1) + (x)), mode=m)
    # the first three ops are Shape_i, Shape_i, and Dimshuffle
    topo = f.maker.fgraph.toposort()
    assert topo[-1].op == tensor.alloc
    assert T.log1p in [node.op for node in topo]

    f = function([x, y], T.log(0 + (x) + tensor.fill(y, 1.0)), mode=m)
    topo = f.maker.fgraph.toposort()
    assert topo[-1].op == tensor.alloc
    assert T.log1p in [node.op for node in topo]

    f = function([x, y], T.log(2 + (x) - tensor.fill(y, 1.0)), mode=m)
    topo = f.maker.fgraph.toposort()
    assert topo[-1].op == tensor.alloc
    assert T.log1p in [node.op for node in topo]

    f([1e-7, 10], [[0, 0], [0, 0]])  # debugmode will verify values

    # should work for int
    z = tensor.imatrix()
    f = function([z], T.log(1 + (z)), mode=m)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.log1p]


def test_log_add():
    m = theano.config.mode
    if m == 'FAST_COMPILE':
        m = 'FAST_RUN'
    m = compile.mode.get_mode(m)
    m = m.excluding('fusion')
    m = copy.copy(m)
    # No need to put them back as we have a new object
    m.check_isfinite = False

    # check some basic cases
    x = dvector()
    y = dvector()
    f = function([x, y], T.log(T.exp(x) + T.exp(y)), mode=m)

    f([10000], [10000])  # causes overflow if handled incorrectly
    assert np.isfinite(f([10000], [10000]))
    utt.assert_allclose(f([10000], [10000]), 10000 + np.log1p(1))

    # test that it give the same result when it don't overflow
    f([10], [10])  # don't causes overflow
    utt.assert_allclose(f([10], [10]), 10 + np.log1p(1))

    # test that it also works with more than two args, (this currently fails)
    x = dvector()
    y = dvector()
    f = function([x, y], T.log(T.exp(x) + T.exp(y) + T.exp(x - y) + T.exp(
        x + y)), mode=m)

    try:
        f([10000], [10000])  # causes overflow if handled incorrectly
        utt.assert_allclose(f([10000], [10000]), 20000)
    except utt.WrongValue:
        raise SkipTest("log(add(exp)) is not stabilized when adding "
                       "more than 2 elements, see #623")

    # TODO: test that the optimization works in the presence of broadcasting.

    # TODO: (write and) test that the optimization works with Sum in addition to working with Add.


def test_local_useless_slice():
    # test a simple matrix
    x = tensor.matrix('x')
    mode_unopt = compile.get_default_mode().excluding("local_useless_slice", "local_mul_canonizer")
    mode_opt = compile.get_default_mode().including("local_useless_slice").excluding("local_mul_canonizer")

    # test with and without the useless slice
    o = 2 * x[0, :]
    f_unopt = theano.function([x], o, mode=mode_unopt)
    f_opt = theano.function([x], o, mode=mode_opt)
    test_inp = np.random.randint(-10, 10, (4, 4)).astype('float32')
    assert all(f_opt(test_inp) == f_unopt(test_inp)),\
        "The optimization caused a mismatch in the result"
    # test to see if the slice is truly gone
    apply_node = f_opt.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(isinstance(idx, slice) for idx in subtens.idx_list), "Slice should be gone"

    # Now test that the stack trace is copied over properly,
    # before before and after optimization.
    assert check_stack_trace(f_unopt, ops_to_check='all')
    assert check_stack_trace(f_opt, ops_to_check='all')

    # test a 4d tensor
    z = tensor.tensor4('z')
    o2 = z[1, :, :, 1]
    o3 = z[0, :, :, :]
    f_opt_check = theano.function([z], o2, mode=mode_opt)
    f_opt_check_apply = theano.function([z], o3, mode=mode_opt)

    # The optimization shouldn't apply here
    apply_node = f_opt_check.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert [isinstance(idx, slice) for idx in subtens.idx_list].count(True) == 2
    # But it should here
    apply_node = f_opt_check_apply.maker.fgraph.toposort()[0]
    subtens = apply_node.op
    assert not any(isinstance(idx, slice) for idx in subtens.idx_list)

    # Finally, test that the stack trace is copied over properly,
    # before before and after optimization.
    assert check_stack_trace(f_opt_check, ops_to_check=Subtensor)
    assert check_stack_trace(f_opt_check_apply, ops_to_check=Subtensor)


def test_local_useless_inc_subtensor():
    x = tensor.matrix('x')
    y = tensor.matrix('y')
    mode = compile.get_default_mode().including("local_useless_inc_subtensor")
    for sub in [slice(None), slice(None, None, -1)]:
        o = tensor.set_subtensor(x[::, sub], y)
        f = theano.function([x, y], o, mode=mode)
        o_shape = tensor.set_subtensor(x[::, sub],
                                       tensor.specify_shape(y, x.shape))
        f_shape = theano.function([x, y], o_shape, mode=mode)

        # Test with shape info
        topo = f_shape.maker.fgraph.toposort()
        assert not any(isinstance(n.op, tensor.IncSubtensor) for n in topo)
        out = f_shape([[2, 3]], [[3, 4]])
        assert (out == np.asarray([[3, 4]])[::, sub]).all()

        # Test that without shape info, we don't apply the opt.
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, tensor.IncSubtensor)
        out = f([[2, 3]], [[3, 4]])
        assert (out == np.asarray([[3, 4]])[::, sub]).all()

        # Test that we don't remove shape error
        try:
            f([[2, 3]], [[3, 4], [4, 5]])
            assert False
        except (ValueError, AssertionError):
            pass

        # Test that we don't remove broadcastability
        out = f([[2, 3], [3, 4]], [[5, 6]])
        assert (out == np.asarray([[5, 6], [5, 6]])[::, sub]).all()

    # Test that we do not optimize others strides even when sub and y
    # have same shapes
    sub = x[::, ::2]
    o_shape = tensor.set_subtensor(sub,
                                   tensor.specify_shape(y, sub.shape))
    f_shape = theano.function([x, y], o_shape)
    topo = f_shape.maker.fgraph.toposort()
    # theano.printing.debugprint(f_shape)
    assert any(isinstance(n.op, tensor.IncSubtensor) for n in topo)
    out = f_shape([[2, 3, 6, 7]], [[8, 9]])
    assert (out == np.asarray([[8, 3, 9, 7]])).all()


def test_local_useless_subtensor():
    x = tensor.matrix('x')

    # Test default
    for dims in [(slice(0, None), ),
                 (slice(0, None), slice(0, None)),
                 ]:
        f = function([x], tensor.exp(x).__getitem__(dims), mode=mode_opt)
        # theano.printing.debugprint(f)
        prog = f.maker.fgraph.toposort()
        assert prog[0].op == tensor.exp
        assert len(prog) == 1
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something

    x_c = tensor.specify_shape(x, (2, 3))
    # Test constant
    for dims, res in [((slice(0, 2), ), True),
                      ((slice(0, 2), slice(0, None)), True),
                      ((slice(0, 2), slice(0, 3)), True),
                      ((slice(0, None), slice(0, 3)), True),
                      ((slice(0, 3), slice(0, 13)), True),
                      ((slice(0, 3), slice(0, 2)), False),
                      ((slice(0, 1), slice(0, None)), False),
                      ((slice(0, 1), 1), False)]:
        f = function([x], tensor.exp(x_c).__getitem__(dims), mode=mode_opt)
        # theano.printing.debugprint(f)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, theano.tensor.SpecifyShape), dims
            assert prog[1].op == tensor.exp, (dims, prog)
            assert len(prog) == 2, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something

    # Test Variable
    for idx, (dims, res) in enumerate([
            ((slice(0, x.shape[0]), ), True),
            ((slice(0, x.shape[1]), ), False),
            ((slice(0, x.shape[0]), slice(0, x.shape[1]), ), True),
            ((slice(0, x.shape[0]), slice(0, x.shape[0]), ), False),
            ((slice(0, x.shape[1]), slice(0, x.shape[0]), ), False),
            ((slice(0, x.shape[1]), slice(0, x.shape[1]), ), False),
            ((slice(0, x.shape[1]), 2), False),
            ((slice(0, x.shape[1]), slice(x.shape[0] - x.shape[0],
                                          x.shape[1]),), False),
            ((slice(0, T.scalar_from_tensor(x.shape[0])), ), True),
            ]):
        f = function([x], tensor.exp(x).__getitem__(dims), mode=mode_opt)
        # theano.printing.debugprint(f)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == tensor.exp, dims
            assert len(prog) == 1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something
    # Test mix Variable and Constant
    # Currently not supported
    for idx, (dims, res) in enumerate([
            ((slice(0, x.shape[0]), slice(0, 3)), False),
            ((slice(0, 3), slice(0, x.shape[1])), False),
            ]):
        f = function([x], tensor.exp(x_c).__getitem__(dims), mode=mode_opt)
        # theano.printing.debugprint(f)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == tensor.exp, dims
            assert len(prog) == 1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something

    # Test scalar variable
    s = scal.int32('s')
    for idx, (dims, res) in enumerate([
            ((slice(0, s), ), False),
            ]):
        f = function([x, s], tensor.exp(x).__getitem__(dims), mode=mode_opt)
        # theano.printing.debugprint(f)
        prog = f.maker.fgraph.toposort()
        if res:
            assert prog[0].op == tensor.exp, dims
            assert len(prog) == 1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[1, 2, 3], [4, 5, 6]], 1)
        f([[1, 2, 3], [4, 5, 6]], 3)

    # Test AdvancedSubtensor1 case when all rows are selected by a list/vector
    # or ARange op
    for dims, res in (([0, 1], True),
                      ([1, 0], False),
                      ([0, 0], False),
                      ([0, 0, 1], False),
                      (T.arange(2), True),
                      (T.arange(0, 2), True),
                      (T.arange(0, 2, 2), False),
                      (T.arange(0, 2, -1), False),
                      (T.arange(1, 2), False)):
        f = function([x], tensor.exp(x_c).__getitem__(dims), mode=mode_opt)
        # theano.printing.debugprint(f)
        prog = f.maker.fgraph.toposort()
        if res:
            assert isinstance(prog[0].op, theano.tensor.SpecifyShape), dims
            assert prog[1].op == tensor.exp, dims
            assert len(prog) == 2, dims
        else:
            assert any([isinstance(node.op, AdvancedSubtensor1)
                        for node in prog])
        f([[0, 1, 2], [3, 4, 5]])  # let debugmode test something


def test_local_subtensor_remove_broadcastable_index():
    # testing local_subtensor_remove_broadcastable_index optimization
    #
    # tests removing broadcastable dimensions with index 0 or -1,
    # otherwise the optimzation should not be applied

    mode = theano.compile.mode.get_default_mode()
    mode = mode.including("local_subtensor_remove_broadcastable_index")
    x = T.dmatrix('x')
    y1 = x.dimshuffle(0, 'x', 1)
    y2 = x.dimshuffle('x', 1, 0, 'x')
    y3 = x.dimshuffle('x', 1, 'x', 0, 'x')

    # testing for cases that the optimzation should be applied
    z1 = y1[:, 0, :]
    z2 = y1[:, -1, :]
    z3 = y2[0, :, :, -1]
    z4 = y2[0, :, :, 0]
    z5 = y2[-1, :, :, -1]
    z6 = y3[-1, :, 0, :, -1]
    z7 = y3[-1, :, -1, :, -1]
    z8 = y3[0, :, 0, :, 0]
    f = theano.function([x], [z1, z2, z3, z4, z5, z6, z7, z8], mode=mode)
    for elem in f.maker.fgraph.toposort():
        assert type(elem.op) not in [Subtensor, AdvancedSubtensor,
                                     AdvancedSubtensor1, IncSubtensor,
                                     AdvancedIncSubtensor,
                                     AdvancedIncSubtensor1]

    rng = np.random.RandomState(seed=utt.fetch_seed())
    xn = rng.rand(5, 5)
    f(xn)

    # testing for cases that the optimzation should not be applied
    # to verify that other subtensor usage are passed without errors
    w1 = y1[3, 0, :]
    w2 = y1[2:4, -1, :]
    w3 = y2[0, :, 4:, -1]
    w4 = y2[:, :, 0, -1]
    w5 = y2[0, 2:4, :, 0]
    w6 = y2[0, -1, :, -1]
    w7 = y2[-1, 4:, :, -1]
    w8 = y2[-1, :, :3, -1]
    w9 = y2[-1, :, -1, -1]
    w10 = y3[-1, 2, 0, :, -1]
    w11 = y3[-1, 0, -1, :, -1]
    w12 = y3[-1, :, -1, -1, -1]
    w13 = y3[0, 0, 0, :, 0]
    w14 = y3[-1, 2:4, 0, 1:5, -1]
    w15 = y3[-1, 0, -1, 0, -1]
    w16 = y3[0, 2, 0, 4, 0]
    w17 = y3[:, 0, :, 1]
    w18 = y3[0, :, :, 2]
    w19 = y3[:, 2, 0]
    w20 = y3[:, 3]
    f2 = theano.function([x], [w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11,
                               w12, w13, w14, w15, w16, w17, w18, w19, w20],
                         mode=mode)
    f2(xn)


class Test_subtensor_inc_subtensor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mode = theano.compile.mode.get_default_mode().including('local_subtensor_inc_subtensor')

    def test_basic(self):
        # basic test
        x = tensor.matrix('x')
        i = tensor.iscalar('i')
        v = tensor.vector('v')
        y = tensor.set_subtensor(x[i], v)
        z = y[i]
        f = theano.function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # basic test, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(size=[4, ]).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_)

    def test_multiple_idx(self):
        # complicated test
        x = tensor.tensor4('x')
        i1 = tensor.iscalar('i1')
        i2 = tensor.iscalar('i2')
        i3 = tensor.iscalar('i3')
        i4 = tensor.iscalar('i4')
        v = tensor.tensor3('v')
        y = tensor.set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i2, i3:, ::i4]
        f = theano.function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, DeepCopyOp)
        # complicated test, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), v_)

    def test_not_applied(self):
        # case not use this optimization
        x = tensor.tensor4('x')
        i1 = tensor.iscalar('i1')
        i2 = tensor.iscalar('i2')
        i3 = tensor.iscalar('i3')
        i4 = tensor.iscalar('i4')
        v = tensor.tensor3('v')
        y = tensor.set_subtensor(x[i1, :i2, i3:, ::i4], v)
        z = y[i1, :i3, i2:, ::i4]
        f = theano.function([x, i1, i2, i3, i4, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) != 1
        assert any(isinstance(x.op, tensor.IncSubtensor) for x in prog)
        assert any(isinstance(x.op, tensor.Subtensor) for x in prog)
        # case not use this optimization, numerical check
        x_ = np.random.uniform(size=[3, 4, 5, 6]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 2, 2]).astype(config.floatX)
        i1_, i2_, i3_, i4_ = 1, 2, 3, 4
        x_[i1_, :i2_, i3_:, ::i4_] = v_
        assert np.array_equal(f(x_, i1_, i2_, i3_, i4_, v_), x_[i1_, :i3_, i2_:, ::i4_])

    def test_fewer_dims(self):
        # case when v has fewer dimensions
        x = tensor.matrix('x')
        i1 = tensor.iscalar('i')
        i2 = tensor.iscalar('i')
        v = tensor.vector('v')
        y = tensor.set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = theano.function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, tensor.Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, ]).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_broadcasted(self):
        # case when v has the same number of dimensions, some broadcastable
        x = tensor.matrix('x')
        i1 = tensor.iscalar('i')
        i2 = tensor.iscalar('i')
        v = tensor.col('v')
        y = tensor.set_subtensor(x[:i1, :i2], v)
        z = y[:i1, :i2]
        f = theano.function([x, i1, i2, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert any(isinstance(x.op, tensor.Alloc) for x in prog)
        # case when v is broadcastable, numerical check
        x_ = np.random.uniform(size=[3, 4]).astype(config.floatX)
        v_ = np.random.uniform(size=[2, 1]).astype(config.floatX)
        i1_, i2_ = 2, 2
        x_[:i1_, :i2_] = v_
        assert np.array_equal(f(x_, i1_, i2_, v_), x_[:i1_, :i2_])

    def test_different_dtypes(self):
        # Case when the dtype differs
        x = tensor.bmatrix('x')
        i = tensor.iscalar('i')
        v = tensor.vector('v')
        y = tensor.set_subtensor(x[i], v)
        z = y[i]
        f = theano.function([x, i, v], z, mode=self.mode)
        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert prog[0].op == tensor.basic._convert_to_int8
        # basic test, numerical check
        x_ = np.random.randint(12, size=[3, 4]).astype('int8')
        v_ = np.random.uniform(12, size=[4, ]).astype(config.floatX)
        i_ = 1
        assert np.array_equal(f(x_, i_, v_), v_.astype('int8'))


class test_local_subtensor_make_vector(unittest.TestCase):
    def test_scalar_idx(self):
        x, y, z = tensor.lscalars('xyz')
        v = make_vector(x, y, z)
        f = function([x, y, z], v[0], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, theano.compile.ops.DeepCopyOp)
        assert f(0, 1, 2) == 0

    def test_slice_idx_stop(self):
        x, y, z = tensor.lscalars('xyz')
        v = make_vector(x, y, z)
        f = function([x, y, z], v[:2], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 1

    def test_slice_idx_step(self):
        x, y, z = tensor.lscalars('xyz')
        v = make_vector(x, y, z)
        f = function([x, y, z], v[::2], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_AdvancedSubtensor1_idx(self):
        x, y, z = tensor.lscalars('xyz')
        v = make_vector(x, y, z)
        f = function([x, y, z], v[[0, 2]], mode=mode_opt)

        prog = f.maker.fgraph.toposort()
        assert len(prog) == 1
        assert isinstance(prog[0].op, MakeVector)
        assert len(prog[0].inputs) == 2
        r = f(0, 1, 2)
        assert r[0] == 0 and r[1] == 2

    def test_stack_trace(self):
        x, y, z = tensor.lscalars('xyz')
        v = make_vector(x, y, z)

        mode = theano.compile.mode.get_default_mode().including(
            "local_subtensor_make_vector")

        # list of subtensor cases, where local_subtensor_make_vector
        # inserts a new MakeVector node
        v_subtensors = [v[:2], v[::2], v[[0, 2]]]

        for v_subtensor in v_subtensors:
            f = function([x, y, z], v_subtensor, mode=mode)
            self.assertTrue(check_stack_trace(f, ops_to_check='all'))


class test_local_subtensor_lift(unittest.TestCase):
    def test0(self):
        # basic test that the Op works
        x = tensor.matrix('x')
        f = function([x], tensor.exp(x)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check='all'))

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)  # first subtensor
        assert prog[1].op == tensor.exp
        assert len(prog) == 2
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test0b(self):
        # as test0, but we reuse the output of the elemwise
        # So we should not lift the subtensor
        x = tensor.matrix('x')
        f = function([x], [tensor.exp(x)[0], tensor.exp(x)], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=[
            Subtensor, tensor.Elemwise]))

        prog = f.maker.fgraph.toposort()
        assert prog[0].op == tensor.exp
        assert isinstance(prog[1].op, tensor.Subtensor)  # first subtensor
        assert isinstance(prog[2].op, DeepCopyOp)
        assert len(prog) == 3
        f([[0, 1], [2, 3]])  # let debugmode test something

    def test1(self):
        # basic test that the optimization work with scalar broadcasted
        x = tensor.matrix('x')
        y = tensor.scalar('y')
        z = tensor.matrix('z')
        f = function([x, y, z], tensor.exp(x + y + z)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=[
            Subtensor, tensor.DimShuffle]))

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.DimShuffle)
        assert isinstance(prog[2].op, tensor.Subtensor)
        assert isinstance(prog[3].op.scalar_op, theano.scalar.
                          Composite)  # Composite{add,add}
        assert len(prog) == 4
        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test2(self):
        # as 1, but take a slice
        x = tensor.matrix('x')
        y = tensor.scalar('y')
        z = tensor.matrix('z')
        f = function([x, y, z], tensor.exp(x + y + z)[0:2], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=[
            Subtensor, tensor.DimShuffle]))

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.DimShuffle)
        assert isinstance(prog[2].op, tensor.Subtensor)
        assert isinstance(prog[3].op.scalar_op, theano.scalar.
                          Composite)  # Composite{add,add}
        assert len(prog) == 4
        # let debugmode test something
        f([[0, 1], [2, 3]], 4, [[4, 5], [6, 7]])

    def test3(self):
        # basic test that the optimization does work with broadcasting
        # for unary elemwise.
        y = tensor.vector('y')
        f = function([y], tensor.exp(y.dimshuffle(0, 'x'))[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check='all'))

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.DimShuffle)
        assert isinstance(prog[1].op, tensor.Subtensor)
        assert prog[2].op == tensor.exp
        assert len(prog) == 3
        f([4, 5])  # let debugmode test something

    @utt.assertFailure_fast
    def test4(self):
        # basic test that the optimization doesn't work with broadcasting
        # ... It *could* be extended to,
        # ... but right now it doesn't, so it shouldn't try.
        x = tensor.matrix('x')
        y = tensor.vector('y')
        f = function([x, y], tensor.exp(x + y)[0], mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # self.assertTrue(check_stack_trace(f, ops_to_check='all'))

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.DimShuffle)
        assert prog[1].op == tensor.add
        assert isinstance(prog[2].op, tensor.Subtensor)  # first subtensor
        assert prog[3].op == inplace.exp_inplace
        assert len(prog) == 4
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test5(self):
        # test that we don't lift when we reuse the output of the
        # elemwise for other computation.
        x = tensor.matrix('x')
        y = tensor.vector('y')
        f = function([x, y], [tensor.exp(x + y)[0], tensor.exp(x + y) + x],
                     mode=mode_opt)

        # Opt doesn't apply, so no need for check_stack_trace
        # self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.DimShuffle)
        assert isinstance(prog[1].op.scalar_op, theano.scalar.
                          Composite)  # Composite{add,exp}
        assert prog[2].op == tensor.add or prog[3].op == tensor.add
        # first subtensor
        assert isinstance(prog[2].op, tensor.Subtensor) or isinstance(prog[3].op, tensor.Subtensor)
        assert len(prog) == 4
        f([[0, 1], [2, 3]], [4, 5])  # let debugmode test something

    def test6(self):
        # basic test that the optimization works with a scalar as input,
        # and a scalar as output (no broadcasting of the scalar needed).
        # The optimization used to fail and display an ERROR message.

        x = tensor.vector('x')
        y = tensor.scalar('y')
        f = function([x, y], tensor.exp(x + y)[0], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        prog = f.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        # Composite{add,exp}
        assert isinstance(prog[1].op.scalar_op, theano.scalar.Composite)
        assert len(prog) == 2
        f([1, 2, 3], 4)  # let debugmode test something

    def test7(self):
        # Test that Subtensor(Rebroadcast(x)) gets optimized into
        # Rebroadcast(Subtensor(x)).

        # test basic case
        x = tensor.matrix('x')
        xval = np.random.rand(1, 10).astype(config.floatX)
        assert x.broadcastable == (False, False)
        newx = tensor.Rebroadcast((0, True), (1, False))(x)
        assert newx.broadcastable == (True, False)

        f1 = function([x], newx[:2, :5], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f1, ops_to_check=[
                        Subtensor, tensor.Rebroadcast]))
        prog = f1.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f1(xval) == xval[:2, :5]).all()

        # corner case 1: rebroadcast changes dims which are dropped through subtensor
        y = tensor.tensor4('x')
        yval = np.random.rand(1, 10, 1, 3).astype(config.floatX)
        assert y.broadcastable == (False, False, False, False)
        newy = tensor.Rebroadcast((0, True), (2, True))(y)
        assert newy.broadcastable == (True, False, True, False)

        f2 = function([y], newy[:, 3, 0, :], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f2, ops_to_check=[
                        Subtensor, tensor.Rebroadcast]))
        prog = f2.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f2(yval) == yval[:, 3, 0, :]).all()

        # corner case 2: subtensor idx_list is shorter than resulting broadcast pattern
        f3 = function([y], newy[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f3, ops_to_check=[
            Subtensor, tensor.Rebroadcast]))
        prog = f3.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f3(yval) == yval[:, 3, 0]).all()

        # corner case 3: subtensor idx_list is shorter than rebroadcast.axis
        z = tensor.tensor4('x')
        zval = np.random.rand(4, 10, 3, 1).astype(config.floatX)
        assert z.broadcastable == (False, False, False, False)
        newz = tensor.Rebroadcast((3, True))(z)
        assert newz.broadcastable == (False, False, False, True)

        f4 = function([z], newz[:, 3, 0], mode=mode_opt)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f4, ops_to_check=[
            Subtensor, tensor.Rebroadcast]))
        prog = f4.maker.fgraph.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f4(zval) == zval[:, 3, 0]).all()


class test_local_subtensor_merge(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        self.x_shapes = [(2, 2), (5, 3), (4, 1), (1, 2),
                         (0, 2), (2, 0), (1, 0), (0, 0)]
        self.rng = np.random.RandomState(seed=utt.fetch_seed())

    def test_const(self):
        # var[const::][-1] -> var[-1]
        x = tensor.matrix('x')
        for idx in xrange(-7, 6):
            f = function([x], x[idx::][-1], mode=mode_opt)
            g = function([x], x[idx::][-1], mode=mode_opt.excluding(
                'local_subtensor_merge'))

            # Check stacktrace was copied over correctly after opt was applied
            self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

            topo = f.maker.fgraph.toposort()
            assert len([t for t in topo
                        if isinstance(t.op, tensor.Subtensor)]) == 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)

                if idx < x_s[0] and x_s[0] > 0:
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    self.assertRaises(IndexError, f, x_val)
                    self.assertRaises(IndexError, g, x_val)

    def test_scalar(self):
        # var[int::][-1] -> var[-1]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        f = function([x, y], x[y::][-1], mode=mode_opt)
        g = function([x, y], x[y::][-1],
                     mode=mode_opt.excluding('local_subtensor_merge'))
        # theano.printing.debugprint(f, print_type=True)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        topo = f.maker.fgraph.toposort()
        # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo
                    if isinstance(t.op, tensor.Subtensor)]) == 1
        # print topo[-1].op
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in xrange(-9, 8):
                if (idx < x_s[0]) and (x_s[0] > 0):
                    # The first subtensor is non-empty
                    f(x_val, idx)  # let debugmode test something
                else:
                    self.assertRaises(IndexError, f, x_val, idx)
                    self.assertRaises(IndexError, g, x_val, idx)

    @attr('slow')
    def test_const2(self):
        # var[::-1][const] -> var[-1]
        x = tensor.matrix('x')
        for idx in xrange(-8, 7):
            f = function([x], x[::-1][idx], mode=mode_opt)
            g = function([x], x[::-1][idx],
                         mode=mode_opt.excluding('local_subtensor_merge'))

            # Check stacktrace was copied over correctly after opt was applied
            self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

            # theano.printing.debugprint(f, print_type=True)
            topo = f.maker.fgraph.toposort()
            # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo
                        if isinstance(t.op, tensor.Subtensor)]) == 1
            # print topo[-1].op
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                if (idx < x_s[0]) and (idx >= -x_s[0]):
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val)  # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be
                    # an IndexError
                    self.assertRaises(IndexError, f, x_val)
                    self.assertRaises(IndexError, g, x_val)

    def test_scalar2(self):
        # var[::-1][int] -> var[-1]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        f = function([x, y], x[::-1][y], mode=mode_opt)
        g = function([x, y], x[::-1][y],
                     mode=mode_opt.excluding('local_subtensor_merge'))
        # theano.printing.debugprint(f, print_type=True)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        topo = f.maker.fgraph.toposort()
        # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo
                    if isinstance(t.op, tensor.Subtensor)]) == 1
        # print topo[-1].op
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in xrange(-x_s[0], x_s[0]):
                f(x_val, idx)  # let debugmode test something
            for idx in (list(range(x_s[0], 9)) + list(range(-9, -x_s[0]))):
                self.assertRaises(IndexError, f, x_val, idx)
                self.assertRaises(IndexError, g, x_val, idx)

    def test_const3(self):
        # var[::-1][:const] -> var[-1]
        x = tensor.matrix('x')
        for idx in xrange(-9, 8):
            f = function([x], x[::-1][:idx], mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

            # theano.printing.debugprint(f, print_type=True)
            topo = f.maker.fgraph.toposort()
            # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo
                        if isinstance(t.op, tensor.Subtensor)]) == 1
            # print topo[-1].op
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                f(x_val)  # let debugmode test something

    def test_scalar3(self):
        # var[::-1][:int] -> var[-1]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        f = function([x, y], x[::-1][:y], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        # theano.printing.debugprint(f, print_type=True)

        topo = f.maker.fgraph.toposort()
        # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo
                    if isinstance(t.op, tensor.Subtensor)]) == 1
        # print topo[-1].op
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx in xrange(-7, 7):
                f(x_val, idx)  # let debugmode test something

    def test_const4(self):
        # var[const1::][:const2]
        x = tensor.matrix('x')
        for idx1 in xrange(-7, 7):
            for idx2 in xrange(-7, 7):
                f = function([x], x[idx1:][:idx2], mode=mode_opt)

                # Check stacktrace was copied over correctly after opt was applied
                self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

                # theano.printing.debugprint(f, print_type=True)
                topo = f.maker.fgraph.toposort()
                # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
                assert len([t for t in topo
                            if isinstance(t.op, tensor.Subtensor)]) == 1
                # print topo[-1].op
                assert isinstance(topo[-1].op, DeepCopyOp)

                for x_s in self.x_shapes:
                    x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                    f(x_val)  # let debugmode test something

    def test_scalar4(self):
        # var[int1:][:int2]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        z = tensor.iscalar('y')
        f = function([x, y, z], x[y:][:z], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        # theano.printing.debugprint(f, print_type=True)

        topo = f.maker.fgraph.toposort()
        # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo
                    if isinstance(t.op, tensor.Subtensor)]) == 1
        # print topo[-1].op
        assert isinstance(topo[-1].op, DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx1 in xrange(-11, 11):
                for idx2 in xrange(-11, 11):
                    f(x_val, idx1, idx2)  # let debugmode test something

    def test_const_general(self):
        # Some cases of merge: shape, (start, stop, step) of first,
        # (start, stop, step) of second subtensor
        cases = [
            ((2, 3), (None, None, None), (None, None, -1)),
            ((12, 1), (None, None, -4), (None, None, 1)),
            ((5, 3), (1, 4, 2), (None, None, -1)),
        ]
        x = tensor.matrix('x')

        for shape, sl1, sl2 in cases:
            z = x[slice(*sl1)][slice(*sl2)]
            f = function([x], z, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

            x_val = self.rng.uniform(size=shape).astype(config.floatX)
            f(x_val)

    def test_scalar5(self):
        # General case with two real slices
        # var[b1:e1:s1][b2:e2:s2]
        x = tensor.matrix('x')
        b1 = tensor.iscalar('b1')
        e1 = tensor.iscalar('e1')
        s1 = tensor.iscalar('s1')
        b2 = tensor.iscalar('b2')
        e2 = tensor.iscalar('e2')
        s2 = tensor.iscalar('s2')
        f = function([x, b1, e1, s1, b2, e2, s2], x[b1:e1:s1][b2:e2:s2],
                     mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        # theano.printing.debugprint(f, print_type=True)

        topo = f.maker.fgraph.toposort()
        # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.
                                                  Subtensor)]) == 1
        # print topo[-1].op
        assert isinstance(topo[-1].op, DeepCopyOp)

        b1r = self.rng.permutation(list(range(-8, 8)))[:2]
        e1r = self.rng.permutation(list(range(-8, 8)))[:2]
        b2r = self.rng.permutation(list(range(-8, 8)))[:2]
        e2r = self.rng.permutation(list(range(-8, 8)))[:2]

        s1r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1,
                                    2, 3, 4, 5, 6, 7])[:2]
        s2r = self.rng.permutation([-7, -6, -5, -4, -3, -2, -1, 1,
                                    2, 3, 4, 5, 6, 7])[:2]

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b1 in b1r:
                for e1 in e1r:
                    for s1 in s1r:
                        for b2 in b2r:
                            for e2 in e2r:
                                for s2 in s2r:
                                    f(x_val, b1, e1, s1, b2, e2, s2)

    def test_const5(self):
        # Bug reported by Razvan
        data = np.asarray(np.arange(8),
                          dtype=theano.config.floatX)
        x = theano.tensor.vector('x')
        y = x[7:1:-1]
        t = theano.shared(np.int64(0))

        fun = theano.function([x], y[t])

        val = fun(data)
        assert val == data[7:1:-1][0]

    def test_const6(self):
        # Bug reported by Graham
        data = self.rng.uniform(size=(8, 8, 8)).astype(theano.config.floatX)
        x = theano.tensor.tensor3('x')

        nops = 1
        if theano.config.mode == "FAST_COMPILE":
            nops = 2

        # test 1)
        y = x[3:6, 2:6, 1:7][1]
        fun = theano.function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2:6, 1:7][1])
        assert len([n for n in fun.maker.fgraph.toposort()
                    if isinstance(n.op, Subtensor)]) == nops

        # test 2)
        y = x[2, 3][1]
        fun = theano.function([x], y)
        val = fun(data)
        assert np.all(val == data[2, 3][1])
        assert len([n for n in fun.maker.fgraph.toposort()
                    if isinstance(n.op, Subtensor)]) == nops

        # test 3)
        y = x[3:6, 2, 1:7][1]
        fun = theano.function([x], y)
        val = fun(data)
        assert np.all(val == data[3:6, 2, 1:7][1])
        assert len([n for n in fun.maker.fgraph.toposort()
                    if isinstance(n.op, Subtensor)]) == nops

    def test_scalar6(self):
        # General case with one slice and one index
        # var[b:e:s][i]
        x = tensor.matrix('x')
        b = tensor.iscalar('b')
        e = tensor.iscalar('e')
        s = tensor.iscalar('s')
        i = tensor.iscalar('i')
        f = function([x, b, e, s, i], x[b:e:s][i], mode=mode_opt)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

        # theano.printing.debugprint(f, print_type=True)

        topo = f.maker.fgraph.toposort()
        # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.
                                                  Subtensor)]) == 1
        # print topo[-1].op
        assert isinstance(topo[-1].op, DeepCopyOp)

        b_r = self.rng.permutation(list(range(-4, 4)))[:3]
        e_r = self.rng.permutation(list(range(-4, 4)))[:3]
        i_r = self.rng.permutation(list(range(-4, 4)))[:3]

        s_r = self.rng.permutation([-3, -2, -1, 1, 2, 3])[:3]

        for x_s in self.x_shapes:
            n_index_err = 0
            n_ok = 0
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b_v in b_r:
                for e_v in e_r:
                    for s_v in s_r:
                        for i_v in i_r:
                            # The index could be out of bounds
                            # In that case, an Exception should be raised,
                            # otherwise, we let DebugMode check f
                            try:
                                x_val[b_v:e_v:s_v][i_v]
                            except IndexError:
                                n_index_err += 1
                                self.assertRaises(IndexError,
                                                  f, x_val, b_v, e_v, s_v, i_v)
                            else:
                                # Executed if the "try" clause did not raise
                                # any exception
                                n_ok += 1
                                f(x_val, b_v, e_v, s_v, i_v)

            # print 'shape: %s' % (x_s,)
            # print '%% OK: %f' % (float(n_ok) * 100 / (n_ok + n_index_err))

    @attr('slow')
    def test_none_slice(self):
        # Test case of two slices, var[b1:e1:s1][b2:e2:s2]
        # where any of the b, e, and s can be None
        x = tensor.matrix('x')
        b1 = tensor.iscalar('b1')
        e1 = tensor.iscalar('e1')
        s1 = tensor.iscalar('s1')
        b2 = tensor.iscalar('b2')
        e2 = tensor.iscalar('e2')
        s2 = tensor.iscalar('s2')

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is a Theano scalar.
        none_positions = np.ndindex(2, 2, 2, 2, 2, 2)

        # Ranges to be used when not None
        b1r = self.rng.permutation(list(range(-4, 4)))[:]
        e1r = self.rng.permutation(list(range(-4, 4)))[:]
        b2r = self.rng.permutation(list(range(-4, 4)))[:]
        e2r = self.rng.permutation(list(range(-4, 4)))[:]
        s1r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]
        s2r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b1, e1, s1, b2, e2, s2]
        scalar_ranges = [b1r, e1r, s1r, b2r, e2r, s2r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar4
                continue

            for i, none_i in enumerate(none_pos):
                if none_i:
                    slice_inputs.append(None)
                else:
                    slice_inputs.append(scalar_vars[i])
                    input_vars.append(scalar_vars[i])
                    values.append(scalar_ranges[i])

            slice1 = slice(*slice_inputs[:3])
            slice2 = slice(*slice_inputs[3:])
            sub_x = x[slice1][slice2]
            f = theano.function([x] + input_vars, sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            # for some cases, the optimization may remove all Subtensors,
            # which is why we pass "bug_print='ignore'".
            self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor,
                                              bug_print='ignore'))

            topo = f.maker.fgraph.toposort()
            # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo if isinstance(t.op,
                                                      tensor.Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values):
                    f(x_val, *i_val)

    def test_none_index(self):
        # Test the general case of indexing into a subvector,
        # like x[b:e:s][i], where any of b, e, and s can be None
        x = tensor.matrix('x')
        b = tensor.iscalar('b')
        e = tensor.iscalar('e')
        s = tensor.iscalar('s')
        i = tensor.iscalar('i')

        # Generate all possible lists of positions for None in those 6 slots
        # A 1 indicates None is present, 0 that there is a Theano scalar.
        # The last index (i) is never None
        none_positions = np.ndindex(2, 2, 2, 1)

        # Ranges to be used when not None
        b_r = self.rng.permutation(list(range(-4, 4)))[:]
        e_r = self.rng.permutation(list(range(-4, 4)))[:]
        i_r = self.rng.permutation(list(range(-4, 4)))[:]
        s_r = self.rng.permutation([-4, -3, -2, -1, 1, 2, 3, 4])[:]

        scalar_vars = [b, e, s, i]
        scalar_ranges = [b_r, e_r, s_r, i_r]

        # For each case, we will build a graph, function, and list of values
        # Then, we test it on each input shape.
        for none_pos in none_positions:
            slice_inputs = []
            input_vars = []
            values = []
            if sum(none_pos) == 0:
                # Those case are already tested in test_scalar6
                continue

            for j, none_j in enumerate(none_pos):
                if none_j:
                    slice_inputs.append(None)

                else:
                    slice_inputs.append(scalar_vars[j])
                    input_vars.append(scalar_vars[j])
                    values.append(scalar_ranges[j])

            symbol_slice = slice(*slice_inputs[:3])
            sub_x = x[symbol_slice][i]
            f = theano.function([x] + input_vars, sub_x, mode=mode_opt)

            # Check stacktrace was copied over correctly after opt was applied
            self.assertTrue(check_stack_trace(f, ops_to_check=Subtensor))

            topo = f.maker.fgraph.toposort()
            # print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo if isinstance(t.op,
                                                      tensor.Subtensor)]) <= 1
            assert isinstance(topo[-1].op, DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                for i_val in zip(*values):
                    # The index could be out of bounds
                    # In that case, an Exception should be raised,
                    # otherwise, we let DebugMode check f
                    # For that, we need to create a numerical slice.
                    i_val_idx = 0
                    num_slice_inputs = []
                    for none_j in none_pos:
                        if none_j:
                            num_slice_inputs.append(None)
                        else:
                            num_slice_inputs.append(i_val[i_val_idx])
                            i_val_idx += 1
                    num_slice = slice(*num_slice_inputs[:3])
                    num_i = num_slice_inputs[3]

                    try:
                        x_val[num_slice][num_i]
                    except IndexError:
                        self.assertRaises(IndexError, f, x_val, *i_val)
                    else:
                        # Executed if the "try" clause did not raise
                        # any exception
                        f(x_val, *i_val)


class test_local_adv_sub1_adv_inc_sub1(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        mode = theano.compile.mode.get_default_mode()
        self.mode = mode.including("local_adv_sub1_adv_inc_sub1").excluding("fusion")
        self.mode_no_assert = self.mode.including("local_remove_all_assert")

    def test0(self):
        for dtype1, dtype2 in [("float32", "float32"),
                               ("float32", "float64"),
                               ("float64", "float32"),
                               ("float64", "float64")]:
            x = tensor.matrix(dtype=dtype1)
            y = tensor.matrix(dtype=dtype2)
            idx = tensor.ivector()

            dx = np.random.rand(4, 5).astype(dtype1)
            dy = np.random.rand(2, 5).astype(dtype2)
            # Duplicate the last row of dy
            dy = np.vstack([dy, dy[-1:]])
            # Use the same index twice, with the same corresponding value.
            # That makes set_subtensor well-defined, and tests
            # duplication for inc_subtensor.
            didx = np.asarray([1, 3, 3], "int32")

            # set_subtensor
            inc = tensor.set_subtensor(x[idx], y)
            o = inc[idx]
            f = theano.function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(dy, res)
            topo = f.maker.fgraph.toposort()
            assert len(topo) == 1
            assert isinstance(topo[0].op, (compile.DeepCopyOp, T.Elemwise))

            # inc_subtensor(data[idx], y)
            inc = tensor.inc_subtensor(x[idx], y)
            o = inc[idx]
            f = theano.function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            _dx = dx.copy()
            np.add.at(_dx, didx, dy)
            utt.assert_allclose(_dx[didx], res)
            topo = f.maker.fgraph.toposort()
            len(topo) == 2

            # inc_subtensor(0[idx], y)
            inc = tensor.inc_subtensor(x.zeros_like()[idx], y)
            o = inc[idx]
            f = theano.function([x, y, idx], o, self.mode_no_assert)

            res = f(dx, dy, didx)
            utt.assert_allclose(np.vstack([dy[0], 2 * dy[1], 2 * dy[2]]), res)

    def test_assert(self):
            x = tensor.matrix("x")
            y = tensor.matrix("y")
            idx = tensor.ivector()

            dx = np.random.rand(4, 5).astype(config.floatX)
            dy = np.random.rand(2, 5).astype(config.floatX)

            # set_subtensor
            inc = tensor.set_subtensor(x[idx], y)
            o = inc[idx]
            f = theano.function([x, y, idx], o, self.mode)
            # test wrong index
            for i in [dx.shape[0], -dx.shape[0] - 1]:
                self.assertRaises((AssertionError, IndexError),
                                  f, dx, dy, [i, i])
            # test wrong shape
            self.assertRaises((AssertionError, ValueError),
                              f, dx, dy, [1])

    def test_stack_trace(self):
        x = tensor.matrix("x")
        # test cases with y.dtype
        # - equal to x.dtype
        # - different from x.dtype (to trigger the cast in
        #   local_adv_sub1_adv_inc_sub1)
        ys = [tensor.matrix("y"), tensor.dmatrix("y")]
        idx = tensor.ivector()

        # set_subtensor and then subtensor with both ys
        incs = [tensor.set_subtensor(x[idx], y) for y in ys]
        outs = [inc[idx] for inc in incs]

        for y, out in zip(ys, outs):
            f = theano.function([x, y, idx], out, self.mode)
            self.assertTrue(check_stack_trace(
                f, ops_to_check=(Assert, scal.Cast)))


class Test_alloc_zero(unittest.TestCase):
    def setUp(self):
        mode = theano.compile.mode.get_default_mode()
        self.mode = mode.including("local_incsubtensor_of_zeros",
                                   "local_setsubtensor_of_constants",
                                   "local_0_dot_x")

    def test_setsubtensor_allocs0(self):
        x = tensor.matrix()
        y = tensor.matrix()
        x0 = tensor.zeros_like(x)
        y0 = tensor.zeros_like(y)
        z = tensor.set_subtensor(x0[:4], y0)
        f = theano.function([x, y], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.IncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_setsubtensor_allocs1(self):
        y = tensor.matrix()
        x0 = tensor.constant(np.asarray(np.zeros((4, 4)),
                                        dtype=config.floatX))
        y0 = tensor.zeros_like(y)
        z = tensor.set_subtensor(x0[:4], y0)
        f = theano.function([y], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.IncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_setsubtensor_allocs1t(self):
        y = tensor.matrix()
        x0 = tensor.constant(np.asarray(np.zeros((4, 4)),
                                        dtype=config.floatX))
        y0 = tensor.zeros_like(y)
        z = tensor.set_subtensor(x0[:4], y0.T)
        f = theano.function([y], z, mode=mode_opt)
        assert np.all([not isinstance(n.op, tensor.IncSubtensor)
                      for n in f.maker.fgraph.toposort()])

    def test_setsubtensor_allocs2(self):
        x = tensor.matrix()
        y0 = tensor.constant(np.asarray(np.zeros_like((4, 4)),
                                        dtype=config.floatX))
        x0 = tensor.zeros_like(x)
        z = tensor.set_subtensor(x0[:4], y0)
        f = theano.function([x], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.IncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_incsubtensor_allocs0(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z = tensor.inc_subtensor(x[:4], y0)
        f = theano.function([x, y], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.IncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_incsubtensor_allocs0t(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z = tensor.inc_subtensor(x[:4], y0.T)
        f = theano.function([x, y], z, mode=mode_opt)
        assert np.all([not isinstance(n.op, tensor.IncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_incsubtensor_allocs1(self):
        x = tensor.matrix()
        y0 = tensor.constant(np.asarray(np.zeros_like((4, 4)),
                                        dtype=config.floatX))
        z = tensor.inc_subtensor(x[:4], y0)
        f = theano.function([x], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.IncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_incsubtensor_x_zeros(self):
        x = tensor.constant(np.asarray(np.zeros((4, 4)),
                                       dtype=config.floatX))
        y = tensor.matrix()
        z = tensor.inc_subtensor(x[:4], y)
        f = theano.function([y], z)
        inc_nodes = [n for n in f.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.IncSubtensor)]

        assert(len(inc_nodes) == 1)
        node_is_set_instead_of_inc = inc_nodes[0].op.set_instead_of_inc
        mode = theano.config.mode
        assert((mode != "FAST_COMPILE" and node_is_set_instead_of_inc) or
               (mode == "FAST_COMPILE" and not node_is_set_instead_of_inc))
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X)

        # also check the flag doesn't get set if first input is not zeros:
        not_all_zeros = np.zeros((4, 4))
        not_all_zeros[1, 0] = 0.001
        x = tensor.constant(np.asarray(not_all_zeros, dtype=config.floatX))
        y = tensor.matrix()
        z = tensor.inc_subtensor(x[:4], y)
        f = theano.function([y], z)
        inc_nodes = [n for n in f.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.IncSubtensor)]
        assert(len(inc_nodes) == 1)
        assert(inc_nodes[0].op.set_instead_of_inc is False)
        test_X = np.random.random((4, 4)).astype(config.floatX)
        utt.assert_allclose(f(test_X), test_X + not_all_zeros)

    def test_advancedincsubtensor1_allocs0(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z = tensor.inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = theano.function([x, y], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.AdvancedIncSubtensor1)
                       for n in f.maker.fgraph.toposort()])

    def test_advancedincsubtensor1_allocs0t(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z = tensor.inc_subtensor(x[[0, 1, 2, 3]], y0.T)
        f = theano.function([x, y], z, mode=mode_opt)
        assert np.all([not isinstance(n.op, tensor.AdvancedIncSubtensor1)
                       for n in f.maker.fgraph.toposort()])

    def test_advancedincsubtensor1_allocs1(self):
        x = tensor.matrix()
        y0 = tensor.constant(np.asarray(np.zeros_like((4, 4)),
                                        dtype=config.floatX))
        z = tensor.inc_subtensor(x[[0, 1, 2, 3]], y0)
        f = theano.function([x], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.AdvancedIncSubtensor1)
                       for n in f.maker.fgraph.toposort()])

    def test_advancedincsubtensor_allocs0(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z = tensor.inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = theano.function([x, y], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.AdvancedIncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_advancedincsubtensor_allocs0t(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z = tensor.inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0.T)
        f = theano.function([x, y], z, mode=mode_opt)
        assert np.all([not isinstance(n.op, tensor.AdvancedIncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_advancedincsubtensor_allocs1(self):
        x = tensor.matrix()
        y0 = tensor.constant(np.asarray(np.zeros_like((2, 2)),
                                        dtype=config.floatX))
        z = tensor.inc_subtensor(x[[[0, 0], [1, 1]], [[0, 1], [0, 1]]], y0)
        f = theano.function([x], z, mode=self.mode)
        assert np.all([not isinstance(n.op, tensor.AdvancedIncSubtensor)
                       for n in f.maker.fgraph.toposort()])

    def test_dot_allocs_0(self):
        v1 = tensor.vector('v1')
        v2 = tensor.vector('v2')
        m1 = tensor.matrix('m1')
        m2 = tensor.matrix('m2')
        vv2 = np.asarray([0, 1], dtype=theano.config.floatX)
        vm2 = np.asarray([[1, 2], [4, 5]],
                         dtype=theano.config.floatX)
        vv3 = np.asarray([0, 1, 2], dtype=theano.config.floatX)
        vm3 = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                         dtype=theano.config.floatX)
        for _e1 in [(v1, vv2, vv3), (m1, vm2, vm3)]:
            for _e2 in [(v2, vv2, vv3), (m2, vm2, vm3)]:
                for p in [0, 1]:
                    if p == 0:
                        e1 = tensor.zeros_like(_e1[0])
                        e2 = _e2[0]
                    else:
                        e1 = _e1[0]
                        e2 = tensor.zeros_like(_e2[0])
                    o = tensor.dot(e1, e2)
                    f = theano.function([_e1[0], _e2[0]], o, mode=self.mode)
                    f(_e1[1], _e2[1])
                    f(_e1[2], _e2[2])
                    assert np.all([not isinstance(n.op, tensor.Dot) for n in
                                   f.maker.fgraph.toposort()])

                    # test that we don't remove shape errors
                    self.assertRaises((ValueError, AssertionError), f,
                                      _e1[1], _e2[2])
                    self.assertRaises((ValueError, AssertionError), f,
                                      _e1[2], _e2[1])


def test_local_IncSubtensor_serialize():
    d = np.random.normal(0, 0.01, size=(100, 100))
    d = d.astype(theano.config.floatX)

    W = theano.shared(d, name='W')
    i = T.vector('i', dtype='int64')
    j = T.vector('j', dtype='int64')
    t = T.scalar('t')
    y = (W[i] + W[j] + W[1] + W[i, j]).sum()
    cost = T.sqr(t - y)
    dW = theano.grad(cost, W)
    mode = theano.compile.mode.get_default_mode().excluding('fusion')
    mode = mode.including("local_IncSubtensor_serialize")
    f = theano.function([i, j, t], updates=[(W, W - 0.01 * dW)], mode=mode)
    topo = f.maker.fgraph.toposort()
    adds = [n for n in topo if isinstance(n.op, T.Elemwise) and
            isinstance(n.op.scalar_op, theano.scalar.Add)]
    for a in adds:
        assert not any([inp.owner and
                        isinstance(inp.owner.op,
                                   (tensor.IncSubtensor,
                                    tensor.AdvancedIncSubtensor,
                                    tensor.AdvancedIncSubtensor1))
                        for inp in a.inputs])

    # Now test that the stack trace is copied over properly,
    # if we return the gradients. We need to use same mode as before.
    f = theano.function([i, j, t], dW, mode=mode)
    assert check_stack_trace(f, ops_to_check=[
        tensor.IncSubtensor, tensor.AdvancedIncSubtensor,
        tensor.AdvancedIncSubtensor1])


def test_local_set_to_inc_subtensor():
    v = theano.tensor.fmatrix()
    s = v[[2, 1]]
    g = s + 3
    r = theano.tensor.set_subtensor(s, g)
    moder = compile.get_default_mode().excluding('local_set_to_inc_subtensor')
    modet = compile.get_default_mode().including('local_set_to_inc_subtensor')
    f1 = theano.function([v], r, mode=moder)
    f2 = theano.function([v], r, mode=modet)

    advi1 = [n for n in f1.maker.fgraph.toposort()
             if isinstance(n.op, tensor.AdvancedIncSubtensor1)]

    advi2 = [n for n in f2.maker.fgraph.toposort()
             if isinstance(n.op, tensor.AdvancedIncSubtensor1)]

    # We only have SetSubtensor in f1
    assert all(n.op.set_instead_of_inc for n in advi1)
    # We don't have any SetSubtensor in f2
    assert all(not n.op.set_instead_of_inc for n in advi2)

    val = np.random.randn(3, 2).astype('float32')

    r1 = f1(val)
    r2 = f2(val)

    utt.assert_allclose(r1, r2)

    # Finally, test that the stack trace is copied over properly,
    # before and after optimization.
    assert check_stack_trace(f1, ops_to_check=tensor.AdvancedIncSubtensor1)
    assert check_stack_trace(f2, ops_to_check='all')


def test_local_subtensor_of_dot():
    m1 = theano.tensor.matrix()
    m2 = theano.tensor.matrix()
    d1 = np.arange(6).reshape((3, 2)).astype(config.floatX)
    d2 = np.arange(8).reshape((2, 4)).astype(config.floatX) + 10
    mode = compile.get_default_mode().including("local_subtensor_of_dot")

    def test_equality(a, b):
        return a.shape == b.shape and np.allclose(a, b)

    # [cst]
    f = theano.function([m1, m2], theano.dot(m1, m2)[1], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert test_equality(f(d1, d2), np.dot(d1, d2)[1])
    # DimShuffle happen in FAST_COMPILE
    assert isinstance(topo[-1].op, (T.blas_c.CGemv, T.blas.Gemv, T.DimShuffle))

    # slice
    f = theano.function([m1, m2], theano.dot(m1, m2)[1:2], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert test_equality(f(d1, d2), np.dot(d1, d2)[1:2])
    assert isinstance(topo[-1].op, (T.blas.Dot22))

    m1 = theano.tensor.tensor3()
    m2 = theano.tensor.tensor3()
    idx = theano.tensor.iscalar()
    d1 = np.arange(30).reshape(2, 5, 3).astype(config.floatX)
    d2 = np.arange(72).reshape(4, 3, 6).astype(config.floatX) + 100

    f = theano.function([m1, m2, idx], theano.dot(m1, m2)[idx, 1:4, :, idx:], mode=mode)
    assert test_equality(f(d1, d2, 1), np.dot(d1, d2)[1, 1:4, :, 1:])
    # if we return the gradients. We need to use same mode as before.
    assert check_stack_trace(f, ops_to_check='last')

    f = theano.function([m1, m2, idx], theano.dot(m1, m2)[1:4, :, idx:, idx], mode=mode)
    assert test_equality(f(d1, d2, 1), np.dot(d1, d2)[1:4, :, 1:, 1])

    # Now test that the stack trace is copied over properly,
    # if we return the gradients. We need to use same mode as before.
    assert check_stack_trace(f, ops_to_check='last')


class Test_local_elemwise_alloc(unittest.TestCase):
    dtype = config.floatX

    def setUp(self):
        self.fast_compile_mode = get_mode('FAST_COMPILE')
        self.fast_run_mode = get_mode('FAST_RUN')

        self.vec = T.vector('vec', dtype=self.dtype)
        self.mat = T.matrix('mat', dtype=self.dtype)
        self.tens = T.tensor3('tens', dtype=self.dtype)

        self.alloc_wo_dep = T.alloc(self.vec, 2, 2)
        self.alloc_wo_dep_broad = T.alloc(self.vec, 1, 2)
        self.alloc_w_dep = T.alloc(self.vec, *self.mat.shape)
        self.alloc_w_dep_broad = T.alloc(self.vec, 1, *self.mat.shape)
        self.alloc_w_dep_broad2 = T.alloc(self.vec, self.mat.shape[0],
                                          self.mat.shape[1], 1)
        self.alloc_w_dep_tens = T.alloc(
            self.vec,
            self.tens.shape[0],
            self.tens.shape[1]
        )
        self.tv_wo_dep = T.alloc(self.vec, 5, 5)
        self.tm_wo_dep = T.alloc(self.mat, 5, 5, 5)
        self.s = T.iscalar('s')
        self.tv_w_dep = T.alloc(self.vec, self.s, self.s)
        self.tm_w_dep = T.alloc(self.mat, 5, 5, 5)
        self.row = theano.tensor.row(dtype=self.dtype)
        self.o = T.alloc(self.row, 5, 5)

    def _verify_alloc_count(self, f, count):
        assert(
            sum([isinstance(elem.op, T.Alloc)
                 for elem in f.maker.fgraph.toposort()
                 if elem.op is not None]) == count
        )

    def _verify_assert_count(self, f, count):
        assert(
            sum([isinstance(elem.op, T.opt.Assert)
                 for elem in f.maker.fgraph.toposort()
                 if elem.op is not None]) == count
        )

    def test_remove_alloc_wo_dimshuffle(self):
        # Exclude local_useless_alloc, since it does not introduce
        # assert in all the same cases.
        self.fast_run_mode = self.fast_run_mode.excluding(
            'local_useless_alloc', 'local_canonicalize_alloc')
        # No optimization on alloc
        func = function(
            [self.vec, self.mat],
            self.alloc_wo_dep + self.mat,
            mode=self.fast_compile_mode
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(func, ops_to_check='all'))

        # Optimization on alloc with assert
        func = function(
            [self.vec, self.mat],
            self.alloc_wo_dep + self.mat,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)

        # Optimization on alloc with assert and broadcast
        func = function(
            [self.vec, self.mat],
            self.alloc_wo_dep_broad + self.mat,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)

        # No optimization on alloc without assert
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep + self.mat,
            mode=self.fast_compile_mode
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # Optimization on alloc without assert
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep + self. mat,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 0)

        # Optimization on alloc without assert and with broadcast
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep_broad + self. mat,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 0)

        # Not optimized case on alloc and with broadcast
        func = function(
            [self.vec, self.mat],
            self.alloc_w_dep_broad2 + self. mat,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

    def test_remove_alloc_w_dimshuffle(self):
        # No optimization on dimshuffle with assert
        func = function(
            [self.vec, self.tens],
            self.alloc_wo_dep.dimshuffle(0, 1, 'x') + self.tens,
            mode=self.fast_compile_mode
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle with assert
        func = function(
            [self.vec, self.tens],
            self.alloc_wo_dep.dimshuffle(0, 1, 'x') + self.tens,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)

        # No optimization on dimshuffle without assert
        func = function(
            [self.vec, self.tens],
            self.alloc_w_dep_tens.dimshuffle(0, 1, 'x') + self.tens,
            mode=self.fast_compile_mode
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle without assert
        func = function(
            [self.vec, self.tens],
            self.alloc_w_dep_tens.dimshuffle(0, 1, 'x') + self.tens,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 0)

    def test_multi_input_single_alloc(self):
        # No optimization on dimshuffle with assert
        func = function(
            [self.vec, self.mat],
            self.tv_wo_dep + self.tm_wo_dep,
            mode=self.fast_compile_mode
        )
        self._verify_alloc_count(func, 2)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle with assert
        func = function(
            [self.vec, self.mat],
            self.tv_wo_dep + self.tm_wo_dep,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 0)

        # No optimization on dimshuffle without assert
        func = function(
            [self.vec, self.mat, self.s],
            self.tv_w_dep + self.tm_w_dep,
            mode=self.fast_compile_mode
        )
        self._verify_alloc_count(func, 2)
        self._verify_assert_count(func, 0)

        # Optimization on dimshuffle without assert
        func = function(
            [self.vec, self.mat, self.s],
            self.tv_w_dep + self.tm_w_dep,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 1)
        self._verify_assert_count(func, 1)

    def test_error(self):
        t3fft = theano.tensor.tensor(dtype=self.dtype,
                                     broadcastable=(False, False, True))
        o = self.o.dimshuffle(0, 1, 'x') + t3fft
        func = function(
            [t3fft, self.row],
            o,
            mode=self.fast_run_mode
        )
        self._verify_alloc_count(func, 0)
        self._verify_assert_count(func, 1)
        d = np.random.rand(5, 5, 1).astype(self.dtype)
        r = np.random.rand(1, 5).astype(self.dtype)
        func(d, r)


def test_local_subtensor_of_alloc():

    # DebugMode should detect if something goes wrong.
    # test shape combination of odd and event shape.
    for shape in [(3, 5), (4, 6), (3, 8), (4, 7),
                  (1, 5), (5, 1)]:
        x = tensor.tensor(dtype=theano.config.floatX,
                          broadcastable=(shape[0] == 1, shape[1] == 1))

        xval = np.zeros(shape, dtype=config.floatX)
        yval = np.arange(shape[1], dtype=config.floatX)

        for y in [theano.shared(yval), tensor.constant([1.])]:

            # The rows of yx are copies of y
            yx = tensor.alloc(y, x.shape[0], x.shape[1])

            # Slice of each row
            z_mat = yx[:, 3:]
            assert z_mat.ndim == 2

            # Only one column
            z_vec = yx[:, 3]
            assert z_vec.ndim == 1
            # results are vector
            slicess = []
            if shape[0] != 1:
                slicess.append((2, slice(None)))
            if shape[1] != 1:
                slicess.append((slice(None), 3))

            # results are matrix
            slicess += [
                (slice(None), slice(3, None)),
                (slice(3, None), ),
                (slice(3, None), slice(3, None)),
                (slice(1, 3), slice(None, -1)),
                (slice(None, None, 2)),
                (slice(1, None, 2)),
            ]
            for slices in slicess:
                z = yx.__getitem__(slices)
                f = theano.function([x], z)
                if theano.config.mode != 'FAST_COMPILE':
                    # Subtensor can be in the input of Alloc
                    assert not isinstance(f.maker.fgraph.toposort()[-1].op,
                                          Subtensor)
                val = f(xval)
                assert xval.__getitem__(slices).shape == val.shape


def test_local_fill_useless():
    # Test opt local_fill_useless
    x = dvector()
    y = dvector()
    z = lvector()
    m = dmatrix()

    x_ = np.random.rand(5,)
    y_ = np.random.rand(5,)
    z_ = (np.random.rand(5,) * 5).astype("int64")
    m_ = np.random.rand(5, 5)

    # basic case
    f = function([x], T.fill(x, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.mul]
    f(x_)

    # basic case
    f = function([x, y], T.second(y, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.mul]
    f(x_, y_)

    # basic case
    f = function([x, y], T.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.mul]
    f(x_, y_)

    # now with different type(cast)
    f = function([x, z], T.fill(z, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.mul]
    f(x_, z_)

    # now with different type(cast)
    f = function([x, z], T.fill(x, z) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.mul]
    f(x_, z_)

    # now cutting out the input ??
    f = function([x, y], T.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.fgraph.toposort()] == [T.mul]
    f(x_, y_)

    # Test with different number of dimensions
    # The fill is not useless, so it should stay
    f = function([m, x], T.fill(m, x) * 2, mode=mode_opt)
    ops = [node.op.__class__ for node in f.maker.fgraph.toposort()]
    assert T.Alloc in ops
    f(m_, x_)


def test_local_elemwise_sub_zeros():
    # Test opt local_elemwise_sub_zeros
    # We test separately for scalars, vectors and matrices
    scalar = T.scalar()
    vect = T.vector()
    mat = T.matrix()

    rng = np.random.RandomState(seed=utt.fetch_seed())
    scalar_val = rng.rand(1).astype(config.floatX)[0]
    vect_val = rng.rand(5).astype(config.floatX)
    mat_val = rng.rand(3, 2).astype(config.floatX)

    mode = theano.compile.get_default_mode()\
        .excluding('canonicalize', 'uncanonicalize',
                   'ShapeOpt', 'local_fill_to_alloc',
                   'local_elemwise_alloc')\
        .including('local_elemwise_sub_zeros')

    # Test scalar minus scalar
    f = function([scalar], scalar - scalar, mode=mode)
    # Check optimized graph is correct
    assert isinstance(f.maker.fgraph.toposort()[0].op, T.Elemwise)
    assert isinstance(f.maker.fgraph.toposort()[0].op.scalar_op,
                      theano.scalar.Second)
    assert isinstance(f.maker.fgraph.toposort()[0].inputs[1],
                      T.TensorConstant) or\
        isinstance(f.maker.fgraph.toposort()[0].inputs[1],
                   T.TensorConstant)
    utt.assert_allclose(f(scalar_val), 0.0)
    # Check stack trace is copied over
    assert check_stack_trace(f, ops_to_check='all')

    # Test vector minus vector
    f = function([vect], vect - vect, mode=mode)
    # Check optimized graph is correct
    assert isinstance(f.maker.fgraph.toposort()[0].op, T.Elemwise)
    assert isinstance(f.maker.fgraph.toposort()[0].op.scalar_op,
                      theano.scalar.Second)
    assert isinstance(f.maker.fgraph.toposort()[0].inputs[1],
                      T.TensorConstant) or\
        isinstance(f.maker.fgraph.toposort()[0].inputs[1],
                   T.TensorConstant)
    utt.assert_allclose(f(vect_val), np.zeros(vect_val.shape))
    # Check stack trace is copied over
    assert check_stack_trace(f, ops_to_check='all')

    # Test vector minus vector
    f = function([mat], mat - mat, mode=mode)
    # Check optimized graph is correct
    assert isinstance(f.maker.fgraph.toposort()[0].op, T.Elemwise)
    assert isinstance(f.maker.fgraph.toposort()[0].op.scalar_op,
                      theano.scalar.Second)
    assert isinstance(f.maker.fgraph.toposort()[0].inputs[1],
                      T.TensorConstant) or\
        isinstance(f.maker.fgraph.toposort()[0].inputs[1],
                   T.TensorConstant)
    utt.assert_allclose(f(mat_val), np.zeros(mat_val.shape))
    # Check stack trace is copied over
    assert check_stack_trace(f, ops_to_check='all')


class Test_local_useless_elemwise_comparison(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    def test_local_useless_elemwise_comparison(self):
        # TODO: test each case individually.
        # The following case is what made me discover those cases.
        X = T.matrix('X')
        Y = T.vector('Y')
        X_sum, updates = theano.scan(fn=lambda x: x.sum(),
                                     outputs_info=None,
                                     sequences=[X],
                                     non_sequences=None)
        Z = X_sum + Y
        # theano.printing.debugprint(Z)
        # here is the output for the debug print:
        """
        Elemwise{add,no_inplace} [id A] ''
         |for{cpu,scan_fn} [id B] ''
         | |Subtensor{int64} [id C] ''
         | | |Shape [id D] ''
         | | | |Subtensor{int64::} [id E] 'X[0:]'
         | | |   |X [id F]
         | | |   |Constant{0} [id G]
         | | |Constant{0} [id H]
         | |Subtensor{:int64:} [id I] ''
         | | |Subtensor{int64::} [id E] 'X[0:]'
         | | |ScalarFromTensor [id J] ''
         | |   |Subtensor{int64} [id C] ''
         | |Subtensor{int64} [id C] ''
         |Y [id K]

        Inner graphs of the scan ops:

        for{cpu,scan_fn} [id B] ''
         >Sum{acc_dtype=float64} [id L] ''
         > |X[t] [id M] -> [id I]
        """

        mode = theano.compile.get_default_mode().excluding('fusion')
        f = theano.function([X, Y], Z, mode=mode)
        f(self.rng.rand(2, 3).astype(config.floatX),
          self.rng.rand(2).astype(config.floatX))
        # theano.printing.debugprint(f, print_type=True)
        # here is the output for the debug print:
        """
        Elemwise{Add}[(0, 0)] [id A] <TensorType(float64, vector)> ''   7
         |for{cpu,scan_fn} [id B] <TensorType(float64, vector)> ''   6
         | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         | | |X [id D] <TensorType(float64, matrix)>
         | |Subtensor{int64:int64:int8} [id E] <TensorType(float64, matrix)> ''   5
         | | |X [id D] <TensorType(float64, matrix)>
         | | |ScalarFromTensor [id F] <int64> ''   4
         | | | |Elemwise{switch,no_inplace} [id G] <TensorType(int64, scalar)> ''   3
         | | |   |Elemwise{le,no_inplace} [id H] <TensorType(int8, scalar)> ''   2
         | | |   | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         | | |   | |TensorConstant{0} [id I] <TensorType(int8, scalar)>
         | | |   |TensorConstant{0} [id I] <TensorType(int8, scalar)>
         | | |   |TensorConstant{0} [id J] <TensorType(int64, scalar)>
         | | |ScalarFromTensor [id K] <int64> ''   1
         | | | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         | | |Constant{1} [id L] <int8>
         | |Shape_i{0} [id C] <TensorType(int64, scalar)> ''   0
         |Y [id M] <TensorType(float64, vector)>

        Inner graphs of the scan ops:

        for{cpu,scan_fn} [id B] <TensorType(float64, vector)> ''
         >Sum{acc_dtype=float64} [id N] <TensorType(float64, scalar)> ''
         > |X[t] [id O] <TensorType(float64, vector)> -> [id E]
        """

    def assert_eqs_const(self, f, val, op=deep_copy_op):
        topo = f.maker.fgraph.toposort()
        elem = topo[0]
        assert len(topo) == 1, topo
        assert elem.op == op, elem.op
        if op == deep_copy_op:
            assert len(elem.inputs) == 1, elem.inputs
            assert isinstance(elem.inputs[0], T.TensorConstant), elem
            assert T.extract_constant(elem.inputs[0]) == val, val
        else:
            assert len(elem.inputs) == 2, elem.inputs
            assert isinstance(elem.inputs[0], T.TensorConstant), elem
            assert T.extract_constant(elem.inputs[0]) == val, val

    def assert_identity(self, f):
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        if f.outputs[0].variable.dtype == 'bool':
            x_vals = [0, 1]
        else:
            x_vals = [0, 1, 10]
        for x_val in x_vals:
            assert f(x_val) == x_val

    def test_inequality_with_self(self):
        x = T.scalar('x', dtype=config.floatX)
        mode = theano.compile.get_default_mode().including('local_useless_elemwise_comparison')

        f = theano.function([x], T.lt(x, x), mode=mode)
        self.assert_eqs_const(f, 0)

        f = theano.function([x], T.le(x, x), mode=mode)
        self.assert_eqs_const(f, 1)

        f = theano.function([x], T.gt(x, x), mode=mode)
        self.assert_eqs_const(f, 0)

        f = theano.function([x], T.ge(x, x), mode=mode)
        self.assert_eqs_const(f, 1)

        f = theano.function([x], T.minimum(x, x), mode=mode)
        self.assert_identity(f)

        f = theano.function([x], T.maximum(x, x), mode=mode)
        self.assert_identity(f)

    def test_shape_inequality_with_self(self):
        x = T.vector('x', dtype=config.floatX)
        mode = theano.compile.get_default_mode().including(
            'local_useless_elemwise_comparison',
            'local_shape_to_shape_i',
            'local_track_shape_i',
            'local_subtensor_make_vector')
        f = theano.function([x], T.lt(x.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 0)

        f = theano.function([x], T.ge(x.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 1)

        f = theano.function([x], T.maximum(x.shape[0], 0), mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Shape_i), topo[0].op
        x_val = np.ones(100, dtype=config.floatX)
        assert f(x_val) == x_val.shape[0]

        f = theano.function([x], T.maximum(0, x.shape[0]), mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, Shape_i), topo[0].op
        x_val = np.ones(100, dtype=config.floatX)
        assert f(x_val) == x_val.shape[0]

        f = theano.function([x], T.minimum(x.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 0)
        assert f(x_val) == 0

        f = theano.function([x], T.minimum(0, x.shape[0]), mode=mode)
        self.assert_eqs_const(f, 0)
        assert f(x_val) == 0
        f = theano.function([x], T.minimum([0, 0], x.shape[0]), mode=mode)
        # This case isn't optimized.
        # self.assert_eqs_const(f, 0)
        utt.assert_allclose(f(x_val), [0, 0])

    def test_shape_add_inequality(self):
        x = T.vector('x', dtype=config.floatX)
        mode = theano.compile.get_default_mode().including(
            'local_useless_elemwise_comparison',
            'local_shape_to_shape_i',
            'local_track_shape_i',
            'local_subtensor_make_vector')

        y = T.vector('y', dtype=config.floatX)

        f = theano.function([x, y], T.lt(x.shape[0] + y.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 0)

        f = theano.function([x, y], T.ge(x.shape[0] + y.shape[0], 0), mode=mode)
        self.assert_eqs_const(f, 1)

    def test_equality_shapes(self):
        # Test equality where one sides contain only shapes related
        # stuff.
        if theano.config.mode == "FAST_COMPILE":
            raise SkipTest("Skip opt test as the opt is disabled")
        x = T.vector('x', dtype=config.floatX)
        for g in [x.shape[0],
                  Shape_i(0)(x)]:
            f = theano.function([x], T.eq(g, 0))
            assert f([3, 3]) == 0
            assert f([]) == 1

            f = theano.function([x], T.eq(g, -1))
            self.assert_eqs_const(f, 0)
            assert f([3, 3]) == 0

        g = join(0,
                 x.shape[0:],  # todo test reshape, dimshuffle
                 x.shape[0:1])
        f = theano.function([x], T.eq(g, 0))
        assert (f([3, 3]) == 0).all()
        assert (f([]) == 1).all()

        f = theano.function([x], T.eq(g, -1))
        self.assert_eqs_const(f, 0, op=T.alloc)
        assert (f([3, 3]) == 0).all()

    def test_and(self):
        # bitwise "and" with 0 should give 0 for both bool and int
        # bitwise "and" with 1 should only simplify for bool
        mode = theano.compile.get_default_mode().including('canonicalize')
        for dtype, zero, one in [('bool', np.array(False), np.array(True)),
                                 ('int8', np.int8(0), np.int8(1)),
                                 ('int8', 0, 1)]:
            x = T.scalar('x', dtype=dtype)

            f = theano.function([x], T.and_(x, zero), mode=mode)
            self.assert_eqs_const(f, 0)

            f = theano.function([x], T.and_(zero, x), mode=mode)
            self.assert_eqs_const(f, 0)

            f = theano.function([x], T.and_(x, one), mode=mode)
            if dtype == 'bool':
                self.assert_identity(f)

            f = theano.function([x], T.and_(one, x), mode=mode)
            if dtype == 'bool':
                self.assert_identity(f)

    def test_and_int(self):
        # Test that bitwise "and" is correctly computed on int constants.
        f = theano.function([], T.and_(5, 6))
        assert f() == 4

    def test_or(self):
        # bitwise "or" with 0 should simplify for both bool and int
        # bitwise "or" with 1 should only give 1 for bool
        mode = theano.compile.get_default_mode().including('canonicalize')
        for dtype, zero, one in [('bool', np.array(False), np.array(True)),
                                 ('int8', np.int8(0), np.int8(1)),
                                 ('int8', 0, 1)]:
            x = T.scalar('x', dtype=dtype)

            f = theano.function([x], T.or_(x, one), mode=mode)
            if dtype == 'bool':
                self.assert_eqs_const(f, 1)

            f = theano.function([x], T.or_(one, x), mode=mode)
            if dtype == 'bool':
                self.assert_eqs_const(f, 1)

            f = theano.function([x], T.or_(x, zero), mode=mode)
            self.assert_identity(f)

            f = theano.function([x], T.or_(zero, x), mode=mode)
            self.assert_identity(f)

    def test_or_int(self):
        # Test that bitwise "or" is correctly computed on int constants.
        f = theano.function([], T.or_(5, 6))
        assert f() == 7

    def test_xor(self):
        # bitwise "xor" with itself should always give 0 for both bool and int.
        mode = theano.compile.get_default_mode().including('canonicalize')
        for dtype in ('bool', 'int8'):
            x = T.scalar('x', dtype=dtype)

            f = theano.function([x], T.xor(x, x), mode=mode)
            self.assert_eqs_const(f, 0)

    def test_stacktrace(self):
        mode = theano.compile.get_default_mode().including(
            'local_useless_elemwise_comparison')

        x = T.vector('x', dtype=config.floatX)
        f = theano.function([x], T.gt(x, x), mode=mode)
        self.assertTrue(check_stack_trace(f, ops_to_check='last'))

        f = theano.function([x], T.le(x, x), mode=mode)
        self.assertTrue(check_stack_trace(f, ops_to_check='last'))


class Test_local_canonicalize_alloc(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    @change_flags(compute_test_value='off')
    def test0(self):
        x = shared(self.rng.randn(3, 7))
        a = tensor.alloc(x, 6, 7)

        # It is a bad idea to have tensor.alloc return x directly,
        # because the shape mismatch cannot be caught.
        assert a.owner and isinstance(a.owner.op, tensor.Alloc)

        f = function([], a, mode=mode_opt)
        # The optimization should then be applied, and remove Alloc
        assert ([node.op for node in f.maker.fgraph.toposort()] ==
                [deep_copy_op])

        # In DebugMode, the shape mismatch should be detected
        if isinstance(mode_opt, compile.DebugMode):
            self.assertRaises(ValueError, f)

        # No need to check_stack_trace as the optimization
        # local_canonicalize_alloc only removes nodes.

    def test1(self):
        # Test that alloc never gets instantiated during optimization
        mode = mode_opt.excluding('local_canonicalize_alloc')

        x = tensor.matrix('x')
        xx = tensor.fill(x, x)

        # The optimization 'locall_fill_to_alloc' should call tensor.alloc,
        # which should return x and not alloc(x, ...)
        f = function([x], [xx], mode=mode)
        op_classes = [node.op.__class__ for node in f.maker.fgraph.toposort()]
        assert tensor.Alloc not in op_classes

        # No need to check_stack_trace as the optimization
        # local_canonicalize_alloc only removes nodes.

    def test2(self):
        # Test that alloc never gets instantiated during optimization
        mode = mode_opt.excluding('local_canonicalize_alloc')

        x = tensor.matrix('x')
        y = tensor.tile(x, (1,) * 2)

        f = function([x], [y], mode=mode)
        op_classes = [node.op.__class__ for node in f.maker.fgraph.toposort()]
        print(op_classes)

        # We are supposed to test if tensr.Alloc is not in op_classes,
        # but since the proper proper optimization is not currently
        # implemented it will fail. Once the correct optimization is in place,
        # we have to change the following we should not see tensor.Alloc
        # in op_classes and we have to change the assert.
        assert tensor.Alloc in op_classes
        # The correct opt removes nodes, no need for check_stack_trace

    def test_useless_alloc_with_shape_one(self):
        alloc_lift = out2in(local_canonicalize_alloc)
        x = shared(self.rng.randn(2,))
        y = shared(self.rng.randn())
        z = shared(self.rng.randn(1, 1))
        w = shared(self.rng.randn(1, 1))
        alloc_x = tensor.alloc(x, 1, 3, 2)
        alloc_y = tensor.alloc(y, 1, 1)
        alloc_z = tensor.alloc(z, 1, 1, 2)
        alloc_w = tensor.alloc(w, 1, 2)

        g = FunctionGraph([x, y, z, w], [alloc_x, alloc_y, alloc_z, alloc_w])
        self.assertTrue(str(g) == ("[Alloc(<TensorType(float64, vector)>, "
                                   "TensorConstant{1}, "
                                   "TensorConstant{3}, "
                                   "TensorConstant{2}), "

                                   "Alloc(<TensorType(float64, scalar)>, "
                                   "TensorConstant{1}, "
                                   "TensorConstant{1}), "

                                   "Alloc(<TensorType(float64, matrix)>, "
                                   "TensorConstant{1}, "
                                   "TensorConstant{1}, "
                                   "TensorConstant{2}), "

                                   "Alloc(<TensorType(float64, matrix)>, "
                                   "TensorConstant{1}, "
                                   "TensorConstant{2})]"))

        alloc_lift.optimize(g)
        self.assertTrue(str(g) == "[InplaceDimShuffle{x,0,1}"
                                  "(Alloc(<TensorType(float64, vector)>, "
                                  "TensorConstant{3}, "
                                  "TensorConstant{2})), "

                                  "InplaceDimShuffle{x,x}"
                                  "(<TensorType(float64, scalar)>), "

                                  "InplaceDimShuffle{x,0,1}"
                                  "(Alloc(<TensorType(float64, matrix)>, "
                                  "TensorConstant{1}, "
                                  "TensorConstant{2})), "

                                  "Alloc(<TensorType(float64, matrix)>, "
                                  "TensorConstant{1}, "
                                  "TensorConstant{2})]")

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(g, ops_to_check='all'))


class Test_local_useless_inc_subtensor_alloc(unittest.TestCase):
    opt_name = 'local_useless_inc_subtensor_alloc'

    def setUp(self):
        # The optimization requires the shape feature so we need to compile in
        # FAST_RUN mode.
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        self.mode = compile.mode.get_mode(mode)

    def test_advanced_inc_subtensor(self):
        x = tensor.vector('x')
        y = tensor.scalar('y')
        i = tensor.matrix('i', dtype='int64')
        z = tensor.advanced_inc_subtensor(x, T.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = theano.function([x, i, y], z, mode=mode1)
        f2 = theano.function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (len([n for n in f1.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.Alloc)]) == 1)
        # the alloc op should have been removed
        assert (len([n for n in f2.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.Alloc)]) == 0)

        x_value = np.random.randn(5).astype(config.floatX)
        y_value = np.random.randn()
        i_value = np.random.randint(0, 3, size=(2, 3))

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f1, ops_to_check=tensor.AdvancedIncSubtensor))
        self.assertTrue(check_stack_trace(f2, ops_to_check=tensor.AdvancedIncSubtensor))

    def test_advanced_inc_subtensor1(self):
        x = tensor.vector('x')
        y = tensor.scalar('y')
        i = tensor.vector('i', dtype='int64')
        z = tensor.advanced_inc_subtensor1(x, T.alloc(y, *i.shape), i)
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = theano.function([x, i, y], z, mode=mode1)
        f2 = theano.function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (len([n for n in f1.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.Alloc)]) == 1)
        # the alloc op should have been removed
        assert (len([n for n in f2.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.Alloc)]) == 0)

        x_value = np.random.randn(5).astype(config.floatX)
        y_value = np.random.randn()
        i_value = np.random.randint(0, 3, size=2)

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(
            f1, ops_to_check=tensor.AdvancedIncSubtensor1))
        self.assertTrue(check_stack_trace(f2, ops_to_check='all'))

    def test_incsubtensor(self):
        x = tensor.vector('x')
        y = tensor.scalar('y')
        i = tensor.scalar('i', dtype='int64')
        z = tensor.inc_subtensor(x[:i], T.alloc(y, i))
        mode1 = self.mode.excluding(self.opt_name)
        mode2 = self.mode.including(self.opt_name)
        f1 = theano.function([x, i, y], z, mode=mode1)
        f2 = theano.function([x, i, y], z, mode=mode2)

        # the alloc op should still be there
        assert (len([n for n in f1.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.Alloc)]) == 1)
        # the alloc op should have been removed
        assert (len([n for n in f2.maker.fgraph.toposort()
                     if isinstance(n.op, tensor.Alloc)]) == 0)

        x_value = np.random.randn(5).astype(config.floatX)
        y_value = np.random.randn()
        i_value = 3

        r1 = f1(x_value, i_value, y_value)
        r2 = f2(x_value, i_value, y_value)

        utt.assert_allclose(r1, r2)

        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(f1, ops_to_check='last'))
        self.assertTrue(check_stack_trace(f2, ops_to_check='last'))


class test_shapeoptimizer(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test0(self):
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        v = T.vector()
        m = T.matrix()
        f = function([v, m], (v + m).shape, mode=mode)
        for node in f.maker.fgraph.toposort():
            assert node.op != T.add

    def test_constant(self):
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'

        v = T.vector()
        f = function([v], v.dimshuffle('x', 'x', 0).shape[1], mode=mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

    @staticmethod
    def max_pool_c01b(c01b, pool_shp, pool_stride, img_shp):
        """
        Like max_pool but with input using axes ('c', 0, 1, 'b')
          (Alex Krizhevsky format)

        pool_shp, pool_stride and img_shp are int that represent
        the same shp in x and y.
        """
        mx = None

        # Compute index in pooled space of last needed pool
        # (needed = each input pixel must appear in at least one pool)
        def last_pool(im_shp, p_shp, p_strd):
            rval = int(np.ceil(float(im_shp - p_shp) / p_strd))
            assert p_strd * rval + p_shp >= im_shp
            assert p_strd * (rval - 1) + p_shp < im_shp
            return rval
        # Compute starting row of the last pool
        last_pool_r = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        # Compute number of rows needed in img for all indexes to work out
        required_r = last_pool_r + pool_shp

        last_pool_c = last_pool(img_shp, pool_shp, pool_stride) * pool_stride
        required_c = last_pool_c + pool_shp

        wide_infinity = T.alloc(-np.inf, c01b.shape[0],
                                required_r, required_c, c01b.shape[3])

        c01b = T.set_subtensor(wide_infinity[:, 0:img_shp, 0:img_shp, :], c01b)

        for row_within_pool in xrange(pool_shp):
            row_stop = last_pool_r + row_within_pool + 1
            for col_within_pool in xrange(pool_shp):
                col_stop = last_pool_c + col_within_pool + 1
                cur = c01b[:, row_within_pool:row_stop:pool_stride,
                           col_within_pool:col_stop:pool_stride, :]
                if mx is None:
                    mx = cur
                else:
                    mx = T.maximum(mx, cur)
        return mx

    def test_broadcasted_dims(self):
        # This test a case that caused a crash during optimization
        shp = (1, 1, 1, 1)
        rng = np.random.RandomState(utt.fetch_seed())
        a = shared(rng.rand(*shp).astype(config.floatX))
        out = self.max_pool_c01b(a, 1, 1, 1)

        # max_pool_c01b use -inf and this will trigger DebugMode error.
        mode = copy.copy(theano.compile.get_default_mode())
        mode.check_isfinite = False
        f = theano.function([], out, mode=mode)
        f()

    def test_constant_merge(self):
        # This test the error in gh-1122 that is a caused by the
        # combination of merge optimizer and ShapeFeature.

        x = tensor.constant([0, 0])
        y = x[1:]
        x1 = x - tensor.join(0, y, y)
        x1.eval()

    def test_local_track_shape_i(self):
        class IdentityNoShape(gof.Op):
            '''Op that does not infer the output shape from the input one'''
            def make_node(self, x):
                x = as_tensor_variable(x)
                return gof.Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                x, = inp
                out, = out_
                out[0] = x.copy()
            # def infer_shape(self, node, (xshp,)):
                # return [tuple([self.shape_i(i)(r) for i in xrange(r.ndim)])]
        identity_noshape = IdentityNoShape()

        class IdentityShape(gof.Op):
            '''Op that does infer the output shape from the input one'''
            def make_node(self, x):
                x = as_tensor_variable(x)
                return gof.Apply(self, [x], [x.type()])

            def perform(self, node, inp, out_):
                x, = inp
                out, = out_
                out[0] = x.copy()

            def infer_shape(self, node, xshp_):
                # Could also just return.
                xshp, = xshp_
                return (xshp,)
        identity_shape = IdentityShape()

        @gof.local_optimizer([IdentityNoShape])
        def local_identity_noshape_to_identity_shape(node):
            '''Optimization transforming the first Op into the second'''
            if isinstance(node.op, IdentityNoShape):
                return [identity_shape(node.inputs[0])]

        mode = theano.compile.get_default_mode().including(
            'ShapeOpt', 'specialize')
        rng = np.random.RandomState(utt.fetch_seed())
        x = T.tensor3('x')
        ins_x = identity_noshape(x)

        # Without the optimization
        f = theano.function([x], ins_x.shape, mode=mode)
        xval = rng.randn(3, 4, 7).astype(config.floatX)
        assert np.all(f(xval) == [3, 4, 7])
        f_ops = [node.op for node in f.maker.fgraph.toposort()]
        assert len(f_ops) == 5
        assert identity_noshape in f_ops
        assert identity_shape not in f_ops

        # Register the optimization
        opt.register_specialize(local_identity_noshape_to_identity_shape)

        mode = theano.compile.get_default_mode().including(
            'ShapeOpt', 'specialize')
        # With the optimization
        # The identity_shape op should not be needed anymore to compute
        # the shape
        g = theano.function([x], ins_x.shape, mode=mode)
        xval = rng.randn(6, 1, 2).astype(config.floatX)
        assert np.all(g(xval) == [6, 1, 2])
        g_ops = [node.op for node in g.maker.fgraph.toposort()]
        assert len(g_ops) == 4
        assert identity_noshape not in g_ops
        assert identity_shape not in g_ops

        # test multiple level of op without infer_shape
        ins_x3 = identity_noshape(identity_noshape(identity_noshape(x)))
        h = theano.function([x], ins_x3.shape, mode=mode)
        xval = rng.randn(6, 1, 2).astype(config.floatX)
        assert np.all(h(xval) == [6, 1, 2])
        h_ops = [node.op for node in h.maker.fgraph.toposort()]
        assert len(h_ops) == 4
        assert identity_noshape not in h_ops
        assert identity_shape not in h_ops

    def test_no_shapeopt(self):
        # Test that a basic example works even when ShapeOpt is excluded
        X = T.matrix()
        expr = X.shape[0]

        mode = theano.compile.get_default_mode().excluding('ShapeOpt')
        f = theano.function([X], expr, mode=mode)
        print(f([[1, 2], [2, 3]]))


class test_assert(utt.InferShapeTester):

    def setUp(self):
        super(test_assert, self).setUp()

    def test0(self):
        x = T.scalar()
        y = T.scalar()
        f = theano.function([x, y], theano.tensor.opt.assert_op(x, T.eq(x, y)))
        f(1, 1)
        self.assertRaises(AssertionError, f, 1, 0)

    def test_local_remove_useless_assert1(self):
        # remove assert that are always true
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode)

        x = T.scalar()
        f = theano.function([x], theano.tensor.opt.assert_op(x, 1), mode=mode)
        assert f(1) == 1
        assert f(5) == 5
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op

    def test_test_local_remove_useless_assert2(self):
        # remove assert condition that are always true
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode)

        x = T.scalar()
        y = T.scalar()
        f = theano.function([x, y], theano.tensor.opt.assert_op(x, y, 1),
                            mode=mode)
        assert f(1, 1) == 1
        assert f(5, 1) == 5
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert len(topo[0].inputs) == 2
        assert topo[1].op == deep_copy_op

    def test_local_remove_useless_assert3(self):
        # don't remove assert condition that are always false
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode)

        x = T.scalar()
        y = T.scalar()
        f = theano.function([x, y], theano.tensor.opt.assert_op(x, y, 0),
                            mode=mode)
        self.assertRaises(AssertionError, f, 1, 0)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert len(topo[0].inputs) == 3
        assert topo[1].op == deep_copy_op

    def test_local_remove_all_assert1(self):
        # remove assert condition that are unknown
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode).including('local_remove_all_assert')

        x = T.scalar()
        y = T.scalar()
        f = theano.function([x, y], theano.tensor.opt.assert_op(x, y),
                            mode=mode)
        if isinstance(mode, theano.compile.debugmode.DebugMode):
            # DebugMode will run the original version with the Assert
            self.assertRaises(AssertionError, f, 1, 0)
        else:
            f(1, 0)  # Without opt, it should fail.
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1, topo
        assert topo[0].op == deep_copy_op, topo

        mode = compile.mode.get_default_mode()
        a = theano.tensor.opt.assert_op(x, T.eq(x, 0).any())
        f = theano.function([x], a, mode=mode.excluding('unsafe'))
        topo = f.maker.fgraph.toposort()
        a_op = [n for n in topo if isinstance(n.op, T.opt.Assert)]
        assert len(a_op) == 1

    def test_infer_shape(self):

        adscal = dscalar()
        bdscal = dscalar()
        adscal_val = np.random.rand()
        bdscal_val = np.random.rand() + 1
        out = theano.tensor.opt.assert_op(adscal, bdscal)
        self._compile_and_check([adscal, bdscal], [out],
                                [adscal_val, bdscal_val], Assert)

        admat = dmatrix()
        admat_val = np.random.rand(3, 4)
        adscal_val += 1
        out = theano.tensor.opt.assert_op(admat, adscal, bdscal)
        self._compile_and_check([admat, adscal, bdscal], [out],
                                [admat_val, adscal_val, bdscal_val], Assert)


def test_local_mul_specialize():
    mode = theano.config.mode
    if mode == 'FAST_COMPILE':
        mode = 'FAST_RUN'
    mode = compile.mode.get_mode(mode)
    mode = mode.excluding('fusion')

    v = T.vector()
    m = T.vector()

    f = function([v], v * 1, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    nodes == [deep_copy_op]

    f = function([v], v * 0, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [Shape_i(0), T.alloc]

    f = function([v], v * (-1), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [T.neg]

    f = function([v, m], v * 1 * (-m), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [T.mul]

    f = function([v, m], v * 0 * (-m), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [Shape_i(0), T.alloc]

    f = function([v, m], v * (-1) * (-m), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [T.mul]

    f = function([v, m], v * (-1) * m, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [T.mul]


class T_Tile(unittest.TestCase):
    def test_local_useless_tile(self):
        v = T.vector()
        m = T.matrix()
        mode = None
        if theano.config.mode == "FAST_COMPILE":
            mode = "FAST_RUN"
        for var, data in [(v, [1, 2, 3]), (m, [[1, 2], [3, 4]])]:
            # When len(repeat pattern) <= var.ndim, everything is removed
            # for ndim in range(1, var.ndim):
            for ndim in range(var.ndim + 1):
                f = theano.function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) == 1
                assert isinstance(topo[0].op, compile.DeepCopyOp)
                f(data)
                # In this case the opt only removes nodes,
                # no need to check_stack_trace
            # When len(repeat pattern) > var.ndim, only a dimshuffle should be
            # left, but there can be a DeepCopy as well
            for ndim in range(var.ndim + 1, var.ndim + 3):
                f = theano.function([var], tile(var, (1,) * ndim), mode=mode)
                topo = f.maker.fgraph.toposort()
                assert len(topo) <= 2
                assert isinstance(topo[0].op, DimShuffle)
                assert check_stack_trace(f, ops_to_check=[DimShuffle])
                f(data)


def speed_local_pow_specialize_range():
    val = np.random.rand(1e7)
    v = T.vector()
    mode = compile.mode.get_default_mode()
    mode_without_pow_opt = mode.excluding('local_pow_specialize')
    for i in xrange(500, 513):
        f1 = function([v], v ** i, mode=mode)
        f2 = function([v], v ** i, mode=mode_without_pow_opt)
        assert len(f1.maker.fgraph.toposort()) == 1
        t1 = time.time()
        f1(val)
        t2 = time.time()
        f2(val)
        t3 = time.time()
        print(i, t2 - t1, t3 - t2, t2 - t1 < t3 - t2)
        if not t2 - t1 < t3 - t2:
            print("WARNING WE ARE SLOWER")
    for i in xrange(-3, -1500, -1):
        f1 = function([v], v ** i, mode=mode)
        f2 = function([v], v ** i, mode=mode_without_pow_opt)
        assert len(f1.maker.fgraph.toposort()) == 1
        t1 = time.time()
        f1(val)
        t2 = time.time()
        f2(val)
        t3 = time.time()
        print(i, t2 - t1, t3 - t2, t2 - t1 < t3 - t2)
        if not t2 - t1 < t3 - t2:
            print("WARNING WE ARE SLOWER")


def test_local_pow_specialize():
    mode = theano.config.mode
    if mode == 'FAST_COMPILE':
        mode = 'FAST_RUN'
    mode = compile.mode.get_mode(mode)
    mode = mode.excluding('fusion')

    v = T.vector()
    val = np.arange(10, dtype=theano.config.floatX)
    val_no0 = np.arange(1, 10, dtype=theano.config.floatX)

    f = function([v], v ** 0, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [Shape_i(0), T.alloc]
    utt.assert_allclose(f(val), val ** 0)

    f = function([v], v ** 1, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    nodes == [deep_copy_op]
    utt.assert_allclose(f(val), val ** 1)

    f = function([v], v ** (-1), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [T.inv]
    utt.assert_allclose(f(val_no0), val_no0 ** (-1))

    f = function([v], v ** 2, mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [T.sqr]
    utt.assert_allclose(f(val), val ** 2)

    f = function([v], v ** (-2), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert nodes[0] == T.sqr
    assert isinstance(nodes[1].scalar_op, theano.scalar.basic.Inv)
#    assert nodes == [T.sqr,T.inv]#Why this don't work?
    utt.assert_allclose(f(val_no0), val_no0 ** (-2))

    f = function([v], v ** (.5), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert nodes == [T.sqrt]
    utt.assert_allclose(f(val), val ** (.5))

    f = function([v], v ** (-.5), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert nodes[0] == T.sqrt
    assert isinstance(nodes[1].scalar_op, theano.scalar.basic.Inv)
#    assert nodes == [T.sqrt,T.inv]#Why this don't work?
    utt.assert_allclose(f(val_no0), val_no0 ** (-.5))


def test_local_pow_specialize_device_more_aggressive_on_cpu():
    mode = theano.config.mode
    if mode == 'FAST_COMPILE':
        mode = 'FAST_RUN'
    mode = compile.mode.get_mode(mode)
    mode = mode.excluding('fusion').excluding('gpu')

    v = T.vector()
    val = np.arange(10, dtype=theano.config.floatX)
    val_no0 = np.arange(1, 10, dtype=theano.config.floatX)
    f = function([v], v ** (15), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 1
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 6
    assert isinstance(nodes[0].scalar_op, theano.scalar.Composite)
    utt.assert_allclose(f(val), val ** 15)

    f = function([v], v ** (-15), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 6
    assert isinstance(nodes[0].scalar_op, theano.scalar.Composite)
    assert isinstance(nodes[-1].scalar_op, theano.scalar.basic.Inv)
    utt.assert_allclose(f(val_no0), val_no0 ** (-15))

    f = function([v], v ** (16), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 1
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 4
    assert isinstance(nodes[0].scalar_op, theano.scalar.Composite)
    utt.assert_allclose(f(val), val ** 16)

    f = function([v], v ** (-16), mode=mode)
    nodes = [node.op for node in f.maker.fgraph.toposort()]
    assert len(nodes) == 2
    assert len(f.maker.fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes) == 4
    assert isinstance(nodes[0].scalar_op, theano.scalar.Composite)
    assert isinstance(nodes[-1].scalar_op, theano.scalar.basic.Inv)
    utt.assert_allclose(f(val_no0), val_no0 ** (-16))


class T_Rebroadcast(unittest.TestCase):
    def test_local_useless_rebroadcast(self):
        mode = theano.compile.get_default_mode().including('canonicalize')
        v1 = T.vector()
        v2 = T.vector()
        j = T.join(0, v1, v2)
        f = theano.function([v1, v2], j, mode=mode)
        f([1, 2], [3, 4, 5])
        e = f.maker.fgraph.toposort()
        assert len([n for n in e if isinstance(n.op, T.Rebroadcast)]) == 0

        assert check_stack_trace(f, ops_to_check='all')

    def test_rebroadcast_rebroadcast(self):
        mode = theano.compile.get_default_mode().including('canonicalize')
        m = T.matrix()
        s = T.addbroadcast(m, 0, 1)
        v = T.unbroadcast(s, 1)
        f = theano.function([m], v, mode=mode)
        f([[76]])
        e = f.maker.fgraph.toposort()
        rebroadcast_nodes = [n for n in e if isinstance(n.op, T.Rebroadcast)]
        assert len(rebroadcast_nodes) == 1
        assert rebroadcast_nodes[0].op.axis == {0: True}


class T_useless_elemwise(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including(
            'canonicalize', 'local_fill_to_alloc')

    def test_eq(self):
        x = T.dmatrix()
        y = T.dmatrix()
        f = theano.function([x, y], T.eq(x, y), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, T.Elemwise)
        assert isinstance(topo[0].op.scalar_op, theano.scalar.EQ)
        f2 = theano.function([x], T.eq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.ones((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        # Shape_i{1}(<TensorType(float64, matrix)>), Shape_i{0}(<TensorType(float64, matrix)>), Alloc([[1]], Shape_i{0}.0, Shape_i{1}.0
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, T.Alloc)

    def test_neq(self):
        x = T.dmatrix()
        y = T.dmatrix()
        f = theano.function([x, y], T.neq(x, y), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx, vy)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, T.Elemwise)
        assert isinstance(topo[0].op.scalar_op, theano.scalar.NEQ)
        f2 = theano.function([x], T.neq(x, x), mode=self.mode)
        assert np.all(f2(vx) == np.zeros((5, 4)))
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 3
        assert isinstance(topo2[-1].op, T.Alloc)

    def test_mul(self):
        x = T.dmatrix()
        y = T.dmatrix()
        f = theano.function([x], T.mul(x), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = theano.function([x, y], T.mul(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx * vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, T.Elemwise)
        assert isinstance(topo2[0].op.scalar_op, theano.scalar.Mul)

    def test_add(self):
        x = T.dmatrix()
        y = T.dmatrix()
        f = theano.function([x], T.add(x), mode=self.mode)
        vx = np.random.rand(5, 4)
        vy = np.random.rand(5, 4)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op
        f2 = theano.function([x, y], T.add(x, y), mode=self.mode)
        assert np.all(f2(vx, vy) == vx + vy)
        topo2 = f2.maker.fgraph.toposort()
        assert len(topo2) == 1
        assert isinstance(topo2[0].op, T.Elemwise)
        assert isinstance(topo2[0].op.scalar_op, theano.scalar.Add)

    def test_identity(self):
        # scalar.identity is used in 2 Elemwise functions:
        # tensor_copy, and view
        x = T.matrix()
        f = theano.function([x], T.tensor_copy(x), mode=self.mode)
        vx = np.random.rand(5, 4).astype(config.floatX)
        f(vx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert topo[0].op == deep_copy_op


class T_cast_cast(unittest.TestCase):
    def setUp(self):
        mode = theano.compile.get_default_mode()
        self.mode = mode.including('local_cast_cast')

    def test_consecutive(self):
        x = T.fmatrix()
        o = T.Elemwise(scal.Cast(scal.Scalar("float64")))(x.astype("float64"))
        f = theano.function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, scal.basic.Cast)

        x = T.dmatrix()
        o = T.Elemwise(scal.Cast(scal.Scalar("float32")))(x.astype("float32"))
        f = theano.function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4)
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, scal.basic.Cast)

    def test_upcast(self):
        # Upcast followed by any other cast
        x = T.fmatrix()
        o = T.Elemwise(scal.Cast(scal.Scalar("complex128")))(x.astype("complex64"))
        f = theano.function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4).astype("float32")
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op.scalar_op, scal.basic.Cast)

        # Upcast followed by a downcast back to the base type
        x = T.fmatrix()
        o = T.Elemwise(scal.Cast(scal.Scalar("float32")))(x.astype("float64"))
        f = theano.function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4).astype('float32')
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 1
        assert isinstance(topo[0].op, DeepCopyOp)

        # Downcast followed by an upcast back to the base type
        # Optimization shouldn't be applied
        x = T.dmatrix()
        o = T.Elemwise(scal.Cast(scal.Scalar("float64")))(x.astype("float32"))
        f = theano.function([x], o, mode=self.mode)
        dx = np.random.rand(5, 4)
        f(dx)
        topo = f.maker.fgraph.toposort()
        assert (len(topo) == 1 and isinstance(topo[0].op.scalar_op, scal.basic.Composite)) or (len(topo) > 1)


class T_func_inverse(unittest.TestCase):

    def setUp(self):
        mode = theano.compile.get_default_mode()
        self.mode = mode.including('local_func_inv')

    def assert_func_pair_optimized(self, func1, func2, data,
                                   should_copy=True, is_complex=False):
        # Check that a pair of funcs is optimized properly

        x = T.cmatrix() if is_complex else T.fmatrix()
        o = func2(func1(x))
        f = theano.function([x], o, mode=self.mode)
        delta = f(data) - data
        topo = f.maker.fgraph.toposort()

        if should_copy:
            acceptable_topo_lens = [1]
        else:
            # The 2 funcs can be split apart if they are not inverses
            acceptable_topo_lens = [1, 2]

        if should_copy:
            delta_condition = np.all(delta == 0)
        else:
            delta_condition = np.all(delta != 0)

        self.assertTrue(len(topo) in acceptable_topo_lens)
        self.assertTrue(delta_condition)
        self.assertEqual(isinstance(topo[0].op, DeepCopyOp), should_copy,
                         "Inverse functions not removed!")

    def test(self):
        # test optimization for consecutive functional inverses

        dx = np.random.rand(5, 4).astype("float32")
        self.assert_func_pair_optimized(T.deg2rad, T.rad2deg, dx)
        dx = np.random.rand(5, 4).astype("float32") * 180
        self.assert_func_pair_optimized(T.rad2deg, T.deg2rad, dx)

        # Test the other functional inverses
        dx = np.random.rand(5, 4).astype("float32")
        self.assert_func_pair_optimized(T.cosh, T.arccosh, dx)
        self.assert_func_pair_optimized(T.arcsinh, T.sinh, dx)
        self.assert_func_pair_optimized(T.arctanh, T.tanh, dx)
        self.assert_func_pair_optimized(T.inv, T.inv, dx)
        self.assert_func_pair_optimized(T.neg, T.neg, dx)
        cx = dx + complex(0, 1) * (dx + 0.01)
        self.assert_func_pair_optimized(T.conj, T.conj, cx, is_complex=True)

        # Test that non-inverse functions are ran normally
        self.assert_func_pair_optimized(T.conj, T.neg, cx,
                                        should_copy=False, is_complex=True)
        dx = np.random.rand(5, 4).astype("float32") + 0.01
        self.assert_func_pair_optimized(T.rad2deg, T.rad2deg, dx,
                                        should_copy=False)
        self.assert_func_pair_optimized(T.rad2deg, T.cosh, dx,
                                        should_copy=False)


def test_constant_folding():
    # Test that constant folding get registered at fast_compile
    # An error removed that registration during the registration.

    x = tensor.dvector()
    mode = theano.compile.get_mode("FAST_COMPILE").excluding("fusion")
    f = theano.function([x], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2

    # Test that we do not crash when constant folding elemwise scalar
    # as they should not generate c code.

    x = tensor.constant(3)
    assert x.ndim == 0
    mode = theano.compile.get_mode("FAST_COMPILE").excluding("fusion")
    f = theano.function([], [x * 2, x + x], mode=mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 2
    assert all([isinstance(n.op, DeepCopyOp) for n in topo])


def test_constant_get_stabilized():
    # Currently Theano enable the constant_folding optimization before stabilization optimization.
    # This cause some stabilization optimization not being implemented and thus cause inf value to appear
    # when it should not.
    #
    # .. note: we can't simply move the constant_folding optimization to specialize as this break other optimization!
    # We will need to partially duplicate some canonicalize optimzation to specialize to fix this issue.

    x2 = T.scalar()
    y2 = T.log(1 + T.exp(x2))
    mode = theano.compile.get_default_mode()
    mode.check_isfinite = False
    f2 = theano.function([x2], y2, mode=mode)
    try:
        assert len(f2.maker.fgraph.toposort()) == 1
        assert (f2.maker.fgraph.toposort()[0].op ==
                theano.tensor.nnet.sigm.softplus)
        assert f2(800) == 800

        x = T.as_tensor_variable(800)
        y = T.log(1 + T.exp(x))
        f = theano.function([], y, mode=mode)
        assert len(f.maker.fgraph.toposort()) == 0
        assert np.isinf(f())

        # When this error is fixed, the following line should be ok.
        assert f() == 800, f()

    except AssertionError:
        raise SkipTest('Theano optimizes constant before stabilization. '
                       'This breaks stabilization optimization in some '
                       'cases. See #504.')


class T_local_switch_sink(unittest.TestCase):
    def setUp(self):
        # condition values
        self.condm = np.asarray([[0.1, 0, 1, -1],
                                 [0., 0., 0., 0.],
                                 [1, 1, 1, 1]])
        self.condv = np.asarray([0.1, 0, 1, -1])
        self.conds = [0.1, 0, 1, -1]

        # x values
        self.xm = np.ones((3, 4))
        self.xv = np.ones((4,))
        self.xs = 1.

        # expected results
        self.resm = (
            [np.asarray([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]])] * 3 +
            [np.asarray([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])] +
            2 * [np.asarray([[1, 0, 1, 0]])] +
            [[np.ones((3, 4)), np.zeros((3, 4)), np.ones((3, 4)), np.zeros((3, 4))]] +
            [[np.ones((4,)), np.zeros((4,)), np.ones((4,)), np.zeros((4,))]] +
            [[np.asarray(1.0), np.asarray(0.0), np.asarray(1.0), np.asarray(0.0)]])

        self.mode = theano.compile.mode.get_default_mode().including(
            'canonicalize', 'fast_run').excluding('gpu', 'fusion')
        self.mode = copy.copy(self.mode)
        self.mode.check_isfinite = False

    def function_remove_nan(self, *args, **kwargs):
        """
        Wrapper around theano.function for this test.

        It disables checking for NaN removed by optimizations in DebugMode
        (it has false positives in that case).
        """
        f = theano.function(*args, **kwargs)

        def wrapped_f(*args, **kwargs):
            # This is a bit ugly since it changes the global value of
            # TensorType.values_eq_approx.
            old_values_eq_approx = staticmethod(TensorType.values_eq_approx)
            TensorType.values_eq_approx = staticmethod(values_eq_approx_remove_nan)
            try:
                out = f(*args, **kwargs)
            finally:
                TensorType.values_eq_approx = old_values_eq_approx
            return out

        return wrapped_f

    def test_local_mul_switch_sink(self):
        c = T.dscalar()
        idx = 0
        for condition in [(T.dmatrix('cond'), self.condm),
                          (T.dvector('cond'), self.condv),
                          (T.dscalar('cond'), self.conds)]:
            for x in [(T.dmatrix('x'), self.xm), (T.dvector('x'), self.xv),
                      (T.dscalar('x'), self.xs)]:
                y = T.mul(T.switch(condition[0] > 0, 1. * x[0], 0. * x[0]),
                          T.switch(condition[0] > 0,
                                   1. * x[0], T.log(c) * x[0]))
                f = self.function_remove_nan([condition[0], x[0], c],
                                             [y], mode=self.mode)
                if type(condition[1]) is list:
                    for i in xrange(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert (res == np.asarray(
                            self.resm[idx][i])).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1], x[1], -1)
                    assert ((res == np.asarray(self.resm[idx])).sum() ==
                            self.resm[idx].size)
                idx += 1

        # This case caused a missed optimization in the past.
        x = T.dscalar('x')
        y = T.switch(x < 7, x, T.sqrt(x - 7))
        f = self.function_remove_nan([x], T.grad(y, x), self.mode)
        assert f(5) == 1, f(5)

    @attr('slow')
    def test_local_div_switch_sink(self):
        c = T.dscalar()
        idx = 0
        for condition in [(T.dmatrix('cond'), self.condm), (T.dvector('cond'), self.condv), (T.dscalar('cond'), self.conds)]:
            for x in [(T.dmatrix('x'), self.xm), (T.dvector('x'), self.xv), (T.dscalar('x'), self.xs)]:
                y = T.true_div(
                    T.switch(condition[0] > 0, 1. * x[0], 0. * x[0]),
                    T.switch(condition[0] > 0, 1. * x[0], T.log(c) * x[0]))
                f = self.function_remove_nan([condition[0], x[0], c],
                                             [y], mode=self.mode)
                if type(condition[1]) is list:
                    for i in xrange(len(condition[1])):
                        res = f(condition[1][i], x[1], -1)
                        assert ((res == np.asarray(self.resm[idx][i])).sum() ==
                                self.resm[idx][i].size)
                else:
                    res = f(condition[1], x[1], -1)
                    assert ((res == np.asarray(self.resm[idx])).sum() ==
                            self.resm[idx].size)
                idx += 1


class T_local_erf(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.mode.get_default_mode().including(
            'canonicalize', 'fast_run').excluding('gpu', 'fusion')
        self.mode._optimizer.position_cutoff = 1.50001
        if theano.config.cxx == '' and not theano.scalar.basic_scipy.imported_scipy_special:
            raise SkipTest("erf need a c++ compiler or scipy")

    def test_local_one_plus_erf(self):
        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30],
                         dtype=config.floatX)
        x = T.vector()

        f = theano.function([x], 1 + T.erf(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [
            T.mul, T.erfc], f.maker.fgraph.toposort()
        f(val)

        f = theano.function([x], T.erf(x) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [
            T.mul, T.erfc], f.maker.fgraph.toposort()
        f(val)

        f = theano.function([x], T.erf(x) + 2, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert topo[0].op == T.erf
        assert isinstance(topo[1].op, T.Elemwise)
        assert isinstance(topo[1].op.scalar_op, scal.Add)
        f(val)

    def test_local_one_minus_erf(self):
        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30],
                         dtype=config.floatX)
        x = T.vector()

        f = theano.function([x], 1 - T.erf(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erfc],\
            f.maker.fgraph.toposort()
        print(f(val))

        f = theano.function([x], 1 + (-T.erf(x)), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erfc],\
            f.maker.fgraph.toposort()
        print(f(val))

        f = theano.function([x], (-T.erf(x)) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erfc],\
            f.maker.fgraph.toposort()
        print(f(val))

        f = theano.function([x], 2 - T.erf(x), mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2, f.maker.fgraph.toposort()
        assert topo[0].op == T.erf, f.maker.fgraph.toposort()
        assert isinstance(topo[1].op, T.Elemwise), f.maker.fgraph.toposort()
        assert isinstance(topo[1].op.scalar_op, scal.Add)\
            or isinstance(topo[1].op.scalar_op, scal.Sub), f.maker.fgraph.toposort()
        print(f(val))

    def test_local_erf_minus_one(self):
        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30],
                         dtype=config.floatX)
        x = T.vector()

        f = theano.function([x], T.erf(x) - 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erfc, T.mul]
        print(f(val))

        f = theano.function([x], T.erf(x) + (-1), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erfc, T.mul]
        print(f(val))

        f = theano.function([x], -1 + T.erf(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erfc, T.mul]
        print(f(val))

        f = theano.function([x], T.erf(x) - 2, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2
        assert topo[0].op == T.erf
        assert isinstance(topo[1].op, T.Elemwise)
        assert isinstance(topo[1].op.scalar_op, scal.Add)\
            or isinstance(topo[1].op.scalar_op, scal.Sub)
        print(f(val))


class T_local_erfc(unittest.TestCase):
    def setUp(self):
        self.mode_fusion = theano.compile.mode.get_default_mode().including(
            'canonicalize').including('fast_run').excluding('gpu')
        self.mode = self.mode_fusion.excluding('fusion')
        self.mode._optimizer.position_cutoff = 1.50001
        if (theano.config.cxx == '' and
                not theano.scalar.basic_scipy.imported_scipy_special):
            raise SkipTest("erfc need a c++ compiler or scipy")

    def test_local_one_minus_erfc(self):
        # test opt: 1-erfc(x) => erf(x) and -erfc(x)+1 => erf(x)

        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30],
                         dtype=config.floatX)
        x = T.vector('x')

        f = theano.function([x], 1 - T.erfc(x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erf],\
            f.maker.fgraph.toposort()
        print(f(val))

        f = theano.function([x], (-T.erfc(x)) + 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erf],\
            f.maker.fgraph.toposort()
        print(f(val))

        f = theano.function([x], 2 - T.erfc(x), mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert len(topo) == 2, f.maker.fgraph.toposort()
        assert topo[0].op == T.erfc, f.maker.fgraph.toposort()
        assert isinstance(topo[1].op, T.Elemwise), f.maker.fgraph.toposort()
        assert isinstance(topo[1].op.scalar_op, scal.Sub),\
            f.maker.fgraph.toposort()
        print(f(val))

    def test_local_erf_neg_minus_one(self):
        # test opt: (-1)+erfc(-x)=>erf(x)

        val = np.asarray([-30, -3, -2, -1, 0, 1, 2, 3, 30],
                         dtype=config.floatX)
        x = T.vector('x')

        f = theano.function([x], -1 + T.erfc(-x), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erf],\
            f.maker.fgraph.toposort()
        print(f(val))

        f = theano.function([x], T.erfc(-x) - 1, mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erf],\
            f.maker.fgraph.toposort()
        print(f(val))

        f = theano.function([x], T.erfc(-x) + (-1), mode=self.mode)
        assert [n.op for n in f.maker.fgraph.toposort()] == [T.erf],\
            f.maker.fgraph.toposort()
        print(f(val))

    def test_local_log_erfc(self):
        val = [-30, -27, -26, -11, -10, -3, -2, -1, 0, 1, 2, 3, 10,
               11, 26, 27, 28, 30]
        if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            # python mode don't like the inv(0)
            val.remove(0)
        val = np.asarray(val, dtype=config.floatX)
        x = T.vector('x')

        # their is some nan that will happear in the graph for the log of the negatives values
        mode = copy.copy(self.mode)
        mode.check_isfinite = False
        mode_fusion = copy.copy(self.mode_fusion)
        mode_fusion.check_isfinite = False

        f = theano.function([x], T.log(T.erfc(x)), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 23, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX
        assert all(np.isfinite(f(val)))

        f = theano.function([x], T.log(T.erfc(-x)), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 24, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX
        assert all(np.isfinite(f(-val)))

        f = theano.function([x], T.log(T.erfc(x)), mode=mode_fusion)
        assert len(f.maker.fgraph.apply_nodes) == 1, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX
        assert len(f.maker.fgraph.toposort()[0].fgraph.toposort()[
            0].op.scalar_op.fgraph.apply_nodes) == 22, len(f.maker.fgraph.toposort()[0].fgraph.toposort()[0].op.scalar_op.fgraph.apply_nodes)
        # TODO: fix this problem
        if theano.config.floatX == "float32" and theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            raise SkipTest('The python code upcast somewhere internally '
                           'some value of float32 to python float for '
                           'part of its computation. That make that the '
                           'c and python code don\'t generate the same value. '
                           'You can ignore this error.')
        assert all(np.isfinite(f(val)))

    def test_local_grad_log_erfc_neg(self):
        val = [-100, -30, -27, -26.4, -26.2, -26, -11, -10, -9, -3, -2, -1, 0,
               1, 2, 3, 9, 10, 11, 27, 26.4, 26.2, 26, 28, 30, 100]
        if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            # python mode don't like the inv(0) in computation,
            # but the switch don't select this value.
            # So it is computed for no good reason.
            val.remove(0)
        if theano.config.mode in ["DebugMode", "DEBUG_MODE"] and theano.config.floatX == 'float32':
            # In float32 their is a plage of values close to 10 that we stabilize as it give bigger error then the stabilized version.
            # The orig value in float32 -30.0, the stab value -20.1 the orig value in float64 -18.1.
            val.remove(10)
        val = np.asarray(val, dtype=config.floatX)
        x = T.vector('x')
        y = T.vector('y')

        # their is some nan that will happear in the graph for the log of the negatives values
        mode = copy.copy(self.mode)
        mode.check_isfinite = False
        mode_fusion = copy.copy(self.mode_fusion)
        mode_fusion.check_isfinite = False

        f = theano.function([x], T.grad(T.log(T.erfc(x)).sum(), x), mode=mode)

        assert len(f.maker.fgraph.apply_nodes) == 23, len(f.maker.fgraph.apply_nodes)
        assert all(np.isfinite(f(val)))
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX

        # test with a different mul constant
        f = theano.function(
            [x],
            T.mul(T.exp(T.neg(T.sqr(x))), - 10.12837917) / T.erfc(x),
            mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 23, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX
        assert all(np.isfinite(f(val)))

        # test that we work without the mul
        f = theano.function([x], T.exp(T.neg(T.sqr(x))) / T.erfc(x), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 22, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX
        assert all(np.isfinite(f(val)))

        # test that we don't work if x!=y
        f = theano.function([x, y], T.exp(T.neg(T.sqr(x))) / T.erfc(
            y), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 5, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX
        f(val, val - 3)

        # test that we work without the sqr and neg
        f = theano.function([x], T.exp(T.mul(-1, x, x)) / T.erfc(x), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 21, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX
        assert all(np.isfinite(f(val)))

        # test that it work correctly if x is x*2 in the graph.
        f = theano.function([x], T.grad(T.log(T.erfc(2 * x)).sum(), x), mode=mode)
        assert len(f.maker.fgraph.apply_nodes) == 23, len(f.maker.fgraph.apply_nodes)
        assert np.isfinite(f(val)).all()
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX

        f = theano.function([x], T.grad(T.log(T.erfc(x)).sum(), x), mode=mode_fusion)
        assert len(f.maker.fgraph.apply_nodes) == 1, len(f.maker.fgraph.apply_nodes)
        assert f.maker.fgraph.outputs[0].dtype == theano.config.floatX

        # TODO: fix this problem
        if theano.config.floatX == "float32" and theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            # The python code upcast somewhere internally some value of float32
            # to python float for part of its computation. That make that the c
            # and python code do not generate the same value. You can ignore
            # this error. This happen in an intermediate step that don't show
            # in the final result.

            # Showing this test error is a duplicate of the one in test_local_log_erfc. We hide it.
            pass
        else:
            assert all(np.isfinite(f(val)))

    def speed_local_log_erfc(self):

        val = np.random.rand(1e6)
        x = T.vector()
        mode = theano.compile.mode.get_mode("FAST_RUN")
        f1 = theano.function([x], T.log(T.erfc(x)),
                             mode=mode.excluding("local_log_erfc"))
        f2 = theano.function([x], T.log(T.erfc(x)), mode=mode)
        print(f1.maker.fgraph.toposort())
        print(f2.maker.fgraph.toposort())
        t0 = time.time()
        f1(val)
        t1 = time.time()
        f2(val)
        t2 = time.time()
        print(t1 - t0, t2 - t1)


class test_local_useless_switch(unittest.TestCase):
    def setUp(self):
        self.mode = mode_opt.excluding('constant_folding')

    def test_const0(self):

        for dtype1 in ['int32', 'int64']:
            for dtype2 in ['int32', 'int64']:
                x = theano.tensor.matrix('x', dtype=dtype1)
                y = theano.tensor.matrix('y', dtype=dtype2)
                z = theano.tensor.switch(0, x, y)
                f = theano.function([x, y], z, mode=self.mode)
                assert len([node.op for node in f.maker.fgraph.toposort() if
                            (isinstance(node.op, theano.tensor.Elemwise) and
                             isinstance(node.op.scalar_op,
                                        theano.scalar.basic.Switch))]) == 0
                vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
                vy = np.array([[7, 8, 9], [10, 11, 12]], dtype=dtype2)
                assert np.all(f(vx, vy) == vy)

    def test_const1(self):

        for dtype1 in ['int32', 'int64']:
            for dtype2 in ['int32', 'int64']:
                x = theano.tensor.matrix('x', dtype=dtype1)
                y = theano.tensor.matrix('y', dtype=dtype2)
                z = theano.tensor.switch(1, x, y)
                f = theano.function([x, y], z, mode=self.mode)
                assert len([node.op for node in f.maker.fgraph.toposort() if
                            (isinstance(node.op, theano.tensor.Elemwise) and
                             isinstance(node.op.scalar_op,
                                        theano.scalar.basic.Switch))]) == 0
                vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
                vy = np.array([[7, 8, 9], [10, 11, 12]], dtype=dtype2)
                assert np.all(f(vx, vy) == vx)

    def test_left_is_right(self):

        for dtype1 in ['int32', 'int64']:
            x = theano.tensor.matrix('x', dtype=dtype1)
            varc = theano.tensor.matrix('varc', dtype=dtype1)
            z1 = theano.tensor.switch(1, x, x)
            z0 = theano.tensor.switch(0, x, x)
            z2 = theano.tensor.switch(varc, x, x)
            f1 = theano.function([x], z1, mode=self.mode)
            f0 = theano.function([x], z0, mode=self.mode)
            f2 = theano.function([x, varc], z2, mode=self.mode)

            topo = f1.maker.fgraph.toposort()
            assert len(topo) == 1
            assert topo[0].op == deep_copy_op

            topo = f0.maker.fgraph.toposort()
            assert len(topo) == 1
            assert topo[0].op == deep_copy_op

            topo = f2.maker.fgraph.toposort()
            assert len(topo) == 1
            assert topo[0].op == deep_copy_op

            vx = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
            vc = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype1)
            assert np.all(f1(vx) == vx)
            assert np.all(f0(vx) == vx)
            assert np.all(f2(vx, vc) == vx)

    def test_shape_le_0(self):

        for dtype1 in ['float32', 'float64']:
            x = theano.tensor.matrix('x', dtype=dtype1)
            z0 = theano.tensor.switch(theano.tensor.le(x.shape[0], 0), 0, x.shape[0])
            f0 = theano.function([x], z0, mode=self.mode)
            assert isinstance(f0.maker.fgraph.toposort()[0].op, Shape_i)

            z1 = theano.tensor.switch(theano.tensor.le(x.shape[1], 0), 0, x.shape[1])
            f1 = theano.function([x], z1, mode=self.mode)
            assert isinstance(f1.maker.fgraph.toposort()[0].op, Shape_i)

            vx = np.random.randn(0, 5).astype(dtype1)
            assert f0(vx) == 0
            assert f1(vx) == 5

    def test_broadcast1(self):
        # test switch(cst, matrix, row)
        x = theano.tensor.matrix('x', dtype='int32')
        y = theano.tensor.vector('y', dtype='int64')

        z = theano.tensor.switch(1, x, y)
        f = theano.function([x, y], z, mode=self.mode)
        assert len([node.op for node in f.maker.fgraph.toposort() if
                    isinstance(node.op, theano.tensor.Elemwise) and
                    not isinstance(node.op.scalar_op, theano.scalar.basic.Cast)]) == 0
        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype='int32')
        vy = np.array([10, 11, 12], dtype='int64')
        assert np.all(f(vx, vy) == vx)

        z = theano.tensor.switch(0, x, y)
        f = theano.function([x, y], z, mode=self.mode)
        assert len([node.op for node in f.maker.fgraph.toposort() if
                    isinstance(node.op, theano.tensor.Elemwise)]) == 0
        vx = np.array([[1, 2, 3], [4, 5, 6]], dtype='int32')
        vy = np.array([10, 11, 12], dtype='int64')
        assert np.all(f(vx, vy) == vy)

    def test_broadcast2(self):
        # test switch(cst, vector, matrix)

        # This case is not optimized for now.
        x = theano.tensor.vector('x', dtype='int32')
        y = theano.tensor.matrix('y', dtype='int64')
        z = theano.tensor.switch(1, x, y)
        f = theano.function([x, y], z, mode=self.mode)
        assert len([node.op for node in f.maker.fgraph.toposort() if
                    isinstance(node.op, theano.tensor.Elemwise) and
                    not isinstance(node.op.scalar_op, theano.scalar.basic.Cast)]) == 0
        vx = np.array([4, 5, 6], dtype='int32')
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype='int64')
        assert np.all(f(vx, vy) == vx)

        z = theano.tensor.switch(0, x, y)
        f = theano.function([x, y], z, mode=self.mode)
        assert len([node.op for node in f.maker.fgraph.toposort() if
                    isinstance(node.op, theano.tensor.Elemwise)]) == 0
        vx = np.array([4, 5, 6], dtype='int32')
        vy = np.array([[7, 8, 9], [10, 11, 12]], dtype='int64')
        assert np.all(f(vx, vy) == vy)

    def test_broadcast3(self):
        # test switch(matrix, same_vector, same_vector)

        x = theano.tensor.matrix('x', dtype='int32')
        y = theano.tensor.vector('y', dtype='int64')
        z = theano.tensor.switch(x, y, y)
        f = theano.function([x, y], z, mode=self.mode)
        vx = np.array([[0, 1], [1, 0]], dtype='int32')
        vy = np.array([7, 8], dtype='int64')
        utt.assert_allclose(f(vx, vy), np.where(vx, vy, vy))
        assert len([node.op for node in f.maker.fgraph.toposort() if
                    isinstance(node.op, theano.tensor.Elemwise)]) == 0


class test_local_merge_switch_same_cond(unittest.TestCase):
    def test_elemwise(self):
        # float Ops
        mats = theano.tensor.matrices('cabxy')
        c, a, b, x, y = mats
        s1 = T.switch(c, a, b)
        s2 = T.switch(c, x, y)
        for op in (T.add, T.sub, T.mul, T.true_div, T.int_div, T.floor_div,
                   T.minimum, T.maximum, T.gt, T.lt, T.ge, T.le, T.eq, T.neq,
                   T.pow):
            g = optimize(FunctionGraph(mats, [op(s1, s2)]))
            assert str(g).count('Switch') == 1
        # integer Ops
        mats = theano.tensor.imatrices('cabxy')
        c, a, b, x, y = mats
        s1 = T.switch(c, a, b)
        s2 = T.switch(c, x, y)
        for op in (T.and_, T.or_, T.xor,
                   T.bitwise_and, T.bitwise_or, T.bitwise_xor):
            g = optimize(FunctionGraph(mats, [op(s1, s2)]))
            assert str(g).count('Switch') == 1
        # add/mul with more than two inputs
        u, v = theano.tensor.matrices('uv')
        s3 = T.switch(c, u, v)
        for op in (T.add, T.mul):
            g = optimize(FunctionGraph(mats + [u, v], [op(s1, s2, s3)]))
            assert str(g).count('Switch') == 1


class T_local_sum_prod(unittest.TestCase):
    """
    Test sum/prod opts in opt.py
    """
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('canonicalize',
                                                                'specialize')

    def test_local_sum_prod_mul_by_scalar(self):
        # Test the optimization local_sum_prod_mul_by_scalar for both Sum and
        # Prod ops in six cases each :
        # 1-the inputs to the mul contain a scalar and no non-scalar
        # 2-the inputs to the mul contain a scalar and one non-scalar
        # 3-the inputs to the mul contain a scalar and two non-scalars
        # 4-the inputs to the mul contain two scalars and no non-scalar
        # 5-the inputs to the mul contain two scalars and one non-scalar
        # 6-the inputs to the mul contain two scalars and two non-scalars

        vect = T.dvector()
        mat = T.dmatrix()
        scalar1 = T.dscalar()
        scalar2 = T.dscalar()

        v_val = np.random.rand(2)
        m_val = np.random.rand(2, 2)
        s1_val = np.random.rand()
        s2_val = np.random.rand()

        def test_reduction_opt(inputs, inputs_val, reduction_op,
                               expected_output, nb_expected_sum_nodes):
            mul_out = T.mul(*inputs)
            f = theano.function(inputs, reduction_op()(mul_out),
                                mode=self.mode)
            out = f(*inputs_val)
            utt.assert_allclose(out, expected_output)

            # Ensure that the optimization has been applied properly by
            # ensuring that the optimized graph contains the expected number
            # of apply nodes for the sum op
            prod_nodes = [n for n in f.maker.fgraph.toposort()
                          if isinstance(n.op, reduction_op)]
            assert len(prod_nodes) == nb_expected_sum_nodes

        # Test sum

        # Case 1
        test_reduction_opt([scalar1], [s1_val], T.Sum, s1_val, 0)

        # Case 2
        test_reduction_opt([vect, scalar1], [v_val, s1_val], T.Sum,
                           s1_val * v_val.sum(), 1)

        # Case 3
        test_reduction_opt([vect, mat, scalar1], [v_val, m_val, s1_val], T.Sum,
                           s1_val * (v_val * m_val).sum(), 1)

        # Case 4
        test_reduction_opt([scalar1, scalar2], [s1_val, s2_val], T.Sum,
                           s1_val * s2_val, 0)

        # Case 5
        test_reduction_opt([vect, scalar1, scalar2], [v_val, s1_val, s2_val],
                           T.Sum, s1_val * s2_val * v_val.sum(), 1)

        # Case 6
        test_reduction_opt([vect, mat, scalar1, scalar2],
                           [v_val, m_val, s1_val, s2_val], T.Sum,
                           s1_val * s2_val * (v_val * m_val).sum(), 1)

        # Test prod

        # Case 1
        test_reduction_opt([scalar1], [s1_val], T.elemwise.Prod, s1_val, 0)

        # Case 2
        test_reduction_opt([vect, scalar1], [v_val, s1_val], T.elemwise.Prod,
                           (s1_val * v_val).prod(), 1)

        # Case 3
        test_reduction_opt([vect, mat, scalar1], [v_val, m_val, s1_val],
                           T.elemwise.Prod, (s1_val * v_val * m_val).prod(), 2)

        # Case 4
        test_reduction_opt([scalar1, scalar2], [s1_val, s2_val],
                           T.elemwise.Prod, s1_val * s2_val, 0)

        # Case 5
        test_reduction_opt([vect, scalar1, scalar2], [v_val, s1_val, s2_val],
                           T.elemwise.Prod, (s1_val * s2_val * v_val).prod(),
                           1)

        # Case 6
        test_reduction_opt([vect, mat, scalar1, scalar2],
                           [v_val, m_val, s1_val, s2_val], T.elemwise.Prod,
                           (s1_val * s2_val * v_val * m_val).prod(), 2)

    def test_local_sum_prod_all_to_none(self):
        a = T.tensor3()
        input = np.arange(3 * 4 * 5, dtype=config.floatX).reshape(3, 4, 5)
        # test sum
        f = theano.function([a], a.sum(), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.sum())
        # test prod
        f = theano.function([a], a.prod(), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.prod())
        # test sum
        f = theano.function([a], a.sum([0, 1, 2]), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.sum())
        # test prod
        f = theano.function([a], a.prod([0, 1, 2]), mode=self.mode)
        assert len(f.maker.fgraph.apply_nodes) == 1
        utt.assert_allclose(f(input), input.prod())

        backup = config.warn.sum_sum_bug
        config.warn.sum_sum_bug = False
        try:
            f = theano.function([a], a.sum(0).sum(0).sum(0), mode=self.mode)
            assert len(f.maker.fgraph.apply_nodes) == 1
            utt.assert_allclose(f(input), input.sum())
        finally:
            config.warn.sum_sum_bug = backup

    def test_local_sum_sum_prod_prod(self):
        a = T.tensor3()
        input = np.arange(3 * 4 * 5, dtype=config.floatX).reshape(3, 4, 5)
        dims = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1),
                ((0, 1), 0), ((1, 2), 0), (0, (0, 1)),
                (1, (0, 1)), (2, (0, 1))]

        backup = config.warn.sum_sum_bug
        config.warn.sum_sum_bug = False

        def my_prod(data, d, dd):
            # This prod when d or dd is a tuple of 2 dimensions.
            if not isinstance(d, tuple) and not isinstance(dd, tuple):
                return data.prod(d).prod(dd)
            if isinstance(d, tuple):
                d = sorted(d)
                return data.prod(d[1]).prod(d[0]).prod(dd)
            else:
                dd = sorted(dd)
                return data.prod(d).prod(dd[1]).prod(dd[0])

        def my_sum(data, d, dd):
            # This sum when d or dd is a tuple of 2 dimensions.
            if not isinstance(d, tuple) and not isinstance(dd, tuple):
                return data.sum(d).sum(dd)
            if isinstance(d, tuple):
                d = sorted(d)
                return data.sum(d[1]).sum(d[0]).sum(dd)
            else:
                dd = sorted(dd)
                return data.sum(d).sum(dd[1]).sum(dd[0])

        def my_sum_prod(data, d, dd):
            # This sum when d or dd is a tuple of 2 dimensions.
            if not isinstance(d, tuple) and not isinstance(dd, tuple):
                return data.sum(d).prod(dd)
            if isinstance(d, tuple):
                d = sorted(d)
                return data.sum(d[1]).sum(d[0]).prod(dd)
            else:
                dd = sorted(dd)
                return data.sum(d).prod(dd[1]).prod(dd[0])

        try:
            for d, dd in dims:
                expected = my_sum(input, d, dd)
                f = theano.function([a], a.sum(d).sum(dd), mode=self.mode)
                utt.assert_allclose(f(input), expected)
                assert len(f.maker.fgraph.apply_nodes) == 1
            for d, dd in dims[:6]:
                f = theano.function([a], a.sum(d).sum(dd).
                                    sum(0), mode=self.mode)
                utt.assert_allclose(f(input), input.sum(d).sum(dd).sum(0))
                assert len(f.maker.fgraph.apply_nodes) == 1
            for d in [0, 1, 2]:
                f = theano.function([a], a.sum(d).sum(None), mode=self.mode)
                utt.assert_allclose(f(input), input.sum(d).sum())
                assert len(f.maker.fgraph.apply_nodes) == 1
            f = theano.function([a], a.sum(None).sum(), mode=self.mode)
            utt.assert_allclose(f(input), input.sum())
            assert len(f.maker.fgraph.apply_nodes) == 1
        finally:
            config.warn.sum_sum_bug = backup

        # test prod
        for d, dd in dims:
            expected = my_prod(input, d, dd)
            f = theano.function([a], a.prod(d).prod(dd), mode=self.mode)
            utt.assert_allclose(f(input), expected)
            assert len(f.maker.fgraph.apply_nodes) == 1
        for d, dd in dims[:6]:
            f = theano.function([a], a.prod(d).prod(dd).
                                prod(0), mode=self.mode)
            utt.assert_allclose(f(input), input.prod(d).prod(dd).prod(0))
            assert len(f.maker.fgraph.apply_nodes) == 1
        for d in [0, 1, 2]:
            f = theano.function([a], a.prod(d).prod(None), mode=self.mode)
            utt.assert_allclose(f(input), input.prod(d).prod())
            assert len(f.maker.fgraph.apply_nodes) == 1
        f = theano.function([a], a.prod(None).prod(), mode=self.mode)
        utt.assert_allclose(f(input), input.prod())
        assert len(f.maker.fgraph.apply_nodes) == 1

        # test sum prod don't get opt.
        for d, dd in dims:
            expected = my_sum_prod(input, d, dd)
            f = theano.function([a], a.sum(d).prod(dd), mode=self.mode)
            utt.assert_allclose(f(input), expected)
            assert len(f.maker.fgraph.apply_nodes) == 2
        for d, dd in dims[:6]:
            f = theano.function([a], a.sum(d).prod(dd).
                                prod(0), mode=self.mode)
            utt.assert_allclose(f(input), input.sum(d).prod(dd).prod(0))
            assert len(f.maker.fgraph.apply_nodes) == 2
        for d in [0, 1, 2]:
            f = theano.function([a], a.sum(d).prod(None), mode=self.mode)
            utt.assert_allclose(f(input), input.sum(d).prod())
            assert len(f.maker.fgraph.apply_nodes) == 2
        f = theano.function([a], a.sum(None).prod(), mode=self.mode)
        utt.assert_allclose(f(input), input.sum())
        assert len(f.maker.fgraph.apply_nodes) == 1

    def test_local_sum_prod_alloc(self):
        # test local_opt_alloc
        a = T.dtensor3()
        input = np.asarray(np.arange(2 * 3 * 4).reshape(2, 3, 4),
                           dtype='float64')
        mode = self.mode.including('specialize').excluding('fusion')

        for t_like, n_like, nb_nodes in [
                (tensor.zeros_like, np.zeros_like, (1, 3, 3, 2)),
                (tensor.ones_like, np.ones_like, (5, 5, 5, 6))]:
            # test sum
            f = theano.function([a], t_like(a).sum(None), mode=mode)
            utt.assert_allclose(f(input), n_like(input).sum())
            assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            f = theano.function([a], t_like(a).sum([0, 1, 2]), mode=mode)
            utt.assert_allclose(f(input), n_like(input).sum())
            assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            for d in xrange(3):
                f = theano.function([a], t_like(a).sum(d), mode=mode)
                utt.assert_allclose(f(input), n_like(input).sum(d))
                assert len(f.maker.fgraph.apply_nodes) == nb_nodes[1]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == T.alloc
                assert not any([isinstance(node.op, T.Sum) for node in topo])
            for i in xrange(3):
                f = theano.function([a], t_like(a).sum(i), mode=mode)
                utt.assert_allclose(f(input), n_like(input).sum(i))
                assert len(f.maker.fgraph.apply_nodes) == nb_nodes[2]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == T.alloc
                assert not any([isinstance(node.op, T.Sum) for node in topo])

            # test prod
            f = theano.function([a], t_like(a).prod(None), mode=mode)
            utt.assert_allclose(f(input), n_like(input).prod())
            # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            f = theano.function([a], t_like(a).prod([0, 1, 2]), mode=mode)
            utt.assert_allclose(f(input), n_like(input).prod())
            # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[0]

            for d in range(3):
                f = theano.function([a], t_like(a).prod(d), mode=mode)
                utt.assert_allclose(f(input), n_like(input).prod(d))
                # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[1]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == T.alloc
                assert not any([isinstance(node.op, T.elemwise.Prod) for node in topo])
            for i in range(3):
                f = theano.function([a], t_like(a).prod(i), mode=mode)
                utt.assert_allclose(f(input), n_like(input).prod(i))
                # assert len(f.maker.fgraph.apply_nodes) == nb_nodes[2]
                topo = f.maker.fgraph.toposort()
                assert topo[-1].op == T.alloc
                assert not any([isinstance(node.op, T.elemwise.Prod) for node in topo])

            backup = config.warn.sum_sum_bug
            config.warn.sum_sum_bug = False
            try:
                for d, dd in [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]:
                    f = theano.function([a], t_like(a).
                                        sum(d).sum(dd), mode=mode)
                    utt.assert_allclose(f(input),
                                        n_like(input).sum(d).sum(dd))
                    assert len(f.maker.fgraph.apply_nodes) == nb_nodes[3]
                    topo = f.maker.fgraph.toposort()
                    assert topo[-1].op == T.alloc
                    assert not any([isinstance(node.op,
                                               T.Sum) for node in topo])
            finally:
                config.warn.sum_sum_bug = backup

    def test_local_sum_sum_int8(self):
        # Test that local_sum_sum works when combining two sums on an int8 array.
        # This is a regression test for ticket gh-356.

        x = tensor.tensor3(dtype='int8')

        y = x.sum(axis=0).sum(axis=1)
        backup = config.on_opt_error
        config.on_opt_error = 'raise'
        try:
            # This compilation would fail prior to fix.
            theano.function([x], y)
        finally:
            config.on_opt_error = backup

    def test_local_sum_sum_dtype(self):
        # Test that local_sum_sum works when specifying dtypes manually.

        x = tensor.tensor3(dtype='int8')
        y = x.sum(axis=0, dtype='int32').sum(axis=1, dtype='int64')
        backup = config.on_opt_error
        config.on_opt_error = 'raise'
        try:
            # This compilation would fail prior to fix.
            theano.function([x], y)
        finally:
            config.on_opt_error = backup

    def test_local_sum_prod_mul_by_scalar_stack_trace(self):
        # Test that stack trace is copied over correctly for local_sum_prod_mul_by_scalar.
        m0 = theano.compile.get_default_mode()\
            .excluding('inplace_elemwise_opt')\
            .including('canonicalize', 'specialize')

        vect = T.dvector()
        mat = T.dmatrix()
        scalar = T.dscalar()

        f = theano.function([vect, scalar], T.sum(vect * scalar), mode=m0)
        assert check_stack_trace(f, ops_to_check='all')

        f = theano.function([vect], T.sum(-vect), mode=m0)
        assert check_stack_trace(f, ops_to_check=[T.Sum])

        f = theano.function([vect, scalar],
                            T.elemwise.Prod()(vect * scalar), mode=m0)
        assert check_stack_trace(f, ops_to_check=[T.elemwise.Prod])

        f = theano.function([vect], T.elemwise.Prod()(-vect), mode=m0)
        assert check_stack_trace(f, ops_to_check=[T.elemwise.Prod])

        f = theano.function([mat, scalar], T.sum(mat * scalar), mode=m0)
        assert check_stack_trace(f, ops_to_check='all')

        f = theano.function([mat], T.sum(-mat), mode=m0)
        assert check_stack_trace(f, ops_to_check=[T.Sum])


class T_local_opt_alloc(unittest.TestCase):
    dtype = 'float32'

    def test_sum_upcast(self):
        s = theano.tensor.lscalar()
        a = theano.tensor.alloc(np.asarray(5, dtype=self.dtype), s, s)
        orig = theano.config.warn_float64
        theano.config.warn_float64 = "raise"
        try:
            f = theano.function([s], a.sum())
            f(5)
        finally:
            theano.config.warn_float64 = orig

    def test_prod_upcast(self):
        s = theano.tensor.lscalar()
        a = theano.tensor.alloc(np.asarray(5, dtype=self.dtype), s, s)
        orig = theano.config.warn_float64
        theano.config.warn_float64 = "raise"
        try:
            f = theano.function([s], a.prod())
            f(5)
        finally:
            theano.config.warn_float64 = orig

    @change_flags(on_opt_error='raise')
    def test_sum_bool_upcast(self):
        s = theano.tensor.lscalar()
        a = theano.tensor.alloc(np.asarray(True, dtype='bool'), s, s)
        f = theano.function([s], a.sum())
        f(5)
        # test with user specified dtype
        f = theano.function([s], a.sum(dtype=self.dtype))
        f(5)
        # test only 1 axis summed
        f = theano.function([s], a.sum(axis=0, dtype=self.dtype))
        f(5)
        print(self.dtype)


class T_local_opt_alloc_f16(T_local_opt_alloc):
    dtype = 'float16'


class T_local_reduce(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including(
            'canonicalize',
            'specialize',
            'uncanonicalize', 'local_max_and_argmax')

    def test_local_reduce_broadcast_all_0(self):
        for fct in [tensor.sum, tensor.all, tensor.any, tensor.prod,
                    tensor.max, tensor.min]:
            x = T.TensorType('int64', (True, True, True))()
            f = theano.function([x], [fct(x)], mode=self.mode)
            assert not any([
                isinstance(node.op, T.CAReduce)
                for node in f.maker.fgraph.toposort()])

    def test_local_reduce_broadcast_all_1(self):
        for fct in [tensor.sum, tensor.all, tensor.any, tensor.prod,
                    tensor.max, tensor.min]:
            x = T.TensorType('int64', (True, True))()
            f = theano.function([x], [fct(x, axis=[0, 1])], mode=self.mode)
            assert not any([
                isinstance(node.op, T.CAReduce)
                for node in f.maker.fgraph.toposort()])

    def test_local_reduce_broadcast_some_0(self):
        for fct in [tensor.sum, tensor.all, tensor.any, tensor.prod,
                    tensor.max, tensor.min]:
            x = T.TensorType('int64', (True, False, True))()
            f = theano.function([x], [fct(x, axis=[0, 1])], mode=self.mode)

            order = f.maker.fgraph.toposort()
            assert 1 == sum([isinstance(node.op, T.CAReduce)
                             for node in order])

            node = [node for node in order if isinstance(node.op,
                                                         tensor.CAReduce)][0]

            op = node.op
            assert isinstance(op, T.CAReduce)
            # -- the leading broadcastable dimension has been dropped
            #   by the local_reduce_broadcastable optimization
            #   now summation is over the original x's dimension 1.
            assert node.inputs[0].ndim == 2, node
            assert op.axis == (0,), op.axis

    def test_local_reduce_broadcast_some_1(self):
        for fct in [tensor.sum, tensor.all, tensor.any, tensor.prod,
                    tensor.max, tensor.min]:
            x = T.TensorType('int64', (True, True, True))()
            f = theano.function([x], [fct(x, axis=[0, 2])], mode=self.mode)
            assert not any([
                isinstance(node.op, T.CAReduce)
                for node in f.maker.fgraph.toposort()])

    def test_local_reduce_join(self):
        vx = matrix()
        vy = matrix()
        vz = matrix()
        x = np.asarray([[1, 0], [3, 4]], dtype=config.floatX)
        y = np.asarray([[4, 0], [2, 1]], dtype=config.floatX)
        z = np.asarray([[5, 0], [1, 2]], dtype=config.floatX)
        # Test different reduction scalar operation
        for out, res in [
            (T.max((vx, vy), 0), np.max((x, y), 0)),
            (T.min((vx, vy), 0), np.min((x, y), 0)),
            (T.sum((vx, vy, vz), 0), np.sum((x, y, z), 0)),
            (T.prod((vx, vy, vz), 0), np.prod((x, y, z), 0)),
            (T.prod((vx, vy.T, vz), 0), np.prod((x, y.T, z), 0)),
        ]:
            f = theano.function([vx, vy, vz], out,
                                on_unused_input='ignore', mode=self.mode)
            assert (f(x, y, z) == res).all(), out
            topo = f.maker.fgraph.toposort()
            assert len(topo) <= 2, out
            assert isinstance(topo[-1].op, T.Elemwise), out

        # Test different axis for the join and the reduction
        # We must force the dtype, of otherwise, this tests will fail
        # on 32 bit systems
        A = theano.shared(np.array([1, 2, 3, 4, 5], dtype='int64'))

        f = theano.function([], T.sum(T.stack([A, A]), axis=0), mode=self.mode)
        utt.assert_allclose(f(), [2, 4, 6, 8, 10])
        topo = f.maker.fgraph.toposort()
        assert isinstance(topo[-1].op, T.Elemwise)

        # Test a case that was bugged in a old Theano bug
        try:
            old = theano.config.warn.reduce_join
            theano.config.warn.reduce_join = False
            f = theano.function([], T.sum(T.stack([A, A]), axis=1),
                                mode=self.mode)
        finally:
            theano.config.warn.reduce_join = old
        utt.assert_allclose(f(), [15, 15])
        topo = f.maker.fgraph.toposort()
        assert not isinstance(topo[-1].op, T.Elemwise)

        # This case could be optimized
        A = theano.shared(np.array([1, 2, 3, 4, 5]).reshape(5, 1))
        f = theano.function([], T.sum(T.concatenate((A, A), axis=1), axis=1),
                            mode=self.mode)
        utt.assert_allclose(f(), [2, 4, 6, 8, 10])
        topo = f.maker.fgraph.toposort()
        assert not isinstance(topo[-1].op, T.Elemwise)

        A = theano.shared(np.array([1, 2, 3, 4, 5]).reshape(5, 1))
        f = theano.function([], T.sum(T.concatenate((A, A), axis=1), axis=0),
                            mode=self.mode)
        utt.assert_allclose(f(), [15, 15])
        topo = f.maker.fgraph.toposort()
        assert not isinstance(topo[-1].op, T.Elemwise)

        # Test that the optimization does not crash in one case where it
        # is not applied.  Reported at
        # https://groups.google.com/d/topic/theano-users/EDgyCU00fFA/discussion
        old = theano.config.warn.reduce_join
        try:
            theano.config.warn.reduce_join = False
            out = tensor.sum([vx, vy, vz], axis=None)
            f = theano.function([vx, vy, vz], out)
        finally:
            theano.config.warn.reduce_join = old


class T_local_sum_prod_dimshuffle(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('canonicalize')

    def test_local_sum_div_dimshuffle(self):
        a = T.matrix('a')
        b = T.vector('b')
        c = T.tensor3('c')
        d = T.scalar('d')
        sum = tensor.sum
        sums = [
            sum(a / d),
            sum(a / d.dimshuffle('x', 'x')),
            sum(a / d.dimshuffle('x', 'x'), axis=0),
            sum(a / d.dimshuffle('x', 'x'), axis=1),
            sum(b / d),
            sum(b / d.dimshuffle('x')),
            sum(c / d),
            sum(c / d.dimshuffle('x', 'x', 'x')),
            sum(c / d.dimshuffle('x', 'x', 'x'), axis=0),
            sum(c / d.dimshuffle('x', 'x', 'x'), axis=1),
            sum(c / d.dimshuffle('x', 'x', 'x'), axis=2),

            sum(a / b, axis=0),
            sum(a / b.dimshuffle(0, 'x'), axis=1),
            sum(a.dimshuffle(0, 1) / b.dimshuffle(0, 'x'), axis=1),
            sum(a.dimshuffle(1, 0) / b.dimshuffle(0, 'x'), axis=1),
            sum(c / a, axis=0),
            sum(c / a.dimshuffle(1, 0), axis=0),
            sum(c / a.dimshuffle(0, 'x', 1), axis=1),
            sum(c / a.dimshuffle(1, 'x', 0), axis=1),
            sum(c / a.dimshuffle(0, 1, 'x'), axis=2),
            sum(c / a.dimshuffle(1, 0, 'x'), axis=2),
            sum(c / b, axis=0),
            sum(c / b, axis=1),
            sum(c / b, axis=(0, 1)),
            sum(c / b.dimshuffle(0, 'x'), axis=0),
            sum(c / b.dimshuffle(0, 'x'), axis=2),
            sum(c / b.dimshuffle(0, 'x'), axis=(0, 2)),
            sum(c / b.dimshuffle(0, 'x', 'x'), axis=1),
            sum(c / b.dimshuffle(0, 'x', 'x'), axis=2),
            sum(c / b.dimshuffle(0, 'x', 'x'), axis=(1, 2)),
            sum(sum(c, axis=0) / b, axis=0),
            sum(sum(c, axis=1) / b, axis=0),
            ]

        rng = np.random.RandomState(utt.fetch_seed())
        a_val = rng.randn(2, 2).astype(config.floatX)
        b_val = rng.randn(2).astype(config.floatX)
        c_val = rng.randn(2, 2, 2).astype(config.floatX)
        d_val = np.asarray(rng.randn(), config.floatX)

        backup = config.warn.sum_sum_bug, config.warn.sum_div_dimshuffle_bug
        config.warn.sum_sum_bug = False
        config.warn.sum_div_dimshuffle_bug = False
        try:
            for i, s in enumerate(sums):
                print(i)
                f = theano.function([a, b, c, d], s, mode=self.mode,
                                    on_unused_input='ignore')
                g = f.maker.fgraph.toposort()
                assert isinstance(g[-1].op.scalar_op,
                                  theano.scalar.basic.TrueDiv)
                f(a_val, b_val, c_val, d_val)
        finally:
            config.warn.sum_sum_bug, config.warn.sum_div_dimshuffle_bug =\
                backup

    def test_local_prod_div_dimshuffle(self):
        a = T.matrix('a')
        b = T.vector('b')
        c = T.tensor3('c')
        e = T.matrix('e')
        d = T.scalar('d')
        prod = T.prod
        prods = [
            prod(a / d),
            prod(a / d.dimshuffle('x', 'x')),
            prod(a / d.dimshuffle('x', 'x'), axis=0),
            prod(a / d.dimshuffle('x', 'x'), axis=1),
            prod(b / d),
            prod(b / d.dimshuffle('x')),
            prod(c / d),
            prod(c / d.dimshuffle('x', 'x', 'x')),
            prod(c / d.dimshuffle('x', 'x', 'x'), axis=0),
            prod(c / d.dimshuffle('x', 'x', 'x'), axis=1),
            prod(c / d.dimshuffle('x', 'x', 'x'), axis=2),

            prod(a / b, axis=0),
            prod(a / b.dimshuffle(0, 'x'), axis=1),
            prod(a.dimshuffle(0, 1) / b.dimshuffle(0, 'x'), axis=1),
            prod(a.dimshuffle(1, 0) / b.dimshuffle(0, 'x'), axis=1),
            prod(c / a, axis=0),
            prod(c / a.dimshuffle(1, 0), axis=0),
            prod(c / a.dimshuffle(0, 'x', 1), axis=1),
            prod(c / a.dimshuffle(1, 'x', 0), axis=1),
            prod(c / a.dimshuffle(0, 1, 'x'), axis=2),
            prod(c / a.dimshuffle(1, 0, 'x'), axis=2),
            prod(c / b, axis=0),
            prod(c / b, axis=1),
            prod(c / b, axis=(0, 1)),
            prod(c / b.dimshuffle(0, 'x'), axis=0),
            prod(c / b.dimshuffle(0, 'x'), axis=2),
            prod(c / b.dimshuffle(0, 'x'), axis=(0, 2)),
            prod(c / b.dimshuffle(0, 'x', 'x'), axis=1),
            prod(c / b.dimshuffle(0, 'x', 'x'), axis=2),
            prod(c / b.dimshuffle(0, 'x', 'x'), axis=(1, 2)),
            prod(c / b.dimshuffle(0, 'x', 'x'), axis=(0, 1)),
            prod(c / b.dimshuffle(0, 'x', 'x'), axis=(1, 0)),
            prod(prod(c, axis=0) / b, axis=0),
            prod(prod(c, axis=1) / b, axis=0)]

        rng = np.random.RandomState(utt.fetch_seed())
        a_val = rng.randn(2, 2).astype(config.floatX)
        b_val = rng.randn(2).astype(config.floatX)
        c_val = rng.randn(2, 2, 2).astype(config.floatX)
        d_val = np.asarray(rng.randn(), config.floatX)

        default_mode = theano.compile.mode.get_default_mode()
        # FusionOptimizer is included to make sure that expected_outer_operator
        # remains the same for all optimization modes.
        mode_with_opt = default_mode.including('local_sum_prod_div_dimshuffle',
                                               'FusionOptimizer')
        mode_without_opt = default_mode.excluding('local_sum_prod_div_dimshuffle')

        # Numerical tests: tests whether the numerical values with and without
        #                  optimizer are equal or not.
        for i, s in enumerate(prods):
            f = theano.function([a, b, c, d], s,
                                on_unused_input='ignore',
                                mode=mode_without_opt)
            g = theano.function([a, b, c, d], s,
                                on_unused_input='ignore',
                                mode=mode_with_opt)

            utt.assert_allclose(f(a_val, b_val, c_val, d_val),
                                g(a_val, b_val, c_val, d_val))

        # Logical tests: tests whether the optimizer has been appplied or not
        #                by checking graph structure.
        prods = [
            prod(a / e),
            prod(a / d),
            prod(a / d.dimshuffle('x', 'x')),
            prod(c / d.dimshuffle('x', 'x', 'x'), axis=1),
            prod(a.dimshuffle(1, 0) / b.dimshuffle(0, 'x'), axis=1),
            prod(c / b.dimshuffle(0, 'x', 'x'), axis=(1, 0)),
            prod(prod(c, axis=1) / b, axis=0),
            prod(prod(c, axis=(1, 2)) / b, axis=0)]

        expected_outer_operator = [theano.scalar.basic.Mul,
                                   theano.scalar.basic.Composite,
                                   theano.scalar.basic.Composite,
                                   theano.scalar.basic.TrueDiv,
                                   theano.scalar.basic.Composite,
                                   theano.scalar.basic.Mul,
                                   theano.scalar.basic.Composite,
                                   theano.scalar.basic.Mul]

        for i, s in enumerate(prods):
            g = theano.function([a, b, c, d, e], s,
                                on_unused_input='ignore',
                                mode=mode_with_opt)
            assert isinstance(g.maker.fgraph.toposort()[-1].op.scalar_op,
                              expected_outer_operator[i])

    # TODO:
    # test_local_sum_prod_dimshuffle (a * b * c)
    # test_local_sum_divprod_dimshuffle ((a * b) / (c * d))


class TestMakeVector(utt.InferShapeTester):

    def setUp(self):
        super(TestMakeVector, self).setUp()

    def test_make_vector(self):
        b = T.bscalar()
        i = T.iscalar()
        d = T.dscalar()

        # TODO: draw random values instead. Not really important.
        val = {b: 2,
               i: -3,
               d: 0.7}

        # Should work
        for (dtype, inputs) in [("int8", (b, b)),
                                ("int32", (i, b)),
                                ("int32", (b, i)),
                                ("float64", (b, i)),
                                ("float64", (b, d)),
                                ("float64", (d, i)),
                                ("float64", ()),
                                ("int64", ()),
                                ]:
            mv = opt.MakeVector(dtype=dtype)(*inputs)
            assert mv.dtype == dtype
            f = theano.function([b, i, d], mv, on_unused_input='ignore')
            f(val[b], val[i], val[d])

            s = mv.sum()
            gb = T.grad(s, b, disconnected_inputs='ignore')
            gi = T.grad(s, i, disconnected_inputs='ignore')
            gd = T.grad(s, d, disconnected_inputs='ignore')
            # print 'gb =', gb
            # print 'gi =', gi
            # print 'gd =', gd

            g = theano.function([b, i, d], [gb, gi, gd])
            g_val = g(val[b], val[i], val[d])
            # print 'g_val =', g_val

            if dtype in tensor.int_dtypes:
                # The gradient should be 0
                utt.assert_allclose(g_val, 0)
            else:
                for var, grval in zip((b, i, d), g_val):
                    float_inputs = []
                    if var.dtype in tensor.int_dtypes:
                        pass
                        # Currently we don't do any checks on these variables
                        # verify_grad doesn't support integer inputs yet
                        # however, the gradient on them is *not* defined to
                        # be 0
                    elif var not in inputs:
                        assert grval == 0
                    else:
                        float_inputs.append(var)

                # Build a function that takes float_inputs, use fix values for the
                # other inputs, and returns the MakeVector. Use it for verify_grad.
                if float_inputs:
                    def fun(*fl_inputs):
                        f_inputs = []
                        for var in f_inputs:
                            if var in fl_inputs:
                                # use symbolic variable
                                f_inputs.append(var)
                            else:
                                # use constant value
                                f_inputs.append(val[var])
                        return opt.MakeVector(dtype=dtype)(*f_inputs)

                    utt.verify_grad(fun, [val[ri] for ri in float_inputs])

        # should fail
        for (dtype, inputs) in [("int8", (b, i)),
                                ("int8", (i, b)),
                                ("int8", (b, d)),
                                ("int8", (i, i)),
                                ("int32", (d, i)),
                                ("int32", (i, d)),
                                ("float32", (i, d)),
                                ]:
            try:
                opt.MakeVector(dtype=dtype)(*inputs)
                raise Exception("Theano should have raised an error")
            except AssertionError:
                pass

    def test_infer_shape(self):
        adscal = dscalar()
        bdscal = dscalar()
        aiscal = iscalar()
        biscal = iscalar()
        ciscal = iscalar()
        discal = iscalar()
        adscal_val = np.random.rand()
        bdscal_val = np.random.rand()
        aiscal_val = np.random.randint(10)
        biscal_val = np.random.randint(10)
        ciscal_val = np.random.randint(10)
        discal_val = np.random.randint(10)
        self._compile_and_check([adscal, aiscal],
                                [MakeVector('float64')(adscal, aiscal)],
                                [adscal_val, aiscal_val], MakeVector)

        self._compile_and_check([adscal, bdscal, aiscal],
                                [MakeVector('float64')(adscal, bdscal, aiscal)],
                                [adscal_val, bdscal_val, aiscal_val], MakeVector)

        self._compile_and_check([aiscal, biscal, ciscal, discal],
                                [MakeVector('int32')(aiscal, biscal, ciscal, discal)],
                                [aiscal_val, biscal_val, ciscal_val, discal_val],
                                MakeVector)


def test_local_join_1():
    # test for vector
    a = tensor.vector('a')
    s = tensor.stack([a])
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(0,a)
    a = tensor.matrix('a')
    s = join(0, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    s = join(1, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test we don't apply when their is 2 inputs
    s = join(1, a, a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_empty():
    # test for vector, vector, empty to vector
    empty_vec = np.asarray([], dtype=config.floatX)
    a = tensor.vector('a')
    s = tensor.join(0, a, a, empty_vec)
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all([not isinstance(n.op, Join) or len(n.inputs) == 3
                for n in e if isinstance(n.op, Join)])
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    # test for matrix join(1,a)
    empty_mat = np.asarray([[]], dtype=config.floatX)
    m = tensor.matrix('m')
    s = join(1, empty_mat, m, m, m)
    f = function([m], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all([not isinstance(n.op, Join) or len(n.inputs) == 4
                for n in e if isinstance(n.op, Join)])
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for vector, vector, empty to matrix
    # We can't optimize this case.
    s = tensor.stack([a, a, empty_vec])
    f = function([a], s, mode=mode_opt)
    val = f([])
    assert np.all(val == [1])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all([not isinstance(n.op, Join) or len(n.inputs) == 4
                for n in e if isinstance(n.op, Join)])
    assert f.maker.fgraph.outputs[0].dtype == config.floatX
    # test for matrix join(0,a)
    # We can't optimize this case.
    s = join(0, m, np.asarray([[2.]], dtype=config.floatX), m)
    f = function([m], s, mode=mode_opt)
    val = f([[1]])
    assert np.all(val == [[1], [2], [1]])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all([not isinstance(n.op, Join) or len(n.inputs) == 4
                for n in e if isinstance(n.op, Join)])
    assert f.maker.fgraph.outputs[0].dtype == config.floatX


def test_local_join_make_vector():
    a, b, c, d, e = tensor.scalars('abcde')
    v = tensor.vector('v')
    mv = MakeVector(config.floatX)
    s = tensor.join(0, mv(a), v, mv(b, c), mv(d, e))
    f = function([a, b, c, d, e, v], s, mode=mode_opt)
    theano.printing.debugprint(f)
    val = f(1, 2, 3, 4, 6, [7, 8])
    assert np.all(val == [1, 7, 8, 2, 3, 4, 6])
    e = f.maker.fgraph.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert all([not isinstance(n.op, Join) or len(n.inputs) == 4
                for n in e if isinstance(n.op, Join)])
    assert f.maker.fgraph.outputs[0].dtype == config.floatX

    assert check_stack_trace(f, ops_to_check='all')


def test_local_add_specialize():
    # test of non-zero dimension
    a = tensor.vector()
    s = tensor.add(tensor.zeros_like(a))
    assert local_add_specialize.transform(s.owner)

    # test of 0-d
    a = tensor.scalar()
    s = tensor.add(tensor.zeros_like(a))
    assert local_add_specialize.transform(s.owner)

    # Test when the 0 input is forcing upcasting
    a = tensor.constant(0, dtype='int64')
    b = tensor.constant(1, dtype='int32')
    s = a + b
    transformed = local_add_specialize.transform(s.owner)
    assert transformed
    assert transformed[0].type == s.type


def test_local_tensor_scalar_tensor():
    dtypes = ['int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64',
              'float32', 'float64',
              'complex64', 'complex128'
              ]

    for dtype in dtypes:
        t_type = TensorType(dtype=dtype, broadcastable=())
        t = t_type()
        s = tensor.scalar_from_tensor(t)
        t2 = tensor.tensor_from_scalar(s)

        f = function([t], t2, mode=mode_opt)
        e = f.maker.fgraph.toposort()
        cast_nodes = [n for n in e
                      if isinstance(n.op, (tensor.TensorFromScalar,
                                           tensor.ScalarFromTensor))]
        assert len(cast_nodes) == 0
        f(0)


def test_local_scalar_tensor_scalar():
    dtypes = ['int8', 'int16', 'int32', 'int64',
              'uint8', 'uint16', 'uint32', 'uint64',
              'float32', 'float64',
              'complex64', 'complex128'
              ]

    for dtype in dtypes:
        s_type = theano.scalar.Scalar(dtype=dtype)
        s = s_type()
        t = tensor.tensor_from_scalar(s)
        s2 = tensor.scalar_from_tensor(t)

        f = function([s], s2, mode=mode_opt)
        e = f.maker.fgraph.toposort()
        cast_nodes = [n for n in e
                      if isinstance(n.op, (tensor.TensorFromScalar,
                                           tensor.ScalarFromTensor))]
        assert len(cast_nodes) == 0
        f(0)


def test_local_div_to_inv():
    num_len_s = tensor.lscalar('num_len')
    denom_s = tensor.scalar('denom')

    num_v = tensor.alloc(1, num_len_s)
    denom_m = denom_s.dimshuffle('x', 'x')

    out = num_v / denom_m
    assert np.all(out.broadcastable == (True, False))

    f = theano.function([num_len_s, denom_s], out)
    out_val = f(3, 2.)
    assert out_val.shape == (1, 3)
    utt.assert_allclose(out_val, 0.5)


def test_local_useless_split():
    x = tensor.matrix('x')
    splits = tensor.ivector('splits')
    opt = tensor.split(x, splits, n_splits=1)
    nonopt = tensor.split(x, splits, n_splits=3)

    mode = compile.get_default_mode().including("local_useless_split")
    f_opt = theano.function([x, splits], opt, mode=mode)
    f_nonopt = theano.function([x, splits], nonopt, mode=mode)

    f_opt(np.random.rand(4, 4).astype(config.floatX), [4])
    f_nonopt(np.random.rand(4, 4).astype(config.floatX), [1, 2, 1])
    graph_opt = f_opt.maker.fgraph.toposort()
    graph_nonopt = f_nonopt.maker.fgraph.toposort()

    assert isinstance(graph_opt[-1].op, DeepCopyOp)
    assert len(graph_nonopt) == 1
    assert isinstance(graph_nonopt[0].op, tensor.Split)

    assert check_stack_trace(f_opt, ops_to_check=[Assert])
    assert check_stack_trace(f_nonopt, ops_to_check='all')


def test_local_flatten_lift():
    for i in xrange(1, 4):
        x = tensor.tensor4()
        out = tensor.flatten(T.exp(x), i)
        assert out.ndim == i
        mode = compile.mode.get_default_mode()
        mode = mode.including('local_reshape_lift')
        f = theano.function([x], out, mode=mode)
        x_np = np.random.rand(5, 4, 3, 2).astype(config.floatX)
        out_np = f(x_np)
        topo = f.maker.fgraph.toposort()
        shape_out_np = tuple(x_np.shape[:i - 1]) + (np.prod(x_np.shape[i - 1:]),)
        assert shape_out_np == out_np.shape

        reshape_nodes = [n for n in topo if isinstance(n.op, tensor.Reshape)]
        assert (len(reshape_nodes) == 1 and
                tensor.is_flat(reshape_nodes[0].outputs[0], ndim=i))
        assert isinstance(topo[-1].op, tensor.Elemwise)


class Test_Reshape(unittest.TestCase):
    def setUp(self):
        self.mode = mode_opt
        self.op = tensor.Reshape

    def test_local_reshape(self):
        a = tensor.fmatrix()
        b = self.op(3)(a, [2, 3, 4])
        c = self.op(1)(b, [24])
        f = theano.function([a], c, mode=self.mode)
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(node.op, self.op) for node in topo) == 1

        # Check stack trace
        self.assertTrue(check_stack_trace(f, ops_to_check=[self.op]))


class Test_local_useless_reshape(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    def test_0(self):
        mode = theano.compile.get_default_mode().including(
            'local_useless_reshape')
        i = T.iscalar('i')
        m = theano.tensor.mgrid[0:i, ]
        f = theano.function([i], m, mode=mode)
        topo = f.maker.fgraph.toposort()
        assert not any(isinstance(n.op, tensor.basic.Reshape) for n in topo)

    def test_1(self):
        x = theano.tensor.matrix('x')
        r = x.reshape(x.shape)

        m0 = theano.compile.get_default_mode()
        m1 = m0.including('local_useless_reshape')
        f1 = theano.function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, tensor.basic.Reshape) for n in topo)

        m2 = m1.excluding('ShapeOpt')
        f2 = theano.function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, tensor.basic.Reshape) for n in topo)

        # We do not need tests checking that stack traces are copied over,
        # because local_useless_reshape only removes nodes from the graph

    def test_2(self):
        x = theano.tensor.matrix('x')
        r = x.reshape([Shape_i(i)(x) for i in xrange(x.ndim)])

        m0 = theano.compile.get_default_mode()
        m1 = m0.including('local_useless_reshape')
        f1 = theano.function([x], r, mode=m1)
        topo = f1.maker.fgraph.toposort()
        assert not any(isinstance(n.op, tensor.basic.Reshape) for n in topo)

        m2 = m1.excluding('ShapeOpt')
        f2 = theano.function([x], r, mode=m2)
        topo = f2.maker.fgraph.toposort()
        assert not any(isinstance(n.op, tensor.basic.Reshape) for n in topo)

    def test_m1(self):
            x = theano.tensor.matrix('x')
            r = x.reshape((x.shape[0], -1))

            m0 = theano.compile.get_default_mode()
            m1 = m0.including('local_useless_reshape')
            f1 = theano.function([x], r, mode=m1)
            topo = f1.maker.fgraph.toposort()
            assert not any(isinstance(n.op, tensor.basic.Reshape) for n in topo)

            m2 = m1.excluding('ShapeOpt')
            f2 = theano.function([x], r, mode=m2)
            topo = f2.maker.fgraph.toposort()
            assert not any(isinstance(n.op, tensor.basic.Reshape) for n in topo)


class Test_local_reshape_to_dimshuffle(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(utt.fetch_seed())

    def test_1(self):
        reshape_lift = out2in(local_reshape_to_dimshuffle)
        useless_reshape = out2in(local_useless_reshape)
        x = shared(self.rng.randn(4,))
        y = shared(self.rng.randn(5, 6))
        reshape_x = tensor.reshape(x, (1, 4))
        reshape_y = tensor.reshape(y, (1, 5, 1, 6, 1, 1))

        g = FunctionGraph([x, y], [reshape_x, reshape_y])
        self.assertTrue(str(g) == ("[Reshape{2}"
                                   "(<TensorType(float64, vector)>, "
                                   "TensorConstant{[1 4]}), "
                                   "Reshape{6}"
                                   "(<TensorType(float64, matrix)>, "
                                   "TensorConstant{[1 5 1 6 1 1]})]"))

        reshape_lift.optimize(g)
        useless_reshape.optimize(g)
        self.assertTrue(str(g) == "[InplaceDimShuffle{x,0}"
                                  "(<TensorType(float64, vector)>), "
                                  "InplaceDimShuffle{x,0,x,1,x,x}"
                                  "(Reshape{2}(<TensorType(float64, matrix)>, "
                                  "TensorConstant{[5 6]}))]")

        # Check stacktrace was copied over correctly after opt was applied
        assert check_stack_trace(g, ops_to_check=(T.DimShuffle, T.Reshape))


def test_local_reshape_lift():
    x = tensor.tensor4()
    out = T.exp(x).reshape([x.size])
    assert out.ndim == 1
    mode = compile.mode.get_default_mode()
    mode = mode.including('local_reshape_lift')
    f = theano.function([x], out, mode=mode)
    f(np.random.rand(5, 4, 3, 2).astype(config.floatX))
    topo = f.maker.fgraph.toposort()
    assert isinstance(topo[-2].op, tensor.Reshape)
    assert isinstance(topo[-1].op, tensor.Elemwise)
    # Check stacktrace was copied over correctly after opt was applied
    assert check_stack_trace(f, ops_to_check='last')


class Test_lift_transpose_through_dot(unittest.TestCase):
    def simple_optimize(self, g):
        out2in(opt.local_useless_elemwise).optimize(g)
        out2in(opt.local_lift_transpose_through_dot).optimize(g)
        out2in(opt.local_useless_elemwise).optimize(g)
        return g

    def test_matrix_matrix(self):
        a, b = matrices('ab')
        g = self.simple_optimize(FunctionGraph([a, b], [tensor.dot(a, b).T]))
        sg = '[dot(InplaceDimShuffle{1,0}(b), InplaceDimShuffle{1,0}(a))]'
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(g, ops_to_check='all'))

    def test_row_matrix(self):
        a = vector('a')
        b = matrix('b')
        g = optimize(FunctionGraph(
            [a, b],
            [tensor.dot(a.dimshuffle('x', 0), b).T]),
            level='stabilize')
        sg = '[dot(InplaceDimShuffle{1,0}(b), InplaceDimShuffle{0,x}(a))]'
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(g, ops_to_check='all'))

    def test_matrix_col(self):
        a = vector('a')
        b = matrix('b')
        g = optimize(FunctionGraph(
            [a, b],
            [tensor.dot(b, a.dimshuffle(0, 'x')).T]),
            level='stabilize')
        sg = '[dot(InplaceDimShuffle{x,0}(a), InplaceDimShuffle{1,0}(b))]'
        assert str(g) == sg, (str(g), sg)
        # Check stacktrace was copied over correctly after opt was applied
        self.assertTrue(check_stack_trace(g, ops_to_check='all'))


def test_local_upcast_elemwise_constant_inputs():
    s = dvector("s")
    x = tensor.sum(tensor.log(10 ** s))
    f = function([s], [tensor.grad(x, s)])
    f([-42, -2.1, -1, -0.5, 0, 0.2, 1, 2, 12])

    # This test a corner where the optimization should not be applied.
    old = theano.config.floatX
    theano.config.floatX = 'float32'
    try:
        v = lvector()
        function([v], theano.tensor.basic.true_div(v, 2))
    finally:
        theano.config.floatX = old


class TestShape_i(utt.InferShapeTester):

    def setUp(self):
        super(TestShape_i, self).setUp()

    def test_perform(self):

        advec = vector()
        advec_val = np.random.rand(3).astype(config.floatX)
        f = function([advec], Shape_i(0)(advec))
        out = f(advec_val)
        utt.assert_allclose(out, advec_val.shape[0])

        admat = matrix()
        admat_val = np.random.rand(4, 3).astype(config.floatX)
        for i in xrange(2):
            f = function([admat], Shape_i(i)(admat))
            out = f(admat_val)
            utt.assert_allclose(out, admat_val.shape[i])

    def test_infer_shape(self):
        admat = matrix()
        admat_val = np.random.rand(3, 4).astype(config.floatX)
        self._compile_and_check([admat], [Shape_i(0)(admat)],
                                [admat_val], Shape_i)

        self._compile_and_check([admat], [Shape_i(1)(admat)],
                                [admat_val], Shape_i)


class TestShapeFeature(unittest.TestCase):
    def test_scalar(self):
        x = scalar()
        cst = T.constant(1).clone()
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = opt.ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_vector(self):
        x = vector()
        cst = T.constant(1).clone()
        o = x + cst
        fgraph = FunctionGraph([x], [o], clone=False)
        shape_feature = opt.ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)

    def test_vector2(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = opt.ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o)
        # The following case isn't implemented
        assert not shape_feature.same_shape(y, o)

    def test_vector_dim(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = opt.ShapeFeature()
        fgraph.attach_feature(shape_feature)
        assert shape_feature.same_shape(x, o, 0, 0)
        # The following case isn't implemented
        assert not shape_feature.same_shape(y, o, 0, 0)

    def test_vector_dim_err(self):
        x = vector()
        y = vector()
        o = x + y
        fgraph = FunctionGraph([x, y], [o], clone=False)
        shape_feature = opt.ShapeFeature()
        fgraph.attach_feature(shape_feature)
        self.assertRaises(IndexError, shape_feature.same_shape, x, o, 1, 0)
        self.assertRaises(IndexError, shape_feature.same_shape, x, o, 0, 1)


def test_assert_op_gradient():
    x = T.vector('x')
    assert_op = Assert()
    cost = T.sum(assert_op(x, x.size < 2))
    grad = T.grad(cost, x)
    func = theano.function([x], grad)

    x_val = np.ones(shape=(1,), dtype=theano.config.floatX)
    assert func(x_val) == 1


class TestIntDivByOne(unittest.TestCase):

    def setUp(self):
        self.mode = theano.compile.mode.get_default_mode()
        self.mode = self.mode.including('local_intdiv_by_one')

    def test1(self):
        # Tests removing the extra floor_div by 1 introduced by
        # local_subtensor_merge optimization

        y = T.tensor4('y')
        self.mode = self.mode.excluding('fusion')
        f = theano.function([y], y[::-1][::-1], mode=self.mode)

        graph = f.maker.fgraph.toposort()
        divs = [node for node in graph
                if isinstance(node.op, T.elemwise.Elemwise) and
                isinstance(node.op.scalar_op, theano.scalar.IntDiv)]
        assert len(divs) == 0

    def test2(self):
        # Simple test case for removing dividing by 1
        y = T.tensor4('y')
        z = y // 1
        f = theano.function([y], z, mode=self.mode)
        graph = f.maker.fgraph.toposort()
        divs = [node for node in graph
                if isinstance(node.op, T.elemwise.Elemwise) and
                isinstance(node.op.scalar_op, theano.scalar.IntDiv)]
        assert len(divs) == 0

    def test3(self):
        # Simple test case for removing dividing by a tensor of ones
        y = T.tensor4('y')
        z = y // np.ones((2, 2, 2, 2))
        f = theano.function([y], z, mode=self.mode)
        graph = f.maker.fgraph.toposort()
        divs = [node for node in graph
                if isinstance(node.op, T.elemwise.Elemwise) and
                isinstance(node.op.scalar_op, theano.scalar.IntDiv)]
        assert len(divs) == 0


def test_local_zero_div():
    # Tests 0/x -> 0

    for t in (T.scalar, T.ivector, T.ftensor4):
        x = t('x')
        for op in (T.int_div, T.true_div):
            y = op(0, x)
            g = optimize(FunctionGraph([x], [y]))
            # the division should be gone
            divs = [node for node in g.toposort()
                    if isinstance(node.op, T.elemwise.Elemwise) and
                    isinstance(node.op.scalar_op, type(op.scalar_op))]
            assert len(divs) == 0
            # the output type should match the unoptimized one
            output = g.outputs[0]
            assert output.ndim == y.ndim
            assert output.type == y.type
            # and the output should be zero
            assert theano.tensor.get_scalar_constant_value(output) == 0


def test_local_sumsqr2dot():
    G = matrix('G')
    W = matrix('W')

    y = T.sqr(W.dimshuffle('x', 0, 1) * G.dimshuffle(0, 'x', 1)).sum(axis=(1, 2))
    MODE = theano.compile.get_default_mode().including('local_sumsqr2dot')

    f = function([W, G], y, mode=MODE)

    w_val = np.random.rand(4, 3).astype(config.floatX)
    g_val = np.random.rand(5, 3).astype(config.floatX)

    f_val = f(w_val, g_val)
    f_test = np.dot(np.square(g_val), np.square(w_val).sum(axis=0))

    utt.assert_allclose(f_val, f_test)
    assert any(isinstance(n.op, (tensor.basic.Dot, tensor.blas.Dot22,
                                 tensor.blas.Gemv, tensor.blas_c.CGemv))
               for n in f.maker.fgraph.toposort())


def test_local_expm1():
    x = matrix('x')
    u = T.scalar('u')

    y = T.exp(x) - 1.
    z = T.exp(x) - 2.
    t = T.exp(x) - x
    s = T.exp(u) - np.ones((4, 3)).astype(config.floatX)
    MODE = theano.compile.get_default_mode().including('local_expm1')
    f = function([x], y, mode=MODE)
    g = function([x], z, mode=MODE)
    h = function([x], t, mode=MODE)
    r = function([u], s, mode=MODE)
    x_val = np.random.rand(4, 3).astype(config.floatX)
    f_val = f(x_val)
    f_test = function([x], T.expm1(x), mode=MODE)

    utt.assert_allclose(f_val, f_test(x_val))

    assert any(isinstance(n.op, T.Elemwise) and isinstance(n.op.scalar_op, theano.scalar.basic.Expm1)
               for n in f.maker.fgraph.toposort())

    assert not any(isinstance(n.op, T.Elemwise) and isinstance(n.op.scalar_op, theano.scalar.basic.Expm1)
                   for n in g.maker.fgraph.toposort())

    assert not any(isinstance(n.op, T.Elemwise) and isinstance(n.op.scalar_op, theano.scalar.basic.Expm1)
                   for n in h.maker.fgraph.toposort())

    assert not any(isinstance(n.op, T.Elemwise) and isinstance(n.op.scalar_op, theano.scalar.basic.Expm1)
                   for n in r.maker.fgraph.toposort())


def test_local_merge_alloc():
    # Add this opt to the default mode,
    # otherwise, FAST_COMPILE fails.
    default_mode = theano.compile.mode.get_default_mode()
    opt_mode = default_mode.including("local_merge_alloc")

    x = T.iscalar('x')
    y = T.iscalar('y')
    y2 = T.iscalar('y2')
    z = T.iscalar('z')
    w = T.iscalar('w')
    m = T.fscalar('m')
    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = T.alloc(T.alloc(m, 1, y, 1, 1), x, y, z, w)
    f = theano.function([m, x, y, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, T.Alloc)
    o = f(0., 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = T.alloc(T.alloc(m, y, 1, 1), x, y, z, w)
    f = theano.function([m, x, y, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, T.Alloc)
    o = f(0., 1, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = T.alloc(T.alloc(m, y, 1, 1), x, y2, z, w)
    f = theano.function([m, x, y, y2, z, w], output, mode=opt_mode)
    topo = f.maker.fgraph.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, T.opt.Assert)
    assert isinstance(topo[-1].op, T.Alloc)
    o = f(0., 1, 2, 2, 3, 4)
    assert o.shape == (1, 2, 3, 4)
    assert_raises((AssertionError, ValueError), f, 0., 1, 2, 5, 3, 4)


def test_local_useless_alloc():

    useless_alloc = out2in(local_useless_alloc)
    merge_alloc = out2in(local_merge_alloc)

    x = T.iscalar('x')
    y = T.iscalar('y')
    y2 = T.iscalar('y2')
    z = T.iscalar('z')
    w = T.iscalar('w')
    m = T.fscalar('m')

    # case 1
    # Alloc(Alloc(m, x, 1, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = T.alloc(T.alloc(m, 1, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, T.Alloc)

    # case 2
    # Alloc(Alloc(m, y, 1, 1), x, y, z, w) -> Alloc(m, x, y, z, w)
    output = T.alloc(T.alloc(m, y, 1, 1), x, y, z, w)
    g = FunctionGraph([m, x, y, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 1
    assert isinstance(topo[0].op, T.Alloc)

    # case 3
    # Alloc(Alloc(m, y1, 1, 1), x, y2, z, w) ->
    #   Alloc(m, x, assert(y1, y1==y2), z, w)
    output = T.alloc(T.alloc(m, y, 1, 1), x, y2, z, w)
    g = FunctionGraph([m, x, y, y2, z, w], [output])

    useless_alloc.optimize(g)
    merge_alloc.optimize(g)
    useless_alloc.optimize(g)

    topo = g.toposort()
    assert len(topo) == 3
    assert isinstance(topo[-2].op, T.opt.Assert)
    assert isinstance(topo[-1].op, T.Alloc)


def compile_graph_log_sum_exp(x, axis, dimshuffle_op=None):
    sum_exp = T.sum(T.exp(x), axis=axis)
    if dimshuffle_op:
        sum_exp = dimshuffle_op(sum_exp)
    y = T.log(sum_exp)
    MODE = theano.compile.get_default_mode().including('local_log_sum_exp')
    return function([x], y, mode=MODE)


def check_max_log_sum_exp(x, axis, dimshuffle_op=None):
    f = compile_graph_log_sum_exp(x, axis, dimshuffle_op)

    fgraph = f.maker.fgraph.toposort()
    for node in fgraph:
        if (hasattr(node.op, 'scalar_op') and
                node.op.scalar_op == theano.scalar.basic.maximum):
            return

        # in mode FAST_COMPILE, the optimisations don't replace the
        # MaxAndArgmax op.
        if isinstance(node.op, theano.tensor.MaxAndArgmax):
            return

    raise Exception('No maximum detected after log_sum_exp optimisation')


def test_local_log_sum_exp1():
    # Tests if optimization is applied by checking the presence of the maximum
    x = tensor3('x')
    check_max_log_sum_exp(x, axis=(0,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(1,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(2,), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(0, 1), dimshuffle_op=None)
    check_max_log_sum_exp(x, axis=(0, 1, 2), dimshuffle_op=None)

    # If a transpose is applied to the sum
    transpose_op = DimShuffle((False, False), (1, 0))
    check_max_log_sum_exp(x, axis=2, dimshuffle_op=transpose_op)

    # If the sum is performed with keepdims=True
    x = TensorType(dtype='floatX', broadcastable=(False, True, False))('x')
    sum_keepdims_op = x.sum(axis=(0, 1), keepdims=True).owner.op
    check_max_log_sum_exp(x, axis=(0, 1), dimshuffle_op=sum_keepdims_op)


def test_local_log_sum_exp2():
    # Tests if the optimization works (result is correct) around 1.0

    x = tensor3('x')
    x_val = 1.0 + np.random.rand(4, 3, 2).astype(config.floatX) / 10.0

    f = compile_graph_log_sum_exp(x, axis=(1,))
    naive_ret = np.log(np.sum(np.exp(x_val), axis=1))
    optimised_ret = f(x_val)
    assert np.allclose(naive_ret, optimised_ret)

    # If a transpose is applied
    transpose_op = DimShuffle((False, False), (1, 0))
    f = compile_graph_log_sum_exp(x, axis=(1,), dimshuffle_op=transpose_op)
    naive_ret = np.log(np.sum(np.exp(x_val), axis=1).T)
    optimised_ret = f(x_val)
    assert np.allclose(naive_ret, optimised_ret)


def test_local_log_sum_exp3():
    # Tests if the optimization works (result is correct) for extreme value 100
    x = vector('x')
    f = compile_graph_log_sum_exp(x, axis=0)

    x_val = np.array([-100., 100.]).astype(config.floatX)

    optimised_ret = f(x_val)

    assert np.allclose(optimised_ret, 100.)
