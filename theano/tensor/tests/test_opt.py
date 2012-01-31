## PENDING REWRITE OF tensor_opt.py

import copy
import time
import unittest

import numpy
from nose.plugins.skip import SkipTest
from numpy.testing import dec
from numpy.testing.noseclasses import KnownFailureTest

import theano
import theano.scalar as scal
from theano import compile
from theano import config
from theano import function
from theano import gof
from theano import pprint
from theano import shared
from theano.gof import Env
from theano.gof.python25 import any, all
import theano.tensor.opt as opt
from theano.tensor.opt import (
        local_add_specialize,
        local_dimshuffle_lift,
        local_greedy_distributor,
        mul_canonizer,
        out2in,
        Shape_i,
        )
from theano import tensor
from theano import tensor as T
from theano.tensor import scalar, iscalar, lscalar, fscalar, dscalar
from theano.tensor import vector, ivector, lvector, fvector, dvector
from theano.tensor import matrix, imatrix, lmatrix, fmatrix, dmatrix
from theano.tensor import scalars, vectors, matrices, fmatrices, dmatrices
from theano.tensor import (
        as_tensor_variable,
        inplace,
        Join,
        join,
        Subtensor,
        TensorType,
        )
from theano.tensor.elemwise import DimShuffle
from theano.tests import unittest_tools as utt

mode_opt = theano.config.mode
if mode_opt == 'FAST_COMPILE':
    mode_opt = 'FAST_RUN'
mode_opt = theano.compile.mode.get_mode(mode_opt)

ds = lambda x, y: DimShuffle(x.type.broadcastable, y)(x)
dimshuffle_lift = out2in(local_dimshuffle_lift)

_optimizer_stabilize = gof.Query(include=['fast_run'])
_optimizer_stabilize.position_cutoff = 1.51
_optimizer_stabilize = compile.optdb.query(_optimizer_stabilize)

_optimizer_specialize = gof.Query(include=['fast_run'])
_optimizer_specialize.position_cutoff = 2.01
_optimizer_specialize = compile.optdb.query(_optimizer_specialize)

_optimizer_fast_run = gof.Query(include=['fast_run'])
_optimizer_fast_run = compile.optdb.query(_optimizer_fast_run)
def optimize(g, level='fast_run'):
    if 'fast_run' is level:
        _optimizer_fast_run.optimize(g)
    elif 'specialize' is level:
        _optimizer_specialize.optimize(g)
    elif 'stabilize' is level:
        _optimizer_stabilize.optimize(g)
    else:
        raise ValueError(level)
    return g


def inputs(xbc = (0, 0), ybc = (0, 0), zbc = (0, 0)):
    x = TensorType(broadcastable = xbc, dtype = 'float64')('x')
    y = TensorType(broadcastable = ybc, dtype = 'float64')('y')
    z = TensorType(broadcastable = zbc, dtype = 'float64')('z')
    return x, y, z


class test_dimshuffle_lift(unittest.TestCase):
    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = Env([x], [e])
        self.assertTrue(str(g) == "[DimShuffle{1,0}(DimShuffle{1,0}(x))]")
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[x]")

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 'x', 0)), (2, 0, 'x', 1))
        g = Env([x], [e])
        self.assertTrue(str(g) == "[DimShuffle{2,0,x,1}(DimShuffle{1,x,0}(x))]", str(g))
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[DimShuffle{0,1,x,x}(x)]", str(g))

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, 'x', 1)), (2, 0, 'x', 1)), (1, 0))
        g = Env([x], [e])
        self.assertTrue(str(g) == "[DimShuffle{1,0}(DimShuffle{2,0,x,1}(DimShuffle{0,x,1}(x)))]", str(g))
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == "[x]", str(g))

    def test_lift(self):
        x, y, z = inputs([False]*1, [False]*2, [False]*3)
        e = x + y + z
        g = Env([x, y, z], [e])
        self.assertTrue(str(g) == ("[Elemwise{add,no_inplace}("
            "InplaceDimShuffle{x,0,1}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{x,0}(x), y)), z)]"), str(g))
        dimshuffle_lift.optimize(g)
        self.assertTrue(str(g) == ("[Elemwise{add,no_inplace}(Elemwise"
            "{add,no_inplace}(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle"
            "{x,0,1}(y)), z)]"), str(g))


def test_add_canonizer_problem0():
    n_segments = 10
    label = lscalar('label')
    segment_labels = label + theano._asarray([0] * n_segments, dtype='int64')

    r = segment_labels * 5
    f = function([label], r)


class test_greedy_distribute(unittest.TestCase):
    def test_main(self):
        a, b, c, d, x, y, z = matrices('abcdxyz')
        e = (a/z + b/x) * x * z
        g = Env([a,b,c,d,x,y,z], [e])
        ##print pprint(g.outputs[0])
        mul_canonizer.optimize(g)
        gof.TopoOptimizer(gof.LocalOptGroup(local_greedy_distributor), order = 'out_to_in').optimize(g)
        ##print pprint(g.outputs[0])

    def test_kording_bug(self):
        x, y = vectors('xy')
        eps = scalar('eps')
        s = scalar('s')

        #r = theano.tensor.mul(theano.tensor.fill(x, 2.*a), x/a , (y+z) , a)
        #r = theano.tensor.mul((x/a+y) , a, z)
        r = tensor.mul(
                s - 1
                , eps + x/s
                , eps + y/s
                , s)

        f = function([s, eps, x,y], r**2)

        s_val = numpy.asarray(4, dtype=config.floatX)
        eps_val = numpy.asarray(1.e-6, dtype=config.floatX)
        x_val = numpy.asarray([1.5,2], dtype=config.floatX)
        y_val = numpy.asarray([2.3,3.1], dtype=config.floatX)

        r0 = f(s_val, eps_val, x_val, y_val)
        r1 = f(s_val, eps_val, x_val, y_val)
        r2 = f(s_val, eps_val, x_val, y_val)

        assert numpy.all(r0 == r1)
        assert numpy.all(r0 == r2)


class test_canonize(unittest.TestCase):
    def test_muldiv(self):
        x, y, z = matrices('xyz')
        a, b, c, d = matrices('abcd')
#        e = (2.0 * x) / (2.0 * y)
#        e = (2.0 * x) / (4.0 * y)
#        e = x / (y / z)
#        e = (x * y) / x
#        e = (x / y) * (y / z) * (z / x)
#        e = (a / b) * (b / c) * (c / d)
#        e = (a * b) / (b * c) / (c * d)
#        e = 2 * x / 2
#        e = x / y / x
#        e = (x / x) * (y / y)
        e = (-1 * x) / y / (-2 * z)
        g = Env([x, y, z, a, b, c, d], [e])
        print pprint(g.outputs[0])
        mul_canonizer.optimize(g)
        print pprint(g.outputs[0])

    def test_elemwise_multiple_inputs_optimisation(self):
        """
        verify that the Canonizer merge sequential Elemwise({mul,add}) part 1
        This part are that case that is done, but don't include case that are not implemented but are suposed to be.
        Test with and without DimShuffle
        """

        shp=(5,5)
        fx, fy, fz = fmatrices('xyz')
        dx, dy, dz = dmatrices('xyz')
        fv = fvector('r').dimshuffle('x',0)
        dv = dvector('s').dimshuffle('x',0)
        fxv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        dxv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dyv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dzv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float64').reshape(1,shp[0])
        cases = [
            (fx+fy,(fx,fy),(fxv,fyv),1,'float32'),
            (fx*fy,(fx,fy),(fxv,fyv),1,'float32'),
#            (fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
#            (dx+dy+dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
#            (fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
#            (dx*dy*dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
#            (fx*fy*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
#            (dx*dy*(dx+dy+dz),(dx,dy,dz),(dxv,dyv,dzv),2,'float64'),
#            (fx*fy*(fx+fy+dz),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),#check mixed type add
#            (dz*fy*(fx+fy),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),#check mixed type mul
            #check with dimshuffle of constant
            (fx+fy+fz+2,(fx,fy,fz),(fxv,fyv,fzv),1, {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            (fx*fy*fz*2,(fx,fy,fz),(fxv,fyv,fzv),1, {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
#            (2+fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
#            (2*fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2+fx+fy+fz+2,(fx,fy,fz),(fxv,fyv,fzv),1, {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            (2*fx*fy*fz*2,(fx,fy,fz),(fxv,fyv,fzv),1, {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
#            (fx*fy*2*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
#            (fx*fy*(2+fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            (fx*fy*2*(fx+fy+fz+2),(fx,fy,fz),(fxv,fyv,fzv),2, {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),

            #check with broadcast of row
#            (fx+fy+fz+fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
#            (fx*fy*fz*fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
#            (fv+fx+fy+fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
#            (fv*fx*fy*fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
#            (fx*fy*fv*(fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
#            (fx*fy*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
#            (fx*fy*fv*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
#            (dx+dy+dz+dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
#            (dx*dy*dz*dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
#            (dv+dx+dy+dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
#            (dv*dx*dy*dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
#            (dx*dy*dv*(dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
#            (dx*dy*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
#            (dx*dy*dv*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            ]#[10:11]
#        print cases


        #We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode=compile.mode.get_default_mode()
        old_optimizer = mode._optimizer
        try:
            mode._optimizer=gof.Query(["canonicalize"])
            mode._optimizer=mode._optimizer.excluding('local_elemwise_fusion')
            for id, [g, sym_inputs, val_inputs, nb_elemwise, out_dtype] in enumerate(cases):
                if isinstance(out_dtype, dict):
                    out_dtype = out_dtype[config.cast_policy]
                f = compile.function(list(sym_inputs), g,
                                     #we need the optimisation enabled, debug do this.
                                     mode=mode)

                out = f(*val_inputs)
                assert(len(f.maker.env.toposort())==nb_elemwise)
                assert(out_dtype==out.dtype)
        finally:
            mode._optimizer = old_optimizer

    def test_elemwise_multiple_inputs_optimisation2(self):
        """
        verify that the Canonizer merge sequential Elemwise({mul,add}) part 2.
        This part are that case that should have been done, but that are not implemented.
        Test with and without DimShuffle
        """
        raise SkipTest("Current implementation of Canonizer does not "
                       "implement all cases. Skip the corresponding test.")

        shp=(5,5)
        fx, fy, fz = fmatrices('xyz')
        dx, dy, dz = dmatrices('xyz')
        fv = fvector('r').dimshuffle('x',0)
        dv = dvector('s').dimshuffle('x',0)
        fxv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        dxv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dyv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dzv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float64').reshape(1,shp[0])
        cases = [
            (fx+fy,(fx,fy),(fxv,fyv),1,'float32'),
            (fx*fy,(fx,fy),(fxv,fyv),1,'float32'),
            (fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (dx+dy+dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
            (fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (dx*dy*dz,(dx,dy,dz),(dxv,dyv,dzv),1,'float64'),
            (fx*fy*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            (dx*dy*(dx+dy+dz),(dx,dy,dz),(dxv,dyv,dzv),2,'float64'),
            (fx*fy*(fx+fy+dz),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),#check mixed type add
            (dz*fy*(fx+fy),(fx,fy,dz),(dxv,dyv,dzv),2,'float64'),#check mixed type mul
            #check with dimshuffle of constant
            (fx+fy+fz+2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (fx*fy*fz*2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2+fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2*fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2+fx+fy+fz+2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2*fx*fy*fz*2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (fx*fy*2*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            (fx*fy*(2+fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            (fx*fy*2*(fx+fy+fz+2),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),

            #check with broadcast of row
            (fx+fy+fz+fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            (fx*fy*fz*fv,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            (fv+fx+fy+fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            (fv*fx*fy*fz,(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),1,'float32'),
            (fx*fy*fv*(fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            (fx*fy*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            (fx*fy*fv*(fv+fx+fy+fz),(fx,fy,fz,fv),(fxv,fyv,fzv,fvv),2,'float32'),
            (dx+dy+dz+dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            (dx*dy*dz*dv,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            (dv+dx+dy+dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            (dv*dx*dy*dz,(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),1,'float64'),
            (dx*dy*dv*(dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            (dx*dy*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),
            (dx*dy*dv*(dv+dx+dy+dz),(dx,dy,dz,dv),(dxv,dyv,dzv,dvv),2,'float64'),

            ]#[10:11]
#        print cases

        #We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode=compile.mode.get_default_mode()
        mode._optimizer=gof.Query(["canonicalize"])
        mode._optimizer=mode._optimizer.excluding('local_elemwise_fusion')
        for id, [g, sym_inputs, val_inputs, nb_elemwise, out_dtype] in enumerate(cases):
            f = compile.function(list(sym_inputs), g,
                                 #we need the optimisation enabled, debug do this.
                                 mode=mode)

            out = f(*val_inputs)
            assert(len(f.maker.env.toposort())==nb_elemwise)
            assert(out_dtype==out.dtype)

    def test_multiple_case(self):
        """ test those case take from the comment in Canonizer
        x / x -> 1
        (x * y) / x -> y
        x / y / x -> 1 / y
        x / y / z -> x / (y * z)
        x / (y / z) -> (x * z) / y
        (a / b) * (b / c) * (c / d) -> a / d
        (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
        2 * x / 2 -> x
        with and without DimShuffle
        TODO: with DimShuffle
        """

        shp=(3,3)
        fx, fy, fz, fw = fmatrices('xyzw')
        dx, dy, dz, dw = dmatrices('xyzw')
        fv = fvector('r').dimshuffle('x',0)
        dv = dvector('s').dimshuffle('x',0)
        fxv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fwv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        dxv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dyv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dzv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dwv = theano._asarray(numpy.random.rand(*shp),dtype='float64')
        dvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float64').reshape(1,shp[0])

        #We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode=compile.mode.get_default_mode()
        old_optimizer = mode._optimizer
        try:
            mode._optimizer=gof.Query(["canonicalize"])
            mode._optimizer=mode._optimizer.including('ShapeOpt')
            mode._optimizer=mode._optimizer.excluding('local_elemwise_fusion')

            #test x / x -> 1
            for id, (g, sym_inputs, val_inputs, out_dtype) in enumerate([(fx/fx,[fx],[fxv],'float32'),
                                                           (dx/dx,[dx],[dxv],'float64'),
                                                           (fv/fv,[fv],[fvv],'float32'),
                                                           (dv/dv,[dv],[dvv],'float64'),
                                                           ]):
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert (out==numpy.ones(shp, dtype=out_dtype)).all()
                topo=f.maker.env.toposort()
                if sym_inputs[0].broadcastable[0]:
                    assert len(topo)==2
                    assert isinstance(topo[0].op, Shape_i)
                    assert isinstance(topo[1].op, tensor.Alloc)
                else:
                    assert len(topo)==3
                    assert isinstance(topo[0].op, Shape_i)
                    assert isinstance(topo[1].op, Shape_i)
                    assert isinstance(topo[2].op, tensor.Alloc)
                assert(out_dtype==out.dtype)

            #test (x * y) / x -> y
            for id,(g, sym_inputs, val_inputs, nb_elemwise, out_dtype) in enumerate([
                                                           ((dx*dy)/dx,[dx,dy],[dxv,dyv],0,'float64'),
                                                           ((fx*fy)/fx,[fx,fy],[fxv,fyv],0,'float32'),
                                                           ((dv*dy)/dv,[dv,dy],[dvv,dyv],0,'float64'),
                                                           ((fv*fy)/fv,[fv,fy],[fvv,fyv],0,'float32'),
                #must broadcast as their is a dimshuffle in the computation
                                                           ((dx*dv)/dx,[dx,dv],[dxv,dvv],1,'float64'),
                #topo: [Elemwise{second,no_inplace}(x, <TensorType(float64, row)>)]
                                                           ((fx*fv)/fx,[fx,fv],[fxv,fvv],1,'float32')
                #topo: [Elemwise{second,no_inplace}(x, <TensorType(float32, row)>)]
                ]):
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert(out_dtype==out.dtype)
                assert numpy.allclose(out,val_inputs[1])
                topo=f.maker.env.toposort()
                print "ID TOPO", id, topo, sym_inputs
                for r,t in f.maker.env.shape_feature.shape_of.items():
                    print '  ', r, t
                if topo and not(len(topo)==1 and topo[0].op==theano.compile.function_module.deep_copy_op):
                    for node in topo[:-1]:
                        assert isinstance(node.op, Shape_i)
                    assert isinstance(topo[-1].op, tensor.Alloc)

            #test x / y / x -> 1 / y
            for id,(g, sym_inputs, val_inputs, nb_elemwise, out_dtype) in enumerate([
                                                           ((dx/dy)/dx,[dx,dy],[dxv,dyv],1,'float64'),
                                                           ((fx/fy)/fx,[fx,fy],[fxv,fyv],1,'float32'),
                                                           ((dv/dy)/dv,[dv,dy],[dvv,dyv],1,'float64'),
                                                           ((fv/fy)/fv,[fv,fy],[fvv,fyv],1,'float32'),
                            #must broadcast as their is a dimshuffle in the computation

                                                           ((dx/dv)/dx,[dx,dv],[dxv,dvv],1,'float64'),
    #topo:            [Shape_i, Shape_i, Elemwise{inv,no_inplace}(<TensorType(float64, row)>), Alloc]
                                                           ((fx/fv)/fx,[fx,fv],[fxv,fvv],1,'float32'),
                #topo:[Shape_i, Shape_i, Elemwise{inv,no_inplace}(<TensorType(float32, row)>), Alloc]
                ]):
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.allclose(out,(1/val_inputs[1]))
                topo=f.maker.env.toposort()
                print topo
                elem = [t for t in topo if isinstance(t.op, T.Elemwise)]
                assert len(elem)==nb_elemwise
                assert isinstance(elem[0].op,(T.Elemwise,))
                assert isinstance(elem[0].op.scalar_op,(theano.scalar.basic.Inv, theano.scalar.basic.TrueDiv))
                assert(out_dtype==out.dtype)

            #test (a / b) * (b / c) * (c / d) -> a / d
            for id,(g, sym_inputs, val_inputs, out_dtype) in enumerate([
                                                           ((dx / dy) * (dy / dz) * (dz / dw),[dx,dy,dz,dw],[dxv,dyv,dzv,dwv],'float64'),
                                                           ((fx / fy) * (fy / fz) * (fz / fw),[fx,fy,fz,fw],[fxv,fyv,fzv,fwv],'float32'),
                                                           ((dv / dy) * (dy / dz) * (dz / dw),[dv,dy,dz,dw],[dvv,dyv,dzv,dwv],'float64'),
                                                           ((fv / fy) * (fy / fz) * (fz / fw),[fv,fy,fz,fw],[fvv,fyv,fzv,fwv],'float32'),
                                                           ((dx / dv) * (dv / dz) * (dz / dw),[dx,dv,dz,dw],[dxv,dvv,dzv,dwv],'float64'),
                                                           ((fx / fv) * (fv / fz) * (fz / fw),[fx,fv,fz,fw],[fxv,fvv,fzv,fwv],'float32'),
                                                           ((dx / dy) * (dy / dv) * (dv / dw),[dx,dy,dv,dw],[dxv,dyv,dvv,dwv],'float64'),
                                                           ((fx / fy) * (fy / fv) * (fv / fw),[fx,fy,fv,fw],[fxv,fyv,fvv,fwv],'float32'),
                                                           ((dx / dy) * (dy / dz) * (dz / dv),[dx,dy,dz,dv],[dxv,dyv,dzv,dvv],'float64'),
                                                           ((fx / fy) * (fy / fz) * (fz / fv),[fx,fy,fz,fv],[fxv,fyv,fzv,fvv],'float32'),
                ]):
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.allclose(out,(val_inputs[0]/val_inputs[3]))
                topo=f.maker.env.toposort()
                assert len(topo)==1
                assert isinstance(topo[0].op,(T.Elemwise,))
                assert isinstance(topo[0].op.scalar_op,theano.scalar.basic.TrueDiv)
                assert len(topo[0].inputs)==2
                assert(out_dtype==out.dtype)

            #test (2.0 * x) / (4.0 * y) -> (0.5 * x) / y
            for id,(g, sym_inputs, val_inputs, out_dtype) in enumerate([
                                                           (((2.0*dx)/(4.0*dy)),[dx,dy],[dxv,dyv],'float64'),
                                                           (((2.0*fx)/(4.0*fy)),[fx,fy],[fxv,fyv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                                                           (((2.0*dv)/(4.0*dy)),[dv,dy],[dvv,dyv],'float64'),
                                                           (((2.0*fv)/(4.0*fy)),[fv,fy],[fvv,fyv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                                                           (((2.0*dx)/(4.0*dv)),[dx,dv],[dxv,dvv],'float64'),
                                                           (((2.0*fx)/(4.0*fv)),[fx,fv],[fxv,fvv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ]):

                if isinstance(out_dtype, dict):
                    out_dtype = out_dtype[config.cast_policy]
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.allclose(out,(0.5*val_inputs[0]/val_inputs[1]))
                topo=f.maker.env.toposort()
                assert len(topo)==2
                assert isinstance(topo[0].op,(T.Elemwise,))
                assert isinstance(topo[0].op.scalar_op,theano.scalar.basic.Mul)
                assert len(topo[0].inputs)==2
                assert isinstance(topo[1].op,(T.Elemwise,))
                assert isinstance(topo[1].op.scalar_op,theano.scalar.basic.TrueDiv)
                assert len(topo[1].inputs)==2
                assert(out_dtype==out.dtype)

            #test 2 * x / 2 -> x
            for id,(g, sym_inputs, val_inputs, out_dtype) in enumerate([
                                                           ((2*dx)/2,[dx],[dxv],'float64'),
                                                           ((2*fx)/2,[fx],[fxv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                                                           ((2*dv)/2,[dv],[dvv],'float64'),
                                                           ((2*fv)/2,[fv],[fvv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ]):
                if isinstance(out_dtype, dict):
                    out_dtype = out_dtype[config.cast_policy]
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.allclose(out,val_inputs[0])
                topo=f.maker.env.toposort()
                assert len(topo)==1
                topo[0].op==theano.compile.function_module.deep_copy_op
                assert(out_dtype==out.dtype)

            #test x / abs(x) -> sign(x)
            for id,(g, sym_inputs, val_inputs, out_dtype) in enumerate([
                                                           (dx/abs(dx),[dx],[0.5-dxv],'float64'),
                                                           (fx/abs(fx),[fx],[0.5-fxv], 'float32'),
                                                           (dx/abs(dx),[dx],[0.1*dxv],'float64'),
                                                           (fx/abs(fx),[fx],[0.1*fxv], 'float32'),
                                                           (dv/abs(dv),[dv],[0.5-dvv],'float64'),
                                                           (fv/abs(fv),[fv],[0.5-fvv], 'float32'),
                ]):
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.all(numpy.isfinite(out))
                assert numpy.allclose(out,numpy.sign(val_inputs[0]))
                assert(out_dtype==out.dtype)
                assert len(f.maker.env.toposort())==1

            #test (2*x) / (3*abs(x)) -> sign(x)
            for id,(g, sym_inputs, val_inputs, out_dtype) in enumerate([
                    ((2*dx)/(3*abs(dx)),[dx],[0.5-dxv],'float64'),
                    ((2*fx)/(3*abs(fx)),[fx],[0.5-fxv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                    ((2*dx)/(3*abs(dx)),[dx],[0.1*dxv],'float64'),
                    ((2*fx)/(3*abs(fx)),[fx],[0.1*fxv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                    ((2*dv)/(3*abs(dv)),[dv],[0.5-dvv],'float64'),
                    ((2*fv)/(3*abs(fv)),[fv],[0.5-fvv], {'custom': 'float32', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
                ]):

                if isinstance(out_dtype, dict):
                    out_dtype = out_dtype[config.cast_policy]
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                topo = f.maker.env.toposort()
                out = f(*val_inputs)
                assert numpy.all(numpy.isfinite(out))
                assert numpy.allclose(out,numpy.sign(val_inputs[0])*2/3)
                assert(out_dtype==out.dtype)
        finally:
            mode._optimizer = old_optimizer

    def test_abs_mul_div(self):
        """
        test that if we have
        4 * x / abs(2*x) it get simplifier during canonicalisation.
        """

        x=T.dscalar()
        a=T.abs_(x)

        if theano.config.mode=='FAST_COMPILE':
            mode = theano.compile.mode.get_mode('FAST_RUN').excluding("local_elemwise_fusion")
        else:
            mode = theano.compile.mode.get_default_mode().excluding("local_elemwise_fusion")

        f=theano.function([x],[(4*x)/abs(2*x)], mode = mode)
        print f.maker.env.toposort()
        print
        f(.1)
        f(-1)
        #some stabilization optimization make the output be finite instead of nan
        #debug_mode will raise an error when he see nan
        if not isinstance(mode,theano.compile.debugmode.DebugMode):
            assert numpy.isfinite(f(0))

        assert len(f.maker.env.toposort())==2
        assert f.maker.env.toposort()[0].op==T.sgn

        f=theano.function([x],[(4*x)/abs(x/2)], mode = mode)
        print f.maker.env.toposort()
        print
        f(.1)
        f(-1)
        #some stabilization optimization make the output be finite instead of nan
        #debug_mode will raise an error when he see nan
        if not isinstance(mode,theano.compile.debugmode.DebugMode):
            assert numpy.isfinite(f(0))

        assert len(f.maker.env.toposort())==2
        assert f.maker.env.toposort()[0].op==T.sgn


    def test_multiple_case_that_fail(self):
        raise SkipTest("Current implementation of Canonizer does not "
                       "implement all cases. Skip the corresponding test.")

        shp=(4,4)
        fx, fy, fz = fmatrices('xyz')
        dx, dy, dz = dmatrices('xyz')
        fxv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        dxv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        dyv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        dzv = theano._asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        #We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode=compile.mode.get_default_mode()
        old_optimizer = mode._optimizer
        try:
            mode._optimizer=gof.Query(["canonicalize"])
            mode._optimizer=mode._optimizer.excluding('local_elemwise_fusion')

    #test fail!
            #test x / y / z -> x / (y * z)
            for (g, sym_inputs, val_inputs, out_dtype) in [
                                                           ((dx/dy)/dz,[dx,dy,dz],[dxv,dyv,dzv],'float64'),
                                                           ((fx/fy)/fz,[fx,fy,fz],[fxv,fyv,fzv],'float32')
                ]:
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.allclose(out,val_inputs[0]/val_inputs[1]/val_inputs[2])
                topo=f.maker.env.toposort()
                print topo
                assert len(topo)==2
                assert isinstance(topo[0].op,(T.Elemwise,))
                assert isinstance(topo[0].op.scalar_op,theano.scalar.basic.Inv)
                assert len(topo[0].inputs)==1
                assert(out_dtype==out.dtype)

            #test x / (y / z) -> (x * z) / y
            for (g, sym_inputs, val_inputs, out_dtype) in [
                                                           (dx/(dy/dz),[dx,dy,dz],[dxv,dyv,dzv],'float64'),
                                                           (fx/(fy/fz),[fx,fy,fz],[fxv,fyv,fzv],'float32')
                ]:
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.allclose(out,val_inputs[0]/(val_inputs[1]/val_inputs[2]))
                topo=f.maker.env.toposort()
                print topo
                assert len(topo)==2
                assert isinstance(topo[0].op,(T.Elemwise,))
                assert isinstance(topo[0].op.scalar_op,theano.scalar.basic.Inv)
                assert len(topo[0].inputs)==1
                assert(out_dtype==out.dtype)

        finally:
            mode._optimizer = old_optimizer

    def test_dont_merge_if_multiple_client(self):
        """ test those case take from the comment in Canonizer
        """
        raise SkipTest("Not implemented")


def test_local_merge_abs():
    x,y,z = T.matrices('xyz')
    x_val = numpy.random.rand(5,5).astype(config.floatX)
    y_val = numpy.random.rand(5,5).astype(config.floatX)
    z_val = numpy.random.rand(5,5).astype(config.floatX)
    mode = theano.config.mode
    if mode == "FAST_COMPILE":
        mode = "FAST_RUN"
    mode = theano.compile.mode.get_mode(mode).excluding("local_elemwise_fusion")

    f = theano.function([x,y,z],(abs(y*z*-2)), mode=mode)
    f(x_val,y_val,z_val)
    theano.printing.debugprint(f)
    assert isinstance(f.maker.env.toposort()[1].op.scalar_op, scal.Abs)
    assert len(f.maker.env.toposort())==2

    f = theano.function([x,y,z],abs(x/y), mode=mode)
    f(x_val,y_val,z_val)
    theano.printing.debugprint(f)
    assert isinstance(f.maker.env.toposort()[1].op.scalar_op, scal.Abs)
    assert len(f.maker.env.toposort())==2


def test_mixeddiv():
    """Test that int division is preserved"""
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

    def sigm(x): return 1./(1+tensor.exp(-x))

    hid = sigm( (tensor.dot(w,input) + hidb) * betas )

    vis_gauss1 = (tensor.dot(w.T, hid) + visb) * betas / (2 * a * a)
    vis_gauss2 = (tensor.dot(w.T, hid) + visb) * betas / (2. * a * a)

    f1 = function([input,w,visb,hidb,betas,a],vis_gauss1)
    f2 = function([input,w,visb,hidb,betas,a],vis_gauss2)

    ival = numpy.random.rand(5,5)
    wval = numpy.random.rand(5,5)
    visbval = numpy.random.rand(5)
    hidbval = numpy.random.rand(5)
    betaval = numpy.random.rand(5)
    aval = numpy.random.rand(5)

    assert numpy.allclose(
        f2(ival, wval, visbval, hidbval, betaval, aval),
        f1(ival, wval, visbval, hidbval, betaval, aval))


class test_fusion(unittest.TestCase):
    def do(self, mode, shared_fn, shp, gpu=False, nb_repeat=1, assert_len_topo=True, slice=None):
        """
        param shared_fn: if None, will use compile.function
        verify that the elemwise fusion work
        Test with and without DimShuffle
        """
        #TODO: disable the canonizer?
        def my_init(shp, dtype='float64', num=0):
            #ret = theano._asarray(numpy.random.rand(*shp),dtype=dtype)
            ret = numpy.zeros(shp, dtype=dtype)+num
            return ret
        fw, fx, fy, fz = [theano.tensor.tensor(dtype='float32',
                                               broadcastable=[False]*len(shp),
                                               name=n) for n in 'wxyz']
        dw, dx, dy, dz = [theano.tensor.tensor(dtype='float64',
                                               broadcastable=[False]*len(shp),
                                               name=n) for n in 'wxyz']
        ix, iy, iz = [theano.tensor.tensor(dtype='int32',
                                           broadcastable=[False]*len(shp),
                                           name=n) for n in 'xyz']
        fv = fvector('r')
        fwv = my_init(shp,'float32',1)
        fxv = my_init(shp,'float32',2)
        fyv = my_init(shp,'float32',3)
        fzv = my_init(shp,'float32',4)
        fvv = theano._asarray(numpy.random.rand(shp[0]),dtype='float32')
        dwv = my_init(shp,'float64',5)
        ixv = theano._asarray(my_init(shp,num=60),dtype='int32')
        iyv = theano._asarray(my_init(shp,num=70),dtype='int32')
        izv = theano._asarray(my_init(shp,num=70),dtype='int32')
        fwx=fw+fx
        ftanx = theano.tensor.tan(fx)
        cases = [
            (fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+fzv,'float32'),#0
            (fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv*fzv,'float32'),#1
            (fx+fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv*fzv,'float32'),#2
            (fx*fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv+fzv,'float32'),#3
            (fw+fx+fy+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            ((fw+fx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),#5
            (((fw+fx)+fy)+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            ((fw+(fx+fy))+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            ((fw+(fx+fy)+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            (fw+(fx+(fy+fz)),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            ((fw+fx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),#10
            (fw*fx*fy*fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv*fxv*fyv*fzv,'float32'),
            (fw+fx*fy*fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv*fyv*fzv,'float32'),
            (fx+fy*fz*fx,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv*fzv*fxv,'float32'),
            (fx*fy+fz+fy,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv+fzv+fyv,'float32'),
            (fx*fy*fz*fw+fx+fy+fz+fw,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fxv*fyv*fzv*fwv+fxv+fyv+fzv+fwv,'float32'),#15
            #test with constant
            ((fw+fx)+(fy+fz)+ 2,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            (((fw+fx)+2+fy)+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            ((fw+(fx+2+fy))+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            ((fw+(fx+fy)+2+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            (fw+(fx+(fy+fz)+2),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),#20
            (2+(fw+fx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            #mix float32 and float64
            (2+(dw+fx)+(fy+fz),(dw,fx,fy,fz),(dwv,fxv,fyv,fzv),1,dwv+fxv+fyv+fzv+2,'float64'),
            (2+(fw+dw)+(fy+fz),(fw,dw,fy,fz),(fwv,dwv,fyv,fzv),1,fwv+dwv+fyv+fzv+2,'float64'),
            (2+(fw+fx)+(dw+fz),(fw,fx,dw,fz),(fwv,fxv,dwv,fzv),1,fwv+fxv+dwv+fzv+2,'float64'),
            (2+(fw+fx)+(fy+dw),(fw,fx,fy,dw),(fwv,fxv,fyv,dwv),1,fwv+fxv+fyv+dwv+2,'float64'),#25
            #test when their is other op then elemwise.
            #the good output for the next test.
#            (Pdb) p f.maker.env.toposort()
#[Elemwise{add,no_inplace}(w, x), Sum(Elemwise{add,no_inplace}.0), InplaceDimShuffle{x,x}(Sum.0), Elemwise{Composite{_impls=[<function <lambda> at 0x2c5c8c0>], nin=4, _c_code={
#npy_float32 V%(id)s_tmp1;
#V%(id)s_tmp1 = %(i2)s + %(i3)s;
#npy_float32 V%(id)s_tmp2;
#V%(id)s_tmp2 = %(i0)s + %(i1)s;
#%(o0)s = V%(id)s_tmp2 + V%(id)s_tmp1;
#}
#, nout=1, env=[add(add(<float32>, <float32>), add(<float32>, <float32>))]}}(InplaceDimShuffle{x,x}.0, Elemwise{add,no_inplace}.0, y, z)]
            ((fwx.sum())+(fwx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),4,(fwv+fxv).sum()+fwv+fxv+fyv+fzv,'float32'),
            #test other elemwise op
            (fx+fy+tensor.cos(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.cos(fzv),'float32'),
            (fx+fy+tensor.cosh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.cosh(fzv),'float32'),
            (fx+fy+abs(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.absolute(fzv),'float32'),
            (ix+iy+abs(iz),(ix,iy,iz),(ixv,iyv,izv),1,ixv+iyv+numpy.absolute(izv),'int32'),#30
            (fx+fy+theano.tensor.log(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.log(fzv),'float32'),
            (fx+fy+theano.tensor.log2(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.log2(fzv),'float32'),
            (fx+fy+theano.tensor.log10(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.log10(fzv),'float32'),
            (fx+fy**fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv**fzv,'float32'),#pow
            (fx+fy+theano.tensor.exp(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.exp(fzv),'float32'),#35
            (fx-fy-fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv-fzv,'float32'),
            (fx-(fy/fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv/fzv),'float32'),
            (fx-theano.tensor.true_div(fy,2),(fx,fy),(fxv,fyv),1,fxv-(fyv/2),'float32'),
            (fx-theano.tensor.true_div(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv/fzv),'float32'),
            (fx-theano.tensor.int_div(ix*100,iy*1000),(fx,ix,iy),(fxv,ixv,iyv),1,fxv-((ixv*100)//(iyv*1000)), {'custom': 'float64', 'numpy+floatX': config.floatX, 'numpy': 'float64'}), #40
            (fx-(fy/2),(fx,fy),(fxv,fyv),1,fxv-(fyv/2),'float32'),
            (fx-(fy%fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv%fzv),'float32'),
            (fx-(fy>fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv>fzv),'float32'),
            (fx-(fy>=fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv>=fzv),'float32'),
            (fx-(fy<fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv<fzv),'float32'),#45
            (fx-(fy<=fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv<=fzv),'float32'),
            (fx-T.eq(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv==fzv),'float32'),
            (fx-T.neq(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv!=fzv),'float32'),
            (fx-fy+tensor.tan(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.tan(fzv),'float32'),
            (fx-fy+tensor.tanh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.tanh(fzv),'float32'),#50
            (fx-fy+tensor.sin(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sin(fzv),'float32'),
            (fx-fy+tensor.sinh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sinh(fzv),'float32'),
            (fx-fy+theano.tensor.sqr(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(fzv*fzv),'float32'),
            (fx-fy+theano.tensor.sqrt(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sqrt(fzv),'float32'),
            (fx-fy+theano.tensor.inv(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(1/fzv),'float32'),#55
            (fx-fy+theano.tensor.neg(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(-fzv),'float32'),
            (fx-fy+theano.tensor.round(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.round(fzv),'float32'),
            (ix-iy+theano.tensor.iround(fz),(ix,iy,fz),(ixv,iyv,fzv),1,ixv-iyv+numpy.round(fzv),'int64'),
            # Bit op
            (fx-theano.tensor.or_(iy,iz),(fx,iy,iz),(fxv,iyv,izv),1,fxv-(iyv|izv), {'custom': 'float64', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            (fx-theano.tensor.xor(iy,iz),(fx,iy,iz),(fxv,iyv,izv),1,fxv-(iyv^izv), {'custom': 'float64', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),#60
            (fx-theano.tensor.and_(iy,iz),(fx,iy,iz),(fxv,iyv,izv),1,fxv-(iyv&izv), {'custom': 'float64', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),
            (fx-theano.tensor.invert(iy),(fx,iy),(fxv,iyv),1,fxv-(~iyv), {'custom': 'float64', 'numpy+floatX': config.floatX, 'numpy': 'float64'}),

            (fx-theano.tensor.cast(fy,dtype='float64'),(fx,fy),(fxv,fyv),1,
                              fxv-numpy.asarray(fyv,'float64'),'float64'),
            (theano.tensor.pow(fx*fy+fz,fx*fy),(fx,fy,fz),(fxv,fyv,fzv),1,numpy.power(fxv*fyv+fzv,fxv*fyv),'float32'),
            (fv+fy**fz,(fv,fy,fz),(fvv,fyv,fzv),2,fvv+fyv**fzv,'float32'),#fused with a dimshuffle #65
            (fv-fy+tensor.tanh(fz),(fv,fy,fz),(fvv,fyv,fzv),2,fvv-fyv+numpy.tanh(fzv),'float32'),#fused with a dimshuffle

            # Cases where the same input is reused many times.
            (theano.tensor.mul(fx,fx,fx,fx),(fx,),(fxv,),1,fxv*fxv*fxv*fxv,'float32'),
            # TODO: This case is not fused!
            (theano.tensor.mul(fx,ftanx,ftanx),(fx,),(fxv,),2,fxv*numpy.tan(fxv)*numpy.tan(fxv),'float32'),
            # TODO: This case is not fused!
            (theano.tensor.mul(fx,ftanx,ftanx,fx),(fx,),(fxv,),2,fxv*numpy.tan(fxv)*numpy.tan(fxv)*fxv,'float32'),
            # The next case test when one variable appear as many inputs to an op.
            # In the past, this was not fused. (TODO) Now it is partially fused.
            (theano.tensor.mul(ftanx,ftanx,fx+fy),(fx,fy),(fxv,fyv),2,numpy.tan(fxv)*numpy.tan(fxv)*(fxv+fyv),'float32'),
            ]
        if slice:
            cases = cases[slice]
        times=numpy.zeros(len(cases))
        fail1=[]
        fail2=[]
        fail3=[]
        fail4=[]
        for id, [g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype] in enumerate(cases):
            if isinstance(out_dtype, dict):
                out_dtype = out_dtype[config.cast_policy]
            if gpu and (out_dtype!='float32' or any(i.dtype != 'float32' for i in g.owner.inputs)):
                print "Skip test %d as the gpu code currently supports only float32" % id
                continue
            print "new cases", id

            if shared_fn == None:
                assert gpu==False
                f = compile.function(list(sym_inputs), g,mode=mode)
                for x in range(nb_repeat):
                    out=f(*val_inputs)
                t1=time.time()
            else:
                out = shared_fn(numpy.zeros(shp, dtype=out_dtype), 'out')
                assert out.dtype == g.dtype
                f = function(sym_inputs,[],updates=[(out, g)],mode=mode)
                t0=time.time()
                for x in range(nb_repeat):
                    f(*val_inputs)
                t1=time.time()
                out=out.get_value()

            #print "CASE2/3", f.maker.env.toposort()
            #print 'CASE2/3', f.maker.env
            #print 'CASE2/3', f.maker.env.toposort()[3].op.scalar_op.env

            times[id]=t1-t0
            atol=1e-8
            if out_dtype=='float32':atol=1e-6
            if not numpy.allclose(out,answer*nb_repeat,atol=atol):
                fail1.append(id)
                print val_inputs
                print out
                print answer*nb_repeat
                #assert 0
            topo=f.maker.env.toposort()
            if gpu:
                import theano.sandbox.cuda as cuda
                topo_ = [x for x in topo if not isinstance(x.op,cuda.basic_ops.GpuFromHost) and not isinstance(x.op,cuda.basic_ops.HostFromGpu)]
                gpu_ = [x for x in topo if isinstance(x.op,cuda.basic_ops.GpuFromHost)]
                if not len(gpu_)==len(sym_inputs):
                    fail2.append((id,gpu_,sym_inputs))
            else: topo_=topo
            if assert_len_topo:
                if not len(topo_)==nb_elemwise:
                    fail3.append((id,topo_,nb_elemwise))
                if nb_elemwise == 1:
                    # check that the number of input to the Composite Elemwise is ok
                    # when there is not variable that appear multiple time the in input
                    # of g
                    assert ((numpy.sum([not isinstance(x, theano.gof.Constant)
                                        for x in topo_[0].inputs]) ==
                             len(sym_inputs)) or
                            len(set(g.owner.inputs)) != len(g.owner.inputs))
            if not out_dtype==out.dtype:
                fail4.append((id,out_dtype,out.dtype))

        failed=len(fail1+fail2+fail3+fail4)
        print "Executed",len(cases),"cases", "failed", failed
        if failed>0:
            raise Exception("Failed %d cases"%failed, fail1, fail2, fail3, fail4)

        return times

    def test_elemwise_fusion(self):
        shp=(5,5)
        mode=copy.copy(compile.mode.get_default_mode())
        #we need the optimisation enabled and the canonicalize.
        #the canonicalize is needed to merge multiplication/addition by constant.
        mode._optimizer=mode._optimizer.including('local_elemwise_fusion','canonicalize')
        self.do(mode, shared, shp)

    def test_elemwise_fusion_4d(self):
        shp=(3,3,3,3)
        mode=copy.copy(compile.mode.get_default_mode())
        #we need the optimisation enabled and the canonicalize.
        #the canonicalize is needed to merge multiplication/addition by constant.
        mode._optimizer=mode._optimizer.including('local_elemwise_fusion','canonicalize')
        self.do(mode, shared, shp)

    def test_gpu_fusion(self):
        shp=(5,5)
        #we need the optimisation enabled, debug do this.
        if theano.config.mode == "FAST_COMPILE":
            mode = theano.compile.mode.get_mode("FAST_RUN").including('local_elemwise_fusion','canonicalize','gpu')
        else:
            mode = theano.compile.mode.get_default_mode().including('local_elemwise_fusion','canonicalize','gpu')
        import theano.sandbox.cuda as cuda
        if not cuda.cuda_available:
            raise SkipTest("cuda not available")

        self.do(mode, cuda.float32_shared_constructor, shp, gpu=True)

    def test_gpu_fusion_Xd(self):
        #we need the optimisation enabled, debug do this.
        if theano.config.mode == "FAST_COMPILE":
            mode = theano.compile.mode.get_mode("FAST_RUN").including('local_elemwise_fusion','canonicalize','gpu')
        else:
            mode = theano.compile.mode.get_default_mode().including('local_elemwise_fusion','canonicalize','gpu')
        import theano.sandbox.cuda as cuda
        if not cuda.cuda_available:
            raise SkipTest("cuda not available")
        sizes = cuda.opt.get_device_type_sizes()
        if sizes['int_size'] == 4:
            shp=(5,5,5,5)
        else:
            shp=(5,5,5)
        self.do(mode, cuda.float32_shared_constructor, shp, gpu=True)

    def speed_fusion(self, shared_fn = shared, gpu = False, s=None):
        """
        param type s: a slice object
        param s: a slice to apply to the case to execute. If None, exec all case.
        """

        shp=(3000,3000)
        shp=(1000,1000)
        nb_repeat=50
#        linker=gof.CLinker
#        linker=gof.OpWiseCLinker

        mode1=copy.copy(compile.get_default_mode())
        mode1._optimizer=mode1._optimizer.including('local_elemwise_fusion')
        #TODO:clinker is much faster... but use to much memory
        #Possible cause: as their is do deletion of intermediate value when we don't keep the fct.
        #More plausible cause: we keep a link to the output data?
        #Follow up. Clinker do the same... second cause?
        mode2=copy.copy(compile.get_default_mode())
        mode2._optimizer=mode2._optimizer.excluding('local_elemwise_fusion')
        print "test with linker", str(mode1.linker)
        times1=self.do(mode1, shared_fn, shp, gpu=gpu, nb_repeat=nb_repeat, assert_len_topo=False,slice=s)
        times2=self.do(mode2, shared_fn, shp, gpu=gpu, nb_repeat=nb_repeat, assert_len_topo=False,slice=s)
        print "times1 with local_elemwise_fusion"
        print times1, times1.min(), times1.max(), times1.sum()
        print "times2 without local_elemwise_fusion"
        print times2, times2.min(), times2.max(), times2.sum()
        d=times2/times1

        print "times2/times1"
        print d
        print "min", d.min(), "argmin", d.argmin(), "max", d.max(), "mean", d.mean(), "std", d.std()

    def test_fusion_inplace(self):
        mode=copy.copy(compile.mode.get_default_mode())
        #we need the optimisation enabled and the canonicalize.
        #the canonicalize is needed to merge multiplication/addition by constant.
        mode._optimizer=mode._optimizer.including('local_elemwise_fusion','canonicalize','inplace')


        x, y, z = dmatrices('xyz')
        f=theano.function([x,y,z],tensor.dot(x,y)+x+y+z,mode=mode)
        topo = f.maker.env.toposort()
        assert len(topo) == 2
        assert f.maker.env.toposort()[-1].op.inplace_pattern
        f(numpy.random.random((5,5)),numpy.random.random((5,5)),numpy.random.random((5,5)))

    def speed_fusion_gpu(self):
        import theano.sandbox.cuda as cuda
        self.speed_fusion(shared_fn=cuda.float32_shared_constructor, gpu=True, s=slice(0,15))

    def speed_log_exp(self):
        s=slice(31,36)
#        linker=gof.CLinker
        linker=gof.OpWiseCLinker
        mode=compile.Mode(linker(), copy.copy(compile.mode.OPT_FAST_RUN))
        mode=compile.ProfileMode()
        print "time", self.do(mode, shared, shp=(1000,1000),gpu=False, assert_len_topo=False,slice=s, nb_repeat=100)


    def tes_memory_leak(self, mode=compile.mode.Mode('c', 'merge'), shared_fn=shared, shp=(3000,3000), gpu=False, nb_repeat=30, assert_len_topo=True, slice=None):
        """
        param shared_fn: if None, will use compile.function
        verify that the elemwise fusion work
        Test with and without DimShuffle
        """
        #TODO: disable the canonizer?
        fx = fmatrices('x')
        fy = fmatrices('y')
        fxv = numpy.zeros(shp, dtype='float32')+ 2
        cases = [
            (fx,(fx),(fxv),'float32'),#1
            ]
        import gc, pdb, objgraph, weakref
        d={}
        dl=[]
        v1=None
        mode=compile.mode.Mode('c', 'merge')
        #TODO: if mode is Mode('py','merge') then their is no memory leak!
        from theano.compile.function_module import orig_function
        for id, [g, sym_inputs, val_inputs, out_dtype] in enumerate(cases):
            for zzzz in range(nb_repeat):
                v=numpy.zeros(shp, dtype=out_dtype)
                gc.collect();gc.collect();gc.collect()
#                print 'v1',v1
                v1=weakref.ref(v)
                pdb.set_trace()
                #f = orig_function([compile.In(fx),compile.In(variable=fy, value=None)],
                #            [fy+fx],mode=mode)#no memory leak
                f = orig_function([compile.In(fx),compile.In(variable=fy, value=v)],
                            [fy+fx],mode=mode)#memory leak
                del v
                gc.collect();gc.collect();gc.collect()
                pdb.set_trace()

                if False:
                    gc.collect();gc.collect();gc.collect()
                    nd=objgraph.typestats()
                    print 'key, old val, new val, diff'
                    for key in set(d.keys()+nd.keys()):
                        if d.has_key(key) and nd.has_key(key) and nd[key]!=d[key]:
                            print key, d.get(key),nd.get(key),
                            if d.has_key(key) and nd.has_key(key): print nd[key]-d[key]
                            else: print None
                    gc.collect();gc.collect();gc.collect()
                    d=nd

#                pdb.set_trace()
                if False:
                    gc.collect();gc.collect();gc.collect()
                    ndl=objgraph.by_type('list')
                    ll=[]
                    if len(dl)>0:
                        nb=0
                        for x in ndl:
                            cmp = not isinstance(x, list)
                            if not cmp and x:
                                cmp=x[0].__class__.__name__!='array_converter'
                                if cmp:
                                    cmp=x[0]!='Option'
                                if cmp:
                                    cmp=x[0]!=270
                                cmp=False
                            if cmp and x in dl:
                                nb+=1
                                ll.append(x)
#                                pdb.set_trace()
                                pass
                        pdb.set_trace()
                    dl=ndl

                gc.collect();gc.collect();gc.collect()
#                objgraph.show_most_common_types(limit=40)
#                f(*val_inputs)
                gc.collect();gc.collect();gc.collect()

#            cases[id]=None #to remove g, that link to out that link to the ndarray!
            #g.owner.inputs[0] is out... make owner a weakref?


class TimesN(theano.scalar.basic.UnaryScalarOp):
    """Used in test TestCompositeCodegen

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

    def c_code(self, node, name, (x, ), (z, ), sub):
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
            assert len(f.maker.env.toposort()) == 1
        fval = f([1, 2, 3])
        assert numpy.all(fval == [6, 12, 18])

    def test_nested_gpu(self):
        import theano.sandbox.cuda as cuda
        if not cuda.cuda_available:
            raise SkipTest("cuda not available")

        import theano.sandbox.cuda.opt

        y = self.times_2(self.x)
        z = self.times_3(y)
        f = theano.function([self.x], cuda.gpu_from_host(z),
                mode=theano.compile.mode.get_default_mode().including('gpu'))
        topo = f.maker.env.toposort()
        if config.mode != "FAST_COMPILE":
            assert len(topo) == 2
            assert topo[1].op == cuda.gpu_from_host
        # topo1 is doing the composite work on the CPU. Auto-generation of
        # GPU code for ops with support code is not possible.
        fval = numpy.asarray(f([1, 2, 3]))
        assert numpy.all(fval == [6, 12, 18]), fval


def test_log1p():
    m = theano.config.mode
    if m == 'FAST_COMPILE':
        m = 'FAST_RUN'
    m = compile.mode.get_mode(m)
    m = m.excluding('fusion')
    # check some basic cases
    x = dvector()
    f = function([x], T.log(1+(x)), mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.log1p]
    f = function([x], T.log(1+(-x)), mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.neg, inplace.log1p_inplace]
    f = function([x], -T.log(1+(-x)), mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.neg, inplace.log1p_inplace, inplace.neg_inplace]


    # check trickier cases (and use different dtype)
    y = fmatrix()
    f = function([x,y], T.log(tensor.fill(y,1)+(x)), mode=m)
    print f.maker.env.toposort()
    # the first three ops are Shape_i, Shape_i, and Dimshuffle
    theano.printing.debugprint(f)
    assert [node.op for node in f.maker.env.toposort()][3:] \
            == [T.log1p, tensor.alloc]
    f = function([x,y], T.log(0+(x) + tensor.fill(y,1.0)), mode=m)
    theano.printing.debugprint(f)
    assert [node.op for node in f.maker.env.toposort()][3:] \
            == [T.log1p, tensor.alloc]
    f = function([x,y], T.log(2+(x) - tensor.fill(y,1.0)), mode=m)
    theano.printing.debugprint(f)
    assert [node.op for node in f.maker.env.toposort()][3:] \
            == [T.log1p, tensor.alloc]

    f([1e-7, 10], [[0, 0], [0, 0]]) #debugmode will verify values

    if 0:
        # at one point this worked, but it has been broken since
        # the constant up-casting made 1 -> 1.0+0.0j
        # I was never sure if this optimization should work on complex numbers or not.
        z = tensor.zmatrix()
        f = function([z], T.log(1+(z)), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [T.log1p]

    if 1:
        # should work for int
        z = tensor.imatrix()
        f = function([z], T.log(1+(z)), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [T.log1p]


def test_log_add():
    m = theano.config.mode
    if m == 'FAST_COMPILE':
        m = 'FAST_RUN'
    m = compile.mode.get_mode(m)
    m = m.excluding('fusion')
    m = copy.copy(m)
    #No need to put them back as we have a new object
    m.check_isfinite=False

    # check some basic cases
    x = dvector()
    y = dvector()
    f = function([x,y], T.log(T.exp(x) + T.exp(y)), mode=m)

    theano.printing.debugprint( f)

    print f([10000], [10000])  # causes overflow if handled incorrectly
    assert numpy.isfinite(f([10000], [10000]))
    assert numpy.allclose(f([10000], [10000]), 10000+numpy.log1p(1))

    #test that it give the same result when it don't overflow
    print f([10], [10])  # don't causes overflow
    assert numpy.allclose(f([10], [10]), 10+numpy.log1p(1))

    # test that it also works with more than two args, (this currently fails)
    x = dvector()
    y = dvector()
    f = function([x,y], T.log(T.exp(x) + T.exp(y) + T.exp(x-y) + T.exp(x+y)), mode=m)
    theano.printing.debugprint( f)

    try:
        print f([10000], [10000])  # causes overflow if handled incorrectly
        assert numpy.allclose(f([10000], [10000]), 20000)
    except AssertionError:
        raise KnownFailureTest(('log(add(exp)) is not stabilized when adding '
                'more than 2 elements, see #623'))

    #TODO: test that the optimization works in the presence of broadcasting.

    #TODO: (write and) test that the optimization works with Sum in addition to working with Add.


def test_local_useless_subtensor():
    x = tensor.matrix('x')

    # Test default
    for dims in [(slice(0,None),),
                 (slice(0,None),slice(0,None)),
                 ]:
        f = function([x], tensor.exp(x).__getitem__(dims), mode=mode_opt)
        #theano.printing.debugprint(f)
        prog=f.maker.env.toposort()
        assert prog[0].op == tensor.exp
        assert len(prog)==1
        f([[0,1,2],[3,4,5]]) # let debugmode test something

    x_c = tensor.specify_shape(x, (2,3))
    # Test constant
    for dims, res in [((slice(0,2),), True),
                 ((slice(0,2),slice(0,None)), True),
                 ((slice(0,2),slice(0,3)), True),
                 ((slice(0,None),slice(0,3)), True),
                 ((slice(0,3),slice(0,13)), True),
                 ((slice(0,3),slice(0,2)), False),
                 ((slice(0,1),slice(0,None)), False),
                 ((slice(0,1),1), False),
                 ]:
        f = function([x], tensor.exp(x_c).__getitem__(dims), mode=mode_opt)
        #theano.printing.debugprint(f)
        prog=f.maker.env.toposort()
        if res:
            assert isinstance(prog[0].op, theano.tensor.basic.SpecifyShape), dims
            assert prog[1].op == tensor.exp, dims
            assert len(prog)==2, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0,1,2],[3,4,5]]) # let debugmode test something

    # Test Variable
    for idx, (dims, res) in enumerate([
            ((slice(0,x.shape[0]),), True),
            ((slice(0,x.shape[1]),), False),
            ((slice(0,x.shape[0]),slice(0,x.shape[1]),), True),
            ((slice(0,x.shape[0]),slice(0,x.shape[0]),), False),
            ((slice(0,x.shape[1]),slice(0,x.shape[0]),), False),
            ((slice(0,x.shape[1]),slice(0,x.shape[1]),), False),
            ((slice(0,x.shape[1]),2), False),
            ((slice(0,x.shape[1]),slice(x.shape[0]-x.shape[0],x.shape[1]),), False),
            ((slice(0,T.scalar_from_tensor(x.shape[0])),), True),
            ]):
        f = function([x], tensor.exp(x).__getitem__(dims), mode=mode_opt)
        #theano.printing.debugprint(f)
        prog=f.maker.env.toposort()
        if res:
            assert prog[0].op == tensor.exp, dims
            assert len(prog)==1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0,1,2],[3,4,5]]) # let debugmode test something
    # Test mix Variable and Constant
    # Currently not supported
    for idx, (dims, res) in enumerate([
            ((slice(0,x.shape[0]),slice(0,3)), False),
            ((slice(0,3),slice(0,x.shape[1])), False),
            ]):
        f = function([x], tensor.exp(x_c).__getitem__(dims), mode=mode_opt)
        #theano.printing.debugprint(f)
        prog=f.maker.env.toposort()
        if res:
            assert prog[0].op == tensor.exp, dims
            assert len(prog)==1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[0,1,2],[3,4,5]]) # let debugmode test something

    # Test scalar variable
    s = scal.int32('s')
    for idx, (dims, res) in enumerate([
            ((slice(0,s),), False),
            ]):
        f = function([x, s], tensor.exp(x).__getitem__(dims), mode=mode_opt)
        #theano.printing.debugprint(f)
        prog=f.maker.env.toposort()
        if res:
            assert prog[0].op == tensor.exp, dims
            assert len(prog)==1, dims
        else:
            assert any([isinstance(node.op, Subtensor) for node in prog])
        f([[1,2,3],[4,5,6]], 1)
        f([[1,2,3],[4,5,6]], 3)


class test_local_subtensor_lift(unittest.TestCase):
    def test0(self):
        # basic test that the Op works
        x = tensor.matrix('x')
        f = function([x], tensor.exp(x)[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor) #first subtensor
        assert prog[1].op == tensor.exp
        assert len(prog)==2
        f([[0,1],[2,3]]) # let debugmode test something

    def test0b(self):
        # as test0, but we reuse the output of the elemwise
        # So we should not lift the subtensor
        x = tensor.matrix('x')
        f = function([x], [tensor.exp(x)[0], tensor.exp(x)], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert prog[0].op == tensor.exp
        assert isinstance(prog[1].op, tensor.Subtensor) #first subtensor
        assert isinstance(prog[2].op, theano.compile.function_module.DeepCopyOp)
        assert len(prog)==3
        f([[0,1],[2,3]]) # let debugmode test something

    def test1(self):
        # basic test that the optimization work with scalar broadcasted
        x = tensor.matrix('x')
        y = tensor.scalar('y')
        z = tensor.matrix('z')
        f = function([x,y,z], tensor.exp(x+y+z)[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[1].op, tensor.DimShuffle)
        assert isinstance(prog[0].op, tensor.Subtensor) #first subtensor
        assert isinstance(prog[2].op, tensor.Subtensor) #first subtensor
        assert isinstance(prog[3].op.scalar_op, theano.scalar.Composite)#Composite{add,add}
        assert len(prog)==4
        f([[0,1],[2,3]], 4, [[4,5],[6,7]]) # let debugmode test something

    def test2(self):
        # as 1, but take a slice
        x = tensor.matrix('x')
        y = tensor.scalar('y')
        z = tensor.matrix('z')
        f = function([x,y,z], tensor.exp(x+y+z)[0:2], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[1].op, tensor.DimShuffle)
        assert isinstance(prog[0].op, tensor.Subtensor) #first subtensor
        assert isinstance(prog[2].op, tensor.Subtensor) #first subtensor
        assert isinstance(prog[3].op.scalar_op, theano.scalar.Composite)#Composite{add,add}
        assert len(prog)==4
        f([[0,1],[2,3]], 4, [[4,5],[6,7]]) # let debugmode test something

    def test3(self):
        # basic test that the optimization does work with broadcasting
        # for unary elemwise.
        y = tensor.vector('y')
        f = function([y], tensor.exp(y.dimshuffle(0,'x'))[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.DimShuffle)
        assert isinstance(prog[1].op, tensor.Subtensor)
        assert prog[2].op == tensor.exp
        assert len(prog)==3
        f([4,5]) # let debugmode test something

    def test4(self):
        # basic test that the optimization doesn't work with broadcasting
        # ... It *could* be extended to,
        # ... but right now it doesn't, so it shouldn't try.
        x = tensor.matrix('x')
        y = tensor.vector('y')
        f = function([x,y], tensor.exp(x+y)[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.DimShuffle)
        assert prog[1].op == tensor.add
        assert isinstance(prog[2].op, tensor.Subtensor) #first subtensor
        assert prog[3].op == inplace.exp_inplace
        assert len(prog)==4
        f([[0,1],[2,3]], [4,5]) # let debugmode test something

    def test5(self):
        # test that we don't lift when we reuse the output of the
        # elemwise for other computation.
        x = tensor.matrix('x')
        y = tensor.vector('y')
        f = function([x,y], [tensor.exp(x+y)[0],tensor.exp(x+y)+x], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.DimShuffle)
        assert isinstance(prog[1].op.scalar_op, theano.scalar.Composite)#Composite{add,exp}
        assert prog[2].op == tensor.add
        assert isinstance(prog[3].op, tensor.Subtensor) #first subtensor
        assert len(prog)==4
        f([[0,1],[2,3]], [4,5]) # let debugmode test something

    def test6(self):
        # basic test that the optimization works with a scalar as input,
        # and a scalar as output (no broadcasting of the scalar needed).
        # The optimization used to fail and display an ERROR message.

        x = tensor.vector('x')
        y = tensor.scalar('y')
        f = function([x,y], tensor.exp(x+y)[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        # Composite{add,exp}
        assert isinstance(prog[1].op.scalar_op, theano.scalar.Composite)
        assert len(prog)==2
        f([1,2,3], 4) # let debugmode test something

    def test7(self):
        # test that Subtensor(Rebroadcast(x)) gets optimized into
        # Rebroadcast(Subtensor(x)).

        # test basic case
        x = tensor.matrix('x')
        xval = numpy.random.rand(1,10).astype(config.floatX)
        assert x.broadcastable == (False,False)
        newx = tensor.Rebroadcast((0,True),(1,False))(x)
        assert newx.broadcastable == (True,False)

        f1 = function([x], newx[:2,:5], mode=mode_opt)
        prog=f1.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f1(xval) == xval[:2,:5]).all()

        # corner case 1: rebroadcast changes dims which are dropped through subtensor
        y = tensor.tensor4('x')
        yval = numpy.random.rand(1,10,1,3).astype(config.floatX)
        assert y.broadcastable == (False,False,False,False)
        newy = tensor.Rebroadcast((0,True),(2,True))(y)
        assert newy.broadcastable == (True,False,True,False)

        f2 = function([y], newy[:,3,0,:], mode=mode_opt)
        prog=f2.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f2(yval) == yval[:,3,0,:]).all()

        # corner case 2: subtensor idx_list is shorter than resulting broadcast pattern
        f3 = function([y], newy[:,3,0], mode=mode_opt)
        prog=f3.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f3(yval) == yval[:,3,0]).all()

        # corner case 3: subtensor idx_list is shorter than rebroadcast.axis
        z = tensor.tensor4('x')
        zval = numpy.random.rand(4,10,3,1).astype(config.floatX)
        assert z.broadcastable == (False,False,False,False)
        newz = tensor.Rebroadcast((3,True))(z)
        assert newz.broadcastable == (False,False,False,True)

        out = newz[:,3,0]
        f4= function([z], newz[:,3,0], mode=mode_opt)
        prog=f4.maker.env.toposort()
        assert isinstance(prog[0].op, tensor.Subtensor)
        assert isinstance(prog[1].op, tensor.Rebroadcast)
        assert (f4(zval) == zval[:,3,0]).all()


class test_local_subtensor_merge(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()
        self.x_shapes = [(2,2), (5,3), (4,1), (1,2), (0,2), (2,0), (1,0), (0,0)]
        self.rng = numpy.random.RandomState(seed=utt.fetch_seed())

    def test_const(self):
        # var[const::][-1] -> var[-1]
        x = tensor.matrix('x')
        for idx in range(-7,6):
            f = function([x], x[idx::][-1], mode=mode_opt)
            g = function([x], x[idx::][-1], mode=mode_opt.excluding('local_subtensor_merge'))

            topo=f.maker.env.toposort()
            assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
            assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)

                if idx < x_s[0] and x_s[0] > 0:
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val) # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be an IndexError
                    self.assertRaises(IndexError, f, x_val)
                    self.assertRaises(IndexError, g, x_val)

    def test_scalar(self):
        # var[int::][-1] -> var[-1]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        f = function([x,y], x[y::][-1], mode=mode_opt)
        g = function([x,y], x[y::][-1], mode=mode_opt.excluding('local_subtensor_merge'))
        #theano.printing.debugprint(f, print_type=True)

        topo=f.maker.env.toposort()
        #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
        #print topo[-1].op
        assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-9, 8):
                if (idx < x_s[0]) and (x_s[0] > 0):
                    # The first subtensor is non-empty
                    f(x_val, idx) # let debugmode test something
                else:
                    self.assertRaises(IndexError, f, x_val, idx)
                    self.assertRaises(IndexError, g, x_val, idx)

    def test_const2(self):
        # var[::-1][const] -> var[-1]
        x = tensor.matrix('x')
        for idx in range(-8,7):
            f = function([x], x[::-1][idx], mode=mode_opt)
            g = function([x], x[::-1][idx], mode=mode_opt.excluding('local_subtensor_merge'))

            #theano.printing.debugprint(f, print_type=True)
            topo=f.maker.env.toposort()
            #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
            #print topo[-1].op
            assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                if (idx < x_s[0]) and (idx >= -x_s[0]):
                    # The first subtensor is non-empty, so it makes sense
                    f(x_val) # let debugmode test something
                else:
                    # A non-empty subtensor of an empty one should be an IndexError
                    self.assertRaises(IndexError, f, x_val)
                    self.assertRaises(IndexError, g, x_val)

    def test_scalar2(self):
        # var[::-1][int] -> var[-1]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        f = function([x,y], x[::-1][y], mode=mode_opt)
        g = function([x,y], x[::-1][y], mode=mode_opt.excluding('local_subtensor_merge'))
        #theano.printing.debugprint(f, print_type=True)

        topo=f.maker.env.toposort()
        #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
        #print topo[-1].op
        assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)

            for idx in range(-x_s[0], x_s[0]):
                f(x_val, idx) # let debugmode test something
            for idx in (range(x_s[0],9) + range(-9,-x_s[0])):
                self.assertRaises(IndexError, f, x_val, idx)
                self.assertRaises(IndexError, g, x_val, idx)


    def test_const3(self):
        # var[::-1][:const] -> var[-1]
        x = tensor.matrix('x')
        for idx in range(-9,8):
            f = function([x], x[::-1][:idx], mode=mode_opt)

            #theano.printing.debugprint(f, print_type=True)
            topo=f.maker.env.toposort()
            #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
            #print topo[-1].op
            assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

            for x_s in self.x_shapes:
                x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                f(x_val) # let debugmode test something

    def test_scalar3(self):
        # var[::-1][:int] -> var[-1]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        f = function([x,y], x[::-1][:y], mode=mode_opt)
        #theano.printing.debugprint(f, print_type=True)

        topo=f.maker.env.toposort()
        #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
        #print topo[-1].op
        assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx in range(-7,7):
                f(x_val, idx) # let debugmode test something

    def test_const4(self):
        # var[const1::][:const2]
        x = tensor.matrix('x')
        for idx1 in range(-7,7):
            for idx2 in range(-7,7):
                f = function([x], x[idx1:][:idx2], mode=mode_opt)

                #theano.printing.debugprint(f, print_type=True)
                topo=f.maker.env.toposort()
                #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
                assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
                #print topo[-1].op
                assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

                for x_s in self.x_shapes:
                    x_val = self.rng.uniform(size=x_s).astype(config.floatX)
                    f(x_val) # let debugmode test something

    def test_scalar4(self):
        # var[int1:][:int2]
        x = tensor.matrix('x')
        y = tensor.iscalar('y')
        z = tensor.iscalar('y')
        f = function([x,y,z], x[y:][:z], mode=mode_opt)
        #theano.printing.debugprint(f, print_type=True)

        topo=f.maker.env.toposort()
        #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
        #print topo[-1].op
        assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for idx1 in range(-11,11):
                for idx2 in range(-11,11):
                    f(x_val, idx1, idx2) # let debugmode test something

    def test_const_general(self):
        # Some cases of merge: shape, (start, stop, step) of first, (start, stop, step) of second subtensor
        cases = [
                ((2,3), (None, None, None), (None, None, -1)),
                ((12, 1), (None, None, -4), (None, None, 1)),
                ((5,3), (1, 4, 2), (None, None, -1)),
                ]
        x = tensor.matrix('x')

        for shape, sl1, sl2 in cases:
            z = x[slice(*sl1)][slice(*sl2)]
            f = function([x], z, mode=mode_opt)

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
        f = function([x,b1,e1,s1,b2,e2,s2], x[b1:e1:s1][b2:e2:s2], mode=mode_opt)
        #theano.printing.debugprint(f, print_type=True)

        topo=f.maker.env.toposort()
        #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
        #print topo[-1].op
        assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

        b1r = self.rng.permutation(range(-8,8))[:2]
        e1r = self.rng.permutation(range(-8,8))[:2]
        b2r = self.rng.permutation(range(-8,8))[:2]
        e2r = self.rng.permutation(range(-8,8))[:2]

        s1r = self.rng.permutation([-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7])[:2]
        s2r = self.rng.permutation([-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7])[:2]

        for x_s in self.x_shapes:
            x_val = self.rng.uniform(size=x_s).astype(config.floatX)
            for b1 in b1r:
                for e1 in e1r:
                    for s1 in s1r:
                        for b2 in b2r:
                            for e2 in e2r:
                                for s2 in s2r:
                                    f(x_val, b1,e1,s1,b2,e2,s2)

    def test_const4(self):
        # Bug reported by Razvan
        data = numpy.asarray(numpy.arange(8),
                             dtype = theano.config.floatX)
        x = theano.tensor.vector('x')
        y = x[7:1:-1]
        t = theano.shared(numpy.int64(0))

        fun = theano.function([x], y[t])
        val = fun(data)
        assert val == data[7:1:-1][0]

    def test_scalar6(self):
        # General case with one slice and one index
        # var[b:e:s][i]
        x = tensor.matrix('x')
        b = tensor.iscalar('b')
        e = tensor.iscalar('e')
        s = tensor.iscalar('s')
        i = tensor.iscalar('i')
        f = function([x,b,e,s,i], x[b:e:s][i], mode=mode_opt)
        #theano.printing.debugprint(f, print_type=True)

        topo=f.maker.env.toposort()
        #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
        assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) == 1
        #print topo[-1].op
        assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

        b_r = self.rng.permutation(range(-4,4))[:3]
        e_r = self.rng.permutation(range(-4,4))[:3]
        i_r = self.rng.permutation(range(-4,4))[:3]

        s_r = self.rng.permutation([-3,-2,-1,1,2,3])[:3]

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

            print 'shape: %s' % (x_s,)
            print '%% OK: %f' % (float(n_ok) * 100 / (n_ok + n_index_err))

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
        none_positions = numpy.ndindex(2, 2, 2, 2, 2, 2)

        # Ranges to be used when not None
        b1r = self.rng.permutation(range(-4,4))[:]
        e1r = self.rng.permutation(range(-4,4))[:]
        b2r = self.rng.permutation(range(-4,4))[:]
        e2r = self.rng.permutation(range(-4,4))[:]
        s1r = self.rng.permutation([-4,-3,-2,-1,1,2,3,4])[:]
        s2r = self.rng.permutation([-4,-3,-2,-1,1,2,3,4])[:]

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

            topo = f.maker.env.toposort()
            #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) <= 1
            assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

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
        none_positions = numpy.ndindex(2, 2, 2, 1)

        # Ranges to be used when not None
        b_r = self.rng.permutation(range(-4,4))[:]
        e_r = self.rng.permutation(range(-4,4))[:]
        i_r = self.rng.permutation(range(-4,4))[:]
        s_r = self.rng.permutation([-4,-3,-2,-1,1,2,3,4])[:]

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

            topo = f.maker.env.toposort()
            #print [t for t in topo if isinstance(t.op, tensor.Subtensor)]
            assert len([t for t in topo if isinstance(t.op, tensor.Subtensor)]) <= 1
            assert isinstance(topo[-1].op, theano.compile.function_module.DeepCopyOp)

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


class Test_alloc_zero(unittest.TestCase):
    def setUp(self):
        mode = theano.compile.mode.get_default_mode()
        self.mode = mode.including("local_incsubtensor_of_allocs", "local_setsubtensor_of_allocs", "local_0_dot_x")

    def test_setsubtensor_allocs0(self):
        x = tensor.matrix()
        y = tensor.matrix()
        x0 = tensor.zeros_like(x)
        y0 = tensor.zeros_like(y)
        z  = tensor.set_subtensor(x0[:4], y0)
        f = theano.function([x,y], z, mode=self.mode)
        assert numpy.all( [ not isinstance(x.op, tensor.IncSubtensor) for x in
                           f.maker.env.toposort() ])

    def test_setsubtensor_allocs1(self):
        y = tensor.matrix()
        x0 = tensor.constant(numpy.asarray(numpy.zeros_like((4,4)), dtype=config.floatX))
        y0 = tensor.zeros_like(y)
        z  = tensor.set_subtensor(x0[:4], y0)
        f = theano.function([y], z, mode=self.mode)
        assert numpy.all( [ not isinstance(x.op, tensor.IncSubtensor) for x in
                           f.maker.env.toposort() ])

    def test_setsubtensor_allocs1t(self):
        y = tensor.matrix()
        x0 = tensor.constant(numpy.asarray(numpy.zeros_like((4,4)), dtype=config.floatX))
        y0 = tensor.zeros_like(y)
        z  = tensor.set_subtensor(x0[:4], y0.T)
        f = theano.function([y], z, mode=mode_opt)
        assert numpy.all( [ not isinstance(x.op, tensor.IncSubtensor) for x in
                           f.maker.env.toposort() ])

    def test_setsubtensor_allocs2(self):
        x = tensor.matrix()
        y0 = tensor.constant(numpy.asarray(numpy.zeros_like((4,4)), dtype=config.floatX))
        x0 = tensor.zeros_like(x)
        z  = tensor.set_subtensor(x0[:4], y0)
        f = theano.function([x], z, mode=self.mode)
        assert numpy.all( [ not isinstance(x.op, tensor.IncSubtensor) for x in
                           f.maker.env.toposort() ])

    def test_incsubtensor_allocs0(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z  = tensor.inc_subtensor(x[:4], y0)
        f = theano.function([x,y], z, mode=self.mode)
        assert numpy.all( [ not isinstance(x.op, tensor.IncSubtensor) for x in
                           f.maker.env.toposort() ])

    def test_incsubtensor_allocs0t(self):
        x = tensor.matrix()
        y = tensor.matrix()
        y0 = tensor.zeros_like(y)
        z  = tensor.inc_subtensor(x[:4], y0.T)
        f = theano.function([x,y], z, mode=mode_opt)
        assert numpy.all( [ not isinstance(x.op, tensor.IncSubtensor) for x in
                           f.maker.env.toposort() ])

    def test_incsubtensor_allocs1(self):
        x = tensor.matrix()
        y0 = tensor.constant(numpy.asarray(numpy.zeros_like((4,4)), dtype=config.floatX))
        z  = tensor.inc_subtensor(x[:4], y0)
        f = theano.function([x], z, mode=self.mode)
        assert numpy.all( [ not isinstance(x.op, tensor.IncSubtensor) for x in
                           f.maker.env.toposort() ])


    def test_dot_allocs_0(self):
        v1 = tensor.vector('v1')
        v2 = tensor.vector('v2')
        m1 = tensor.matrix('m1')
        m2 = tensor.matrix('m2')
        vv = numpy.asarray([0,1,2], dtype = theano.config.floatX)
        vm = numpy.asarray([[1,2,3],[4,5,6],[7,8,9]],
                           dtype=theano.config.floatX)
        for _e1 in [(v1,vv),(m1,vm)]:
            for _e2 in [(v2,vv),(m2,vm)]:
                for p in [0,1]:
                    if p == 0:
                        e1 = tensor.zeros_like(_e1[0])
                        e2 = _e2[0]
                    else:
                        e1 = _e1[0]
                        e2 = tensor.zeros_like(_e2[0])
                    o = tensor.dot(e1,e2)
                    f = theano.function([_e1[0],_e2[0]], o, mode=self.mode)
                    f(_e1[1], _e2[1])
                    assert numpy.all([ not isinstance(x.op, tensor.Dot) for x in
                                      f.maker.env.toposort() ])


def test_local_subtensor_of_alloc():
    x = tensor.matrix('x')

    # DebugMode should detect if something goes wrong.
    # test shape combination of odd and event shape.
    for shape in [(3, 5), (4, 6), (3, 8), (4, 7)]:

        xval = numpy.zeros(shape, dtype=config.floatX)
        yval = numpy.arange(shape[1], dtype=config.floatX)

        for y in [theano.shared(yval), tensor.constant([1.])]:

            # The rows of yx are copies of y
            yx = tensor.alloc(y, x.shape[0], x.shape[1])

            # Slice of each row
            z_mat = yx[:, 3:]
            assert z_mat.ndim == 2

            # Only one column
            z_vec = yx[:, 3]
            assert z_vec.ndim == 1

            for slices in [
                # results are vector
                (slice(None), 3),
                (2, slice(None)),
                # results are matrix
                (slice(None), slice(3, None)),
                (slice(3, None), ),
                (slice(3, None), slice(3, None)),
                (slice(1, 3), slice(None, -1)),
                (slice(None, None, 2)),
                (slice(1, None, 2)),
                ]:
                z = yx.__getitem__(slices)
                f = theano.function([x], z)
                val = f(xval)
                assert xval.__getitem__(slices).shape == val.shape


def test_local_fill_useless():
    #Test opt local_fill_cut
    x = dvector()
    y = dvector()
    z = lvector()
    m = dmatrix()

    x_ = numpy.random.rand(5,)
    y_ = numpy.random.rand(5,)
    z_ = (numpy.random.rand(5,) * 5).astype("int64")
    m_ = numpy.random.rand(5, 5)

    # basic case
    f = function([x], T.fill(x, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]
    f(x_)

    # basic case
    f = function([x, y], T.second(y, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]
    f(x_, y_)

    # basic case
    f = function([x, y], T.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]
    f(x_, y_)

    # now with different type(cast)
    f = function([x, z], T.fill(z, x) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]
    f(x_, z_)

    # now with different type(cast)
    f = function([x, z], T.fill(x, z) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]
    f(x_, z_)

    # now cutting out the input ??
    f = function([x, y], T.fill(x, y) * 2, mode=mode_opt)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]
    f(x_, y_)

    # Test with different number of dimensions
    # The fill is not useless, so it should stay
    f = function([m, x], T.fill(m, x) * 2, mode=mode_opt)
    ops = [node.op.__class__ for node in f.maker.env.toposort()]
    assert T.Alloc in ops
    f(m_, x_)


class test_shapeoptimizer(unittest.TestCase):
    def setUp(self):
        utt.seed_rng()

    def test0(self):
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        v = T.vector()
        m = T.matrix()
        f = function([v,m], (v+m).shape, mode=mode)
        for node in f.maker.env.toposort():
            assert node.op != T.add

    def test_constant(self):
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'

        v = T.vector()
        m = T.matrix()
        f = function([v,m], v.dimshuffle('x','x',0).shape[1], mode=mode)
        topo = f.maker.env.toposort()
        assert len(topo) == 1
        assert topo[0].op == theano.compile.function_module.deep_copy_op

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
            #def infer_shape(self, node, (xshp,)):
                #return [tuple([self.shape_i(i)(r) for i in xrange(r.ndim)])]
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

        mode = theano.compile.get_default_mode().including('ShapeOpt', 'specialize')
        rng = numpy.random.RandomState(utt.fetch_seed())
        x = T.tensor3('x')
        ins_x = identity_noshape(x)

        # Without the optimization
        f = theano.function([x], ins_x.shape, mode=mode)
        xval = rng.randn(3,4,7).astype(config.floatX)
        assert numpy.all(f(xval) == [3,4,7])
        f_ops = [node.op for node in f.maker.env.toposort()]
        assert len(f_ops) == 5
        assert identity_noshape in f_ops
        assert identity_shape not in f_ops

        # Register the optimization
        opt.register_specialize(local_identity_noshape_to_identity_shape)

        # With the optimization
        # The identity_shape op should not be needed anymore to compute
        # the shape
        g = theano.function([x], ins_x.shape, mode=mode)
        xval = rng.randn(6,1,2).astype(config.floatX)
        assert numpy.all(g(xval) == [6,1,2])
        g_ops = [node.op for node in g.maker.env.toposort()]
        assert len(g_ops) == 4
        assert identity_noshape not in g_ops
        assert identity_shape not in g_ops


        ###test multiple level of op without infer_shape
        ins_x3 = identity_noshape(identity_noshape(identity_noshape(x)))
        h = theano.function([x], ins_x3.shape, mode=mode)
        xval = rng.randn(6,1,2).astype(config.floatX)
        assert numpy.all(h(xval) == [6,1,2])
        h_ops = [node.op for node in h.maker.env.toposort()]
        assert len(h_ops) == 4
        assert identity_noshape not in h_ops
        assert identity_shape not in h_ops

    def test_no_shapeopt(self):
        # Test that a basic example works even when ShapeOpt is excluded
        X = T.matrix()
        expr = X.shape[0]

        mode = theano.compile.get_default_mode().excluding('ShapeOpt')
        f = theano.function([X], expr, mode=mode)
        print f([[1, 2], [2, 3]])


class test_assert(unittest.TestCase):
    def test0(self):
        x=T.scalar()
        y=T.scalar()
        f = theano.function([x,y],theano.tensor.opt.assert_(x,T.eq(x,y)))
        f(1,1)
        self.assertRaises(AssertionError, f, 1,0)

    def test1(self):
        #remove assert that are always true
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode)

        x=T.scalar()
        f = theano.function([x],theano.tensor.opt.assert_(x,1),mode=mode)
        assert f(1)==1
        assert f(5)==5
        topo=f.maker.env.toposort()
        assert len(topo)==1
        assert topo[0].op==theano.compile.function_module.deep_copy_op

    def test2(self):
        #remove assert condition that are always true
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode)

        x=T.scalar()
        y=T.scalar()
        f = theano.function([x,y],theano.tensor.opt.assert_(x,y,1),mode=mode)
        assert f(1,1)==1
        assert f(5,1)==5
        topo=f.maker.env.toposort()
        assert len(topo)==2
        assert len(topo[0].inputs)==2
        assert topo[1].op==theano.compile.function_module.deep_copy_op

    def test3(self):
        #don't remove assert condition that are always false
        mode = theano.config.mode
        if mode == 'FAST_COMPILE':
            mode = 'FAST_RUN'
        mode = compile.mode.get_mode(mode)

        x=T.scalar()
        y=T.scalar()
        f = theano.function([x,y],theano.tensor.opt.assert_(x,y,0),mode=mode)
        self.assertRaises(AssertionError, f, 1,0)
        topo=f.maker.env.toposort()
        assert len(topo)==2
        assert len(topo[0].inputs)==3
        assert topo[1].op==theano.compile.function_module.deep_copy_op


def test_local_mul_specialize():
    mode = theano.config.mode
    if mode == 'FAST_COMPILE':
        mode = 'FAST_RUN'
    mode = compile.mode.get_mode(mode)
    mode = mode.excluding('fusion')

    v = T.vector()
    m = T.vector()

    f = function([v,m], v*1, mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    print nodes
    nodes == [theano.compile.function_module.deep_copy_op]

    f = function([v,m], v*0, mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    print nodes
    assert nodes == [Shape_i(0), T.alloc]

    f = function([v,m], v*(-1), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    print nodes
    assert nodes == [T.neg]

    f = function([v,m], v*1*(-m), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    print nodes
    theano.printing.debugprint(f)
    assert nodes == [T.mul, inplace.neg_inplace]

    f = function([v,m], v*0*(-m), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    print nodes
    theano.printing.debugprint(f)
    assert nodes == [Shape_i(0), T.alloc]

    f = function([v,m], v*(-1)*(-m), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    print nodes
    theano.printing.debugprint(f)
    assert nodes == [T.mul]


def speed_local_pow_specialize_range():
    val = numpy.random.rand(1e7)
    v = T.vector()
    mode = compile.mode.get_default_mode()
    mode_without_pow_opt = mode.excluding('local_pow_specialize')
    for i in range(500,513):
        f1 = function([v], v**i, mode=mode)
        f2 = function([v], v**i, mode=mode_without_pow_opt)
        assert len(f1.maker.env.toposort())==1
        t1=time.time()
        f1(val)
        t2=time.time()
        f2(val)
        t3=time.time()
        print i,t2-t1,t3-t2,t2-t1<t3-t2
        if not t2-t1<t3-t2:
            print "WARNING WE ARE SLOWER"
    for i in range(-3,-1500,-1):
        f1 = function([v], v**i, mode=mode)
        f2 = function([v], v**i, mode=mode_without_pow_opt)
        assert len(f1.maker.env.toposort())==1
        t1=time.time()
        f1(val)
        t2=time.time()
        f2(val)
        t3=time.time()
        print i,t2-t1,t3-t2,t2-t1<t3-t2
        if not t2-t1<t3-t2:
            print "WARNING WE ARE SLOWER"


def test_local_pow_specialize():
    mode = theano.config.mode
    if mode == 'FAST_COMPILE':
        mode = 'FAST_RUN'
    mode = compile.mode.get_mode(mode)
    mode = mode.excluding('fusion')

    v = T.vector()
    val = numpy.arange(10,dtype=theano.config.floatX)
    val_no0 = numpy.arange(1,10,dtype=theano.config.floatX)

    f = function([v], v**0, mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert nodes == [Shape_i(0), T.alloc]
    assert numpy.allclose(f(val),val**0)

    f = function([v], v**1, mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    nodes == [theano.compile.function_module.deep_copy_op]
    assert numpy.allclose(f(val),val**1)

    f = function([v], v**(-1), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert nodes == [T.inv]
    assert numpy.allclose(f(val_no0),val_no0**(-1))

    f = function([v], v**2, mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert nodes == [T.sqr]
    assert numpy.allclose(f(val),val**2)

    f = function([v], v**(-2), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert len(nodes)==2
    assert nodes[0] == T.sqr
    assert isinstance(nodes[1].scalar_op,theano.scalar.basic.Inv)
#    assert nodes == [T.sqr,T.inv]#Why this don't work?
    assert numpy.allclose(f(val_no0),val_no0**(-2))

    f = function([v], v**(.5), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert nodes == [T.sqrt]
    assert numpy.allclose(f(val),val**(.5))

    f = function([v], v**(-.5), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert len(nodes)==2
    assert nodes[0] == T.sqrt
    assert isinstance(nodes[1].scalar_op,theano.scalar.basic.Inv)
#    assert nodes == [T.sqrt,T.inv]#Why this don't work?
    assert numpy.allclose(f(val_no0),val_no0**(-.5))


def test_local_pow_specialize_device_more_aggressive_on_cpu():
    mode = theano.config.mode
    if mode == 'FAST_COMPILE':
        mode = 'FAST_RUN'
    mode = compile.mode.get_mode(mode)
    mode = mode.excluding('fusion').excluding('gpu')

    v = T.vector()
    val = numpy.arange(10,dtype=theano.config.floatX)
    val_no0 = numpy.arange(1,10,dtype=theano.config.floatX)
    f = function([v], v**(15), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert len(nodes)==1
    assert len(f.maker.env.toposort()[0].op.scalar_op.env.nodes)==6
    assert isinstance(nodes[0].scalar_op,theano.scalar.Composite)
    assert numpy.allclose(f(val),val**15)

    f = function([v], v**(-15), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert len(nodes)==2
    assert len(f.maker.env.toposort()[0].op.scalar_op.env.nodes)==6
    assert isinstance(nodes[0].scalar_op,theano.scalar.Composite)
    assert isinstance(nodes[-1].scalar_op,theano.scalar.basic.Inv)
    assert numpy.allclose(f(val_no0),val_no0**(-15))

    f = function([v], v**(16), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert len(nodes) == 1
    assert len(f.maker.env.toposort()[0].op.scalar_op.env.nodes)==4
    assert isinstance(nodes[0].scalar_op,theano.scalar.Composite)
    assert numpy.allclose(f(val),val**16)

    f = function([v], v**(-16), mode=mode)
    nodes = [node.op for node in f.maker.env.toposort()]
    assert len(nodes) == 2
    assert len(f.maker.env.toposort()[0].op.scalar_op.env.nodes)==4
    assert isinstance(nodes[0].scalar_op,theano.scalar.Composite)
    assert isinstance(nodes[-1].scalar_op,theano.scalar.basic.Inv)
    assert numpy.allclose(f(val_no0),val_no0**(-16))


class T_Rebroadcast(unittest.TestCase):
    def test_local_useless_rebroadcast(self):
        mode = theano.compile.get_default_mode().including('canonicalize')
        v1 = T.vector()
        v2 = T.vector()
        j = T.join(0, v1, v2)
        f = theano.function([v1, v2], j, mode=mode)
        f([1,2], [3,4,5])
        e = f.maker.env.toposort()
        assert len([n for n in e if isinstance(n.op, T.Rebroadcast)]) == 0

    def test_rebroadcast_rebroadcast(self):
        mode = theano.compile.get_default_mode().including('canonicalize')
        m = T.matrix()
        s = T.addbroadcast(m, 0, 1)
        v = T.unbroadcast(s, 1)
        f = theano.function([m], v, mode=mode)
        f([[76]])
        e = f.maker.env.toposort()
        rebroadcast_nodes = [n for n in e if isinstance(n.op, T.Rebroadcast)]
        assert len(rebroadcast_nodes) == 1
        assert rebroadcast_nodes[0].op.axis == {0: True}


class T_useless_elemwise(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('canonicalize')

    def test_eq(self):
        x=T.dmatrix()
        y=T.dmatrix()
        f=theano.function([x,y],T.eq(x,y), mode=self.mode)
        vx=numpy.random.rand(5,4)
        vy=numpy.random.rand(5,4)
        f(vx,vy)
        topo = f.maker.env.toposort()
        assert len(topo)==1
        assert isinstance(topo[0].op,T.Elemwise)
        assert isinstance(topo[0].op.scalar_op,theano.scalar.EQ)
        f2=theano.function([x],T.eq(x,x), mode=self.mode)
        assert numpy.all(f2(vx)==numpy.ones((5,4)))
        topo2 = f2.maker.env.toposort()
        print topo2
        #Shape_i{1}(<TensorType(float64, matrix)>), Shape_i{0}(<TensorType(float64, matrix)>), Alloc([[1]], Shape_i{0}.0, Shape_i{1}.0
        assert len(topo2)==3
        assert isinstance(topo2[-1].op,T.Alloc)

    def test_neq(self):
        x=T.dmatrix()
        y=T.dmatrix()
        f=theano.function([x,y],T.neq(x,y), mode=self.mode)
        vx=numpy.random.rand(5,4)
        vy=numpy.random.rand(5,4)
        f(vx,vy)
        topo = f.maker.env.toposort()
        assert len(topo)==1
        assert isinstance(topo[0].op,T.Elemwise)
        assert isinstance(topo[0].op.scalar_op,theano.scalar.NEQ)
        f2=theano.function([x],T.neq(x,x), mode=self.mode)
        assert numpy.all(f2(vx)==numpy.zeros((5,4)))
        topo2 = f2.maker.env.toposort()
        print topo2
        assert len(topo2)==3
        assert isinstance(topo2[-1].op,T.Alloc)

    def test_mul(self):
        x=T.dmatrix()
        y=T.dmatrix()
        f=theano.function([x],T.mul(x), mode=self.mode)
        vx=numpy.random.rand(5,4)
        vy=numpy.random.rand(5,4)
        f(vx)
        topo = f.maker.env.toposort()
        assert len(topo)==1
        assert topo[0].op==theano.compile.function_module.deep_copy_op
        f2=theano.function([x,y],T.mul(x,y), mode=self.mode)
        assert numpy.all(f2(vx,vy)==vx*vy)
        topo2 = f2.maker.env.toposort()
        print topo2
        assert len(topo2)==1
        assert isinstance(topo2[0].op,T.Elemwise)
        assert isinstance(topo2[0].op.scalar_op,theano.scalar.Mul)

    def test_add(self):
        x=T.dmatrix()
        y=T.dmatrix()
        f=theano.function([x],T.add(x), mode=self.mode)
        vx=numpy.random.rand(5,4)
        vy=numpy.random.rand(5,4)
        f(vx)
        topo = f.maker.env.toposort()
        assert len(topo)==1
        assert topo[0].op==theano.compile.function_module.deep_copy_op
        f2=theano.function([x,y],T.add(x,y), mode=self.mode)
        assert numpy.all(f2(vx,vy)==vx+vy)
        topo2 = f2.maker.env.toposort()
        print topo2
        assert len(topo2)==1
        assert isinstance(topo2[0].op,T.Elemwise)
        assert isinstance(topo2[0].op.scalar_op,theano.scalar.Add)

    def test_identity(self):
        # scalar.identity is used in 2 Elemwise functions:
        # tensor_copy, and view
        x = T.matrix()
        f = theano.function([x], T.tensor_copy(x), mode=self.mode)
        vx = numpy.random.rand(5,4).astype(config.floatX)
        f(vx)
        topo = f.maker.env.toposort()
        assert len(topo) == 1
        assert topo[0].op == theano.compile.function_module.deep_copy_op


def test_constant_get_stabilized():
    """
    Currently Theano enable the constant_folding optimization before stabilization optimization.
    This cause some stabilization optimization not being implemented and thus cause inf value to appear
    when it should not.

    .. note: we can't simply move the constant_folding optimization to specialize as this break other optimization!
    We will need to partially duplicate some canonicalize optimzation to specialize to fix this issue.
    """
    x2 = T.scalar()
    y2 = T.log(1+T.exp(x2))
    f2 = theano.function([x2],y2)
    try:
        assert len(f2.maker.env.toposort())==1
        assert f2.maker.env.toposort()[0].op==theano.tensor.nnet.sigm.softplus
        assert f2(800)==800

        x = T.as_tensor_variable(800)
        y = T.log(1+T.exp(x))
        f = theano.function([],y)
        assert len(f.maker.env.toposort())==0
        assert numpy.isinf(f())

        #When this error is fixed, the following line should be ok.
        assert f()==800,f()

    except (AssertionError, theano.compile.debugmode.InvalidValueError):
        raise KnownFailureTest((
            "Theano optimizes constant before stabilization. "
            "This breaks stabilization optimization in some cases. See #504."))


class T_local_switch_sink(unittest.TestCase):
    def setUp(self):
        # condition values
        self.condm = numpy.asarray([[0.1,0,1,-1],[0.,0.,0.,0.],[1,1,1,1]])
        self.condv = numpy.asarray([0.1,0,1,-1])
        self.conds = [0.1,0,1,-1]

        # x values
        self.xm = numpy.ones((3,4))
        self.xv = numpy.ones((4,))
        self.xs = 1.

        # expected results
        self.resm = [numpy.asarray([[1,0,1,0],[0,0,0,0],[1,1,1,1]])]*3 + [numpy.asarray([[1,0,1,0],[1,0,1,0],[1,0,1,0]])] + \
                    2*[numpy.asarray([[1,0,1,0]])] + [[numpy.ones((3,4)),numpy.zeros((3,4)),numpy.ones((3,4)),numpy.zeros((3,4))]] + \
                    [[numpy.ones((4,)),numpy.zeros((4,)),numpy.ones((4,)),numpy.zeros((4,))]] + \
                    [[numpy.asarray(1.0),numpy.asarray(0.0),numpy.asarray(1.0),numpy.asarray(0.0)]]

        self.mode = theano.compile.mode.get_default_mode().including('canonicalize','fast_run').excluding('gpu','fusion')
        self.mode = copy.copy(self.mode)
        self.mode.check_isfinite = False

    def test_local_mul_switch_sink(self):
        c = T.dscalar()
        idx = 0
        for condition in [(T.dmatrix('cond'),self.condm),(T.dvector('cond'),self.condv),(T.dscalar('cond'),self.conds)]:
            for x in [(T.dmatrix('x'),self.xm),(T.dvector('x'),self.xv),(T.dscalar('x'),self.xs)]:
                y = T.mul(T.switch(condition[0]>0,1.*x[0],0.*x[0]),T.switch(condition[0]>0,1.*x[0],T.log(c)*x[0]))
                f = theano.function([condition[0],x[0],c],[y], mode=self.mode)
                if type(condition[1]) is list:
                    for i in range(len(condition[1])):
                        res= f(condition[1][i],x[1],-1)
                        assert (res == numpy.asarray(self.resm[idx][i])).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1],x[1],-1)
                    assert (res == numpy.asarray(self.resm[idx])).sum() == self.resm[idx].size
                idx += 1

    def test_local_div_switch_sink(self):
        c = T.dscalar()
        idx = 0
        for condition in [(T.dmatrix('cond'),self.condm),(T.dvector('cond'),self.condv),(T.dscalar('cond'),self.conds)]:
            for x in [(T.dmatrix('x'),self.xm),(T.dvector('x'),self.xv),(T.dscalar('x'),self.xs)]:
                y = T.true_div(T.switch(condition[0]>0,1.*x[0],0.*x[0]),T.switch(condition[0]>0,1.*x[0],T.log(c)*x[0]))
                f = theano.function([condition[0],x[0],c],[y], mode=self.mode)
                if type(condition[1]) is list:
                    for i in range(len(condition[1])):
                        res= f(condition[1][i],x[1],-1)
                        assert (res == numpy.asarray(self.resm[idx][i])).sum() == self.resm[idx][i].size
                else:
                    res = f(condition[1],x[1],-1)
                    assert (res == numpy.asarray(self.resm[idx])).sum() == self.resm[idx].size
                idx += 1


class T_local_erf(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.mode.get_default_mode().including('canonicalize','fast_run').excluding('gpu','fusion')
        self.mode._optimizer.position_cutoff = 1.50001

    def test_local_one_plus_erf(self):
        val = numpy.asarray([-30,-3,-2,-1,0,1,2,3,30], dtype=config.floatX)
        x = T.vector()

        f = theano.function([x],1+T.erf(x), mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.mul,T.erfc], f.maker.env.toposort()
        f(val)

        f = theano.function([x],T.erf(x)+1, mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.mul,T.erfc], f.maker.env.toposort()
        f(val)

        f = theano.function([x],T.erf(x)+2, mode=self.mode)
        topo = f.maker.env.toposort()
        print topo
        assert len(topo)==2
        assert topo[0].op==T.erf
        assert isinstance(topo[1].op,T.Elemwise)
        assert isinstance(topo[1].op.scalar_op,scal.Add)
        f(val)

    def test_local_one_minus_erf(self):
        val = numpy.asarray([-30,-3,-2,-1,0,1,2,3,30], dtype=config.floatX)
        x = T.vector()

        f = theano.function([x],1-T.erf(x), mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.erfc], f.maker.env.toposort()
        print f(val)

        f = theano.function([x],1+(-T.erf(x)), mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.erfc], f.maker.env.toposort()
        print f(val)

        f = theano.function([x],(-T.erf(x))+1, mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.erfc], f.maker.env.toposort()
        print f(val)

        f = theano.function([x],2-T.erf(x), mode=self.mode)
        topo = f.maker.env.toposort()
        print topo
        assert len(topo)==2, f.maker.env.toposort()
        assert topo[0].op==T.erf, f.maker.env.toposort()
        assert isinstance(topo[1].op,T.Elemwise), f.maker.env.toposort()
        assert isinstance(topo[1].op.scalar_op,scal.Add) or isinstance(topo[1].op.scalar_op,scal.Sub), f.maker.env.toposort()
        print f(val)

    def test_local_erf_minus_one(self):
        val = numpy.asarray([-30,-3,-2,-1,0,1,2,3,30], dtype=config.floatX)
        x = T.vector()

        f = theano.function([x],T.erf(x)-1, mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.erfc,T.mul]
        print f(val)

        f = theano.function([x],T.erf(x)+(-1), mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.erfc,T.mul]
        print f(val)

        f = theano.function([x],-1+T.erf(x), mode=self.mode)
        print f.maker.env.toposort()
        assert [n.op for n in f.maker.env.toposort()]==[T.erfc,T.mul]
        print f(val)

        f = theano.function([x],T.erf(x)-2, mode=self.mode)
        topo = f.maker.env.toposort()
        print topo
        assert len(topo)==2
        assert topo[0].op==T.erf
        assert isinstance(topo[1].op,T.Elemwise)
        assert isinstance(topo[1].op.scalar_op,scal.Add) or isinstance(topo[1].op.scalar_op,scal.Sub)
        print f(val)


class T_local_erfc(unittest.TestCase):
    def setUp(self):
        self.mode_fusion = theano.compile.mode.get_default_mode().including('canonicalize').including('fast_run').excluding('gpu')
        self.mode = self.mode_fusion.excluding('fusion')
        self.mode._optimizer.position_cutoff = 1.50001

    def test_local_one_minus_erfc(self):
        """ test opt: 1-erfc(x) => erf(x) and -erfc(x)+1 => erf(x)
        """
        val = numpy.asarray([-30,-3,-2,-1,0,1,2,3,30], dtype=config.floatX)
        x = T.vector('x')

        f = theano.function([x],1-T.erfc(x), mode=self.mode)
        theano.printing.debugprint(f)
        assert [n.op for n in f.maker.env.toposort()]==[T.erf], f.maker.env.toposort()
        print f(val)

        f = theano.function([x],(-T.erfc(x))+1, mode=self.mode)
        theano.printing.debugprint(f)
        assert [n.op for n in f.maker.env.toposort()]==[T.erf], f.maker.env.toposort()
        print f(val)

        f = theano.function([x],2-T.erfc(x), mode=self.mode)
        topo = f.maker.env.toposort()
        theano.printing.debugprint(f)
        assert len(topo)==2, f.maker.env.toposort()
        assert topo[0].op==T.erfc, f.maker.env.toposort()
        assert isinstance(topo[1].op,T.Elemwise), f.maker.env.toposort()
        assert isinstance(topo[1].op.scalar_op,scal.Sub), f.maker.env.toposort()
        print f(val)

    def test_local_erf_neg_minus_one(self):
        """ test opt: (-1)+erfc(-x)=>erf(x)"""
        val = numpy.asarray([-30,-3,-2,-1,0,1,2,3,30], dtype=config.floatX)
        x = T.vector('x')

        f = theano.function([x],-1+T.erfc(-x), mode=self.mode)
        theano.printing.debugprint(f)
        assert [n.op for n in f.maker.env.toposort()]==[T.erf], f.maker.env.toposort()
        print f(val)

        f = theano.function([x],T.erfc(-x)-1, mode=self.mode)
        theano.printing.debugprint(f)
        assert [n.op for n in f.maker.env.toposort()]==[T.erf], f.maker.env.toposort()
        print f(val)

        f = theano.function([x],T.erfc(-x)+(-1), mode=self.mode)
        theano.printing.debugprint(f)
        assert [n.op for n in f.maker.env.toposort()]==[T.erf], f.maker.env.toposort()
        print f(val)

    def test_local_log_erfc(self):
        val = [-30,-27,-26,-11,-10,-3,-2,-1,0,1,2,3,10,11,26,27,28,30]
        if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            #python mode don't like the inv(0)
            val.remove(0)
        val = numpy.asarray(val, dtype=config.floatX)
        x = T.vector('x')

        #their is some nan that will happear in the graph for the log of the negatives values
        mode = copy.copy(self.mode)
        mode.check_isfinite = False
        mode_fusion = copy.copy(self.mode_fusion)
        mode_fusion.check_isfinite = False

        f = theano.function([x],T.log(T.erfc(x)), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==23, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX
        assert all(numpy.isfinite(f(val)))

        f = theano.function([x],T.log(T.erfc(-x)), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==24, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX
        assert all(numpy.isfinite(f(-val)))

        f = theano.function([x],T.log(T.erfc(x)), mode=mode_fusion)
        assert len(f.maker.env.nodes)==1, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX
        assert len(f.maker.env.toposort()[0].env.toposort()[0].op.scalar_op.env.nodes)==2,len(f.maker.env.toposort()[0].env.toposort()[0].op.scalar_op.env.nodes)
        #TODO: fix this problem
        if theano.config.floatX=="float32" and theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            raise KnownFailureTest("the python code upcast somewhere internally some value of float32 to python float for part of its computation. That make that the c and python code don't generate the same value. You can ignore this error.")
        assert all(numpy.isfinite(f(val)))

    def test_local_grad_log_erfc_neg(self):
        val = [-100,-30,-27,-26.4,-26.2,-26,-11,-10,-9,-3,-2,-1,0,1,2,3,9,10,11,27,26.4,26.2,26,28,30,100]
        if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
#python mode don't like the inv(0) in computation, but the switch don't select this value. So it is computed for no good reason.
            val.remove(0)
        if theano.config.mode in ["DebugMode", "DEBUG_MODE"] and theano.config.floatX=='float32':
            # In float32 their is a plage of values close to 10 that we stabilize as it give bigger error then the stabilized version.
            # The orig value in float32 -30.0, the stab value -20.1 the orig value in float64 -18.1.
            val.remove(10)
        val = numpy.asarray(val, dtype=config.floatX)
        x = T.vector('x')
        y = T.vector('y')

        #their is some nan that will happear in the graph for the log of the negatives values
        mode = copy.copy(self.mode)
        mode.check_isfinite = False
        mode_fusion = copy.copy(self.mode_fusion)
        mode_fusion.check_isfinite = False

        f = theano.function([x],T.grad(T.log(T.erfc(x)).sum(),x), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==23, len(f.maker.env.nodes)
        assert all(numpy.isfinite(f(val)))
        assert f.maker.env.outputs[0].dtype==theano.config.floatX

        #test with a different mul constant
        f = theano.function([x],T.mul(T.exp(T.neg(T.sqr(x))),-10.12837917)/T.erfc(x), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==23, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX
        assert all(numpy.isfinite(f(val)))

        #test that we work without the mul
        f = theano.function([x],T.exp(T.neg(T.sqr(x)))/T.erfc(x), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==23, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX
        assert all(numpy.isfinite(f(val)))

        #test that we don't work if x!=y
        f = theano.function([x,y],T.exp(T.neg(T.sqr(x)))/T.erfc(y), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==5, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX
        f(val,val-3)

        #test that we work without the sqr and neg
        f = theano.function([x],T.exp(T.mul(-1,x,x))/T.erfc(x), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==22, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX
        assert all(numpy.isfinite(f(val)))

        #test that it work correctly if x is x*2 in the graph.
        f = theano.function([x],T.grad(T.log(T.erfc(2*x)).sum(),x), mode=mode)
        #theano.printing.debugprint(f)
        assert len(f.maker.env.nodes)==23, len(f.maker.env.nodes)
        assert numpy.isfinite(f(val)).all()
        assert f.maker.env.outputs[0].dtype==theano.config.floatX

        f = theano.function([x],T.grad(T.log(T.erfc(x)).sum(),x), mode=mode_fusion)
        assert len(f.maker.env.nodes)==1, len(f.maker.env.nodes)
        assert f.maker.env.outputs[0].dtype==theano.config.floatX

        #TODO: fix this problem
        if theano.config.floatX=="float32" and theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            #Showing this test error is a duplicate of the one in test_local_log_erfc. We hide it.
            #raise KnownFailureTest("the python code upcast somewhere internally some value of float32 to python float for part of its computation. That make that the c and python code don't generate the same value. You can ignore this error. This happen in an intermediate step that don't show in the final result.")
            pass
        else:
            assert all(numpy.isfinite(f(val)))

    def speed_local_log_erfc(self):

        val = numpy.random.rand(1e6)
        x = T.vector()
        mode=theano.compile.mode.get_mode("FAST_RUN")
        f1 = theano.function([x],T.log(T.erfc(x)), mode=mode.excluding("local_log_erfc"))
        f2 = theano.function([x],T.log(T.erfc(x)), mode=mode)
        print f1.maker.env.toposort()
        print f2.maker.env.toposort()
        t0=time.time()
        f1(val)
        t1=time.time()
        f2(val)
        t2=time.time()
        print t1-t0,t2-t1


class test_local_remove_switch_const_cond(unittest.TestCase):
    def setUp(self):
        self.mode = mode_opt.excluding('constant_folding')

    def test_const0(self):

        for dtype1 in ['int32', 'int64']:
            for dtype2 in ['int32', 'int64']:
                x = theano.tensor.matrix('x', dtype=dtype1)
                y = theano.tensor.matrix('y', dtype=dtype2)
                z = theano.tensor.switch(0, x, y)
                f = theano.function([x,y], z, mode=self.mode)
                assert len([node.op for node in f.maker.env.toposort() if
                            ( isinstance(node.op,theano.tensor.Elemwise)
                           and isinstance(node.op.scalar_op,
                                          theano.scalar.basic.Switch))]) == 0
                vx = numpy.array([[1,2,3],[ 4, 5, 6]], dtype=dtype1)
                vy = numpy.array([[7,8,9],[10,11,12]], dtype=dtype2)
                assert numpy.all(f(vx,vy) == vy)

    def test_const1(self):

        for dtype1 in ['int32', 'int64']:
            for dtype2 in ['int32', 'int64']:
                x = theano.tensor.matrix('x', dtype=dtype1)
                y = theano.tensor.matrix('y', dtype=dtype2)
                z = theano.tensor.switch(1, x, y)
                f = theano.function([x,y], z, mode=self.mode)
                assert len([node.op for node in f.maker.env.toposort() if
                            ( isinstance(node.op,theano.tensor.Elemwise)
                           and isinstance(node.op.scalar_op,
                                          theano.scalar.basic.Switch))]) == 0
                vx = numpy.array([[1,2,3],[ 4, 5, 6]], dtype=dtype1)
                vy = numpy.array([[7,8,9],[10,11,12]], dtype=dtype2)
                assert numpy.all(f(vx,vy) == vx)

    def test_broadcast1(self):
        #test switch(cst, matrix, row)
        x = theano.tensor.matrix('x', dtype='int32')
        y = theano.tensor.vector('y', dtype='int64')

        z = theano.tensor.switch(1, x, y)
        f = theano.function([x,y], z, mode=self.mode)
        #theano.printing.debugprint(f)
        assert len([node.op for node in f.maker.env.toposort() if
                    isinstance(node.op,theano.tensor.Elemwise) and
                    not isinstance(node.op.scalar_op,theano.scalar.basic.Cast)]) == 0
        vx = numpy.array([[1, 2, 3],[ 4, 5, 6]], dtype='int32')
        vy = numpy.array([10,11,12], dtype='int64')
        assert numpy.all(f(vx,vy) == vx)


        z = theano.tensor.switch(0, x, y)
        f = theano.function([x,y], z, mode=self.mode)
        #theano.printing.debugprint(f)
        assert len([node.op for node in f.maker.env.toposort() if
                    isinstance(node.op,theano.tensor.Elemwise) ]) == 0
        vx = numpy.array([[1, 2, 3],[ 4, 5, 6]], dtype='int32')
        vy = numpy.array([10,11,12], dtype='int64')
        assert numpy.all(f(vx,vy) == vy)

    def test_broadcast2(self):
        #test switch(cst, vector, matrix)

        #This case is not optimized for now.
        x = theano.tensor.vector('x', dtype='int32')
        y = theano.tensor.matrix('y', dtype='int64')
        z = theano.tensor.switch(1, x, y)
        f = theano.function([x,y], z, mode=self.mode)
        assert len([node.op for node in f.maker.env.toposort() if
                    isinstance(node.op,theano.tensor.Elemwise) and
                    not isinstance(node.op.scalar_op,theano.scalar.basic.Cast)]) == 0
        vx = numpy.array([ 4, 5, 6], dtype='int32')
        vy = numpy.array([[7,8,9],[10,11,12]], dtype='int64')
        assert numpy.all(f(vx,vy) == vx)

        z = theano.tensor.switch(0, x, y)
        f = theano.function([x,y], z, mode=self.mode)
        assert len([node.op for node in f.maker.env.toposort() if
                    isinstance(node.op,theano.tensor.Elemwise) ]) == 0
        vx = numpy.array([ 4, 5, 6], dtype='int32')
        vy = numpy.array([[7,8,9],[10,11,12]], dtype='int64')
        assert numpy.all(f(vx,vy) == vy)


class T_local_sum(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('canonicalize')

    def test_local_sum_all_to_none(self):
        a = T.tensor3()
        input=numpy.arange(3*3*3, dtype=config.floatX).reshape(3,3,3)
        f = theano.function([a],a.sum(),mode=self.mode)
        assert len(f.maker.env.nodes)==1
        assert numpy.allclose(f(input),input.sum())

        f = theano.function([a],a.sum([0,1,2]),mode=self.mode)
        assert len(f.maker.env.nodes)==1
        assert numpy.allclose(f(input),input.sum())


        backup = config.warn.sum_sum_bug
        config.warn.sum_sum_bug = False
        try:
            f = theano.function([a],a.sum(0).sum(0).sum(0),mode=self.mode)
            assert len(f.maker.env.nodes)==1
            assert numpy.allclose(f(input),input.sum())
        finally:
            config.warn.sum_sum_bug = backup

    def test_local_sum_sum(self):
        a=T.tensor3()
        input=numpy.arange(3*3*3, dtype=config.floatX).reshape(3,3,3)
        dims=[(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

        backup = config.warn.sum_sum_bug
        config.warn.sum_sum_bug = False
        try:
            for d,dd in dims:
                f = theano.function([a],a.sum(d).sum(dd),mode=self.mode)
                assert numpy.allclose(f(input),input.sum(d).sum(dd))
                assert len(f.maker.env.nodes)==1
            for d,dd in dims:
                f = theano.function([a],a.sum(d).sum(dd).sum(0),mode=self.mode)
                assert numpy.allclose(f(input),input.sum(d).sum(dd).sum(0))
                assert len(f.maker.env.nodes)==1
            for d in [0,1,2]:
                f = theano.function([a],a.sum(d).sum(None),mode=self.mode)
                assert numpy.allclose(f(input),input.sum(d).sum())
                assert len(f.maker.env.nodes)==1
            for d in [0,1,2]:
                f = theano.function([a],a.sum(None).sum(),mode=self.mode)
                assert numpy.allclose(f(input),input.sum())
                assert len(f.maker.env.nodes)==1
        finally:
            config.warn.sum_sum_bug = backup

    def test_local_sum_alloc(self):
        a=T.dtensor3()
        input=numpy.asarray(numpy.arange(2*3*4).reshape(2,3,4),dtype='float64')
        mode = self.mode.including('specialize').excluding('fusion')

        for t_like,n_like,nb_nodes in [(tensor.zeros_like,numpy.zeros_like,(1,3,3,2)),
                                       (tensor.ones_like,numpy.ones_like,(5,5,5,6))]:

            f = theano.function([a],t_like(a).sum(None),mode=mode)
            assert numpy.allclose(f(input),n_like(input).sum())
            assert len(f.maker.env.nodes)==nb_nodes[0]

            f = theano.function([a],t_like(a).sum([0,1,2]),mode=mode)
            assert numpy.allclose(f(input),n_like(input).sum())
            assert len(f.maker.env.nodes)==nb_nodes[0]

            for d in range(3):
                f = theano.function([a],t_like(a).sum(d),mode=mode)
                assert numpy.allclose(f(input),n_like(input).sum(d))
                assert len(f.maker.env.nodes)==nb_nodes[1]
                assert f.maker.env.toposort()[-1].op==T.alloc

            for i in range(3):
                f = theano.function([a],t_like(a).sum(i),mode=mode)
                assert numpy.allclose(f(input),n_like(input).sum(i))
                assert len(f.maker.env.nodes)==nb_nodes[2]
                assert f.maker.env.toposort()[-1].op==T.alloc

            backup = config.warn.sum_sum_bug
            config.warn.sum_sum_bug = False
            try:
                for d, dd in [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]:
                    f = theano.function([a],t_like(a).sum(d).sum(dd),mode=mode)
                    print f.maker.env.toposort()
                    assert numpy.allclose(f(input),n_like(input).sum(d).sum(dd))
                    assert len(f.maker.env.nodes)==nb_nodes[3]
                    assert f.maker.env.toposort()[-1].op==T.alloc
            finally:
                config.warn.sum_sum_bug = backup

    def test_local_sum_sum_int8(self):
        """
        Test that local_sum_sum works when combining two sums on an int8 array.

        This is a regression test for ticket gh-356.
        """
        x = tensor.tensor3(dtype='int8')
        y = x.sum(axis=0).sum(axis=1)
        backup = config.on_opt_error
        config.on_opt_error = 'raise'
        try:
            # This compilation would fail prior to fix.
            f = theano.function([x], y)
        finally:
            config.on_opt_error = backup

    def test_local_sum_sum_dtype(self):
        """
        Test that local_sum_sum works when specifying dtypes manually.
        """
        x = tensor.tensor3(dtype='int8')
        y = x.sum(axis=0, dtype='int32').sum(axis=1, dtype='int64')
        backup = config.on_opt_error
        config.on_opt_error = 'raise'
        try:
            # This compilation would fail prior to fix.
            f = theano.function([x], y)
        finally:
            config.on_opt_error = backup


class T_local_sum_dimshuffle(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('canonicalize')

    def test_local_sum_div_dimshuffle(self):
        a = T.matrix('a')
        b = T.vector('b')
        c = T.tensor3('c')
        d = T.scalar('d')
        sum = tensor.sum
        sums = [
            sum(a/d),
            sum(a/d.dimshuffle('x','x')),
            sum(a/d.dimshuffle('x','x'), axis=0),
            sum(a/d.dimshuffle('x','x'), axis=1),
            sum(b/d),
            sum(b/d.dimshuffle('x')),
            sum(c/d),
            sum(c/d.dimshuffle('x','x','x')),
            sum(c/d.dimshuffle('x','x','x'),axis=0),
            sum(c/d.dimshuffle('x','x','x'),axis=1),
            sum(c/d.dimshuffle('x','x','x'),axis=2),

            sum(a / b, axis=0),
            sum(a / b.dimshuffle(0,'x'), axis=1),
            sum(a.dimshuffle(0,1)/ b.dimshuffle(0,'x'), axis=1),
            sum(a.dimshuffle(1,0)/ b.dimshuffle(0,'x'), axis=1),
            sum(c / a, axis=0),
            sum(c / a.dimshuffle(1, 0), axis=0),
            sum(c / a.dimshuffle(0,'x',1), axis=1),
            sum(c / a.dimshuffle(1,'x',0), axis=1),
            sum(c / a.dimshuffle(0, 1, 'x'), axis=2),
            sum(c / a.dimshuffle(1, 0, 'x'), axis=2),
            sum(c / b, axis=0),
            sum(c / b, axis=1),
            sum(c / b, axis=(0,1)),
            sum(c / b.dimshuffle(0,'x'), axis=0),
            sum(c / b.dimshuffle(0,'x'), axis=2),
            sum(c / b.dimshuffle(0,'x'), axis=(0,2)),
            sum(c / b.dimshuffle(0,'x','x'), axis=1),
            sum(c / b.dimshuffle(0,'x','x'), axis=2),
            sum(c / b.dimshuffle(0,'x','x'), axis=(1,2)),
            sum(sum(c, axis=0) / b, axis=0),
            sum(sum(c, axis=1) / b, axis=0),
            ]

        rng = numpy.random.RandomState(utt.fetch_seed())
        a_val = rng.randn(2,2).astype(config.floatX)
        b_val = rng.randn(2).astype(config.floatX)
        c_val = rng.randn(2,2,2).astype(config.floatX)
        d_val = numpy.asarray(rng.randn(), config.floatX)

        backup = config.warn.sum_sum_bug, config.warn.sum_div_dimshuffle_bug
        config.warn.sum_sum_bug = False
        config.warn.sum_div_dimshuffle_bug = False
        try:
            for i,s in enumerate(sums):
                print i
                f = theano.function([a,b,c,d], s, mode=self.mode)
                theano.printing.debugprint(f)
                g = f.maker.env.toposort()
                #print 'g =', g
                assert isinstance(g[-1].op.scalar_op, theano.scalar.basic.TrueDiv)
                f(a_val, b_val, c_val, d_val)
        finally:
            config.warn.sum_sum_bug, config.warn.sum_div_dimshuffle_bug = backup

    # TODO:
    # test_local_sum_prod_dimshuffle (a * b * c)
    # test_local_sum_divprod_dimshuffle ((a * b) / (c * d))


def test_make_vector():
    b = T.bscalar()
    i = T.iscalar()
    d = T.dscalar()

    #TODO: draw random values instead. Not really important.
    val = {b: 2,
           i: -3,
           d: 0.7}

    # Should work
    for (dtype, inputs) in [("int8", (b,b)),
                            ("int32", (i,b)),
                            ("int32", (b,i)),
                            ("float64", (b,i)),
                            ("float64", (b,d)),
                            ("float64", (d,i)),
                            ("float64", ()),
                            ("int64", ()),
                            ]:
        mv = opt.MakeVector(dtype=dtype)(*inputs)
        assert mv.dtype == dtype
        f = theano.function([b,i,d], mv)
        f_val = f(val[b], val[i], val[d])
        #print 'f_val =', f_val


        s = mv.sum()
        gb = T.grad(s, b, disconnected_inputs='ignore')
        gi = T.grad(s, i, disconnected_inputs='ignore')
        gd = T.grad(s, d, disconnected_inputs='ignore')
        #print 'gb =', gb
        #print 'gi =', gi
        #print 'gd =', gd

        g = theano.function([b,i,d], [gb, gi, gd])
        g_val = g(val[b], val[i], val[d])
        #print 'g_val =', g_val

        if dtype.startswith('int'):
            # The gradient should be 0
            assert numpy.allclose(g_val, 0)
        else:
            for var, grval in zip((b,i,d), g_val):
                float_inputs = []
                if var.dtype.startswith('int'):
                    assert grval == 0
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

    #should fail
    for (dtype,inputs) in [("int8",(b,i)),
                           ("int8",(i,b)),
                           ("int8",(b,d)),
                           ("int8",(i,i)),
                           ("int32",(d,i)),
                           ("int32",(i,d)),
                           ("float32",(i,d)),
                           ]:
        try:
            opt.MakeVector(dtype=dtype)(*inputs)
            raise Exception("Theano should have raised an error")
        except AssertionError:
            pass


def test_local_join_1():
    #test for vector
    a = tensor.vector('a')
    s = tensor.stack(a)
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert numpy.all(val == [1])
    e = f.maker.env.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.env.outputs[0].dtype == config.floatX

    #test for matrix join(0,a)
    a = tensor.matrix('a')
    s = join(0,a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert numpy.all(val == [[1]])
    e = f.maker.env.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.env.outputs[0].dtype == config.floatX

    #test for matrix join(1,a)
    s = join(1,a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert numpy.all(val == [[1]])
    e = f.maker.env.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.env.outputs[0].dtype == config.floatX

    #test we don't apply when their is 2 inputs
    s = join(1,a,a)
    f = function([a], s, mode=mode_opt)
    val = f([[1]])
    assert numpy.all(val == [[1]])
    e = f.maker.env.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 1
    assert f.maker.env.outputs[0].dtype == config.floatX


def test_local_mul_to_neg():
    """
    Test that a multiplication by -1 or -1.0 yields the appropriate data type
    """
    a = T.imatrix()
    f1 = theano.function([a], -1*a)
    f2 = theano.function([a], -1.0*a)
    aval = numpy.random.randint(0,10,(2,2)).astype('int32')
    if config.cast_policy == 'custom':
        assert f1(aval).dtype == a.dtype
        assert f2(aval).dtype == 'float64'
    elif config.cast_policy == 'numpy':
        assert f1(aval).dtype == str(numpy.array(0).dtype)
        assert f2(aval).dtype == 'float64'
    elif config.cast_policy == 'numpy+floatX':
        assert f1(aval).dtype == str(numpy.array(0).dtype)
        assert f2(aval).dtype == config.floatX
    else:
        raise NotImplementedError(config.cast_policy)


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
        e = f.maker.env.toposort()
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
        e = f.maker.env.toposort()
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
    theano.printing.debugprint(out, print_type=True)
    print out.broadcastable
    assert numpy.all(out.broadcastable == (True, False))

    f = theano.function([num_len_s, denom_s], out)
    out_val = f(3, 2.)
    assert out_val.shape == (1, 3)
    assert numpy.allclose(out_val, 0.5)


class Test_lift_transpose_through_dot(unittest.TestCase):
    def simple_optimize(self, g):
        out2in(opt.local_useless_elemwise).optimize(g)
        out2in(opt.local_lift_transpose_through_dot).optimize(g)
        out2in(opt.local_useless_elemwise).optimize(g)
        return g

    def test_matrix_matrix(self):
        a, b = matrices('ab')
        g = self.simple_optimize(Env([a, b], [tensor.dot(a, b).T]))
        sg = '[dot(DimShuffle{1,0}(b), DimShuffle{1,0}(a))]'
        assert str(g) == sg

    def test_row_matrix(self):
        a = vector('a')
        b = matrix('b')
        g = optimize(Env(
            [a, b],
            [tensor.dot(a.dimshuffle('x', 0), b).T]),
            level='stabilize')
        sg = '[dot(DimShuffle{1,0}(b), DimShuffle{0,x}(a))]'
        assert str(g) == sg

    def test_matrix_col(self):
        a = vector('a')
        b = matrix('b')
        g = optimize(Env(
            [a, b],
            [tensor.dot(b, a.dimshuffle(0, 'x')).T]),
            level='stabilize')
        sg = '[dot(DimShuffle{x,0}(a), DimShuffle{1,0}(b))]'
        assert str(g) == sg


if __name__ == '__main__':
#    unittest.main()
    test_fusion().tes_memory_leak()
