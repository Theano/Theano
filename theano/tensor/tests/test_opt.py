## PENDING REWRITE OF tensor_opt.py

import copy
import time
import unittest

import numpy
from nose.plugins.skip import SkipTest
from numpy.testing.noseclasses import KnownFailureTest

import theano
from theano import config
from theano import gof
from theano.tensor.opt import *
from theano import tensor  #do not use, there is  an import * below that hides it
from theano import tensor as TT  #ugly but works for now...
from theano.tensor import TensorType, inplace
from theano.gof import Env
from theano.tensor.elemwise import DimShuffle
from theano import pprint, shared
from theano.tests import unittest_tools as utt

from theano import function, compile
mode_opt = theano.config.mode
if mode_opt == 'FAST_COMPILE':
    mode_opt = 'FAST_RUN'

def inputs(xbc = (0, 0), ybc = (0, 0), zbc = (0, 0)):
    x = TensorType(broadcastable = xbc, dtype = 'float64')('x')
    y = TensorType(broadcastable = ybc, dtype = 'float64')('y')
    z = TensorType(broadcastable = zbc, dtype = 'float64')('z')
    return x, y, z


ds = lambda x, y: DimShuffle(x.type.broadcastable, y)(x)
dimshuffle_lift = out2in(local_dimshuffle_lift)

class test_dimshuffle_lift(unittest.TestCase):

    def test_double_transpose(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 0)), (1, 0))
        g = Env([x], [e])
        self.failUnless(str(g) == "[DimShuffle{1,0}(DimShuffle{1,0}(x))]")
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == "[x]")

    def test_merge2(self):
        x, y, z = inputs()
        e = ds(ds(x, (1, 'x', 0)), (2, 0, 'x', 1))
        g = Env([x], [e])
        self.failUnless(str(g) == "[DimShuffle{2,0,x,1}(DimShuffle{1,x,0}(x))]", str(g))
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == "[DimShuffle{0,1,x,x}(x)]", str(g))

    def test_elim3(self):
        x, y, z = inputs()
        e = ds(ds(ds(x, (0, 'x', 1)), (2, 0, 'x', 1)), (1, 0))
        g = Env([x], [e])
        self.failUnless(str(g) == "[DimShuffle{1,0}(DimShuffle{2,0,x,1}(DimShuffle{0,x,1}(x)))]", str(g))
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == "[x]", str(g))

    def test_lift(self):
        x, y, z = inputs([False]*1, [False]*2, [False]*3)
        e = x + y + z
        g = Env([x, y, z], [e])
        self.failUnless(str(g) == ("[Elemwise{add,no_inplace}("
            "InplaceDimShuffle{x,0,1}(Elemwise{add,no_inplace}"
            "(InplaceDimShuffle{x,0}(x), y)), z)]"), str(g))
        dimshuffle_lift.optimize(g)
        self.failUnless(str(g) == ("[Elemwise{add,no_inplace}(Elemwise"
            "{add,no_inplace}(InplaceDimShuffle{x,x,0}(x), InplaceDimShuffle"
            "{x,0,1}(y)), z)]"), str(g))


def test_add_canonizer_problem0():
    #observed in a real graph

    n_segments = 10
    label = lscalar('label')
    segment_labels = label + theano._asarray([0] * n_segments, dtype='int64')

    r = segment_labels * 5
    f = function([label], r)

from theano.tensor import *
# Why is there TWO 'import *' in this file???

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
        r = mul(
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
            (fx+fy+fz+2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (fx*fy*fz*2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
#            (2+fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
#            (2*fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2+fx+fy+fz+2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
            (2*fx*fy*fz*2,(fx,fy,fz),(fxv,fyv,fzv),1,'float32'),
#            (fx*fy*2*(fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
#            (fx*fy*(2+fx+fy+fz),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),
            (fx*fy*2*(fx+fy+fz+2),(fx,fy,fz),(fxv,fyv,fzv),2,'float32'),

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
        raise SkipTest("Current implementation of Canonizer don't implement all case. Skip the corresponding test")

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
        import theano.tensor, theano.compile

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
                    assert isinstance(topo[1].op, TT.Alloc)
                else:
                    assert len(topo)==3
                    assert isinstance(topo[0].op, Shape_i)
                    assert isinstance(topo[1].op, Shape_i)
                    assert isinstance(topo[2].op, TT.Alloc)
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
                    assert isinstance(topo[-1].op, TT.Alloc)

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
                                                           (((2.0*fx)/(4.0*fy)),[fx,fy],[fxv,fyv],'float32'),
                                                           (((2.0*dv)/(4.0*dy)),[dv,dy],[dvv,dyv],'float64'),
                                                           (((2.0*fv)/(4.0*fy)),[fv,fy],[fvv,fyv],'float32'),
                                                           (((2.0*dx)/(4.0*dv)),[dx,dv],[dxv,dvv],'float64'),
                                                           (((2.0*fx)/(4.0*fv)),[fx,fv],[fxv,fvv],'float32'),
                ]):
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
                                                           ((2*fx)/2,[fx],[fxv],'float32'),
                                                           ((2*dv)/2,[dv],[dvv],'float64'),
                                                           ((2*fv)/2,[fv],[fvv],'float32'),
                ]):
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
                                                           (fx/abs(fx),[fx],[0.5-fxv],'float32'),
                                                           (dx/abs(dx),[dx],[0.1*dxv],'float64'),
                                                           (fx/abs(fx),[fx],[0.1*fxv],'float32'),
                                                           (dv/abs(dv),[dv],[0.5-dvv],'float64'),
                                                           (fv/abs(fv),[fv],[0.5-fvv],'float32'),
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
                    ((2*fx)/(3*abs(fx)),[fx],[0.5-fxv],'float32'),
                    ((2*dx)/(3*abs(dx)),[dx],[0.1*dxv],'float64'),
                    ((2*fx)/(3*abs(fx)),[fx],[0.1*fxv],'float32'),
                    ((2*dv)/(3*abs(dv)),[dv],[0.5-dvv],'float64'),
                    ((2*fv)/(3*abs(fv)),[fv],[0.5-fvv],'float32'),
                ]):
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
        import theano.tensor, theano.compile
        raise SkipTest("Current implementation of Canonizer don't implement all case. Skip the corresponding test")

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
    assert 0 == function([i,d], d*(i/(i+1)))(3, 1.0)

def test_const_type_in_mul_canonizer():
    input = dmatrix()
    w = dmatrix()
    visb = dvector()
    hidb = dvector()
    betas = dvector()
    a = dvector()

    def sigm(x): return 1./(1+exp(-x))

    hid = sigm( (dot(w,input) + hidb) * betas )

    vis_gauss1 = (dot(w.T, hid) + visb) * betas / (2 * a * a)
    vis_gauss2 = (dot(w.T, hid) + visb) * betas / (2. * a * a)

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
            (fx+fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv*fzv,'float32'),
            (fx*fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv+fzv,'float32'),
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
            ((fw+fx)+(fy+fz)+2,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
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
            (fx+fy+cos(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.cos(fzv),'float32'),
            (fx+fy+cosh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.cosh(fzv),'float32'),
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
            (fx-theano.tensor.int_div(ix*100,iy*1000),(fx,ix,iy),(fxv,ixv,iyv),4,fxv-((ixv*100)//(iyv*1000)),'float64'),#int32 - float32 = float64 #No c_code for int_div#40
            (fx-(fy/2),(fx,fy),(fxv,fyv),1,fxv-(fyv/2),'float32'),
            (fx-(fy%fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv%fzv),'float32'),
            (fx-(fy>fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv>fzv),'float32'),
            (fx-(fy>=fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv>=fzv),'float32'),
            (fx-(fy<fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv<fzv),'float32'),#45
            (fx-(fy<=fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv<=fzv),'float32'),
            (fx-T.eq(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv==fzv),'float32'),
            (fx-T.neq(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv!=fzv),'float32'),
            (fx-fy+tan(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.tan(fzv),'float32'),
            (fx-fy+tanh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.tanh(fzv),'float32'),#50
            (fx-fy+sin(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sin(fzv),'float32'),
            (fx-fy+sinh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sinh(fzv),'float32'),
            (fx-fy+theano.tensor.sqr(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(fzv*fzv),'float32'),
            (fx-fy+theano.tensor.sqrt(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sqrt(fzv),'float32'),
            (fx-fy+theano.tensor.inv(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(1/fzv),'float32'),#55
            (fx-fy+theano.tensor.neg(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(-fzv),'float32'),
#            (fx-fy+theano.tensor.iround(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.round(fzv),'float32'),#TODO: trouble with the output type. To my understanding, numpy and c round fct return the same type as the input. Why we don't do this?

            #TODO: BIT OP only with ints, xor, or, and, invert, cast
#            (fx-theano.tensor.or_(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fy|fz),'float32'),
#            (fx-theano.tensor.xor(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fy^fz),'float32'),
            (theano.tensor.pow(fx*fy+fz,fx*fy),(fx,fy,fz),(fxv,fyv,fzv),1,numpy.power(fxv*fyv+fzv,fxv*fyv),'float32'),
            (fv+fy**fz,(fv,fy,fz),(fvv,fyv,fzv),2,fvv+fyv**fzv,'float32'),#fused with a dimshuffle
            (fv-fy+tanh(fz),(fv,fy,fz),(fvv,fyv,fzv),2,fvv-fyv+numpy.tanh(fzv),'float32'),#fused with a dimshuffle

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
            if gpu and (out_dtype!='float32' or any(i.dtype != 'float32' for i in g.owner.inputs)):
                print "Skip test %d as the gpu code currently support only float32" % id
                continue
            print "new cases", id

            if shared_fn == None:
                assert gpu==False
                f = compile.function(list(sym_inputs), g,mode=mode)
                for x in range(nb_repeat):
                    out=f(*val_inputs)
                t1=time.time()
            else:
                out=shared_fn(numpy.zeros(shp, dtype=out_dtype),'out')
                f = function(sym_inputs,[],updates=[(out, g)],mode=mode)
                t0=time.time()
                for x in range(nb_repeat):
                    f(*val_inputs)
                t1=time.time()
                out=out.value

            times[id]=t1-t0
            atol=1e-8
            if out_dtype=='float32':atol=1e-6
            if not numpy.allclose(out,answer*nb_repeat,atol=atol):
                fail1.append(id)
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
        f=theano.function([x,y,z],dot(x,y)+x+y+z,mode=mode)
        topo = f.maker.env.toposort()
        assert len(topo) == 2
        assert f.maker.env.toposort()[-1].op.inplace_pattern
        f(numpy.random.random((5,5)),numpy.random.random((5,5)),numpy.random.random((5,5)))

    def speed_fusion_gpu(self):
        import theano.sandbox.cuda as cuda
        self.speed_fusion(shared_fn=tcn.float32_shared_constructor, gpu=True, s=slice(0,15))

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
    f = function([x,y], T.log(fill(y,1)+(x)), mode=m)
    print f.maker.env.toposort()
    # the first three ops are Shape_i, Shape_i, and Dimshuffle
    theano.printing.debugprint(f)
    assert [node.op for node in f.maker.env.toposort()][3:] \
            == [T.log1p, alloc]
    f = function([x,y], T.log(0+(x) + fill(y,1.0)), mode=m)
    theano.printing.debugprint(f)
    assert [node.op for node in f.maker.env.toposort()][3:] \
            == [T.log1p, alloc]
    f = function([x,y], T.log(2+(x) - fill(y,1.0)), mode=m)
    theano.printing.debugprint(f)
    assert [node.op for node in f.maker.env.toposort()][3:] \
            == [T.log1p, alloc]

    f([1e-7, 10], [[0, 0], [0, 0]]) #debugmode will verify values

    if 0:
        # at one point this worked, but it has been broken since
        # the constant up-casting made 1 -> 1.0+0.0j
        # I was never sure if this optimization should work on complex numbers or not.
        z = zmatrix()
        f = function([z], T.log(1+(z)), mode=m)
        theano.printing.debugprint(f)
        assert [node.op for node in f.maker.env.toposort()] == [T.log1p]

    if 1:
        # should work for int
        z = imatrix()
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

class test_local_subtensor_lift(unittest.TestCase):

    def test0(self):
        # basic test that the Op works
        x = TT.matrix('x')
        f = function([x], TT.exp(x)[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, TT.Subtensor) #first subtensor
        assert prog[1].op == TT.exp
        assert len(prog)==2
        f([[0,1],[2,3]]) # let debugmode test something

    def test0b(self):
        # as test0, but we reuse the output of the elemwise
        # So we should not lift the subtensor
        x = TT.matrix('x')
        f = function([x], [TT.exp(x)[0], TT.exp(x)], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert prog[0].op == TT.exp
        assert isinstance(prog[1].op, TT.Subtensor) #first subtensor
        assert isinstance(prog[2].op, theano.compile.function_module.DeepCopyOp)
        assert len(prog)==3
        f([[0,1],[2,3]]) # let debugmode test something

    def test1(self):
        # basic test that the optimization work with scalar broadcasted
        x = TT.matrix('x')
        y = TT.scalar('y')
        z = TT.matrix('z')
        f = function([x,y,z], TT.exp(x+y+z)[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[1].op, TT.DimShuffle)
        assert isinstance(prog[0].op, TT.Subtensor) #first subtensor
        assert isinstance(prog[2].op, TT.Subtensor) #first subtensor
        assert isinstance(prog[3].op.scalar_op, theano.scalar.Composite)#Composite{add,add}
        assert len(prog)==4
        f([[0,1],[2,3]], 4, [[4,5],[6,7]]) # let debugmode test something

    def test2(self):
        # as 1, but take a slice
        x = TT.matrix('x')
        y = TT.scalar('y')
        z = TT.matrix('z')
        f = function([x,y,z], TT.exp(x+y+z)[0:2], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[1].op, TT.DimShuffle)
        assert isinstance(prog[0].op, TT.Subtensor) #first subtensor
        assert isinstance(prog[2].op, TT.Subtensor) #first subtensor
        assert isinstance(prog[3].op.scalar_op, theano.scalar.Composite)#Composite{add,add}
        assert len(prog)==4
        f([[0,1],[2,3]], 4, [[4,5],[6,7]]) # let debugmode test something

    def test3(self):
        # basic test that the optimization does work with broadcasting
        # for unary elemwise.
        y = TT.vector('y')
        f = function([y], TT.exp(y.dimshuffle(0,'x'))[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, TT.DimShuffle)
        assert isinstance(prog[1].op, TT.Subtensor)
        assert prog[2].op == TT.exp
        assert len(prog)==3
        f([4,5]) # let debugmode test something

    def test4(self):
        # basic test that the optimization doesn't work with broadcasting
        # ... It *could* be extended to,
        # ... but right now it doesn't, so it shouldn't try.
        x = TT.matrix('x')
        y = TT.vector('y')
        f = function([x,y], TT.exp(x+y)[0], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, TT.DimShuffle)
        assert prog[1].op == TT.add
        assert isinstance(prog[2].op, TT.Subtensor) #first subtensor
        assert prog[3].op == inplace.exp_inplace
        assert len(prog)==4
        f([[0,1],[2,3]], [4,5]) # let debugmode test something

    def test5(self):
        # test that we don't lift when we reuse the output of the
        # elemwise for other computation.
        x = TT.matrix('x')
        y = TT.vector('y')
        f = function([x,y], [TT.exp(x+y)[0],TT.exp(x+y)+x], mode=mode_opt)

        prog=f.maker.env.toposort()
        assert isinstance(prog[0].op, TT.DimShuffle)
        assert isinstance(prog[1].op.scalar_op, theano.scalar.Composite)#Composite{add,exp}
        assert prog[2].op == TT.add
        assert isinstance(prog[3].op, TT.Subtensor) #first subtensor
        assert len(prog)==4
        f([[0,1],[2,3]], [4,5]) # let debugmode test something

def test_local_fill_useless():
    m = theano.config.mode
    if m == 'FAST_COMPILE':
        m = 'FAST_RUN'

    x = dvector()
    y = dvector()
    z = lvector()

    # basic case
    f = function([x], T.fill(x,x)*2, mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]

    # basic case
    f = function([x,y], T.second(y,x)*2, mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]

    # now with different type
    f = function([x,z], T.fill(z,x)*2, mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]

    # now cutting out the input ??
    f = function([x,y], T.fill(x,y)*2, mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]

    # now filll is serving as a cast
    f = function([x,y], T.fill(x,y)*2, mode=m)
    assert [node.op for node in f.maker.env.toposort()] == [T.mul]

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
        print f.maker.env.toposort()
        assert [] == f.maker.env.toposort()

    def test_local_track_shape_i(self):
        class IdentityNoShape(Op):
            '''Op that does not infer the output shape from the input one'''
            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])
            def perform(self, node, (x,), (out,)):
                out[0] = x.copy()
            #def infer_shape(self, node, (xshp,)):
                #return [tuple([self.shape_i(i)(r) for i in xrange(r.ndim)])]
        identity_noshape = IdentityNoShape()

        class IdentityShape(Op):
            '''Op that does infer the output shape from the input one'''
            def make_node(self, x):
                x = as_tensor_variable(x)
                return Apply(self, [x], [x.type()])
            def perform(self, node, (x,), (out,)):
                out[0] = x.copy()
            def infer_shape(self, node, (xshp,)):
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
        register_specialize(local_identity_noshape_to_identity_shape)

        # With the optimization
        # The identity_shape op is should not be needed anymore to compute
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

class test_assert(unittest.TestCase):
    def test0(self):
        x=T.scalar()
        y=T.scalar()
        f = theano.function([x,y],theano.tensor.opt.assert_(x,T.eq(x,y)))
        f(1,1)
        self.failUnlessRaises(AssertionError, f, 1,0)

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
        self.failUnlessRaises(AssertionError, f, 1,0)
        topo=f.maker.env.toposort()
        assert len(topo)==2
        assert len(topo[0].inputs)==3
        assert topo[1].op==theano.compile.function_module.deep_copy_op

def test_local_mul_specialize():

    # test a few cases to make sure that the basics are covered
    #

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

    # test a few cases to make sure that the basics are covered
    #

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

def test_local_pow_specialize_device():

    # test that on cpu we use more agressive optimization

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


        f = theano.function([a],a.sum(0).sum(0).sum(0),mode=self.mode)
        assert len(f.maker.env.nodes)==1
        assert numpy.allclose(f(input),input.sum())

    def test_local_sum_sum(self):
        a=T.tensor3()
        input=numpy.arange(3*3*3, dtype=config.floatX).reshape(3,3,3)
        dims=[(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

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

    def test_local_sum_alloc(self):
        a=T.dtensor3()
        input=numpy.asarray(numpy.arange(2*3*4).reshape(2,3,4),dtype='float64')
        mode = self.mode.including('specialize').excluding('fusion')

        for t_like,n_like,nb_nodes in [(zeros_like,numpy.zeros_like,(0,3,3,2)),
                                       (ones_like,numpy.ones_like,(5,5,5,6))]:

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

            for d, dd in [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]:
                f = theano.function([a],t_like(a).sum(d).sum(dd),mode=mode)
                print f.maker.env.toposort()
                assert numpy.allclose(f(input),n_like(input).sum(d).sum(dd))
                assert len(f.maker.env.nodes)==nb_nodes[3]
                assert f.maker.env.toposort()[-1].op==T.alloc

class T_local_sum_dimshuffle(unittest.TestCase):
    def setUp(self):
        self.mode = theano.compile.get_default_mode().including('canonicalize')

    def test_local_sum_div_dimshuffle(self):
        a = T.matrix('a')
        b = T.vector('b')
        c = T.tensor3('c')
        d = T.scalar('d')
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

        for i,s in enumerate(sums):
            print i
            f = theano.function([a,b,c,d], s, mode=self.mode)
            theano.printing.debugprint(f)
            g = f.maker.env.toposort()
            #print 'g =', g
            assert isinstance(g[-1].op.scalar_op, theano.scalar.basic.TrueDiv)
            f(a_val, b_val, c_val, d_val)

    # TODO:
    # test_local_sum_prod_dimshuffle (a * b * c)
    # test_local_sum_divprod_dimshuffle ((a * b) / (c * d))

def test_make_vector():
    b = T.bscalar()
    i = T.iscalar()
    d = T.dscalar()

    assert opt.MakeVector(dtype="int8")(b,b).dtype=="int8"
    assert opt.MakeVector(dtype="int32")(i,b).dtype=="int32"
    assert opt.MakeVector(dtype="int32")(b,i).dtype=="int32"
    assert opt.MakeVector(dtype="float64")(b,i).dtype=="float64"
    assert opt.MakeVector(dtype="float64")(b,d).dtype=="float64"
    assert opt.MakeVector(dtype="float64")(d,i).dtype=="float64"
    assert opt.MakeVector(dtype="float64")().dtype=="float64"
    assert opt.MakeVector(dtype="int64")().dtype=="int64"

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
    a = TT.vector('a')
    s = stack(a)
    f = function([a], s, mode=mode_opt)
    val = f([1])
    assert numpy.all(val == [1])
    e = f.maker.env.toposort()
    assert len([n for n in e if isinstance(n.op, Join)]) == 0
    assert f.maker.env.outputs[0].dtype == config.floatX

    #test for matrix join(0,a)
    a = TT.matrix('a')
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

if __name__ == '__main__':
#    unittest.main()
    test_fusion().tes_memory_leak()

def test_local_mul_to_neg():
    """
    Test that a multiplication by -1 or -1.0 yields the appropriate data type
    """
    a = T.imatrix()
    f1 = theano.function([a], -1*a)
    f2 = theano.function([a], -1.0*a)
    aval = numpy.random.randint(0,10,(2,2)).astype('int32')
    assert f1(aval).dtype == a.dtype
    assert f2(aval).dtype == 'float64'

def test_local_add_specialize():

    # test of non-zero dimension
    a = TT.vector()
    s = TT.add(TT.zeros_like(a))
    assert local_add_specialize.transform(s.owner)

    # test of 0-d
    a = TT.scalar()
    s = TT.add(TT.zeros_like(a))
    assert local_add_specialize.transform(s.owner)
