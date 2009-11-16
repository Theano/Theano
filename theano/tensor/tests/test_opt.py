## PENDING REWRITE OF tensor_opt.py


import unittest

from theano import gof
from theano.tensor.opt import *
from theano import tensor
from theano.tensor import TensorType
from theano.gof import Env
from theano.tensor.elemwise import DimShuffle
from theano import pprint
import numpy
#import scalar_opt

from theano import function, compile
from nose.plugins.skip import SkipTest
import unittest, copy
from copy import copy as cp

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
    segment_labels = label + numpy.asarray([0] * n_segments, dtype='int64')

    r = segment_labels * 5
    f = function([label], r)




# class _test_inplace_opt(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = inputs()
#         e = x + y + z
#         g = Env([x, y], [e])
#         self.failUnless(str(g) == "[Broadcast{Add}(Broadcast{Add}(x, y), z)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}{0: 0}(Broadcast{Add}{0: 0}(x, y), z)]")

#     def test_multiple_uses(self):
#         x, y, z = inputs()
#         e0 = x + y
#         e1 = x * y
#         g = Env([x, y], [e0, e1])
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}(x, y)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}{0: 0}(x, y), Broadcast{Mul}(x, y)]" \
#             or str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")

#     def test_user_inplace(self):
#         x, y, z = inputs()
#         e0 = x + y
#         e1 = tensor._mul_inplace(x, y)
#         g = Env([x, y], [e0, e1])
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, y)]")

#     def test_inplace_on_second_argument(self):
#         x, y, z = inputs()
#         e0 = x + y
#         e1 = tensor._mul_inplace(x, z)
#         g = Env([x, y], [e0, e1])
#         self.failUnless(str(g) == "[Broadcast{Add}(x, y), Broadcast{Mul}{0: 0}(x, z)]")
#         inplace_optimizer.optimize(g)
#         self.failUnless(str(g) == "[Broadcast{Add}{0: 1}(x, y), Broadcast{Mul}{0: 0}(x, z)]")





from theano.tensor import *

#from sandbox import pprint

class test_greedy_distribute(unittest.TestCase):
    def test_main(self):
        a, b, c, d, x, y, z = matrices('abcdxyz')
        e = (a/z + b/x) * x * z
        g = Env([a,b,c,d,x,y,z], [e])
        ##print pprint(g.outputs[0])
        mul_canonizer.optimize(g)
        gof.TopoOptimizer(gof.LocalOptGroup(local_fill_cut, local_fill_lift), order = 'out_to_in').optimize(g)
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

        r0 = f(4,1.e-6, [1.5,2], [2.3,3.1])
        r1 = f(4,1.e-6, [1.5,2], [2.3,3.1])
        r2 = f(4,1.e-6, [1.5,2], [2.3,3.1])

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
        gof.TopoOptimizer(gof.LocalOptGroup(local_fill_cut, local_fill_lift), order = 'out_to_in').optimize(g)
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
        fxv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        dxv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dyv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dzv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float64').reshape(1,shp[0])
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
        mode=compile.mode.predefined_modes[compile.mode.default_mode]
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
        fxv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        dxv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dyv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dzv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float64').reshape(1,shp[0])
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
        mode=compile.mode.predefined_modes[compile.mode.default_mode]
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
        fxv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fwv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        dxv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dyv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dzv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dwv = numpy.asarray(numpy.random.rand(*shp),dtype='float64')
        dvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float64').reshape(1,shp[0])

        #We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode=compile.mode.predefined_modes[compile.mode.default_mode]
        old_optimizer = mode._optimizer
        try:
            mode._optimizer=gof.Query(["canonicalize"])
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
                assert len(topo)==1
                assert isinstance(topo[0].op,(T.Elemwise,))
                assert isinstance(topo[0].op.scalar_op,theano.scalar.basic.Second)
                assert len(topo[0].inputs)==2
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
                assert numpy.allclose(out,val_inputs[1])
                topo=f.maker.env.toposort()
                assert len(topo)==nb_elemwise
                assert(out_dtype==out.dtype)

            #test x / y / x -> 1 / y
            for id,(g, sym_inputs, val_inputs, nb_elemwise, out_dtype) in enumerate([
                                                           ((dx/dy)/dx,[dx,dy],[dxv,dyv],1,'float64'),
                                                           ((fx/fy)/fx,[fx,fy],[fxv,fyv],1,'float32'),
                                                           ((dv/dy)/dv,[dv,dy],[dvv,dyv],1,'float64'),
                                                           ((fv/fy)/fv,[fv,fy],[fvv,fyv],1,'float32'),
                            #must broadcast as their is a dimshuffle in the computation

                                                           ((dx/dv)/dx,[dx,dv],[dxv,dvv],2,'float64'),
    #topo:            [Elemwise{inv,no_inplace}(<TensorType(float64, row)>), Elemwise{second,no_inplace}(x, Elemwise{inv,no_inplace}.0)]
                                                           ((fx/fv)/fx,[fx,fv],[fxv,fvv],2,'float32'),
                #topo:[Elemwise{inv,no_inplace}(<TensorType(float32, row)>), Elemwise{second,no_inplace}(x, Elemwise{inv,no_inplace}.0)]
                ]):
                f = compile.function(list(sym_inputs), g,
                                     mode=mode)
                out = f(*val_inputs)
                assert numpy.allclose(out,(1/val_inputs[1]))
                topo=f.maker.env.toposort()
                assert len(topo)==nb_elemwise
                assert isinstance(topo[0].op,(T.Elemwise,))
                assert isinstance(topo[0].op.scalar_op,(theano.scalar.basic.Inv, theano.scalar.basic.TrueDiv))
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
                assert len(topo)==0
                assert(out_dtype==out.dtype)
        finally:
            mode._optimizer = old_optimizer


    def test_multiple_case_that_fail(self):
        import theano.tensor, theano.compile
        raise SkipTest("Current implementation of Canonizer don't implement all case. Skip the corresponding test")

        shp=(4,4)
        fx, fy, fz = fmatrices('xyz')
        dx, dy, dz = dmatrices('xyz')
        fxv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fyv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fzv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        dxv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        dyv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        dzv = numpy.asarray(numpy.random.rand(*shp),dtype='float32')
        fvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        #We must be sure that the Canonizer is working, but that we don't have other
        # optimisation that could hide bug in the Canonizer as local_elemwise_fusion
        mode=compile.mode.predefined_modes[compile.mode.default_mode]
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

def test_mixeddiv():
    """Test that int division is preserved"""
    i = iscalar()
    d = dscalar()
    assert 0 == function([i,d], d*(i/(i+1)))(3, 1.0)

def test_local_shape_lift_dot():
    args_to_result = {
        (fvector, fvector): "[]",
        (fvector, fmatrix): "[<TensorType(float32, matrix)>.shape[1]]",
        (fmatrix, fvector): "[<TensorType(float32, matrix)>.shape[0]]",
        (fmatrix, fmatrix): "[<TensorType(float32, matrix)>.shape[0], <TensorType(float32, matrix)>.shape[1]]",
        }

    for x in [fvector, fmatrix]:
        for y in [fvector, fmatrix]:
            i = x()
            j = y()
            print 'I SHAPE', i.type.shape
            print 'J SHAPE', j.type.shape
            d = shape(dot(i,j))
            if x is fvector and y is fvector:
                assert d == ()
            else:
                g = Env([i,j], [d])
                gof.TopoOptimizer(gof.LocalOptGroup(local_shape_lift_dot), order='out_to_in').optimize(g)
                print pprint(g.outputs[0]), args_to_result[(x,y)]
                assert pprint(g.outputs[0]) == args_to_result[(x,y)]
        
#     def test_plusmin(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
# #        e = x - x
# #        e = (2.0 + x) - (2.0 + y)
# #        e = (2.0 + x) - (4.0 + y)
# #        e = x - (y - z)
# #        e = (x + y) - x
# #        e = (x - y) + (y - z) + (z - x)
# #        e = (a - b) + (b - c) + (c - d)
# #        e = x + -y
# #        e = a - b - b + a + b + c + b - c
# #        e = x + log(y) - x + y
#         e = 2.0 + x + 4.0
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         addfn = lambda *inputs: sum(inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
#         print g

#     def test_both(self):
#         x, y, z = inputs()
#         a, b, c, d = more_inputs()
#         e0 = (x * y / x)
#         e = e0 + e0 - e0
#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         gof.ConstantFinder().optimize(g)
#         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
#         divfn = lambda x, y: x / y
#         invfn = lambda x: 1 / x
#         Canonizer(Mul, Div, Inv, mulfn, divfn, invfn).optimize(g)
#         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(Add, Sub, Neg, addfn, subfn, negfn).optimize(g)
#         print g

#     def test_group_powers(self):
#         x, y, z, a, b, c, d = floats('xyzabcd')

###################
#         c1, c2 = constant(1.), constant(2.)
#         #e = pow(x, c1) * pow(x, y) / pow(x, 7.0) # <-- fucked
#         #f = -- moving from div(mul.out, pow.out) to pow(x, sub.out)
#         e = div(mul(pow(x, 2.0), pow(x, y)), pow(x, 7.0))

#         g = Env([x, y, z, a, b, c, d], [e])
#         print g
#         print g.inputs, g.outputs, g.orphans
#         f = sub(add(2.0, y), add(7.0))
#         g.replace(e, pow(x, f))
#         print g
#         print g.inputs, g.outputs, g.orphans
#         g.replace(f, sub(add(2.0, y), add(7.0))) # -- moving from sub(add.out, add.out) to sub(add.out, add.out)
#         print g
#         print g.inputs, g.outputs, g.orphans
###################

# #        e = x * exp(y) * exp(z)
# #        e = x * pow(x, y) * pow(x, z)
# #        e = pow(x, y) / pow(x, z)
#         e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0) # <-- fucked
# #        e = pow(x - x, y)
# #        e = pow(x, 2.0 + y - 7.0)
# #        e = pow(x, 2.0) * pow(x, y) / pow(x, 7.0) / pow(x, z)
# #        e = pow(x, 2.0 + y - 7.0 - z)
# #        e = x ** y / x ** y
# #        e = x ** y / x ** (y - 1.0)
# #        e = exp(x) * a * exp(y) / exp(z)
#         g = Env([x, y, z, a, b, c, d], [e])
#         g.extend(gof.PrintListener(g))
#         print g, g.orphans
#         mulfn = lambda *inputs: reduce(lambda x, y: x * y, (1,) + inputs)
#         divfn = lambda x, y: x / y
#         invfn = lambda x: 1 / x
#         Canonizer(mul, div, inv, mulfn, divfn, invfn, group_powers).optimize(g)
#         print g, g.orphans
#         addfn = lambda *inputs: reduce(lambda x, y: x + y, (0,) + inputs)
#         subfn = lambda x, y: x - y
#         negfn = lambda x: -x
#         Canonizer(add, sub, neg, addfn, subfn, negfn).optimize(g)
#         print g, g.orphans
#         pow2one_float.optimize(g)
#         pow2x_float.optimize(g)
#         print g, g.orphans
        







# class _test_cliques(unittest.TestCase):

#     def test_straightforward(self):
#         x, y, z = inputs()
#         m = y * z
#         d = tensor.dot(x, m)
#         d.name = 'd'
#         e = x + y + d
#         g = Env([x, y, z], [e])
#         cliques = find_cliques(g)
#         self.failUnless(len(cliques) == 2)
#         (i1, o1), (i2, o2) = cliques
#         self.failUnless(str(Env(i1, o1)) == "[Broadcast{Add}(Broadcast{Add}(x, y), d)]")
#         self.failUnless(str(Env(i2, o2)) == "[Broadcast{Mul}(y, z)]")
# #         print g
# #         for i, o in find_cliques(g):
# #             print "-->", Env(i, [o])

#     def test_broadcasting(self):
#         x, y, z = inputs([0]*1, [0]*2, [0]*3)
#         e = x + y + z
#         g = Env([x, y, z], [e])
#         lift_dimshuffle.optimize(g)
#         self.failUnless(len(find_cliques(g, through_broadcast = True)) == 1)
#         self.failUnless(len(find_cliques(g, through_broadcast = False)) == 2)
# #         print g
# #         for i, o in find_cliques(g, True):
# #             print "-->", Env(i, [o])


# # class _test_clique_opt(unittest.TestCase):

# #     def test_straightforward(self):
# #         x, y, z = inputs()
# #         e = x ** 2.0 #x * x
# #         g = Env([x], [e])
# #         gof.ConstantFinder().optimize(g)
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = scalar_opt.opt2,
# #                               make_composite = False)
# #         print g
# #         opt.optimize(g)
# #         print g

# #     def test_inplace(self):
# #         x, y, z = inputs()
# #         #e = tensor._add_inplace(x, y + z)
# #         e = x + tensor._add_inplace(y, z)
# #         g = Env([x, y, z], [e])
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = None,
# #                               make_composite = True)
# #         print g
# #         opt.optimize(g)
# #         print g
# # #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
# #         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))

# #     def test_straightforward(self):
# #         x, y, z = inputs()
# #         e = x + y + z
# #         g = Env([x, y, z], [e])
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = None,
# #                               make_composite = True)
# #         print g
# #         opt.optimize(g)
# #         print g
# # #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
# #         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))

# #     def test_straightforward2(self):
# #         x, y, z = inputs()
# #         m = y * z
# #         d = tensor.dot(x, m)
# #         d.name = 'd'
# #         e = x + y + d
# #         g = Env([x, y, z], [e])
# #         opt = CliqueOptimizer(through_broadcast = False,
# #                               scalar_optimizer = None,
# #                               make_composite = True)
# #         print g
# #         opt.optimize(g)
# #         print g
# # #        print g.outputs[0].owner.c_code(['x', 'y', 'z'], ['e'], dict(fail = "FAIL;", id = 0))
# #         print gof.OpWiseCLinker(g).make_function()(numpy.ones((5, 5)), numpy.ones((5, 5)), numpy.ones((5, 5)))


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
    
from theano.compile.sandbox.pfunc import pfunc
from theano.compile.sandbox.sharedvalue import shared
import theano

class test_fusion(unittest.TestCase):

    def do(self, mode, shared_fn, shp, gpu=False, nb_repeat=1, assert_len_topo=True, slice=None):
        """
        param shared_fn: if None, will use compile.function
        verify that the elemwise fusion work
        Test with and without DimShuffle
        """
        #TODO: disable the canonizer?
        def my_init(shp, dtype='float64', num=0):
            #ret = numpy.asarray(numpy.random.rand(*shp),dtype=dtype)
            ret = numpy.zeros(shp, dtype=dtype)+num
            return ret
        fw, fx, fy, fz = fmatrices('wxyz')
        dw, dx, dy, dz = dmatrices('wxyz')
        ix, iy, iz = imatrices('xyz')
        fv = fvector('r').dimshuffle('x',0)
        dv = dvector('s').dimshuffle('x',0)
        fwv = my_init(shp,'float32',1)
        fxv = my_init(shp,'float32',2)
        fyv = my_init(shp,'float32',3)
        fzv = my_init(shp,'float32',4)
        fvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        dwv = my_init(shp,'float64',5)
        ixv = numpy.asarray(my_init(shp,num=60),dtype='int32')
        iyv = numpy.asarray(my_init(shp,num=70),dtype='int32')
        izv = numpy.asarray(my_init(shp,num=70),dtype='int32')
#        dxv = my_init(shp,'float64',6)
#        dyv = my_init(shp,'float64',7)
#        dzv = my_init(shp,'float64',8)
#        dvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float64').reshape(1,shp[0])
        fwx=fw+fx
        cases = [
            (fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+fzv,'float32'),#1
            (fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv*fzv,'float32'),
            (fx+fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv*fzv,'float32'),
            (fx*fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv+fzv,'float32'),
            (fw+fx+fy+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),#5
            ((fw+fx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            (((fw+fx)+fy)+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            ((fw+(fx+fy))+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            ((fw+(fx+fy)+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            (fw+(fx+(fy+fz)),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),#10
            ((fw+fx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            (fw*fx*fy*fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv*fxv*fyv*fzv,'float32'),
            (fw+fx*fy*fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv*fyv*fzv,'float32'),
            (fx+fy*fz*fx,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv*fzv*fxv,'float32'),
            (fx*fy+fz+fy,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv+fzv+fyv,'float32'),#15
            (fx*fy*fz*fw+fx+fy+fz+fw,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fxv*fyv*fzv*fwv+fxv+fyv+fzv+fwv,'float32'),
            #test with constant
            ((fw+fx)+(fy+fz)+2,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            (((fw+fx)+2+fy)+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            ((fw+(fx+2+fy))+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            ((fw+(fx+fy)+2+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),#20
            (fw+(fx+(fy+fz)+2),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            (2+(fw+fx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv+2,'float32'),
            #mix float32 and float64
            (2+(dw+fx)+(fy+fz),(dw,fx,fy,fz),(dwv,fxv,fyv,fzv),1,dwv+fxv+fyv+fzv+2,'float64'),
            (2+(fw+dw)+(fy+fz),(fw,dw,fy,fz),(fwv,dwv,fyv,fzv),1,fwv+dwv+fyv+fzv+2,'float64'),
            (2+(fw+fx)+(dw+fz),(fw,fx,dw,fz),(fwv,fxv,dwv,fzv),1,fwv+fxv+dwv+fzv+2,'float64'),#25
            (2+(fw+fx)+(fy+dw),(fw,fx,fy,dw),(fwv,fxv,fyv,dwv),1,fwv+fxv+fyv+dwv+2,'float64'),
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
            (fx+fy+abs(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.absolute(fzv),'float32'),#30
            (ix+iy+abs(iz),(ix,iy,iz),(ixv,iyv,izv),1,ixv+iyv+numpy.absolute(izv),'int32'),
            (fx+fy+theano.tensor.log(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.log(fzv),'float32'),
            (fx+fy+theano.tensor.log2(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.log2(fzv),'float32'),
            (fx+fy+theano.tensor.log10(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.log10(fzv),'float32'),
            (fx+fy**fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv**fzv,'float32'),#pow #35
            (fx+fy+theano.tensor.exp(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+numpy.exp(fzv),'float32'),
            (fx-fy-fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv-fzv,'float32'),
            (fx-(fy/fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv/fzv),'float32'),
            (fx-theano.tensor.true_div(fy,2),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv/2),'float32'),
            (fx-theano.tensor.true_div(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv/fzv),'float32'),#40
            (fx-theano.tensor.int_div(ix*100,iy*1000),(fx,ix,iy),(fxv,ixv,iyv),4,fxv-((ixv*100)//(iyv*1000)),'float64'),#int32 - float32 = float64 #No c_code for int_div
            (fx-(fy/2),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv/2),'float32'),
            (fx-(fy%fz),(fx,fy,fz),(fxv,fyv,fzv),2,fxv-(fyv%fzv),'float32'),
            (fx-(fy>fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv>fzv),'float32'),
            (fx-(fy>=fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv>=fzv),'float32'),
            (fx-(fy<fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv<fzv),'float32'),
            (fx-(fy<=fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv<=fzv),'float32'),#TODO: bugged on the gpu
            (fx-(fy==fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv==fzv),'float32'),#TODO: bugged
            (fx-(fy!=fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fyv!=fzv),'float32'),
            (fx-fy+tan(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.tan(fzv),'float32'),
            (fx-fy+tanh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.tanh(fzv),'float32'),
            (fx-fy+sin(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sin(fzv),'float32'),
            (fx-fy+sinh(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sinh(fzv),'float32'),
            (fx-fy+theano.tensor.sqr(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(fzv*fzv),'float32'),
            (fx-fy+theano.tensor.sqrt(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.sqrt(fzv),'float32'),
            (fx-fy+theano.tensor.inv(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(1/fzv),'float32'),
            (fx-fy+theano.tensor.neg(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+(-fzv),'float32'),
#            (fx-fy+theano.tensor.iround(fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-fyv+numpy.round(fzv),'float32'),#TODO: trouble with the output type. To my understanding, numpy and c round fct return the same type as the input. Why we don't do this?

            #TODO: BIT OP only with ints, xor, or, and, invert, cast
#            (fx-theano.tensor.or_(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fy|fz),'float32'),
#            (fx-theano.tensor.xor(fy,fz),(fx,fy,fz),(fxv,fyv,fzv),1,fxv-(fy^fz),'float32'),

            ]
        if slice:
            cases = cases[slice]
        import time
        times=numpy.zeros(len(cases))
        for id, [g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype] in enumerate(cases):
            print "new cases", id

            if shared_fn == None:
                assert gpu==False
                f = compile.function(list(sym_inputs), g,mode=mode)
                #pre-call to have the data in cache if it fit to don't penalise the first iteration
#                if id==0:
#                    out=f(*val_inputs)
                t0=time.time()
                for x in range(nb_repeat):
                    out=f(*val_inputs)
                t1=time.time()
                nb_repeat=1
            else:
                out=shared_fn(numpy.zeros(shp, dtype=out_dtype),'out')
                f = pfunc(sym_inputs,[],updates=[(out,out+g)],mode=mode)
                #pre-call to have the data in cache if it fit to don't penalise the first iteration
#                if id==0:
#                    f(*val_inputs)
                t0=time.time()
                for x in range(nb_repeat):
                    f(*val_inputs)
                t1=time.time()
                out=out.value
#                if id==0:
#                    nb_repeat+=1

            times[id]=t1-t0
            assert numpy.allclose(out,answer*nb_repeat,atol=1e-6 if out_dtype=='float32' else 1e-8)
            topo=f.maker.env.toposort()
            if gpu:
                import theano_cuda_ndarray as tcn

                topo_ = [x for x in topo if not isinstance(x.op,tcn.basic_ops.GpuFromHost)]
                gpu_ = [x for x in topo if isinstance(x.op,tcn.basic_ops.GpuFromHost)]
                assert len(gpu_)==len(sym_inputs)
            else: topo_=topo
            if assert_len_topo:
                assert(len(topo_)==nb_elemwise)
            assert(out_dtype==out.dtype)
        print "Executed",len(cases),"cases"
        return times
    
    def test_elemwise_fusion(self):
        shp=(5,5)
        #we need the optimisation enabled, debug do this.
        mode=cp(compile.mode.get_default_mode())
        mode._optimizer=mode._optimizer.including('local_elemwise_fusion')
        self.do(mode, shared, shp)

    def gpu_fusion(self):
        shp=(5,5)
        #we need the optimisation enabled, debug do this.
        mode=compile.mode.predefined_modes['FAST_COMPILE']
        mode=compile.mode.predefined_modes['FAST_RUN']
        mode=compile.mode.predefined_modes['DEBUG_MODE']
        import theano_cuda_ndarray as tcn

        self.do(mode, tcn.shared_constructor, shp, gpu=True)

    def speed_fusion(self, shared_fn = shared, gpu = False, s=None):
        """
        param type s: a slice object
        param s: a slice to apply to the case to execute. If None, exec all case.
        """
        
        import copy
        shp=(3000,3000)
        #mode1=copy.copy(compile.mode.predefined_modes['FAST_RUN'])
        linker=gof.CLinker
        linker=gof.OpWiseCLinker
        mode1=compile.Mode(linker(), copy.copy(compile.mode.OPT_FAST_RUN))
        #TODO:clinker is much faster... but use to much memory
        #Possible cause: as their is do deletion of intermediate value when we don't keep the fct.
        #More plausible cause: we keep a link to the output data?
        #Follow up. Clinker do the same... second cause?
        mode2=compile.Mode(linker(), copy.copy(compile.mode.OPT_FAST_RUN))
#        mode2=copy.copy(compile.mode.predefined_modes['FAST_RUN'])
        old_optimizer = mode2._optimizer
        try:
            mode2._optimizer=mode2._optimizer.excluding('local_elemwise_fusion')
    #        mode2=compile.Mode(gof.OpWiseCLinker(allow_gc=True), compile.mode.OPT_FAST_COMPILE)

            if s is None:
                s=slice(0,49)
                #s=slice(49,59)
            nb_repeat=10
            print "test with linker", str(linker)
            times1=self.do(mode1, shared_fn, shp, gpu=gpu, nb_repeat=nb_repeat, assert_len_topo=False,slice=s)
            times2=self.do(mode2, shared_fn, shp, gpu=gpu, nb_repeat=nb_repeat, assert_len_topo=False,slice=s)
            print "times1 FAST_RUN optimisation"
            print times1, times1.min(), times1.max(), times1.sum()
            print "times2 FAST_RUN optimisation without local_elemwise_fusion"
            print times2, times2.min(), times2.max(), times2.sum()
            d=times2/times1
    #        d.sort()
            print "times2/times1",d
            print "min", d.min(), "argmin", d.argmin(), "max", d.max(), "mean", d.mean(), "std", d.std()
        finally:
            mode2._optimizer = old_optimizer

    def speed_fusion_gpu(self):
        import theano_cuda_ndarray as tcn
        self.speed_fusion(shared_fn=tcn.shared_constructor, gpu=True, s=slice(0,15))
        
    def speed_log_exp(self):
        s=slice(31,36)
#        linker=gof.CLinker
        linker=gof.OpWiseCLinker
        mode=compile.Mode(linker(), cp(compile.mode.OPT_FAST_RUN))
        mode=compile.ProfileMode()
        print "time", self.do(mode, shared, shp=(1000,1000),gpu=False, assert_len_topo=False,slice=s, nb_repeat=100)


    def test_memory_leak(self, mode=compile.mode.predefined_modes['FAST_RUN'], shared_fn=shared, shp=(3000,3000), gpu=False, nb_repeat=1, assert_len_topo=True, slice=None):
        """
        param shared_fn: if None, will use compile.function
        verify that the elemwise fusion work
        Test with and without DimShuffle
        """
        #TODO: disable the canonizer?
        def my_init(shp, dtype='float64', num=0):
            #ret = numpy.asarray(numpy.random.rand(*shp),dtype=dtype)
            ret = numpy.zeros(shp, dtype=dtype)+num
            return ret
        fw, fx, fy, fz = fmatrices('wxyz')
        dw, dx, dy, dz = dmatrices('wxyz')
        ix, iy, iz = imatrices('xyz')
       fv = fvector('r').dimshuffle('x',0)
        fwv = my_init(shp,'float32',1)
        fxv = my_init(shp,'float32',2)
        fyv = my_init(shp,'float32',3)
        fzv = my_init(shp,'float32',4)
        fvv = numpy.asarray(numpy.random.rand(shp[0]),dtype='float32').reshape(1,shp[0])
        ixv = numpy.asarray(my_init(shp,num=60),dtype='int32')
        iyv = numpy.asarray(my_init(shp,num=70),dtype='int32')
        izv = numpy.asarray(my_init(shp,num=70),dtype='int32')
        fwx=fw+fx
        cases = [
            (fx+fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv+fzv,'float32'),#1
            (fx*fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv*fzv,'float32'),
            (fx+fy*fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv+fyv*fzv,'float32'),
            (fx*fy+fz,(fx,fy,fz),(fxv,fyv,fzv),1,fxv*fyv+fzv,'float32'),
            (fw+fx+fy+fz,(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),#5
            ((fw+fx)+(fy+fz),(fw,fx,fy,fz),(fwv,fxv,fyv,fzv),1,fwv+fxv+fyv+fzv,'float32'),
            ]
        import gc, pdb, objgraph
        for id, [g, sym_inputs, val_inputs, nb_elemwise, answer, out_dtype] in enumerate(cases):
            if gpu and out_dtype!='float32':
                print "Skip test %d as the gpu code currently support only float32" % id
                continue
            print "new cases", id

            out=shared_fn(numpy.zeros(shp, dtype=out_dtype),'out')
            f = pfunc(sym_inputs,[],updates=[(out,out+g)],mode=mode)

            gc.collect();gc.collect();gc.collect()
            pdb.set_trace()

            for x in range(nb_repeat):
                gc.collect();gc.collect();gc.collect()
                pdb.set_trace()
                objgraph.show__most_common_types(limit=40)
                f(*val_inputs)

#            cases[id]=None #to remove g, that link to out that link to the ndarray!
            #g.owner.inputs[0] is out... make owner a weakref?
            
        

if __name__ == '__main__':
    unittest.main()




