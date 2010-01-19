from theano.tensor import *
import theano.config as config
from theano import function
#from theano.floatx import set_floatX, xscalar, xmatrix, xrow, xcol, xvector, xtensor3, xtensor4
import theano.floatX as FX

def test_floatX():
    def test():
        floatx=config.floatX
    #TODO test other fct then ?vector

    #float64 cast to float64 should not generate an op
        x = dvector('x')
        f = function([x],[cast(x,'float64')])
    #    print f.maker.env.toposort()
        assert len(f.maker.env.toposort())==0

    #float32 cast to float32 should not generate an op
        x = fvector('x')
        f = function([x],[cast(x,'float32')])
    #    print f.maker.env.toposort()
        assert len(f.maker.env.toposort())==0

    #floatX cast to float64
        x = FX.xvector('x')
        f = function([x],[cast(x,'float64')])
    #    print f.maker.env.toposort()
        if floatx=='float64':
            assert len(f.maker.env.toposort()) == 0 
        else:
            assert len(f.maker.env.toposort()) == 1

    #floatX cast to float32
        x = FX.xvector('x')
        f = function([x],[cast(x,'float32')])
    #    print f.maker.env.toposort()
        if floatx=='float32':
            assert len(f.maker.env.toposort()) == 0 
        else:
            assert len(f.maker.env.toposort()) == 1

    #float64 cast to floatX
        x = dvector('x')
        f = function([x],[cast(x,'floatX')])
    #    print f.maker.env.toposort()
        if floatx=='float64':
            assert len(f.maker.env.toposort()) == 0 
        else:
            assert len(f.maker.env.toposort()) == 1

    #float32 cast to floatX
        x = fvector('x')
        f = function([x],[cast(x,'floatX')])
    #    print f.maker.env.toposort()
        if floatx=='float32':
            assert len(f.maker.env.toposort()) == 0 
        else:
            assert len(f.maker.env.toposort()) == 1
            
    #floatX cast to floatX
        x = FX.xvector('x')
        f = function([x],[cast(x,'floatX')])
    #    print f.maker.env.toposort()
        assert len(f.maker.env.toposort()) == 0 

    orig_floatx = config.floatX
    try:
        print 'float32'
        FX.set_floatX('float32')
        test()
        print 'float64'
        FX.set_floatX('float64')
        test()
    finally:
        pass
        FX.set_floatX(orig_floatx)
