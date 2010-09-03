import numpy

try:
    import pycuda
except ImportError:
    from nose.plugins.skip import SkipTest
    raise SkipTest("Pycuda not installed. Skip test of theano op with pycuda code.")

import theano
import theano.tensor as T
from theano.misc.pycuda_example import PycudaElemwiseSourceModuleOp, PycudaElemwiseKernelOp
from theano.sandbox.cuda import GpuContiguous
import theano.misc.pycuda_example

def test_pycuda_elemwise_source_module():
    x=T.fmatrix('x')
    y=T.fmatrix('y')
    f=theano.function([x,y],x*y)
    print f.maker.env.toposort()
    f2 = theano.function([x,y],x*y, mode=theano.compile.mode.get_default_mode().including("local_pycuda_gpu_elemwise"))
    print f2.maker.env.toposort()

    assert any([ isinstance(node.op, theano.sandbox.cuda.GpuElemwise) for node in f.maker.env.toposort()])
    assert any([ isinstance(node.op, PycudaElemwiseSourceModuleOp) for node in f2.maker.env.toposort()])
    
    val1 = numpy.random.rand(5,5)
    val2 = numpy.random.rand(5,5)
    #val1 = numpy.ones((5,5))
    #val2 = numpy.arange(25).reshape(5,5)
    assert (f(val1,val2) == f2(val1,val2)).all()
    print f(val1,val2)
    print f2(val1,val2)

def test_pycuda_elemwise_kernel():
    x=T.fmatrix('x')
    y=T.fmatrix('y')
    f=theano.function([x,y],x+y)
    print f.maker.env.toposort()
    f2 = theano.function([x,y],x+y, mode=theano.compile.mode.get_default_mode().including("local_pycuda_gpu_elemwise_kernel"))
    print f2.maker.env.toposort()

    assert any([ isinstance(node.op, theano.sandbox.cuda.GpuElemwise) for node in f.maker.env.toposort()])
    assert any([ isinstance(node.op, PycudaElemwiseKernelOp) for node in f2.maker.env.toposort()])
    
    val1 = numpy.random.rand(5,5)
    val2 = numpy.random.rand(5,5)
    #val1 = numpy.ones((5,5))
    #val2 = numpy.arange(25).reshape(5,5)
    assert (f(val1,val2) == f2(val1,val2)).all()
    print f(val1,val2)
    print f2(val1,val2)


    x3=T.ftensor3('x')
    y3=T.ftensor3('y')
    z3=T.ftensor3('y')

    f4 = theano.function([x3,y3,z3],x3*y3+z3, mode=theano.compile.mode.get_default_mode().including("local_pycuda_gpu_elemwise_kernel"))
    print f4.maker.env.toposort()
    assert any([ isinstance(node.op, PycudaElemwiseKernelOp) for node in f4.maker.env.toposort()])

    val1 = numpy.random.rand(2,2,2)
    print val1
    print f4(val1,val1,val1)
    assert numpy.allclose(f4(val1,val1,val1),val1*val1+val1)
