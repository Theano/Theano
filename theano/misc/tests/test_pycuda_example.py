from __future__ import absolute_import, print_function, division
import numpy as np

import theano
import theano.misc.pycuda_init

if not theano.misc.pycuda_init.pycuda_available:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest("Pycuda not installed. Skip test of theano op"
                   " with pycuda code.")

import theano.sandbox.cuda as cuda_ndarray
if not cuda_ndarray.cuda_available:  # noqa
    from nose.plugins.skip import SkipTest
    raise SkipTest('Optional package cuda disabled')

import theano.tensor as T
from theano.misc.pycuda_example import (PycudaElemwiseSourceModuleOp,
                                        PycudaElemwiseSourceModuleMakeThunkOp)

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')


def test_pycuda_elemwise_source_module():
    for shape in [(5, 5), (10, 49), (50, 49), (500, 501)]:
        for op in [theano.scalar.basic.mul, theano.scalar.basic.add]:
            x = T.fmatrix('x')
            y = T.fmatrix('y')
            elemwise_op = theano.tensor.Elemwise(op)
            pycuda_op = PycudaElemwiseSourceModuleOp(op)
            pycuda_op_thunk = PycudaElemwiseSourceModuleMakeThunkOp(op)
            f = theano.function([x, y], elemwise_op(x, y), mode=mode_with_gpu)
            f2 = theano.function([x, y],
                                 theano.sandbox.cuda.host_from_gpu(
                                     pycuda_op(x, y)),
                                 mode=mode_with_gpu)
            mode_pycuda = mode_with_gpu.including("local_pycuda_gpu_elemwise")
            f3 = theano.function([x, y], elemwise_op(x, y),
                                 mode=mode_pycuda)
            f4 = theano.function([x, y],
                                 theano.sandbox.cuda.host_from_gpu(
                                     pycuda_op_thunk(x, y)),
                                 mode=mode_with_gpu)

            assert any([isinstance(node.op, theano.sandbox.cuda.GpuElemwise)
                        for node in f.maker.fgraph.toposort()])
            assert any([isinstance(node.op, PycudaElemwiseSourceModuleOp)
                        for node in f2.maker.fgraph.toposort()])
            assert any([isinstance(node.op, PycudaElemwiseSourceModuleOp)
                        for node in f3.maker.fgraph.toposort()])
            assert any([isinstance(node.op,
                                   PycudaElemwiseSourceModuleMakeThunkOp)
                        for node in f4.maker.fgraph.toposort()])

            val1 = np.asarray(np.random.rand(*shape), dtype='float32')
            val2 = np.asarray(np.random.rand(*shape), dtype='float32')
            assert np.allclose(f(val1, val2), f2(val1, val2))
            assert np.allclose(f(val1, val2), f3(val1, val2))
            assert np.allclose(f(val1, val2), f4(val1, val2))
            # print f(val1,val2)
            # print f2(val1,val2)

"""
#commented as it work only with old pycuda version.
def test_pycuda_elemwise_kernel():
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    f = theano.function([x, y], x + y, mode=mode_with_gpu)
    print(f.maker.fgraph.toposort())
    mode_pycuda = mode_with_gpu.including("local_pycuda_gpu_elemwise_kernel")
    f2 = theano.function([x, y], x + y, mode=mode_pycuda)
    print(f2.maker.fgraph.toposort())

    assert any([isinstance(node.op, theano.sandbox.cuda.GpuElemwise)
                for node in f.maker.fgraph.toposort()])
    assert any([isinstance(node.op, PycudaElemwiseKernelOp)
                for node in f2.maker.fgraph.toposort()])

    val1 = np.asarray(np.random.rand(5, 5), dtype='float32')
    val2 = np.asarray(np.random.rand(5, 5), dtype='float32')
    #val1 = np.ones((5,5))
    #val2 = np.arange(25).reshape(5,5)
    assert (f(val1, val2) == f2(val1, val2)).all()
    print(f(val1, val2))
    print(f2(val1, val2))

    x3 = T.ftensor3('x')
    y3 = T.ftensor3('y')
    z3 = T.ftensor3('y')

    f4 = theano.function([x3, y3, z3], x3 * y3 + z3, mode=mode_pycuda)
    print(f4.maker.fgraph.toposort())
    assert any([isinstance(node.op, PycudaElemwiseKernelOp)
                for node in f4.maker.fgraph.toposort()])

    val1 = np.random.rand(2, 2, 2)
    print(val1)
    print(f4(val1, val1, val1))
    assert np.allclose(f4(val1, val1, val1), val1 * val1 + val1)
"""
