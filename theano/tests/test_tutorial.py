""" test code snippet in the Theano tutorials.
"""
from __future__ import print_function

import os
import shutil
import unittest

from nose.plugins.attrib import attr
from nose.plugins.skip import SkipTest
import numpy
from numpy import array

import theano
import theano.tensor as T
from theano import function, compat

from six.moves import xrange
from theano import config
from theano.tests import unittest_tools as utt
from theano.sandbox.rng_mrg import MRG_RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams


class T_using_gpu(unittest.TestCase):
    # All tests here belog to
    # http://deeplearning.net/software/theano/tutorial/using_gpu.html
    # Theano/doc/tutorial/using_gpu.txt
    # Any change you do here also add it to the tutorial !

    def test_using_gpu_1(self):
        # I'm checking if this compiles and runs
        from theano import function, config, shared, sandbox
        import theano.tensor as T
        import numpy
        import time

        vlen = 10 * 30 * 70  # 10 x #cores x # threads per core
        iters = 10

        rng = numpy.random.RandomState(22)
        x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
        f = function([], T.exp(x))
        # print f.maker.fgraph.toposort()
        t0 = time.time()
        for i in xrange(iters):
            r = f()
        t1 = time.time()
        print('Looping %d times took' % iters, t1 - t0, 'seconds')
        print('Result is', r)
        if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
            print('Used the cpu')
        else:
            print('Used the gpu')
        if theano.config.device.find('gpu') > -1:
            assert not numpy.any( [isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()])
        else:
            assert numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()])

    def test_using_gpu_2(self):
        if theano.config.device.find('gpu') > -1:

            from theano import function, config, shared, sandbox
            import theano.tensor as T
            import numpy
            import time

            vlen = 10 * 30 * 70  # 10 x #cores x # threads per core
            iters = 10

            rng = numpy.random.RandomState(22)
            x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
            f = function([], sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)))
            # print f.maker.fgraph.toposort()
            t0 = time.time()
            for i in xrange(iters):
                r = f()
            t1 = time.time()
            print('Looping %d times took' % iters, t1 - t0, 'seconds')
            print('Result is', r)
            print('Numpy result is', numpy.asarray(r))
            if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
                print('Used the cpu')
            else:
                print('Used the gpu')

            assert not numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()])

    def test_using_gpu_3(self):

        if theano.config.device.find('gpu') > -1:

            from theano import function, config, shared, sandbox, Out
            import theano.tensor as T
            import numpy
            import time

            vlen = 10 * 30 * 70  # 10 x #cores x # threads per core
            iters = 10

            rng = numpy.random.RandomState(22)
            x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
            f = function([],
                    Out(sandbox.cuda.basic_ops.gpu_from_host(T.exp(x)),
                        borrow=True))
            # print f.maker.fgraph.toposort()
            t0 = time.time()
            for i in xrange(iters):
                r = f()
            t1 = time.time()
            print('Looping %d times took' % iters, t1 - t0, 'seconds')
            print('Result is', r)
            print('Numpy result is', numpy.asarray(r))
            if numpy.any([isinstance(x.op, T.Elemwise)
                          for x in f.maker.fgraph.toposort()]):
                print('Used the cpu')
            else:
                print('Used the gpu')

            assert not numpy.any([isinstance(x.op, T.Elemwise)
                                  for x in f.maker.fgraph.toposort()])

    def test_using_gpu_pycudaop(self):
        import theano.misc.pycuda_init
        if not theano.misc.pycuda_init.pycuda_available:
            raise SkipTest("Pycuda not installed. Skip test of theano op"
                           " with pycuda code.")
        from pycuda.compiler import SourceModule
        import theano.sandbox.cuda as cuda

        import theano.sandbox.cuda as cuda_ndarray
        if not cuda_ndarray.cuda_available:
            raise SkipTest('Optional package cuda disabled')

        class PyCUDADoubleOp(theano.Op):
            
            __props__ = ()

            def make_node(self, inp):
                inp = cuda.basic_ops.gpu_contiguous(
                    cuda.basic_ops.as_cuda_ndarray_variable(inp))
                assert inp.dtype == "float32"
                return theano.Apply(self, [inp], [inp.type()])

            def make_thunk(self, node, storage_map, _, _2):
                mod = SourceModule("""
    __global__ void my_fct(float * i0, float * o0, int size) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<size){
        o0[i] = i0[i]*2;
    }
  }""")
                pycuda_fct = mod.get_function("my_fct")
                inputs = [storage_map[v] for v in node.inputs]
                outputs = [storage_map[v] for v in node.outputs]

                def thunk():
                    z = outputs[0]
                    if z[0] is None or z[0].shape != inputs[0][0].shape:
                        z[0] = cuda.CudaNdarray.zeros(inputs[0][0].shape)
                        grid = (int(numpy.ceil(inputs[0][0].size / 512.)), 1)
                        pycuda_fct(inputs[0][0], z[0],
                                   numpy.intc(inputs[0][0].size),
                                   block=(512, 1, 1), grid=grid)
                return thunk
        x = theano.tensor.fmatrix()
        f = theano.function([x], PyCUDADoubleOp()(x))
        xv = numpy.ones((4, 5), dtype="float32")
        assert numpy.allclose(f(xv), xv*2)


class T_typedlist(unittest.TestCase):
    # All tests here belong to
    # http://deeplearning.net/software/theano/library/typed_list.html
    # Theano/doc/library/typed_list.txt
    # Any change you do here must also be done in the documentation !

    def test_typedlist_basic(self):
        import theano.typed_list

        tl = theano.typed_list.TypedListType(theano.tensor.fvector)()
        v = theano.tensor.fvector()
        o = theano.typed_list.append(tl, v)
        f = theano.function([tl, v], o)
        output = f([[1, 2, 3], [4, 5]], [2])

        # Validate ouput is as expected
        expected_output = [numpy.array([1, 2, 3], dtype="float32"),
                           numpy.array([4, 5], dtype="float32"),
                           numpy.array([2], dtype="float32")]

        assert len(output) == len(expected_output)
        for i in range(len(output)):
            utt.assert_allclose(output[i], expected_output[i])

    def test_typedlist_with_scan(self):
        import theano.typed_list

        a = theano.typed_list.TypedListType(theano.tensor.fvector)()
        l = theano.typed_list.length(a)
        s, _ = theano.scan(fn=lambda i, tl: tl[i].sum(),
                        non_sequences=[a],
                        sequences=[theano.tensor.arange(l, dtype='int64')])

        f = theano.function([a], s)
        output = f([[1, 2, 3], [4, 5]])

        # Validate ouput is as expected
        expected_output = numpy.array([6, 9], dtype="float32")
        utt.assert_allclose(output, expected_output)
