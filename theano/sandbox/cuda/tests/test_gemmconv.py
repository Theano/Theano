import unittest
import numpy

import theano
from theano.tests import unittest_tools as utt

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if not cuda_ndarray.cuda_available:
    raise SkipTest('Optional package cuda not available')
from theano.sandbox.cuda import float32_shared_constructor as shared
from  theano.sandbox.cuda.blas import GpuCorr3DMM

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


class TestCorr3DMM(unittest.TestCase):

    def run_conv_valid(self, inputs_shape, filters_shape):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        bias = shared(numpy.zeros(filters_shape[0]).astype('float32'))
        conv_ref = theano.tensor.nnet.conv3D(V=inputs, W=filters,
                                             b=bias, d=(1,1,1))
        conv = GpuCorr3DMM(border_mode = "valid")(inputs.dimshuffle(0, 4, 1, 2, 3),
                                                  filters.dimshuffle(0, 4, 1, 2, 3))
        conv = conv.dimshuffle(0, 2, 3, 4, 1)

        f_ref = theano.function([], conv_ref)
        f = theano.function([], conv, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res,  rtol=1e-05, atol=1e-05)



    def run_conv_full(self, inputs_shape, filters_shape):
        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        bias = shared(numpy.zeros(filters_shape[4]).astype('float32'))

        conv_ref = theano.tensor.nnet.convTransp3D(W=filters, b=bias, d=(1,1,1),
                                                   H=inputs)
        filters = filters.dimshuffle(4, 0, 1, 2, 3)
        inputs = inputs.dimshuffle(0, 4, 1, 2, 3)
        filters = filters[:,:,::-1,::-1,::-1]
        conv = GpuCorr3DMM(border_mode = "full")(inputs, filters)
        conv = conv.dimshuffle(0, 2, 3, 4, 1)

        f_ref = theano.function([], conv_ref)
        f = theano.function([], conv, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res,  rtol=1e-04, atol=1e-04)


    def test_valid(self):
        self.run_conv_valid(inputs_shape=(16, 20, 32, 16, 1),
                            filters_shape=(10, 6, 12, 4, 1))
        self.run_conv_valid(inputs_shape=(16, 20, 32, 15, 1),
                            filters_shape=(10, 6, 12, 4, 1))
    def test_full(self):
        self.run_conv_full(inputs_shape=(16, 15, 21, 16, 10),
                           filters_shape=(10, 6, 12, 4, 1))
        self.run_conv_full(inputs_shape=(16, 15, 21, 12, 10),
                           filters_shape=(10, 6, 12, 4, 1))

    def test_opt_conv3d_gemm(self):
        inputs_shape = (16, 20, 32, 16, 1)
        filters_shape = (10, 6, 12, 4, 1)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)
        bias = shared(numpy.zeros(filters_shape[0]).astype('float32'))

        conv = theano.tensor.nnet.conv3D(V=inputs, W=filters,
                                         b=bias, d=(1,1,1))
        mode = mode_with_gpu.including('conv3d_gemm')

        f_ref = theano.function([], conv)
        f = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f.maker.fgraph.toposort()
        #print sum(isinstance(n.op, theano.sandbox.cuda.blas.GpuCorr3DMM) for n in topo)

        assert sum(isinstance(n.op, theano.sandbox.cuda.blas.GpuCorr3DMM) for n in topo) > 0
        res_ref = f()
        res = f()
        utt.assert_allclose(res_ref, res)

    def test_opt_convgrad3d_fft(self):
        inputs_shape = (16, 20, 32, 16, 1)
        filters_shape = (10, 6, 12, 4, 1)
        dCdH_shape = (16, 15, 21, 13, 10)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        dCdH_val = numpy.random.random(dCdH_shape).astype('float32')

        inputs = shared(inputs_val)
        dCdH = shared(dCdH_val)

        conv = theano.tensor.nnet.convGrad3D(V=inputs, dCdH=dCdH,
                                             WShape=filters_shape,
                                             d=(1,1,1))
        mode = mode_with_gpu.including('convgrad3d_gemm')

        f_ref = theano.function([], conv)
        f = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.blas.GpuCorr3DMM) for n in topo) > 0


        res_ref = f_ref()
        res = f()

        utt.assert_allclose(res_ref, res,  rtol=1e-04, atol=1e-04)


    def test_opt_convtransp3d_gemm(self):
        inputs_shape = (16, 15, 21, 12, 10)
        filters_shape = (10, 6, 12, 4, 1)

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')
        bias = shared(numpy.zeros(filters_shape[4]).astype('float32'))

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        conv = theano.tensor.nnet.convTransp3D(W=filters, b=bias, d=(1,1,1),
                                               H=inputs)
        mode = mode_with_gpu.including('convtransp3d_gemm')

        f_ref = theano.function([], conv)
        f = theano.function([], conv, mode=mode)

        # make sure we inserted the fft trickery
        topo = f.maker.fgraph.toposort()
        assert sum(isinstance(n.op, theano.sandbox.cuda.blas.GpuCorr3DMM) for n in topo) > 0



        res_ref = f_ref()
        res = f()

        utt.assert_allclose(res_ref, res, rtol=1e-04, atol=1e-04)

