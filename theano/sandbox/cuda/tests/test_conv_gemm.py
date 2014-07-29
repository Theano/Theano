"""
Tests for Caffe GPU convolution
"""
import sys
import time
import unittest


import numpy

from nose.plugins.skip import SkipTest
imported_scipy_convolve2d = False
try:
    from scipy.signal import correlate
    imported_scipy_convolve2d = True
except ImportError:
    pass


import theano
from theano import tensor
from theano.gof.python25 import any
from theano.tests.unittest_tools import seed_rng

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

#needed as the gpu conv don't have a perform implementation.
if theano.config.mode == 'FAST_COMPILE':
    theano_mode = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    theano_mode = theano.compile.mode.get_default_mode().including('gpu')

cuda_tensor4 = cuda_ndarray.CudaNdarrayType([False] * 4)
cuda_tensor2 = cuda_ndarray.CudaNdarrayType([False] * 2)
device_id = theano.sandbox.cuda.use.device_number
if device_id is None:
    cuda_ndarray.shared_constructor(numpy.zeros(2, dtype='float32'))
device_id = theano.sandbox.cuda.use.device_number
if device_id is None:
    cuda.use("gpu",
             force=False,
             default_to_move_computation_to_gpu=False,
             move_shared_float32_to_gpu=False,
             enable_cuda=False,
             test_driver=True)
    device_id = theano.sandbox.cuda.use.device_number
cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
device_prop = cuda_ndarray.device_properties(device_id)

  
def py_corr_scipy(img, kern, mode, subsample):
    assert img.shape[1] == kern.shape[1]
    if mode == 'valid':
        outshp = (img.shape[0], kern.shape[0],
                img.shape[2] - kern.shape[2] + 1,
                img.shape[3] - kern.shape[3] + 1)
    else:
        outshp = (img.shape[0], kern.shape[0],
                img.shape[2] + kern.shape[2] - 1,
                img.shape[3] + kern.shape[3] - 1)
    out = numpy.zeros(outshp, dtype='float32')
    for b in xrange(out.shape[0]):
        for k in xrange(out.shape[1]):
            for s in xrange(img.shape[1]):
                out[b, k, :, :] += correlate(img[b, s, :, :],
                                              kern[k, s, :, :],
                                              mode)
    return out


def _params_allgood_header():
    print "ishape kshape #Mflops CPU Mflops GPU Mflops Speedup"
     
kH = 3
kW = 3
nInputPlane = 3 #channels
nOutputPlane = 2
padding = 0
batchSize = 4 



inputWidth   = 7 #im.shape[1] 
inputHeight  = 7 #im.shape[0] 
ishape = (batchSize, nInputPlane, inputHeight, inputWidth)
kshape = (nOutputPlane, nInputPlane, kH, kW)
print 'Image shape', ishape
print 'Kernel shape', kshape

im = numpy.random.rand(*ishape) + 1
#plt.imread('lena.bmp')

img_stride = (1, 1)
kern_stride = (1, 1)
outputWidth  = (inputWidth + 2*padding - kW) / img_stride[1] + 1
outputHeight = (inputHeight + 2*padding - kH) / img_stride[0] + 1
oshape=(batchSize, nInputPlane, outputHeight, outputWidth)

npy_img = theano._asarray(numpy.random.rand(*ishape) + 1,
                          dtype='float32')

npy_kern = theano._asarray(numpy.random.rand(*kshape) - 2,
                           dtype='float32')
                
img = cuda_ndarray.CudaNdarray(npy_img)
kern = cuda_ndarray.CudaNdarray(npy_kern)

 #temporary columns
cshape = (nInputPlane*kW*kH, outputHeight*outputWidth)
print 'Columns shape: ', cshape
oshape=(batchSize, nInputPlane, outputHeight, outputWidth)
print 'Output shape: ', oshape

subsample = 1
mode = 'valid'
t0 = time.time()
cpuval = py_corr_scipy(npy_img, npy_kern, mode, subsample)
t1 = time.time()
i = cuda_tensor4()
k = cuda_tensor4()


op = theano.sandbox.cuda.blas.GpuConvMM(border_mode=mode,
                                      subsample=(subsample, subsample),
                                      version=100,
                                      verbose=2, pad=1)(i, k)
                                      
f = theano.function([i, k], op, mode=theano_mode)
gpuval = f(img, kern)
t2 = time.time()
gpuval = numpy.asarray(gpuval)


if gpuval.shape != cpuval.shape:
    print >> sys.stdout, "ERROR: shape mismatch",
    print >> sys.stdout, gpuval.shape, cpuval.shape
     
print '---------------- INPUT VAL -----------------------'
print npy_img
print '---------------- kernel -----------------------'
print npy_kern
print '---------------- GPU VAL -----------------------'
print gpuval
print '---------------- CPU VAL -----------------------'
print cpuval
rval = numpy.allclose(cpuval, gpuval, rtol=1e-4)
print rval
assert numpy.all(numpy.isfinite(gpuval))