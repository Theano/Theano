"""
Tests for GPU convolution
"""
import sys
import time
import unittest


import numpy

from nose.plugins.skip import SkipTest
imported_scipy_convolve2d = False
try:
    from scipy.signal import convolve2d
    imported_scipy_convolve2d = True
except ImportError:
    pass

import theano
from theano import tensor
from theano.gof.python25 import any

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


def py_conv_valid_numpy(img, kern):
    assert img.shape[1] == kern.shape[1]
    outshp = (img.shape[0], kern.shape[0],
            img.shape[2] - kern.shape[2] + 1,
            img.shape[3] - kern.shape[3] + 1)
    out = numpy.zeros(outshp, dtype='float32')
    for b in xrange(out.shape[0]):
        for k in xrange(out.shape[1]):
            for rr in xrange(out.shape[2]):
                for cc in xrange(out.shape[3]):
                    #rr, cc is the upper-left corner of img patches
                    imgpatch = img[b, :, rr:rr + kern.shape[2],
                                   cc:cc + kern.shape[3]]

                    innerprod = (imgpatch[:, ::-1, ::-1] *
                                 kern[k, :, :, :]).sum()
                    out[b, k, rr, cc] = innerprod
    return out


def py_conv_full_numpy(img, kern):
    # manually pad the img with zeros all around, and then run it
    # through py_conv_valid
    pad_rows = 2 * (kern.shape[2] - 1) + img.shape[2]
    pad_cols = 2 * (kern.shape[3] - 1) + img.shape[3]
    padded_img = numpy.zeros((img.shape[0], img.shape[1], pad_rows, pad_cols),
                             dtype=img.dtype)
    padded_img[:, :, kern.shape[2] - 1: kern.shape[2] - 1 + img.shape[2],
                     kern.shape[3] - 1: kern.shape[3] - 1 + img.shape[3]] = img
    return py_conv_valid_numpy(padded_img, kern)


def py_conv(img, kern, mode, subsample):
    """
    use a scipy or numpy implementation depending is scipy is available.
    The scipy version is faster.
    """
    if imported_scipy_convolve2d:
        return py_conv_scipy(img, kern, mode, subsample)
    elif mode == 'valid':
        return py_conv_valid_numpy(img, kern)[:, :, ::subsample[0],
                                                      ::subsample[1]]
    elif mode == 'full':
        return py_conv_full_numpy(img, kern)[:, :, ::subsample[0],
                                                     ::subsample[1]]
    else:
        raise Exception("Can't execute this kernel.")


def py_conv_scipy(img, kern, mode, subsample):
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
                out[b, k, :, :] += convolve2d(img[b, s, :, :],
                                              kern[k, s, :, :],
                                              mode)
    return out[:, :, ::subsample[0], ::subsample[1]]


def _params_allgood_header():
    print "ishape kshape #Mflops CPU Mflops GPU Mflops Speedup"


def _params_allgood(ishape, kshape, mode, subsample=(1, 1), img_stride=(1, 1),
                    kern_stride=(1, 1), version=-1, verbose=0, random=True,
                    print_=None, id=None, rtol=1e-5, atol=1e-8,
                    nb_iter=0, ones=False, compile_kshp=None):
    #
    # This function is the core of several of the big unit-test drivers,
    # but it can also be used very directly on its own to test a specific
    # kind of convolution.
    #
    # See `test_example` (above) for an example of how to use this directly.
    #
    # :param kshape: (4d)The shape of the kernel at run time.
    # :param compile_kshp: (2d) hardcode the shape of the kernel in
    #                      the generated code This is supposed to be
    #                      faster, but we need to check That we raise
    #                      an error if the input have the wrong shape.
    #
    if ones:
        assert not random
        npy_img = theano._asarray(numpy.ones(ishape), dtype='float32')
        npy_kern = -theano._asarray(numpy.ones(kshape), dtype='float32')
    elif random:
        npy_img = theano._asarray(numpy.random.rand(*ishape) + 1,
                                  dtype='float32')
        npy_kern = theano._asarray(numpy.random.rand(*kshape) - 2,
                                   dtype='float32')
    else:
        npy_img = theano._asarray(numpy.arange(
                numpy.prod(ishape)).reshape(ishape), dtype='float32') + 1
        npy_kern = -(theano._asarray(numpy.arange(
                    numpy.prod(kshape)).reshape(kshape), dtype='float32') + 1)

    img = cuda_ndarray.CudaNdarray(npy_img)
    kern = cuda_ndarray.CudaNdarray(npy_kern)

    #we take the stride after the transfert as we make c_contiguous
    #data on the GPU.
    if img_stride != (1, 1):
        img = img[:, :, ::img_stride[0], ::img_stride[1]]
        npy_img = npy_img[:, :, ::img_stride[0], ::img_stride[1]]
    if kern_stride != (1, 1):
        kern = kern[:, :, ::kern_stride[0], ::kern_stride[1]]
        npy_kern = npy_kern[:, :, ::kern_stride[0], ::kern_stride[1]]

    t2 = None
    rval = True
    try:
        t0 = time.time()
        cpuval = py_conv(npy_img, npy_kern, mode, subsample)
        t1 = time.time()
        i = cuda_tensor4()
        k = cuda_tensor4()
        op = theano.sandbox.cuda.blas.GpuConv(border_mode=mode,
                                              subsample=subsample,
                                              version=version,
                                              verbose=verbose,
                                              kshp=compile_kshp)(i, k)
        f = theano.function([i, k], op, mode=theano_mode)
        gpuval = f(img, kern)
        t2 = time.time()
        for i in range(nb_iter):
            gpuval2 = f(img, kern)
            assert numpy.allclose(numpy.asarray(gpuval),
                                  numpy.asarray(gpuval2))
            assert (numpy.asarray(gpuval) == numpy.asarray(gpuval2)).all()
        gpuval = numpy.asarray(gpuval)
        if gpuval.shape != cpuval.shape:
            print >> sys.stdout, "ERROR: shape mismatch",
            print >> sys.stdout, gpuval.shape, cpuval.shape
            rval = False
        if rval:
            rval = numpy.allclose(cpuval, gpuval, rtol=rtol)
            assert numpy.all(numpy.isfinite(gpuval))
    except NotImplementedError, e:
        print >> sys.stdout, '_params_allgood Failed allclose', e
        rval = False

    if (t2 is not None):
        if mode == 'valid':
            approx_fp = cpuval.size * ishape[1] * kshape[2] * kshape[3] * 2
        else:
            approx_fp = (ishape[0] * kshape[0] * kshape[1] * kshape[2] *
                         kshape[3] * ishape[2] * ishape[3] * 2)
        approx_fp /= 1e6
        cpu_mflops = approx_fp / (t1 - t0)
        gpu_mflops = approx_fp / (t2 - t1)
        if verbose > 0:
            print >> sys.stdout, '%15s' % str(ishape), '%15s' % str(kshape),
            print >> sys.stdout, '%12.5f  %7.2f %7.2f %7.1f' % (approx_fp,
                    cpu_mflops, gpu_mflops, (t1 - t0) / (t2 - t1))
    if not rval:
        print >> sys.stdout, ('test_' + mode + ' id=' + str(id) +
                              ' FAILED for ishape, kshape, mode, subsample,' +
                              ' img_stride, kern_stride, version', ishape,
                              kshape, mode, subsample, img_stride, kern_stride,
                              version)
        diff = cpuval - gpuval
        diffabs = numpy.absolute(diff)
        pr_diff = diffabs / numpy.absolute(cpuval)
        nb_close = (diffabs <= (atol + rtol * numpy.absolute(gpuval))).sum()
        print "max absolute diff:", (diffabs.max(), "avg abs diff:",
                                     numpy.average(diffabs))
        print "median abs diff:", (numpy.median(diffabs), "nb close:",
                                   nb_close, "/", diff.size)
        print "max relatif diff:", (pr_diff.max(), "avg rel diff:",
                                    numpy.average(pr_diff))
    if not rval and print_ != False:
        if npy_img.shape[0] > 5:
            print "img", npy_img[0]
            print "kern", npy_kern[0]
            print "gpu", gpuval[0][0]
            print "cpu", cpuval[0][0]
            print "diff", diff[0][0]
        else:
            print "img", npy_img
            print "kern", npy_kern
            print "gpu", gpuval
            print "cpu", cpuval
            print "diff", diff

    return rval


def exec_conv(version, shapes, verbose, random, mode,
              print_=None, rtol=1e-5, ones=False):
    if verbose > 0:
        _params_allgood_header()
    nb_failed = 0
    nb_tests = 0

    failed_version = set()
    failed_id = []
    # I put -1 in case we forget to add version in the test to.
    for ver in version:
        for id, (ishape, kshape, subshape,
                 istride, kstride) in enumerate(shapes):
            ret = False
            try:
                ret = _params_allgood(ishape,
                        kshape,
                        mode,
                        subsample=subshape,
                        img_stride=istride,
                        kern_stride=kstride,
                        version=ver,
                        verbose=verbose,
                        random=random,
                        id=id,
                        print_=print_,
                        rtol=rtol,
                        ones=ones)
            except Exception, e:
                print ver, id, (ishape, kshape, subshape, istride, kstride)
                print e
                pass
            if not ret:
                failed_version.add(ver)
                failed_id.append(id)
                nb_failed += 1
            nb_tests += 1
    if nb_failed > 0:
        print "nb_failed", nb_failed, "on", nb_tests,
        print "failed_version", failed_version, "failed_id", failed_id
        assert nb_failed == 0, nb_failed
    else:
        print 'Executed', nb_tests, 'different shapes'


def get_basic_shapes():
        #basic test of image and kernel shape
    return [((1, 1, 1, 1), (1, 1, 1, 1), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 2, 2), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 3, 3), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
        #basic test for unsquare kernel and image
            ((1, 1, 2, 4), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 3, 4), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 4, 3), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 4, 4), (1, 1, 3, 2), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 4, 4), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1))]


def get_shapes(imshp=(1, 1), kshp=(1, 1), subsample=(1, 1),
               img_stride=(1, 1), kern_stride=(1, 1)):
    """ all possible case if we one or more of stack size, batch size,
    nkern. We use the gived image shape, kernel shape and subsmaple
    shape."""
    return [
        #stack only
        ((1, 2) + imshp, (1, 2) + kshp, subsample, img_stride, kern_stride),
        #batch only
        ((3, 1) + imshp, (1, 1) + kshp, subsample, img_stride, kern_stride),
        #nkern only
        ((1, 1) + imshp, (2, 1) + kshp, subsample, img_stride, kern_stride),
        #batch and nkern
        ((3, 1) + imshp, (2, 1) + kshp, subsample, img_stride, kern_stride),
        #batch and stack
        ((3, 2) + imshp, (1, 2) + kshp, subsample, img_stride, kern_stride),
        #stack and nkern
        ((1, 2) + imshp, (2, 2) + kshp, subsample, img_stride, kern_stride),
        #batch, nkern and stack
        ((2, 2) + imshp, (2, 2) + kshp, subsample, img_stride, kern_stride),
        #batch, nkern and stack
        ((3, 2) + imshp, (4, 2) + kshp, subsample, img_stride, kern_stride)
    ]


def get_shapes2(scales_img=(1, 1), scales_kern=(1, 1), subsample=(1, 1),
                img_stride=(1, 1), kern_stride=(1, 1)):
    #basic test of stack, batch and nkern paramter
    shapes = get_shapes((1 * scales_img[0], 1 * scales_img[1]),
                        (1 * scales_kern[0], 1 * scales_kern[1]),
                        subsample, img_stride, kern_stride)
    #basic test of stack, batch and nkern paramter with image and kernel shape
    shapes += get_shapes((2 * scales_img[0], 2 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    #basic test of stack, batch and nkern paramter with image and kernel shape
    shapes += get_shapes((3 * scales_img[0], 3 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    #basic test of stack, batch and nkern paramter with not square image.
    shapes += get_shapes((4 * scales_img[0], 3 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    #basic test of stack, batch and nkern paramter with not square image.
    shapes += get_shapes((3 * scales_img[0], 4 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    #basic test of stack, batch and nkern paramter with not square kernel.
    shapes += get_shapes((4 * scales_img[0], 4 * scales_img[1]),
                         (3 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    #basic test of stack, batch and nkern paramter with not square kernel.
    shapes += get_shapes((4 * scales_img[0], 4 * scales_img[1]),
                         (2 * scales_kern[0], 3 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    return shapes


def get_valid_shapes():

    #          img shape,     kern shape, subsample shape

    shapes = get_basic_shapes()
    shapes += get_shapes2()

    #test image stride
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(1, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 1))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(-1, -1))
    shapes += get_shapes2(scales_img=(2, 2), kern_stride=(-1, -1))

    #test subsample done in a separate fct

    shapes += [
         #other test
              ((2, 1, 2, 2), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1))
            , ((3, 2, 4, 4), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 10, 10), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1))
            , ((1, 1, 4, 4), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 10, 10), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 10, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 20, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1))
            , ((3, 2, 8, 8), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize
            , ((3, 2, 8, 6), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize, non-square image
            , ((3, 2, 8, 6), (4, 2, 4, 3), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize, non-square image, non-square kern
            , ((3, 2, 8, 6), (4, 2, 4, 6), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize ,non-square image, non-square kern, kernsize==imgsize on one dim
            , ((16, 5, 64, 64), (8, 5, 8, 8), (1, 1), (1, 1), (1, 1)) # a big one
            , ((16, 1, 28, 28), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)) # MNIST LeNET layer 1
            , ((20, 16, 32, 32), (1, 16, 28, 28), (1, 1), (1, 1), (1, 1)) # layer 1 backprop to weights
            , ((60,20,28,28), (10,20,5,5), (1, 1), (2,2), (1, 1))#added a test case that fail from test_nnet.py.test_conv_nnet2
            , ((10,5,28,28), (10,5,5,5), (1, 1), (2,2), (1, 1))#test precedent but reduced that triger the error
            #Test more than maxThreadsDim0
            , ((2,4,13,1050), (3,4,10, 11), (1, 1), (1, 1), (1, 1))
            , ((2,4,1050,13), (3,4,10, 11), (1, 1), (1, 1), (1, 1))
            ]

    shapes += [ ((60,1,28,28),(20,1,5,5), (1, 1), (1, 1), (1, 1))#test_lenet_28 1 layers
            , ((60,20,12,12),(30,20,5,5), (1, 1), (1, 1), (1, 1))#test_lenet_28 2 layers
            , ((60,30,8,8),(20,30,5,5), (1, 1), (1, 1), (1, 1))#test_lenet_28 bprop 1 full
            , ((20,60,12,12),(30,60,8,8), (1, 1), (1, 1), (1, 1))#test_lenet_28 bprop 2 valid
#            , ((1,60,28,28),(20,60,24,24), (1, 1), (1, 1), (1, 1))#test_lenet_28 bprop 2 valid
            , ((10,1,64,64),(20,1,7,7), (1, 1), (1, 1), (1, 1))#test_lenet_64 1 layers
            , ((10,20,29,29),(30,20,7,7), (1, 1), (1, 1), (1, 1))#test_lenet_64 2 layers
            , ((10,30,23,23),(20,30,7,7), (1, 1), (1, 1), (1, 1))#test_lenet_64 full
#            , ((20,10,29,29),(30,10,23,23), (1, 1), (1, 1), (1, 1))#test_lenet_64 bprop 1
#            , ((1,10,64,64),(20,10,58,58), (1, 1), (1, 1), (1, 1))#test_lenet_64 bprop 2
            ]
    return shapes


def test_valid_0_2():
    shapes = get_valid_shapes()
    version = [0, 2]
    verbose = 0

    random = True
    print_ = False
    ones = False
    if ones:
        random = False
    shapes2 = []

    for id, (ishape, kshape, subshape, istride, kstride) in enumerate(shapes):
        oshape = [ishape[0]] + [kshape[0]] + list(numpy.asarray(ishape[2:]) -
                                                  numpy.asarray(kshape[2:]) +
                                                  numpy.asarray([1, 1]))
        if oshape[3] > device_prop['maxThreadsDim0']:
            continue
        if ishape[1] > 1:
            continue
        if ((numpy.prod(ishape[2:]) + numpy.prod(kshape[2:])) * 4 >
            (16 * 1024 - 150)):
            continue
        if subshape == (1, 1):
            shapes2.append((ishape, kshape, subshape, istride, kstride))
    shapes = shapes2

    exec_conv(version, shapes, verbose, random, 'valid',
              print_=print_, ones=ones, rtol=1.1e-5)


def test_valid_1_3_11_12():
    shapes = get_valid_shapes()
    version = [1, 3, 11, 12]
    verbose = 0

    random = True
    print_ = False
    ones = False
    if ones:
        random = False
    shapes2 = []

    for id, (ishape, kshape, subshape, istride, kstride) in enumerate(shapes):
        oshape = [ishape[0]] + [kshape[0]] + list(numpy.asarray(ishape[2:]) -
                                                  numpy.asarray(kshape[2:]) +
                                                  numpy.asarray([1, 1]))
        if oshape[3] > device_prop['maxThreadsDim0']:
            continue
        if ((numpy.prod(ishape[2:]) + numpy.prod(kshape[2:])) * 4 >
            (16 * 1024 - 150)):
            continue
        if subshape == (1, 1):
            shapes2.append((ishape, kshape, subshape, istride, kstride))
    shapes = shapes2

    exec_conv(version, shapes, verbose, random, 'valid',
              print_=print_, ones=ones, rtol=1.1e-5)


def test_valid_4():
    shapes = get_valid_shapes()
    version = [4]
    verbose = 0

    random = True
    print_ = False
    ones = False
    if ones:
        random = False
    shapes2 = []

    for id, (ishape, kshape, subshape, istride, kstride) in enumerate(shapes):
        oshape = [ishape[0]] + [kshape[0]] + list(numpy.asarray(ishape[2:]) -
                                                  numpy.asarray(kshape[2:]) +
                                                  numpy.asarray([1, 1]))
        if oshape[3] > device_prop['maxThreadsDim0']:
            continue
        if ishape[1] > 1:
            continue
        if ((kshape[2] * ishape[3] * 4 + numpy.prod(kshape[2:]) * 4) >
            (16 * 1024 - 150)):
            continue
        if subshape == (1, 1):
            shapes2.append((ishape, kshape, subshape, istride, kstride))
    shapes = shapes2

    exec_conv(version, shapes, verbose, random, 'valid',
              print_=print_, ones=ones, rtol=1.1e-5)


def test_valid_5():
    shapes = get_valid_shapes()
    version = [5]
    verbose = 0

    random = True
    print_ = False
    ones = False
    if ones:
        random = False
    shapes2 = []

#    print len(shapes)
    for id, (ishape, kshape, subshape, istride, kstride) in enumerate(shapes):
        oshape = [ishape[0]] + [kshape[0]] + list(numpy.asarray(ishape[2:]) -
                                                  numpy.asarray(kshape[2:]) +
                                                  numpy.asarray([1, 1]))
        if oshape[3] > device_prop['maxThreadsDim0']:
            continue
        if ((kshape[2] * ishape[3] * 4 + numpy.prod(kshape[2:]) * 4) >
            (16 * 1024 - 150)):
            continue
        if subshape == (1, 1):
            shapes2.append((ishape, kshape, subshape, istride, kstride))
    shapes = shapes2
#    print len(shapes2)

    exec_conv(version, shapes, verbose, random, 'valid',
              print_=print_, ones=ones, rtol=1.1e-5)


def test_valid_7_8_13():
    shapes = get_valid_shapes()
    # This is to test the "new" lower shared memory usage.
    shapes.append(((10, 30, 60, 60), (20, 30, 40, 40),
                   (1, 1), (1, 1), (1, 1)))
    version = [7, 8, 13]
    verbose = 0

    random = True
    print_ = False
    ones = False
    if ones:
        random = False
    shapes2 = []

#    print len(shapes)
    for id, (ishape, kshape, subshape, istride, kstride) in enumerate(shapes):
        oshape = [ishape[0]] + [kshape[0]] + list(numpy.asarray(ishape[2:]) -
                                                  numpy.asarray(kshape[2:]) +
                                                  numpy.asarray([1, 1]))
        if oshape[2] * oshape[3] > device_prop['maxThreadsDim0']:
            continue
        if max(numpy.prod(ishape[2:]) * 4 + 2 * kshape[3] * 4,
               oshape[2] * oshape[3] * 4 * 2) > (16 * 1024 - 150):
            continue
        if subshape == (1, 1):
            shapes2.append((ishape, kshape, subshape, istride, kstride))
    shapes = shapes2
#    print len(shapes2)

    exec_conv(version, shapes, verbose, random, 'valid',
              print_=print_, ones=ones, rtol=1.1e-5)


def test_valid_9_10():
    shapes = get_valid_shapes()
    version = [9, 10]
    verbose = 0

    random = True
    print_ = False
    ones = False
    if ones:
        random = False
    shapes2 = []

#    print len(shapes)
    for id, (ishape, kshape, subshape, istride, kstride) in enumerate(shapes):
        oshape = [ishape[0]] + [kshape[0]] + list(numpy.asarray(ishape[2:]) -
                                                  numpy.asarray(kshape[2:]) +
                                                  numpy.asarray([1, 1]))
        if oshape[3] > device_prop['maxThreadsDim0']:
            continue
        if (kshape[3] * 4 + ishape[3]) > (16 * 1024 - 150):
            continue
        if subshape == (1, 1):
            shapes2.append((ishape, kshape, subshape, istride, kstride))
    shapes = shapes2
#    print len(shapes2)

    exec_conv(version, shapes, verbose, random, 'valid',
              print_=print_, ones=ones, rtol=1.1e-5)


def test_valid():
    shapes = get_valid_shapes()

    #shapes=shapes[400:426]
    # I put -1 in case we forget to add version in the test to.
    # I put -2 to test the reference version.
    version = [-2, -1, 6]
    verbose = 0
#    version=[1]

    random = True
    print_ = False
    ones = False
    if ones:
        random = False

    exec_conv(version, shapes, verbose, random, 'valid',
              print_=print_, ones=ones, rtol=1.1e-5)


def test_full():
    shapes = get_basic_shapes()
    shapes += get_shapes2()
    #test image stride
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(1, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 1))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(-1, -1))
    shapes += get_shapes2(scales_img=(2, 2), kern_stride=(-1, -1))

    #test subsample done in a separate fct

    shapes += [
        #other test
              ((2, 1, 2, 2), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1))
            , ((3, 2, 4, 4), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 10, 10), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1))
            , ((1, 1, 4, 4), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 10, 10), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 10, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1))
            , ((4, 1, 20, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1))
            , ((3, 2, 8, 8), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize
            , ((3, 2, 8, 6), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize, non-square image
            , ((3, 2, 8, 6), (4, 2, 4, 3), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize, non-square image, non-square kern
            , ((3, 2, 8, 6), (4, 2, 4, 6), (1, 1), (1, 1), (1, 1)) #stack, nkern, bsize ,non-square image, non-square kern, kernsize==imgsize on one dim
            , ((16, 5, 64, 64), (8, 5, 8, 8), (1, 1), (1, 1), (1, 1)) # a big one
            , ((16, 1, 28, 28), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)) # MNIST LeNET layer 1
            , ((20, 16, 32, 32), (1, 16, 28, 28), (1, 1), (1, 1), (1, 1)) # layer 1 backprop to weights

        #other test
            , ((3, 1, 1, 1), (2, 1, 5, 3), (1, 1), (1, 1), (1, 1))#kernel bigger then image
            , ((3, 2, 1, 1), (4, 2, 1, 1), (1, 1), (1, 1), (1, 1))
            , ((3, 2, 4, 4), (4, 2, 2, 6), (1, 1), (1, 1), (1, 1))
            , ((3, 2, 4, 4), (4, 2, 8, 6), (1, 1), (1, 1), (1, 1))#kernel bigger then image
            , ((4, 2, 10, 10), (3, 2, 2, 12), (1, 1), (1, 1), (1, 1))
            ]
    shapes += [
#        ((60,1,28,28),(20,1,5,5), (1, 1), (1, 1), (1, 1))#test_lenet_28 1 layers
#            , ((60,20,12,12),(30,20,5,5), (1, 1), (1, 1), (1, 1))#test_lenet_28 2 layers
             ((60,30,8,8),(20,30,5,5), (1, 1), (1, 1), (1, 1))#test_lenet_28 bprop 1 full
#            , ((20,60,12,12),(30,60,8,8), (1, 1), (1, 1), (1, 1))#test_lenet_28 bprop 2 valid
#            , ((1,60,28,28),(20,60,24,24), (1, 1), (1, 1), (1, 1))#test_lenet_28 bprop 2 valid
#            , ((10,1,64,64),(20,1,7,7), (1, 1), (1, 1), (1, 1))#test_lenet_64 1 layers
#            , ((10,20,29,29),(30,20,7,7), (1, 1), (1, 1), (1, 1))#test_lenet_64 2 layers
            , ((10,30,23,23),(20,30,7,7), (1, 1), (1, 1), (1, 1))#test_lenet_64 full
#            , ((20,10,29,29),(30,10,23,23), (1, 1), (1, 1), (1, 1))#test_lenet_64 bprop 1
#            , ((1,10,64,64),(20,10,58,58), (1, 1), (1, 1), (1, 1))#test_lenet_64 bprop 2
            #Test more than maxThreadsDim0
            , ((2,4,13,1050), (3,4,10, 11), (1, 1), (1, 1), (1, 1))
            , ((2,4,1050,13), (3,4,10, 11), (1, 1), (1, 1), (1, 1))
            ]

#    shapes=shapes[:277]
    version = [-2, -1, 0, 1, 2, 3, 4, 5]
    verbose = 0
#    version=[4]
    random = True

    exec_conv(version, shapes, verbose, random, 'full')


def test_subsample():
    # implement when
    shapes = [((1, 1, 1, 1), (1, 1, 1, 1), (1, 1), (1, 1), (1, 1)),
              ((1, 1, 1, 1), (1, 1, 1, 1), (2, 2), (1, 1), (1, 1)),
              ((4, 2, 10, 10), (3, 2, 2, 2), (1, 3), (1, 1), (1, 1)),
              ((4, 2, 10, 10), (3, 2, 2, 2), (3, 3), (1, 1), (1, 1)),
              ((4, 2, 10, 10), (3, 2, 2, 2), (3, 1), (1, 1), (1, 1))
          ]
    shapes += get_shapes2(scales_img=(2, 2), subsample=(1, 1))
    shapes += get_shapes2(scales_img=(2, 2), subsample=(1, 2))
    shapes += get_shapes2(scales_img=(2, 2), subsample=(2, 1))
    shapes += get_shapes2(scales_img=(2, 2), subsample=(2, 2))

#We put only the version that implement the subsample to make the test faster.
    version_valid = [-2, -1, 1, 3, 11, 12]
    version_full = [-2, -1]
    verbose = 0
    random = True
    print_ = False
    ones = False
    if ones:
        random = False

    exec_conv(version_valid, shapes, verbose, random, 'valid',
              print_=print_, ones=ones)
    exec_conv(version_full, shapes, verbose, random, 'full',
              print_=print_, ones=ones)


class TestConv2DGPU(unittest.TestCase):
    def test_logical_shapes(self):
        for stride in range(1, 4):
            kshp = (10, 2, 10, 10)
            featshp = (3, 10, 11, 11)

            a = tensor.ftensor4()
            A = tensor.ftensor4()

            # Need to transpose first two dimensions of kernel, and reverse
            # index kernel image dims (for correlation)
            kernel_rotated = tensor.transpose(A, axes=[1, 0, 2, 3])

            featshp_logical = (featshp[0], featshp[1], featshp[2] * stride,
                               featshp[3] * stride)
            kshp_rotated = (kshp[1], kshp[0], kshp[2], kshp[3])
            print featshp, kshp_rotated, featshp_logical[1:], kshp[2:]
            image_estimate = tensor.nnet.conv2d(a, kernel_rotated,
                                                border_mode='full',
                                                image_shape=featshp,
                                                filter_shape=kshp_rotated,
                                                imshp_logical=featshp_logical[1:],
                                                kshp_logical=kshp[2:])

            func = theano.function([a, A], image_estimate, mode=theano_mode)
            theano.printing.debugprint(func,)
            assert any([isinstance(node.op, theano.sandbox.cuda.blas.GpuConv)
                        for node in func.maker.fgraph.toposort()])

            a_in = numpy.random.randn(*featshp).astype("float32")
            A_in = numpy.random.randn(*kshp).astype("float32")

            func(a_in, A_in)

    def test_invalid_input_shape(self):
        """
        Tests that when the shape gived at build time is not the same as
        run time we raise an error
        """
        verbose = 0
        random = True
        print_ = False
        ones = False
        if ones:
            random = False

        global theano_mode
        theano_mode_orig = theano_mode
        try:
            if theano.config.mode in ['DebugMode', 'DEBUG_MODE']:
                theano_mode = theano.compile.mode.get_mode(
                    'FAST_RUN').including('gpu')
                for mode in ['valid', 'full']:
                    for shapes in [((3, 2, 8, 8), (4, 2, 5, 5), (8, 8)),
                                   ((3, 2, 8, 8), (4, 2, 5, 5), (5, 8)),
                                   #((3, 2, 8, 8), (4, 2, 5, 5), (8, 5)),
                                   # We use only the number of columns.
                                   ]:

                        self.assertRaises(ValueError, _params_allgood,
                                          shapes[0], shapes[1],
                                          verbose=verbose, random=random,
                                          mode=mode,
                                          print_=print_, ones=ones,
                                          compile_kshp=shapes[2])
        finally:
            theano_mode = theano_mode_orig


def _test_dummy():
    ishape = (1, 1, 5, 5)
    kshape = (1, 1, 3, 3)
    mode = 'valid'
    subsample = (1, 1)

    npy_img = theano._asarray(numpy.random.rand(*ishape), dtype='float32')
    npy_kern = theano._asarray(numpy.random.rand(*kshape), dtype='float32')

    img = cuda_ndarray.CudaNdarray(npy_img)
    kern = cuda_ndarray.CudaNdarray(npy_kern)

    #print >> sys.stdout, '_params_allgood trying ', ishape, kshape, mode
    t2 = None
    rval = True

    t0 = time.time()
    cpuval = py_conv(npy_img, npy_kern, mode, subsample)
    t1 = time.time()
    gpuval = cuda_ndarray.conv(img, kern, mode, subsample)
    t2 = time.time()
    gpuval = numpy.asarray(gpuval)
    print gpuval
    print cpuval


def benchmark():

    shapes_valid = [
        #test_lenet_28 shape
        ((20, 60,12,12), (30,60,8,8), (1, 1), (1, 1), (1, 1))#valid
        ,((60, 20,12,12), (30,20,5,5), (1, 1), (1, 1), (1, 1))#valid
        ,((60, 1,28,28), (20,1,5,5), (1, 1), (1, 1), (1, 1))#valid
        ,((1, 60,28,28), (20,60,24,24), (1, 1), (1, 1), (1, 1))#valid
        #test_lenet_32 shape
        ,((20, 60,14,14), (30,60,10,10), (1, 1), (1, 1), (1, 1))#valid
        ,((60, 20,14,14), (30,20,5,5), (1, 1), (1, 1), (1, 1))#valid
        ,((60, 1,32,32), (20,1,5,5), (1, 1), (1, 1), (1, 1))#valid
        ,((1, 60,32,32), (20,60,28,28), (1, 1), (1, 1), (1, 1))#valid
        #test_lenet_64 shape
        ,((10, 20,29,29), (30,20,7,7), (1, 1), (1, 1), (1, 1))#valid
        ,((20, 10,29,29), (30,10,23,23), (1, 1), (1, 1), (1, 1))#valid
        ,((10, 1,64,64), (20,1,7,7), (1, 1), (1, 1), (1, 1))#valid
        ,((1, 10,64,64), (20,10,58,58), (1, 1), (1, 1), (1, 1))#valid
        #test_lenet_108 shape
        ,((10, 20,51,51), (30,20,7,7), (1, 1), (1, 1), (1, 1))#valid
        ,((20, 10,51,51), (30,10,45,45), (1, 1), (1, 1), (1, 1))#valid
        ,((10, 1,108,108), (20,1,7,7), (1, 1), (1, 1), (1, 1))#valid
        ,((1, 10,108,108), (20,10,102,102), (1, 1), (1, 1), (1, 1))#valid
        #test_lenet_256 shape
        ,((2, 20,124,124), (30,20,9,9), (1, 1), (1, 1), (1, 1))#valid
        ,((20, 2,124,124), (30,2,116,116), (1, 1), (1, 1), (1, 1))#valid
        ,((2, 1,256,256), (20,1,9,9), (1, 1), (1, 1), (1, 1))#valid
        ,((1, 2,256,256), (20,2,248,248), (1, 1), (1, 1), (1, 1))#valid
            ]

    shapes_full = [
        #test_lenet_28 shape
         ((60, 30,8,8), (20, 30, 5, 5), (1, 1), (1, 1), (1, 1))#full
        #test_lenet_32 shape
         ,((60, 30,10,10), (20, 30, 5, 5), (1, 1), (1, 1), (1, 1))#full conv_full_patch_stack_padded' N=1
        #test_lenet_64 shape
         ,((10, 30,23,23), (20, 30, 7, 7), (1, 1), (1, 1), (1, 1))#full conv_full_patch_stack_padded' N=3
        #test_lenet_108 shape
         ,((10, 30,45,45), (20, 30, 7, 7), (1, 1), (1, 1), (1, 1))#full 'conv_full_patch_stack_padded' N=9
        #test_lenet_256 shape
         ,((2, 30,116,116), (20, 30, 9,9), (1, 1), (1, 1), (1, 1))#full conv_reference_full
            ]

#    shapes_valid=shapes_valid[-1:]
#    shapes_full=shapes_full[-1:]
    version = [-1]
    verbose = 1
    random = True

    exec_conv(version, shapes_valid, verbose, random, 'valid',
              print_=None, rtol=1e-3)
    exec_conv(version, shapes_full, verbose, random, 'full')


def test_stack_rows_segfault_070312():
    # 07/03/2012
    # Running this unittest with cuda-memcheck exposes an illegal read.
    # THEANO_FLAGS=device=gpu cuda-memcheck nosetests \
    # test_conv_cuda_ndarray.py:test_stack_rows_segfault_070312
    img = theano.shared(numpy.random.rand(1, 80, 96, 96).astype('float32'))
    kern = theano.shared(numpy.random.rand(1, 80, 9, 9).astype('float32'))
    out = theano.shared(numpy.random.rand(1, 2, 2, 3).astype('float32'))
    op = theano.tensor.nnet.conv.ConvOp(imshp=(80, 96, 96), kshp=(9, 9),
            nkern=1, bsize=1)
    f = theano.function([], [], updates=[(out, op(img, kern))])
    f()
