"""
Tests for GPU convolution
"""
from __future__ import absolute_import, print_function, division
import time
import unittest
import theano
from theano import tensor
from theano.tests.unittest_tools import seed_rng, assert_allclose
from theano.sandbox import cuda
import numpy
from six.moves import xrange

from theano.sandbox.cuda.dnn import GpuDnnConv, DnnBase, dnn_conv
from nose.plugins.skip import SkipTest
from nose.tools import assert_raises
imported_scipy_convolve2d = False
try:
    from scipy.signal import convolve2d
    imported_scipy_convolve2d = True
except ImportError:
    pass

# Skip test if cuda is not available.
if cuda.cuda_available is False:
    raise SkipTest('Optional package cuda disabled')


# needed as the gpu conv don't have a perform implementation.
if theano.config.mode == 'FAST_COMPILE':
    theano_mode = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
else:
    theano_mode = theano.compile.mode.get_default_mode().including('gpu')

device_id = theano.sandbox.cuda.use.device_number
if device_id is None:
    cuda.shared_constructor(numpy.zeros(2, dtype='float32'))
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
                    # rr, cc is the upper-left corner of img patches
                    imgpatch = img[b, :, rr:rr + kern.shape[2],
                                   cc:cc + kern.shape[3]]

                    innerprod = (imgpatch[:, ::-1, ::-1] *
                                 kern[k, :, :, :]).sum()
                    out[b, k, rr, cc] = innerprod
    return out


def py_conv_pad_img(img, pad_h, pad_w):
    assert pad_h >= 0 and pad_w >= 0
    padded_img = numpy.zeros(
        (img.shape[0], img.shape[1],
         pad_h * 2 + img.shape[2], pad_w * 2 + img.shape[3]),
        dtype=img.dtype)
    padded_img[:, :,
               pad_h: pad_h + img.shape[2],
               pad_w: pad_w + img.shape[3]] = img
    return padded_img


def py_conv_full_numpy(img, kern):
    # manually pad the img with zeros all around, and then run it
    # through py_conv_valid
    padded_img = py_conv_pad_img(img, kern.shape[2] - 1, kern.shape[3] - 1)
    return py_conv_valid_numpy(padded_img, kern)


def py_conv(img, kern, mode, subsample):
    """
    use a scipy or numpy implementation depending is scipy is available.
    The scipy version is faster.
    """
    if isinstance(mode, int):
        mode = (mode, mode)
    if isinstance(mode, tuple):
        pad_h, pad_w = map(int, mode)
        img = py_conv_pad_img(img, pad_h, pad_w)
        mode = 'valid'
    if imported_scipy_convolve2d:
        return py_conv_scipy(img, kern, mode, subsample)
    elif mode == 'valid':
        return py_conv_valid_numpy(img, kern)[
            :, :, ::subsample[0], ::subsample[1]]
    elif mode == 'full':
        return py_conv_full_numpy(img, kern)[
            :, :, ::subsample[0], ::subsample[1]]
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
                # convolve2d or correlate
                out[b, k, :, :] += convolve2d(img[b, s, :, :],
                                              kern[k, s, :, :],
                                              mode)
    return out[:, :, ::subsample[0], ::subsample[1]]


def _params_allgood_header():
    print("ishape kshape #Mflops CPU Mflops GPU Mflops Speedup")


def _params_allgood(ishape, kshape, mode, subsample=(1, 1), img_stride=(1, 1),
                    kern_stride=(1, 1), version=-1, verbose=0, random=True,
                    print_=None, id=None, rtol=1e-5, atol=1e-8,
                    nb_iter=0, ones=False, compile_kshp=None,
                    theano_mode=None, cls=None):
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
        npy_img = theano._asarray(
            numpy.arange(numpy.prod(ishape)).reshape(ishape),
            dtype='float32') + 1
        npy_kern = -(theano._asarray(
            numpy.arange(numpy.prod(kshape)).reshape(kshape),
            dtype='float32') + 1)

    img = cuda_ndarray.CudaNdarray(npy_img)
    kern = cuda_ndarray.CudaNdarray(npy_kern)

    # we take the stride after the transfert as we make c_contiguous
    # data on the GPU.
    if img_stride != (1, 1):
        img = img[:, :, ::img_stride[0], ::img_stride[1]]
        npy_img = npy_img[:, :, ::img_stride[0], ::img_stride[1]]
    if kern_stride != (1, 1):
        kern = kern[:, :, ::kern_stride[0], ::kern_stride[1]]
        npy_kern = npy_kern[:, :, ::kern_stride[0], ::kern_stride[1]]

    i = cuda.CudaNdarrayType(
        broadcastable=[sh == 1 for sh in npy_img.shape])()
    k = cuda.CudaNdarrayType(
        broadcastable=[sh == 1 for sh in npy_kern.shape])()
    op = theano.sandbox.cuda.blas.GpuConv(border_mode=mode,
                                          subsample=subsample,
                                          version=version,
                                          verbose=verbose,
                                          kshp=compile_kshp)(i, k)
    f = theano.function([i, k], op, mode=theano_mode)
    if cls is not None:
        assert any([isinstance(node.op, cls)
                    for node in f.maker.fgraph.toposort()]), "Cannot find class %r in %r" % (cls, f.maker.fgraph.toposort())
    t2 = time.time()
    gpuval = f(img, kern)
    t3 = time.time()
    for i in range(nb_iter):
        gpuval2 = f(img, kern)
        assert (numpy.asarray(gpuval) == numpy.asarray(gpuval2)).all()
    gpuval = numpy.asarray(gpuval)

    # CPU val computed after GPU val to get the GPU errors.
    t0 = time.time()
    cpuval = py_conv(npy_img, npy_kern, mode, subsample)
    t1 = time.time()

    assert gpuval.shape == cpuval.shape, ("shape mismatch", gpuval.shape, cpuval.shape)
    assert_allclose(cpuval, gpuval, rtol=rtol, atol=atol)
    assert numpy.all(numpy.isfinite(gpuval)), gpuval
    assert [(sh == 1) is br for
            sh, br in zip(cpuval.shape[:2], op.type.broadcastable[:2])]

    if (t2 is not None and verbose > 0):
        if mode == 'valid':
            approx_fp = cpuval.size * ishape[1] * kshape[2] * kshape[3] * 2
        else:
            approx_fp = (ishape[0] * kshape[0] * kshape[1] * kshape[2] *
                         kshape[3] * ishape[2] * ishape[3] * 2)
        approx_fp /= 1e6
        if t1 - t0 != 0:
            cpu_mflops = approx_fp / (t1 - t0)
        else:
            cpu_mflops = float('inf')
        if t3 - t2 != 0:
            gpu_mflops = approx_fp / (t3 - t2)
        else:
            gpu_mflops = float('inf')

        if t2 - t1 != 0:
            div = (t1 - t0) / (t2 - t1)
        else:
            div = float('inf')
        print('%15s' % str(ishape), '%15s' % str(kshape), end=' ')
        print('%12.5f  %7.2f %7.2f %7.1f' % (
            approx_fp, cpu_mflops, gpu_mflops, div))


def exec_conv(version, shapes, verbose, random, mode,
              print_=None, rtol=1e-5, ones=False,
              theano_mode=theano_mode, cls=None):
    if verbose > 0:
        _params_allgood_header()

    for ver in version:
        for id, (ishape, kshape, subshape,
                 istride, kstride) in enumerate(shapes):
            yield (_params_allgood, ishape, kshape, mode, subshape,
                   istride, kstride, ver, verbose, random, print_, id,
                   rtol, 1e-8, 0, ones, None, theano_mode, cls)


def get_basic_shapes():
        # basic test of image and kernel shape
    return [((1, 1, 1, 1), (1, 1, 1, 1), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 2, 2), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
            ((1, 1, 3, 3), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
            # basic test for unsquare kernel and image
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
        # stack only
        ((1, 2) + imshp, (1, 2) + kshp, subsample, img_stride, kern_stride),
        # batch only
        ((3, 1) + imshp, (1, 1) + kshp, subsample, img_stride, kern_stride),
        # nkern only
        ((1, 1) + imshp, (2, 1) + kshp, subsample, img_stride, kern_stride),
        # batch and nkern
        ((3, 1) + imshp, (2, 1) + kshp, subsample, img_stride, kern_stride),
        # batch and stack
        ((3, 2) + imshp, (1, 2) + kshp, subsample, img_stride, kern_stride),
        # stack and nkern
        ((1, 2) + imshp, (2, 2) + kshp, subsample, img_stride, kern_stride),
        # batch, nkern and stack
        ((2, 2) + imshp, (2, 2) + kshp, subsample, img_stride, kern_stride),
        # batch, nkern and stack
        ((3, 2) + imshp, (4, 2) + kshp, subsample, img_stride, kern_stride)
        ]


def get_shapes2(scales_img=(1, 1), scales_kern=(1, 1), subsample=(1, 1),
                img_stride=(1, 1), kern_stride=(1, 1)):
    # basic test of stack, batch and nkern paramter
    shapes = get_shapes((1 * scales_img[0], 1 * scales_img[1]),
                        (1 * scales_kern[0], 1 * scales_kern[1]),
                        subsample, img_stride, kern_stride)
    # basic test of stack, batch and nkern paramter with image and kernel shape
    shapes += get_shapes((2 * scales_img[0], 2 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    # basic test of stack, batch and nkern paramter with image and kernel shape
    shapes += get_shapes((3 * scales_img[0], 3 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    # basic test of stack, batch and nkern paramter with not square image.
    shapes += get_shapes((4 * scales_img[0], 3 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    # basic test of stack, batch and nkern paramter with not square image.
    shapes += get_shapes((3 * scales_img[0], 4 * scales_img[1]),
                         (2 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    # basic test of stack, batch and nkern paramter with not square kernel.
    shapes += get_shapes((4 * scales_img[0], 4 * scales_img[1]),
                         (3 * scales_kern[0], 2 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    # basic test of stack, batch and nkern paramter with not square kernel.
    shapes += get_shapes((4 * scales_img[0], 4 * scales_img[1]),
                         (2 * scales_kern[0], 3 * scales_kern[1]),
                         subsample, img_stride, kern_stride)
    return shapes


def get_valid_shapes():

    #          img shape,     kern shape, subsample shape

    shapes = get_basic_shapes()
    shapes += get_shapes2()

    # test image stride
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(1, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 1))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(-1, -1))
    shapes += get_shapes2(scales_img=(2, 2), kern_stride=(-1, -1))

    # test subsample done in a separate fct

    shapes += [
        # other test
        ((2, 1, 2, 2), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
        ((3, 2, 4, 4), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 10, 10), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
        ((1, 1, 4, 4), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 10, 10), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 10, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 20, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1)),
        ((3, 2, 8, 8), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize,
        ((3, 2, 8, 6), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize, non-square image,
        ((3, 2, 8, 6), (4, 2, 4, 3), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize, non-square image, non-square kern,
        ((3, 2, 8, 6), (4, 2, 4, 6), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize ,non-square image, non-square kern, kernsize==imgsize on one dim,
        ((16, 5, 64, 64), (8, 5, 8, 8), (1, 1), (1, 1), (1, 1)),  # a big one
        ((16, 1, 28, 28), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)),  # MNIST LeNET layer 1
        ((20, 16, 32, 32), (1, 16, 28, 28), (1, 1), (1, 1), (1, 1)),  # layer 1 backprop to weights
        ((60, 20, 28, 28), (10, 20, 5, 5), (1, 1), (2, 2), (1, 1)),  # added a test case that fail from test_nnet.py.test_conv_nnet2
        ((10, 5, 28, 28), (10, 5, 5, 5), (1, 1), (2, 2), (1, 1)),  # test precedent but reduced that triger the error
        # Test more than maxThreadsDim0
        ((2, 4, 13, 1050), (3, 4, 10, 11), (1, 1), (1, 1), (1, 1)),
        ((2, 4, 1050, 13), (3, 4, 10, 11), (1, 1), (1, 1), (1, 1))
        ]

    shapes += [((60, 1, 28, 28), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 1 layers
               ((60, 20, 12, 12), (30, 20, 5, 5), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 2 layers
               ((60, 30, 8, 8), (20, 30, 5, 5), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 bprop 1 full
               ((20, 60, 12, 12), (30, 60, 8, 8), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 bprop 2 valid
               # ((1,60,28,28),(20,60,24,24), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 bprop 2 valid
               ((10, 1, 64, 64), (20, 1, 7, 7), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 1 layers
               ((10, 20, 29, 29), (30, 20, 7, 7), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 2 layers
               ((10, 30, 23, 23), (20, 30, 7, 7), (1, 1), (1, 1), (1, 1))  # test_lenet_64 full
               # ((20,10,29,29),(30,10,23,23), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 bprop 1
               # ((1,10,64,64),(20,10,58,58), (1, 1), (1, 1), (1, 1))  # test_lenet_64 bprop 2
               ]
    return shapes


def _test_valid(cls, mode=None, extra_shapes=[], version=[-1]):
    seed_rng()
    shapes = get_valid_shapes()

    verbose = 0

    random = True
    print_ = False
    ones = False
    if ones:
        random = False

    shapes += extra_shapes

    return exec_conv(version, shapes, verbose, random, 'valid',
                     print_=print_, ones=ones, rtol=1.1e-5,
                     theano_mode=mode, cls=cls)


def test_valid():
    for t in _test_valid(None,
                         mode=theano_mode,
                         version=[-1]):
        yield t


def test_gemm_valid():
    extra_shapes = get_shapes2(scales_img=(2, 2), img_stride=(2, 2))
    extra_shapes += get_shapes2(scales_kern=(2, 2), kern_stride=(2, 2))

    for t in _test_valid(cuda.blas.BaseGpuCorrMM,
                         mode=theano_mode.excluding("cudnn"),
                         extra_shapes=extra_shapes):
        yield t


def test_dnn_valid():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    for t in _test_valid(DnnBase, mode=theano_mode.including("cudnn")):
        yield t


def test_dnn_valid_err():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    assert_raises(ValueError, _params_allgood, (1, 2, 4, 4), (1, 1, 2, 2),
                  'valid', theano_mode=theano_mode.including("cudnn"),
                  cls=DnnBase)


def test_default_conv():
    """Just test that we introduce the right GPU convolution
    version.

    """
    img = theano.tensor.ftensor4()
    fil = theano.tensor.ftensor4()

    c = theano.tensor.nnet.conv2d(img, fil)
    f = theano.function([img, fil], c, mode=theano_mode)

    if cuda.dnn.dnn_available():
        assert any([isinstance(a.op, GpuDnnConv)
                    for a in f.maker.fgraph.apply_nodes])
    else:
        assert any([isinstance(a.op, cuda.blas.GpuCorrMM)
                    for a in f.maker.fgraph.apply_nodes])


def _test_full(cls, mode=None, version=[-1], extra_shapes=[],
               test_bigger_kernels=True):
    seed_rng()
    shapes = get_basic_shapes()
    shapes += get_shapes2()
    # test image stride
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(1, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 1))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(2, 2))
    shapes += get_shapes2(scales_img=(2, 2), img_stride=(-1, -1))
    shapes += get_shapes2(scales_img=(2, 2), kern_stride=(-1, -1))

    # test subsample done in a separate fct

    shapes += [
        # other test
        ((2, 1, 2, 2), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
        ((3, 2, 4, 4), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 10, 10), (1, 1, 2, 2), (1, 1), (1, 1), (1, 1)),
        ((1, 1, 4, 4), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 10, 10), (1, 1, 2, 3), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 10, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1)),
        ((4, 1, 20, 10), (1, 1, 2, 10), (1, 1), (1, 1), (1, 1)),
        ((3, 2, 8, 8), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize
        ((3, 2, 8, 6), (4, 2, 4, 4), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize, non-square image
        ((3, 2, 8, 6), (4, 2, 4, 3), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize, non-square image, non-square kern
        ((3, 2, 8, 6), (4, 2, 4, 6), (1, 1), (1, 1), (1, 1)),  # stack, nkern, bsize ,non-square image, non-square kern, kernsize==imgsize on one dim
        ((16, 5, 64, 64), (8, 5, 8, 8), (1, 1), (1, 1), (1, 1)),  # a big one
        ((16, 1, 28, 28), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)),  # MNIST LeNET layer 1
        ((20, 16, 32, 32), (1, 16, 28, 28), (1, 1), (1, 1), (1, 1))  # layer 1 backprop to weights
        ]

    if test_bigger_kernels:
        # Shapes where the kernel is larger than the image in some dimension
        shapes += [
            ((3, 1, 1, 1), (2, 1, 5, 3), (1, 1), (1, 1), (1, 1)),
            ((3, 2, 1, 1), (4, 2, 1, 1), (1, 1), (1, 1), (1, 1)),
            ((3, 2, 4, 4), (4, 2, 2, 6), (1, 1), (1, 1), (1, 1)),
            ((3, 2, 4, 4), (4, 2, 8, 6), (1, 1), (1, 1), (1, 1)),
            ((4, 2, 10, 10), (3, 2, 2, 12), (1, 1), (1, 1), (1, 1))
            ]

    shapes += [((60, 1, 28, 28), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 1 layers
               # ((60, 20, 12, 12),(30, 20, 5, 5), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 2 layers
               ((60, 30, 8, 8), (20, 30, 5, 5), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 bprop 1 full
               # ((20,60,12,12),(30,60,8,8), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 bprop 2 valid
               # ((1,60,28,28),(20,60,24,24), (1, 1), (1, 1), (1, 1)),  # test_lenet_28 bprop 2 valid
               # ((10,1,64,64),(20,1,7,7), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 1 layers
               # ((10,20,29,29),(30,20,7,7), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 2 layers
               ((10, 30, 23, 23), (20, 30, 7, 7), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 full
               # ((20,10,29,29),(30,10,23,23), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 bprop 1
               # ((1,10,64,64),(20,10,58,58), (1, 1), (1, 1), (1, 1)),  # test_lenet_64 bprop 2
               # Test more than maxThreadsDim0
               ((2, 4, 13, 1050), (3, 4, 10, 11), (1, 1), (1, 1), (1, 1)),
               ((2, 4, 1050, 13), (3, 4, 10, 11), (1, 1), (1, 1), (1, 1)),
               ((1, 1, 44800, 1), (6, 1, 1, 1), (1, 1), (1, 1), (1, 1))  # This caused crash
               ]

    verbose = 0
    random = True

    shapes += extra_shapes

    return exec_conv(version, shapes, verbose, random, 'full',
                     theano_mode=mode, cls=cls)


def test_full():

    # If using cuDNN version before v3, only run the tests where the
    # kernels are not larger than the input in any spatial dimension.
    if cuda.dnn.dnn_available() and cuda.dnn.version() < (3000, 3000):
        test_bigger_kernels = False
    else:
        test_bigger_kernels = True

    for t in _test_full(None, mode=theano_mode, version=[-1],
                        test_bigger_kernels=test_bigger_kernels):
        yield t


def test_gemm_full():
    for t in _test_full(cuda.blas.BaseGpuCorrMM,
                        mode=theano_mode.excluding("cudnn")):
        yield t


def test_dnn_full():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

    # If using cuDNN version before v3, only run the tests where the
    # kernels are not larger than the input in any spatial dimension.
    if cuda.dnn.version() < (3000, 3000):
        test_bigger_kernels = False
    else:
        test_bigger_kernels = True

    for t in _test_full(DnnBase, mode=theano_mode.including("cudnn"),
                        test_bigger_kernels=test_bigger_kernels):
        yield t


def _test_subsample(cls, mode, version_valid=[-1], version_full=[-1]):
    seed_rng()
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

    # We put only the version that implement the subsample to make the
    # test faster.
    verbose = 0
    random = True
    print_ = False
    ones = False
    if ones:
        random = False

    for t in exec_conv(version_valid, shapes, verbose, random, 'valid',
                       print_=print_, ones=ones,
                       theano_mode=mode, cls=cls):
        yield t
    for t in exec_conv(version_full, shapes, verbose, random, 'full',
                       print_=print_, ones=ones,
                       theano_mode=mode, cls=cls):
        yield t


def test_subsample():
    for t in _test_subsample(None, theano_mode,
                             version_valid=[-1],
                             version_full=[-1]):
        yield t


def test_gemm_subsample():
    for t in _test_subsample(cuda.blas.BaseGpuCorrMM,
                             theano_mode.excluding("cudnn")):
        yield t


def test_dnn_subsample():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    for t in _test_subsample(DnnBase, theano_mode.including('cudnn')):
        yield t


class TestConv2DGPU(unittest.TestCase):
    conv_ops = (cuda.blas.GpuConv,
                cuda.dnn.DnnBase,
                cuda.blas.BaseGpuCorrMM)

    def test_logical_shapes(self):
        # Logical shapes are not supported anymore, so we check that it
        # raises an Exception.
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
            self.assertRaises(ValueError, tensor.nnet.conv2d,
                              a, kernel_rotated,
                              border_mode='full',
                              image_shape=featshp,
                              filter_shape=kshp_rotated,
                              imshp_logical=featshp_logical[1:],
                              kshp_logical=kshp[2:])

    def test_invalid_input_shape(self):
        """
        Tests that when the shape gived at build time is not the same as
        run time we raise an error
        """
        seed_rng()
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
                                   # ((3, 2, 8, 8), (4, 2, 5, 5), (8, 5)),
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


class TestConvWithPadding(object):
    """test conv ops that support arbitrary padding via border_mode
    note that in order to make the yield work, we can not subclass from
    unittest.TestCase
    """

    @staticmethod
    def gemm_conv_op(img, kern, border_mode):
        kern = theano.sandbox.cuda.basic_ops.gpu_contiguous(
            kern[:, :, ::-1, ::-1])
        y = theano.sandbox.cuda.blas.GpuCorrMM(border_mode=border_mode)(
            img, kern)
        return y

    conv_ops = []

    @classmethod
    def setup_class(cls):
        cls.conv_ops.append(cls.gemm_conv_op)
        if cuda.dnn.dnn_available():
            cls.conv_ops.append(cuda.dnn.dnn_conv)

    def test_invalid_arg(self):
        img = theano._asarray(numpy.empty((1, 1, 1, 1)), dtype='float32')
        kern = theano._asarray(numpy.empty((1, 1, 1, 1)), dtype='float32')
        for i in self.conv_ops:
            assert_raises(ValueError, i, img, kern,
                          border_mode=(-1, 0))
            assert_raises(ValueError, i, img, kern,
                          border_mode=(0, -1))
            assert_raises(ValueError, i, img, kern,
                          border_mode='not border')

    def _run_onecase(self, img_shape, kern_shape, padding, op):
        npy_img = numpy.random.rand(*img_shape).astype('float32')
        npy_kern = numpy.random.rand(*kern_shape).astype('float32')
        img = theano._asarray(npy_img, dtype='float32')
        kern = theano.shared(npy_kern)
        border_mode = padding
        cpuval = py_conv(npy_img, npy_kern, border_mode, (1, 1))
        X = tensor.ftensor4()
        Y = op(X, kern, border_mode=border_mode)
        func = theano.function([X], Y, mode=theano_mode)
        gpuval = numpy.asarray(func(img))
        assert_allclose(cpuval, gpuval, rtol=1e-5, atol=1e-5)

    def test_numeric_value(self):
        params = [
            ((5, 10, 4, 4), (12, 10, 4, 4), (2, 1)),
            ((5, 10, 8, 8), (12, 10, 4, 4), 3),
            ((5, 10, 6, 8), (12, 10, 3, 4), 'full'),
            ((5, 10, 9, 6), (12, 10, 9, 4), 'valid')
        ]
        for img_shape, kern_shape, padding in params:
            for op in self.conv_ops:
                yield self._run_onecase, img_shape, kern_shape, padding, op


def gemm_directly(bs, ch, nf, rImg1, rImg2, rFlt1, rFlt2, subsx, subsy,
                  direction):
    ishape = (bs, ch, rImg1, rImg2)
    kshape = (nf, ch, rFlt1, rFlt2)
    subsample = (subsx, subsy)

    npy_img = theano._asarray(numpy.random.rand(*ishape), dtype='float32')
    npy_kern = theano._asarray(numpy.random.rand(*kshape), dtype='float32')

    if direction == 'fprop':
        i = cuda.CudaNdarrayType(
            broadcastable=[sh == 1 for sh in npy_img.shape])()
        k = cuda.CudaNdarrayType(
            broadcastable=[sh == 1 for sh in npy_kern.shape])()

        cpuval = py_conv(npy_img, npy_kern, 'valid', subsample)
        op = theano.sandbox.cuda.blas.GpuCorrMM(border_mode='valid',
                                                subsample=subsample)(i, k)
        f = theano.function([i, k], op, mode=theano_mode)
        gpuval = f(npy_img, npy_kern[:, :, ::-1, ::-1])
    elif direction == 'bprop img':
        i = cuda.CudaNdarrayType(
            broadcastable=[sh == 1 for sh in
                           npy_kern.transpose(1, 0, 2, 3).shape])()
        k = cuda.CudaNdarrayType(
            broadcastable=[sh == 1 for sh in npy_img.shape])()

        cpuval = py_conv(npy_img, npy_kern, 'full', subsample)
        op = theano.sandbox.cuda.blas.GpuCorrMM_gradInputs(
            border_mode='valid', subsample=subsample)(i, k)
        f = theano.function([i, k], op, mode=theano_mode)
        gpuval = f(npy_kern.transpose(1, 0, 2, 3), npy_img)
    elif direction == 'bprop kern':
        i = cuda.CudaNdarrayType(
            broadcastable=[sh == 1 for sh in
                           npy_img.transpose(1, 0, 2, 3).shape])()
        k = cuda.CudaNdarrayType(
            broadcastable=[sh == 1 for sh in
                           npy_kern.transpose(1, 0, 2, 3).shape])()

        cpuval = py_conv(npy_img, npy_kern, 'valid', subsample)
        op = theano.sandbox.cuda.blas.GpuCorrMM_gradWeights(
            border_mode='valid', subsample=subsample)(i, k)
        f = theano.function([i, k], op, mode=theano_mode)
        gpuval = numpy.array(f(
            npy_img.transpose(1, 0, 2, 3),
            npy_kern.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1])
            ).transpose(1, 0, 2, 3)

    assert_allclose(cpuval, gpuval, rtol=1e-4)


def test_gemm_directly():
    for bs in range(1, 5):
        for ch in range(1, 4):
            for nf in range(1, 4):
                for rImg1 in range(5, 9):
                    for rImg2 in range(5, 9):
                        for rFlt1 in range(2, 4):
                            for rFlt2 in range(2, 4):
                                for direction in ['bprop img', 'bprop kern']:
                                    yield (gemm_directly, bs, ch, nf, rImg1,
                                           rImg2, rFlt1, rFlt2, 1, 1,
                                           direction)

                                for subsx in range(1, 3):
                                    for subsy in range(1, 3):
                                        yield (gemm_directly, bs, ch, nf,
                                               rImg1, rImg2, rFlt1, rFlt2,
                                               subsx, subsy, 'fprop')


def gemm_op(mode, subsample):
    return theano.sandbox.cuda.blas.GpuCorrMM(mode, subsample)


def dnn_op(mode, subsample):
    def f(img, kern):
        return dnn_conv(img, kern, border_mode=mode, conv_mode='cross',
                        subsample=subsample)
    return f


def conv_grad(mode, bs, ch, nf, rImg1, rImg2, rFlt1, rFlt2, subsample, op):
    ishape = (bs, ch, rImg1, rImg2)
    kshape = (nf, ch, rFlt1, rFlt2)

    npy_img = theano._asarray(numpy.random.rand(*ishape), dtype='float32')
    npy_kern = theano._asarray(numpy.random.rand(*kshape), dtype='float32')

    i = cuda.CudaNdarrayType(
        broadcastable=[sh == 1 for sh in npy_img.shape])()
    k = cuda.CudaNdarrayType(
        broadcastable=[sh == 1 for sh in npy_kern.shape])()

    # TODO: also test custom pad values
    corr_op = op(mode, subsample)(i, k)
    conv_op = tensor.nnet.conv2d(i, k[:, :, ::-1, ::-1],
                                 border_mode=mode, subsample=subsample)
    conv_op_di = theano.grad(conv_op.sum(), i)
    conv_op_dk = theano.grad(conv_op.sum(), k)
    corr_op_di = theano.grad(corr_op.sum(), i)
    corr_op_dk = theano.grad(corr_op.sum(), k)
    outputs = [corr_op, conv_op,
               corr_op_di, conv_op_di,
               corr_op_dk, conv_op_dk]

    conv_op_dik = theano.grad(conv_op_di.sum(), k)
    conv_op_dki = theano.grad(conv_op_dk.sum(), i)
    corr_op_dik = theano.grad(corr_op_di.sum(), k)
    corr_op_dki = theano.grad(corr_op_dk.sum(), i)
    outputs.extend([corr_op_dik, conv_op_dik,
                    corr_op_dki, conv_op_dki])

    if not theano.config.blas.ldflags:
        # Some of the operations are not transferred to the GPU,
        # and withoug BLAS, the abstract Op will not be optimized
        # to CorrMM either, so we have to accept the use of the
        # slow Python convolution in that case.
        mode = theano_mode.excluding('AbstractConvCheck')
    else:
        mode = theano_mode

    f = theano.function([i, k], outputs, mode=mode)

    allvals = f(npy_img, npy_kern)

    for a, b, oa, ob, p in zip(allvals[::2], allvals[1::2],
                               outputs[::2], outputs[1::2],
                               ('top', 'dtop/dbottom', 'dtop/dweight',
                                'dtop/dbottom/dweight', 'dtop/dweight/dbottom')):
        assert oa.type.broadcastable[:2] == ob.type.broadcastable[:2]

        assert_allclose(a, b, rtol=1e-4)


def test_conv_grads():
    if (not cuda.dnn.dnn_available() or
            cuda.device_properties(cuda.active_device_number())['major'] < 3):
        ops = [gemm_op]
    else:
        ops = [gemm_op, dnn_op]
    for mode in 'valid', 'full':
        for bs in [1, 5]:
            for ch in [4]:
                for nf in [3]:
                    for rImg1 in [2, 5]:
                        for rImg2 in [2, 8]:
                            for rFlt1 in [1, 2]:
                                for rFlt2 in [1, 2]:
                                    for subsample in (1, 1), (1, 2), (2, 2):
                                        for op in ops:
                                            yield (conv_grad, mode, bs, ch, nf,
                                                   rImg1, rImg2, rFlt1, rFlt2,
                                                   subsample, op)


def benchmark():

    shapes_valid = [
        # test_lenet_28 shape
        ((20, 60, 12, 12), (30, 60, 8, 8), (1, 1), (1, 1), (1, 1)),  # valid
        ((60, 20, 12, 12), (30, 20, 5, 5), (1, 1), (1, 1), (1, 1)),  # valid
        ((60, 1, 28, 28), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)),  # valid
        ((1, 60, 28, 28), (20, 60, 24, 24), (1, 1), (1, 1), (1, 1)),  # valid
        # test_lenet_32 shape
        ((20, 60, 14, 14), (30, 60, 10, 10), (1, 1), (1, 1), (1, 1)),  # valid
        ((60, 20, 14, 14), (30, 20, 5, 5), (1, 1), (1, 1), (1, 1)),  # valid
        ((60, 1, 32, 32), (20, 1, 5, 5), (1, 1), (1, 1), (1, 1)),  # valid
        ((1, 60, 32, 32), (20, 60, 28, 28), (1, 1), (1, 1), (1, 1)),  # valid
        # test_lenet_64 shape
        ((10, 20, 29, 29), (30, 20, 7, 7), (1, 1), (1, 1), (1, 1)),  # valid
        ((20, 10, 29, 29), (30, 10, 23, 23), (1, 1), (1, 1), (1, 1)),  # valid
        ((10, 1, 64, 64), (20, 1, 7, 7), (1, 1), (1, 1), (1, 1)),  # valid
        ((1, 10, 64, 64), (20, 10, 58, 58), (1, 1), (1, 1), (1, 1)),  # valid
        # test_lenet_108 shape
        ((10, 20, 51, 51), (30, 20, 7, 7), (1, 1), (1, 1), (1, 1)),  # valid
        ((20, 10, 51, 51), (30, 10, 45, 45), (1, 1), (1, 1), (1, 1)),  # valid
        ((10, 1, 108, 108), (20, 1, 7, 7), (1, 1), (1, 1), (1, 1)),  # valid
        ((1, 10, 108, 108), (20, 10, 102, 102), (1, 1), (1, 1), (1, 1)),  # valid
        # test_lenet_256 shape
        ((2, 20, 124, 124), (30, 20, 9, 9), (1, 1), (1, 1), (1, 1)),  # valid
        ((20, 2, 124, 124), (30, 2, 116, 116), (1, 1), (1, 1), (1, 1)),  # valid
        ((2, 1, 256, 256), (20, 1, 9, 9), (1, 1), (1, 1), (1, 1)),  # valid
        ((1, 2, 256, 256), (20, 2, 248, 248), (1, 1), (1, 1), (1, 1))  # valid
        ]

    shapes_full = [
        # test_lenet_28 shape
        ((60, 30, 8, 8), (20, 30, 5, 5), (1, 1), (1, 1), (1, 1)),  # full
        # test_lenet_32 shape
        ((60, 30, 10, 10), (20, 30, 5, 5), (1, 1), (1, 1), (1, 1)),  # full conv_full_patch_stack_padded' N=1
        # test_lenet_64 shape
        ((10, 30, 23, 23), (20, 30, 7, 7), (1, 1), (1, 1), (1, 1)),  # full conv_full_patch_stack_padded' N=3
        # test_lenet_108 shape
        ((10, 30, 45, 45), (20, 30, 7, 7), (1, 1), (1, 1), (1, 1)),  # full 'conv_full_patch_stack_padded' N=9
        # test_lenet_256 shape
        ((2, 30, 116, 116), (20, 30, 9, 9), (1, 1), (1, 1), (1, 1))  # full conv_reference_full
        ]

    version = [-1]
    verbose = 1
    random = True

    for t in exec_conv(version, shapes_valid, verbose, random, 'valid',
                       print_=None, rtol=1e-3):
        t[0](*t[1:])
    for t in exec_conv(version, shapes_full, verbose, random, 'full'):
        t[0](*t[1:])


def test_stack_rows_segfault_070312():
    seed_rng()
    # 07/03/2012
    # Running this unittest with cuda-memcheck exposes an illegal read.
    # THEANO_FLAGS=device=gpu cuda-memcheck nosetests \
    # test_conv_cuda_ndarray.py:test_stack_rows_segfault_070312
    img = theano.shared(numpy.random.rand(1, 80, 96, 96).astype('float32'))
    kern = theano.shared(numpy.random.rand(1, 80, 9, 9).astype('float32'))
    out = theano.shared(numpy.random.rand(1, 2, 2, 3).astype('float32'))
    op = theano.tensor.nnet.conv.ConvOp(imshp=(80, 96, 96), kshp=(9, 9),
                                        nkern=1, bsize=1)
    f = theano.function([], [], updates=[(out, op(img, kern))], mode=theano_mode)
    f()
