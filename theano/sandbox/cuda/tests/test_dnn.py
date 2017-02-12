from __future__ import absolute_import, print_function, division
from collections import OrderedDict
import logging
import os
import sys

from nose.plugins.skip import SkipTest
from nose_parameterized import parameterized
from itertools import chain, product
import six.moves.cPickle as pickle
from six import StringIO
from six import reraise

import numpy

import theano
import theano.tensor as T
import theano.tests.unittest_tools as utt
from theano.tensor.signal.pool import pool_2d, pool_3d
from theano.tensor.signal.pool import Pool, MaxPoolGrad, AveragePoolGrad
from theano.tensor.nnet.abstract_conv import get_conv_output_shape
from theano.tensor.nnet import bn
import theano.sandbox.cuda.dnn as dnn
from theano.sandbox.cuda.basic_ops import GpuAllocEmpty, gpu_alloc_empty
from theano.sandbox.cuda import float32_shared_constructor as shared

from . import test_nnet

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda
if not cuda.cuda_available:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode().excluding('gpu')
    mode_with_gpu.check_py_code = False
    mode_without_gpu.check_py_code = False


def test_dnn_conv_desc_merge():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    img_shp = T.as_tensor_variable(
        numpy.asarray([2, 1, 8, 8]).astype('int64'))
    kern_shp = T.as_tensor_variable(
        numpy.asarray([3, 1, 2, 2]).astype('int64'))
    desc1 = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(2, 2),
                               conv_mode='conv')(img_shp, kern_shp)
    desc2 = dnn.GpuDnnConvDesc(border_mode='full', subsample=(1, 1),
                               conv_mode='cross')(img_shp, kern_shp)
    # CDataType is not DeepCopyable so this will crash if we don't use
    # borrow=True
    f = theano.function([], [theano.Out(desc1, borrow=True),
                             theano.Out(desc2, borrow=True)],
                        mode=mode_with_gpu)

    d1, d2 = f()

    # This will be the case if they are merged, which would be bad.
    assert d1 != d2

    desc1v2 = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(2, 2),
                                 conv_mode='conv')(img_shp, kern_shp)
    f = theano.function([], [theano.Out(desc1, borrow=True),
                             theano.Out(desc1v2, borrow=True)],
                        mode=mode_with_gpu)
    assert len([n for n in f.maker.fgraph.apply_nodes
                if isinstance(n.op, dnn.GpuDnnConvDesc)]) == 1

    # CDATA type don't equal even if they represent the same object
    # So we can't use debugmode with it.
    if theano.config.mode not in ["DebugMode", "DEBUG_MODE"]:
        d1, d2 = f()

        # They won't be equal if they aren't merged.
        assert d1 == d2


def test_dnn_conv_merge():
    """This test that we merge correctly multiple dnn_conv.

    This can is more difficult due to GpuEmptyAlloc that aren't
    merged.

    """
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    img_shp = [2, 5, 6, 8]
    kern_shp = [3, 5, 5, 6]
    img = T.ftensor4('img')
    kern = T.ftensor4('kern')
    out = T.ftensor4('out')
    desc = dnn.GpuDnnConvDesc(
        border_mode='valid')(img.shape, kern.shape)

    # Test forward op
    o1 = dnn.dnn_conv(img, kern)
    o2 = dnn.dnn_conv(img, kern)
    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
    d1, d2 = f(numpy.random.rand(*img_shp).astype('float32'),
               numpy.random.rand(*kern_shp).astype('float32'))
    topo = f.maker.fgraph.toposort()
    assert len([n for n in topo if isinstance(n.op, dnn.GpuDnnConv)]) == 1

    # Test grad w op
    o1 = dnn.GpuDnnConvGradW()(img, kern, out, desc)
    o2 = dnn.GpuDnnConvGradW()(img, kern, out, desc)
    f = theano.function([img, kern, out], [o1, o2], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len([n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradW)]) == 1

    # Test grad i op
    o1 = dnn.GpuDnnConvGradI()(img, kern, out, desc)
    o2 = dnn.GpuDnnConvGradI()(img, kern, out, desc)
    f = theano.function([img, kern, out], [o1, o2], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    assert len([n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradI)]) == 1


def test_dnn_conv_inplace():
    """This test that we have inplace work correctly even when
    GpuAllocEmpty get merged together.

    """
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    img_shp = [2, 5, 6, 8]
    kern_shp = [3, 5, 5, 6]
    img = T.ftensor4('img')
    kern = T.ftensor4('kern')
    out = T.ftensor4('out')
    desc1 = dnn.GpuDnnConvDesc(border_mode='valid', conv_mode='conv')(
        img.shape, kern.shape)
    desc2 = dnn.GpuDnnConvDesc(
        border_mode='valid', conv_mode='cross')(img.shape, kern.shape)

    # Test forward op
    o1 = dnn.dnn_conv(img, kern, conv_mode='conv')
    o2 = dnn.dnn_conv(img, kern, conv_mode='cross')
    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
    d1, d2 = f(numpy.random.rand(*img_shp).astype('float32'),
               numpy.random.rand(*kern_shp).astype('float32'))
    topo = f.maker.fgraph.toposort()
    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConv)]
    assert len(convs) == 2
    assert all([node.op.inplace for node in convs])
    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2

    # Test grad w op
    out = gpu_alloc_empty(*kern.shape)
    o1 = dnn.GpuDnnConvGradW()(img, kern, out, desc1)
    o2 = dnn.GpuDnnConvGradW()(img, kern, out, desc2)
    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradW)]
    assert len(convs) == 2
    assert all([node.op.inplace for node in convs])
    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2

    # Test grad i op
    out = gpu_alloc_empty(*img.shape)
    o1 = dnn.GpuDnnConvGradI()(img, kern, out, desc1)
    o2 = dnn.GpuDnnConvGradI()(img, kern, out, desc2)
    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradI)]
    assert len(convs) == 2
    assert all([node.op.inplace for node in convs])
    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2


def test_pooling():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

    # 'average_exc_pad' is disabled for versions < 4004
    if cuda.dnn.version() < (4004, 4004):
        modes = ('max', 'average_inc_pad')
    else:
        modes = ('max', 'average_inc_pad', 'average_exc_pad')

    x = T.ftensor4()
    for mode, pad in product(modes,
                             ((0, 0), (1, 0), (0, 1), (2, 3), (3, 2))):
        if pad != (0, 0) and mode == 'average_exc_pad':
            # Not implemented
            continue

        for ws in (4, 2, 5):
            for stride in (2, 3):
                if stride > ws:
                    continue
                if pad[0] > stride or pad[1] > stride:
                    # Not implemented
                    continue
                # We will check that the opt introduced it.
                out = pool_2d(x, (ws, ws),
                              stride=(stride, stride),
                              ignore_border=True,
                              pad=pad, mode=mode)
                mode_without_gpu2 = mode_without_gpu.including()
                mode_without_gpu2.check_isfinite = False

                # GPU implementation
                f_gpu = theano.function([x], out, mode=mode_with_gpu)
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                            for node in f_gpu.maker.fgraph.apply_nodes])

                # CPU implementation
                f_cpu = theano.function([x], out, mode=mode_without_gpu2)
                assert not any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                                for node in f_cpu.maker.fgraph.apply_nodes])
                assert any([isinstance(node.op, Pool)
                            for node in f_cpu.maker.fgraph.apply_nodes])

                for shp in [(1, 10, 100, 100),
                            (1, 3, 99, 99),
                            (32, 1, 147, 197),
                            ]:
                    data = numpy.random.normal(0, 1, shp).astype("float32")
                    a = f_cpu(data).__array__()
                    b = f_gpu(data).__array__()
                    utt.assert_allclose(a, b)

        # Test the grad
        for shp in [(1, 1, 2, 2),
                    (1, 1, 3, 3)]:
            data = numpy.random.normal(0, 1, shp).astype("float32") * 10

            ws = 2
            stride = 2
            if pad[0] > stride or pad[1] > stride:
                # Not implemented
                continue

            # This tests the CPU grad + opt + GPU implementation
            def fn(x):
                return pool_2d(x, (ws, ws), ignore_border=True,
                               pad=pad, mode=mode)
            utt.verify_grad(fn, [data], mode=mode_with_gpu)
            # Confirm that the opt would have inserted it.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                        for node in fg.maker.fgraph.toposort()])

            # Test the GPU grad + GPU implementation
            def fn(x):
                dnn_op = cuda.dnn.dnn_pool(
                    x, ws=(ws, ws),
                    stride=(stride, stride),
                    pad=pad,
                    mode=mode)
                return dnn_op
            utt.verify_grad(fn, [data], mode=mode_with_gpu)
            # Confirm that we get the good op.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                        for node in fg.maker.fgraph.toposort()])


def test_pooling_with_tensor_vars():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    x = T.ftensor4()
    ws = theano.shared(numpy.array([2, 2], dtype='int32'))
    stride = theano.shared(numpy.array([1, 1], dtype='int32'))
    pad = theano.shared(numpy.array([0, 0], dtype='int32'))
    mode = 'max'

    def fn(x):
        dnn_op = cuda.dnn.dnn_pool(
            x, ws=ws,
            stride=stride,
            pad=pad,
            mode=mode)
        return dnn_op

    for shp in [(1, 1, 2, 2),
                (1, 1, 3, 3)]:
        data = numpy.random.normal(0, 1, shp).astype("float32") * 10
        theano.tests.unittest_tools.verify_grad(
            fn, [data], mode=mode_with_gpu)

    mode_without_gpu2 = mode_without_gpu.including()
    mode_without_gpu2.check_isfinite = False

    # GPU implementation
    f_gpu = theano.function([x], fn(x), mode=mode_with_gpu)
    assert any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                for node in f_gpu.maker.fgraph.apply_nodes])

    # CPU implementation
    out_cpu = pool_2d(x, ws, ignore_border=True, stride=stride, pad=pad, mode=mode)
    f_cpu = theano.function([x], out_cpu, mode=mode_without_gpu2)
    assert not any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                   for node in f_cpu.maker.fgraph.apply_nodes])
    assert any([isinstance(node.op, Pool)
                for node in f_cpu.maker.fgraph.apply_nodes])

    i = 1
    for shp in [(1, 10, 100, 100),
                (1, 3, 99, 99),
                (32, 1, 147, 197)]:
        data = numpy.random.normal(0, 1, shp).astype("float32")

        # Change the window size dynamically
        ws.set_value(numpy.array([i, i]).astype('int32'))
        a = f_gpu(data).__array__()
        b = f_cpu(data).__array__()
        utt.assert_allclose(a, b)
        i += 1


def test_old_pool_interface():
    if not cuda.dnn.dnn_available() or cuda.dnn.version() > (5000, 5000):
        raise SkipTest(cuda.dnn.dnn_available.msg)

    testfile_dir = os.path.dirname(os.path.realpath(__file__))
    fname = 'old_pool_interface.pkl'
    with open(os.path.join(testfile_dir, fname), 'rb') as fp:
        try:
            pickle.load(fp)
        except ImportError:
            # Windows sometimes fail with nonsensical errors like:
            #   ImportError: No module named type
            #   ImportError: No module named copy_reg
            # when "type" and "copy_reg" are builtin modules.
            if sys.platform == 'win32':
                exc_type, exc_value, exc_trace = sys.exc_info()
                reraise(SkipTest, exc_value, exc_trace)
            raise


def test_pooling3d():
    # cuDNN 3d pooling requires cuDNN v3. Don't test if the cuDNN version is
    # too old.
    if not cuda.dnn.dnn_available() or cuda.dnn.version() < (3000, 3000):
        raise SkipTest(cuda.dnn.dnn_available.msg)

    # We force the FAST_RUN as we don't want the reference to run in DebugMode.
    mode_without_gpu_ref = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpu')

    # 'average_exc_pad' is disabled for versions < 4004
    if cuda.dnn.version() < (4004, 4004):
        modes = ('max', 'average_inc_pad')
    else:
        modes = ('max', 'average_inc_pad', 'average_exc_pad')

    x = T.ftensor5()
    for mode, pad in product(modes,
                             ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                              (2, 3, 2), (3, 2, 2), (2, 2, 3))):
        if pad != (0, 0, 0) and mode == 'average_exc_pad':
            # Not implemented
            continue

        for ws in (4, 2, 5):
            for stride in (2, 3):
                if stride > ws:
                    continue
                if pad[0] > stride or pad[1] > stride or pad[2] > stride:
                    # Not implemented
                    continue
                out = pool_3d(x, (ws, ws, ws),
                              stride=(stride, stride, stride),
                              ignore_border=True,
                              pad=pad, mode=mode)

                # GPU implementation
                f_gpu = theano.function([x], out, mode=mode_with_gpu)
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                            for node in f_gpu.maker.fgraph.apply_nodes])

                # CPU implementation
                f_cpu = theano.function([x], out, mode=mode_without_gpu_ref)
                assert not any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                                for node in f_cpu.maker.fgraph.apply_nodes])
                assert any([isinstance(node.op, Pool)
                            for node in f_cpu.maker.fgraph.apply_nodes])

                for shp in [(1, 5, 50, 20, 50),
                            (1, 3, 99, 99, 29),
                            (2, 1, 147, 97, 37),
                            ]:
                    data = numpy.random.normal(0, 1, shp).astype("float32")
                    a = f_cpu(data).__array__()
                    b = f_gpu(data).__array__()
                    utt.assert_allclose(a, b,
                                        atol=numpy.finfo(numpy.float32).eps)

        # Test the grad
        for shp in [(1, 1, 2, 2, 2),
                    (1, 1, 3, 3, 3),
                    (1, 1, 3, 3, 4),
                    (1, 1, 3, 4, 3),
                    (1, 1, 4, 3, 3),
                    (1, 1, 4, 4, 4),
                    (1, 1, 5, 5, 5)]:
            data = numpy.random.normal(0, 1, shp).astype("float32") * 10

            ws = 2
            stride = 2
            if pad[0] > stride or pad[1] > stride or pad[2] > stride:
                # Not implemented
                continue

            # Test the GPU grad + GPU implementation
            def fn(x):
                dnn_op = cuda.dnn.dnn_pool(
                    x, ws=(ws, ws, ws),
                    stride=(stride, stride, stride),
                    pad=pad,
                    mode=mode)
                return dnn_op
            utt.verify_grad(fn, [data], mode=mode_with_gpu)
            # Confirm that we get the good op.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                        for node in fg.maker.fgraph.toposort()])


def test_pooling_opt():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

    # 2D pooling
    x = T.fmatrix()

    f = theano.function(
        [x],
        pool_2d(x, ws=(2, 2), mode='average_inc_pad', ignore_border=True),
        mode=mode_with_gpu)

    assert any([isinstance(n.op, cuda.dnn.GpuDnnPool)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10), dtype='float32'))

    # gradient of 2D pooling
    f = theano.function(
        [x],
        T.grad(pool_2d(x, ws=(2, 2), mode='average_inc_pad',
                       ignore_border=True).sum(), x),
        mode=mode_with_gpu.including("cudnn"))

    assert any([isinstance(n.op, cuda.dnn.GpuDnnPoolGrad)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10), dtype='float32'))

    # Test sum pooling
    f = theano.function(
        [x],
        pool_2d(x, ws=(2, 3), mode='sum',
                ignore_border=True),
        mode=mode_with_gpu)

    assert any([isinstance(n.op, dnn.GpuDnnPool)
                for n in f.maker.fgraph.toposort()])
    data = numpy.random.rand(10, 10).astype('float32')
    f(data)

    # 3D pooling
    x = T.ftensor3()

    f = theano.function(
        [x],
        pool_3d(x, ws=(2, 2, 2), mode='average_inc_pad', ignore_border=True),
        mode=mode_with_gpu)

    assert any([isinstance(n.op, cuda.dnn.GpuDnnPool)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10, 10), dtype='float32'))

    # gradient of 3D pooling
    f = theano.function(
        [x],
        T.grad(pool_3d(x, ws=(2, 2, 2), mode='average_inc_pad',
                       ignore_border=True).sum(), x),
        mode=mode_with_gpu.including("cudnn"))

    assert any([isinstance(n.op, cuda.dnn.GpuDnnPoolGrad)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10, 10), dtype='float32'))


def test_pooling_opt_arbitrary_dimensions():
    # test if input with an arbitrary number of non-pooling dimensions
    # is correctly reshaped to run on the GPU

    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

    # 'average_exc_pad' is disabled for versions < 4004
    if cuda.dnn.version() < (4004, 4004):
        modes = ('max', 'average_inc_pad')
    else:
        modes = ('max', 'average_inc_pad', 'average_exc_pad')

    for n_non_pool_dims in (0, 1, 2, 3):
        for ws in ((2, 2), (3, 3, 3)):
            # create input shape: non-pooling dimensions
            # followed by 2 or 3 pooling dimensions
            shp = tuple(range(2, 2 + n_non_pool_dims)) + tuple(range(5, 5 + len(ws)))
            data = numpy.random.normal(0, 1, shp).astype('float32')
            input = shared(data)

            for mode in modes:
                out_pool = Pool(ndim=len(ws), mode=mode, ignore_border=True)(input, ws)
                out_pool_grad = T.grad(T.sum(out_pool), wrt=input)
                out = [out_pool, out_pool_grad]

                # run on GPU
                fg = theano.function([], out, mode=mode_with_gpu)
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                           for node in fg.maker.fgraph.toposort()])
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                           for node in fg.maker.fgraph.toposort()])
                res_gpu = fg()

                # run on CPU
                fc = theano.function([], out, mode=mode_without_gpu)
                assert any([isinstance(node.op, Pool)
                           for node in fc.maker.fgraph.toposort()])
                if mode == 'max':
                    assert any([isinstance(node.op, MaxPoolGrad)
                               for node in fc.maker.fgraph.toposort()])
                else:
                    assert any([isinstance(node.op, AveragePoolGrad)
                               for node in fc.maker.fgraph.toposort()])
                res_cpu = fg()

                # check for similarity
                utt.assert_allclose(res_gpu[0], res_cpu[0])
                utt.assert_allclose(res_gpu[1], res_cpu[1])


def test_pooling_empty_batch():
    img_shp = (0, 5, 6, 8)
    img = T.ftensor4('img')

    o = dnn.dnn_pool(img, (2, 2), (2, 2))
    f = theano.function([img], o, mode=mode_with_gpu)
    d = f(numpy.random.rand(*img_shp).astype('float32'))
    assert d.shape == (0, 5, 3, 4)

    g = T.grad(T.sum(o), wrt=img)
    f = theano.function([img], g, mode=mode_with_gpu)
    d = f(numpy.random.rand(*img_shp).astype('float32'))
    # Not sure what to assert, it should just pass, that's all.
    assert d.shape == (0, 5, 6, 8)


class test_DnnSoftMax(test_nnet.test_SoftMax):
    gpu_op = dnn.GpuDnnSoftmax
    gpu_grad_op = dnn.GpuDnnSoftmaxGrad
    mode = mode_with_gpu
    do_0 = False
    topo_idx = -3

    def setUp(self):
        if not cuda.dnn.dnn_available():
            raise SkipTest(cuda.dnn.dnn_available.msg)
        utt.seed_rng()

    def test_dnn_softmax_grad(self):
        softmax_op = dnn.GpuDnnSoftmax('bc01', 'accurate', 'channel')

        x_val = numpy.random.normal(0, 1, (3, 4, 2, 5)).astype('float32')
        x_val2 = numpy.random.normal(0, 1, (3, 4, 1, 1)).astype('float32')

        utt.verify_grad(softmax_op, [x_val], mode=mode_with_gpu)

        # Gradient is broken for (n, c, 1, 1) in v3 rc1
        if cuda.dnn.version() != (3000, 3000):
            utt.verify_grad(softmax_op, [x_val2], mode=mode_with_gpu)

    def test_local_softmax_dnn_grad(self):
        """
        Check for optimization error when grad of summed
        softmax is taken over tensor with fixed shape.
        """
        x = T.fvector('x')
        xp = x.reshape((5, 5))
        y = T.nnet.softmax(xp.flatten()).sum()
        g = T.grad(y, x)
        f = theano.function(inputs=[x], outputs=g, mode=self.mode)
        assert(any(n for n in f.maker.fgraph.toposort() if
                   isinstance(n.op, dnn.GpuDnnSoftmaxGrad)))

    def test_cudnn_softmax_grad_opt(self):
        # Verify that the SoftmaxGrad -> GpuDnnSoftmaxGrad optimization is
        # applied when cudnn is required
        y = T.fvector('y')
        f = theano.function(
            [y],
            T.grad(T.nnet.softmax(y).mean(), y),
            mode=mode_with_gpu
        )
        sorted_f = f.maker.fgraph.toposort()
        val = numpy.random.rand(5).astype('float32')
        out_dnn = f(val)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.sandbox.cuda.dnn.GpuDnnSoftmaxGrad
                    )]) == 1)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.tensor.nnet.SoftmaxGrad
                    )]) == 0)

        # Verify that the SoftmaxGrad -> GpuDnnSoftmaxGrad optimization is not
        # applied when cudnn is excluded or not available
        mode_wo_cudnn = mode_with_gpu.excluding("cudnn")
        y = T.fvector('y')
        f = theano.function(
            [y],
            T.grad(T.nnet.softmax(y).mean(), y),
            mode=mode_wo_cudnn
        )
        sorted_f = f.maker.fgraph.toposort()
        out_cpu = f(val)
        utt.assert_allclose(out_dnn, out_cpu)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.sandbox.cuda.dnn.GpuDnnSoftmaxGrad
                    )]) == 0)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.tensor.nnet.SoftmaxGrad
                    )]) == 1)

        # Verify that the SoftmaxGrad -> GpuDnnSoftmaxGrad do not
        # crash with manual graph
        y = T.fvector('y')
        o = theano.tensor.nnet.SoftmaxGrad()(y, y * 2)
        f = theano.function([y], o, mode=mode_with_gpu)
        sorted_f = f.maker.fgraph.toposort()
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.sandbox.cuda.dnn.GpuDnnSoftmaxGrad
                    )]) == 1)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.tensor.nnet.SoftmaxGrad
                    )]) == 0)

    def test_log_softmax(self):
        # This is a test for an optimization that depends on cuDNN v3 or
        # more recent. Don't test if the cuDNN version is too old.
        if cuda.dnn.version() < (3000, 3000):
            raise SkipTest("Log-softmax is only in cudnn v3+")

        x = T.ftensor4()
        softmax_out = dnn.GpuDnnSoftmax('bc01', 'accurate', 'channel')(x)
        log_out = T.log(T.as_tensor_variable(softmax_out))

        f = theano.function([x], log_out, mode=mode_with_gpu)

        # Ensure that the optimization has been applied
        dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                             isinstance(n.op, cuda.dnn.GpuDnnSoftmax)]
        assert len(dnn_softmax_nodes) == 1
        assert dnn_softmax_nodes[0].op.algo == "log"

        # Ensure that the output of the function is valid
        input_shapes = [(3, 4, 5, 6),
                        (1025, 2, 3, 4),
                        (2, 1025, 3, 4),
                        (2, 3, 1025, 4),
                        (2, 3, 4, 1025),
                        (66000, 2, 3, 4),
                        (2, 66000, 3, 4),
                        (2, 3, 66000, 4),
                        (2, 3, 4, 66000)]

        for inp_shape in input_shapes:
            input_val = numpy.random.normal(0, 1, inp_shape).astype("float32")

            out = f(input_val)
            expected_out = numpy.log(
                numpy.exp(input_val) /
                numpy.exp(input_val).sum(1)[:, None, :, :])

            utt.assert_allclose(out, expected_out)

    def test_log_softmax2(self):
        # Test that the op LogSoftmax is correctly replaced by the op
        # DnnSoftmax with the 'log' mode.

        # Compile a reference function, on the CPU, to be used to validate the
        # results of the other function.
        x = T.fmatrix()
        f_ref = theano.function([x], T.nnet.LogSoftmax()(x))

        # Build the first graph and ensure that the optimization is applied
        log_softmax_out = T.nnet.LogSoftmax()(x)
        f = theano.function([x], log_softmax_out, mode=mode_with_gpu)

        dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                             isinstance(n.op, cuda.dnn.GpuDnnSoftmax)]
        assert len(dnn_softmax_nodes) == 1
        assert dnn_softmax_nodes[0].op.algo == "log"

        # Compare the output of the function with the reference function
        inp = numpy.random.normal(0, 1, (5, 6)).astype("float32")
        utt.assert_allclose(f(inp), f_ref(inp))

        # Build the first graph and ensure that the optimization is applied
        log_softmax_out = T.log(T.nnet.Softmax()(x))
        f = theano.function([x], log_softmax_out, mode=mode_with_gpu)

        dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                             isinstance(n.op, cuda.dnn.GpuDnnSoftmax)]
        assert len(dnn_softmax_nodes) == 1
        assert dnn_softmax_nodes[0].op.algo == "log"

        # Compare the output of the function with the reference function
        inp = numpy.random.normal(0, 1, (5, 6)).astype("float32")
        utt.assert_allclose(f(inp), f_ref(inp))


def test_batchnorm_train():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    if cuda.dnn.version() < (5000, 5000):
        raise SkipTest("batch normalization requires cudnn v5+")
    utt.seed_rng()

    tensor6 = T.TensorType('float32', (False,) * 6)

    for mode in ('per-activation', 'spatial'):
        for vartype in (tensor6, T.ftensor5, T.ftensor4, T.ftensor3, T.fmatrix, T.fvector):
            x, scale, bias, running_mean, running_var = (vartype(n)
                                                         for n in ('x', 'scale', 'bias',
                                                                   'running_mean',
                                                                   'running_var'))
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used
            running_average_factor = 0.3

            # forward pass, direct interface
            out_gpu, x_mean_gpu, x_invstd_gpu, \
                out_running_mean_gpu, out_running_var_gpu = \
                dnn.dnn_batch_normalization_train(x, scale, bias, mode, eps,
                                                  running_average_factor,
                                                  running_mean, running_var)
            # forward pass, abstract interface
            out_abstract, x_mean_abstract, x_invstd_abstract, \
                out_running_mean_abstract, out_running_var_abstract = \
                bn.batch_normalization_train(x, scale, bias, mode, eps,
                                             running_average_factor,
                                             running_mean, running_var)
            # reference forward pass
            if mode == 'per-activation':
                axes = (0,)
            elif mode == 'spatial':
                axes = (0,) + tuple(range(2, ndim))
            x_mean_ref = x.mean(axis=axes, keepdims=True)
            x_var_ref = x.var(axis=axes, keepdims=True)
            x_invstd_ref = T.inv(T.sqrt(x_var_ref + eps))
            scale_ref = T.addbroadcast(scale, *axes)
            bias_ref = T.addbroadcast(bias, *axes)
            m = T.cast(T.prod(x.shape) / T.prod(scale.shape), 'float32')
            out_ref = (x - x_mean_ref) * (scale_ref * x_invstd_ref) + bias_ref
            out_running_mean_ref = running_mean * (1 - running_average_factor) + \
                x_mean_ref * running_average_factor
            out_running_var_ref = running_var * (1 - running_average_factor) + \
                (m / (m - 1)) * x_var_ref * running_average_factor
            # backward pass
            dy = vartype('dy')
            grads_gpu = T.grad(None, wrt=[x, scale, bias], known_grads={out_gpu: dy})
            grads_abstract = T.grad(None, wrt=[x, scale, bias], known_grads={out_abstract: dy})
            # reference backward pass
            grads_ref = T.grad(None, wrt=[x, scale, bias], known_grads={out_ref: dy})
            # compile
            f_gpu = theano.function([x, scale, bias, running_mean, running_var, dy],
                                    [out_gpu, x_mean_gpu, x_invstd_gpu,
                                     out_running_mean_gpu, out_running_var_gpu] + grads_gpu,
                                    mode=mode_with_gpu)
            f_abstract = theano.function([x, scale, bias, running_mean, running_var, dy],
                                         [out_abstract, x_mean_abstract, x_invstd_abstract,
                                          out_running_mean_abstract, out_running_var_abstract] +
                                         grads_abstract,
                                         mode=mode_with_gpu)
            f_ref = theano.function([x, scale, bias, running_mean, running_var, dy],
                                    [out_ref, x_mean_ref, x_invstd_ref,
                                     out_running_mean_ref, out_running_var_ref] + grads_ref)
            # check if the abstract Ops have been replaced
            assert any([isinstance(n.op, dnn.GpuDnnBatchNorm) for n
                        in f_abstract.maker.fgraph.toposort()])
            assert any([isinstance(n.op, dnn.GpuDnnBatchNormGrad) for n
                        in f_abstract.maker.fgraph.toposort()])
            assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                              bn.AbstractBatchNormInference,
                                              bn.AbstractBatchNormTrainGrad)) for n
                            in f_abstract.maker.fgraph.toposort()])
            # run
            for data_shape in ((5, 2, 30, 4, 10, 5), (4, 3, 1, 1, 1, 1), (2, 3, 5, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(1 if d in axes else s
                                    for d, s in enumerate(data_shape))
                X = 4 + 3 * numpy.random.randn(*data_shape).astype('float32')
                Dy = -1 + 2 * numpy.random.randn(*data_shape).astype('float32')
                Scale = numpy.random.randn(*param_shape).astype('float32')
                Bias = numpy.random.randn(*param_shape).astype('float32')
                Running_mean = numpy.random.randn(*param_shape).astype('float32')
                Running_var = numpy.random.randn(*param_shape).astype('float32')
                outputs_gpu = f_gpu(X, Scale, Bias, Running_mean, Running_var, Dy)
                outputs_abstract = f_abstract(X, Scale, Bias, Running_mean, Running_var, Dy)
                outputs_ref = f_ref(X, Scale, Bias, Running_mean, Running_var, Dy)
                # compare outputs
                utt.assert_allclose(outputs_gpu[0], outputs_ref[0])  # out
                utt.assert_allclose(outputs_gpu[1], outputs_ref[1])  # mean
                utt.assert_allclose(outputs_gpu[2], outputs_ref[2])  # invstd
                utt.assert_allclose(outputs_gpu[3], outputs_ref[3])  # running_mean
                utt.assert_allclose(numpy.nan_to_num(outputs_gpu[4]),
                                    numpy.nan_to_num(outputs_ref[4]))  # running_var
                utt.assert_allclose(outputs_abstract[0], outputs_ref[0])  # out
                utt.assert_allclose(outputs_abstract[1], outputs_ref[1])  # mean
                utt.assert_allclose(outputs_abstract[2], outputs_ref[2])  # invstd
                utt.assert_allclose(outputs_abstract[3], outputs_ref[3])  # running_mean
                utt.assert_allclose(numpy.nan_to_num(outputs_abstract[4]),
                                    numpy.nan_to_num(outputs_ref[4]))  # running_var
                # compare gradients
                utt.assert_allclose(outputs_gpu[5], outputs_ref[5], atol=2e-4)  # dx
                utt.assert_allclose(outputs_gpu[6], outputs_ref[6], rtol=4e-4, atol=1e-4)  # dscale
                utt.assert_allclose(outputs_gpu[7], outputs_ref[7])  # dbias
                utt.assert_allclose(outputs_abstract[5], outputs_ref[5], atol=2e-4)  # dx
                utt.assert_allclose(outputs_abstract[6], outputs_ref[6], rtol=4e-4, atol=1e-4)  # dscale
                utt.assert_allclose(outputs_abstract[7], outputs_ref[7])  # dbias


def test_dnn_batchnorm_train_without_running_averages():
    # compile and run batch_normalization_train without running averages
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    if cuda.dnn.version() < (5000, 5000):
        raise SkipTest("batch normalization requires cudnn v5+")
    utt.seed_rng()

    x, scale, bias, dy = T.ftensor4('x'), T.ftensor4('scale'), T.ftensor4('bias'), T.ftensor4('dy')
    data_shape = (5, 10, 30, 25)
    param_shape = (1, 10, 30, 25)

    # forward pass
    out_gpu, x_mean_gpu, x_invstd_gpu = \
        dnn.dnn_batch_normalization_train(x, scale, bias, 'per-activation')
    out_abstract, x_mean_abstract, x_invstd_abstract = \
        bn.batch_normalization_train(x, scale, bias, 'per-activation')
    # backward pass
    grads_gpu = T.grad(None, wrt=[x, scale, bias], known_grads={out_gpu: dy})
    grads_abstract = T.grad(None, wrt=[x, scale, bias], known_grads={out_gpu: dy})
    # compile
    f_gpu = theano.function([x, scale, bias, dy],
                            [out_gpu, x_mean_gpu, x_invstd_gpu] +
                            grads_gpu,
                            mode=mode_with_gpu)
    f_abstract = theano.function([x, scale, bias, dy],
                                 [out_abstract, x_mean_abstract, x_invstd_abstract] +
                                 grads_abstract,
                                 mode=mode_with_gpu)
    # check if the abstract Ops have been replaced
    assert any([isinstance(n.op, dnn.GpuDnnBatchNorm)
                for n in f_abstract.maker.fgraph.toposort()])
    assert any([isinstance(n.op, dnn.GpuDnnBatchNormGrad)
                for n in f_abstract.maker.fgraph.toposort()])
    assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                      bn.AbstractBatchNormInference,
                                      bn.AbstractBatchNormTrainGrad))
                    for n in f_abstract.maker.fgraph.toposort()])
    # run
    X = 4 + 3 * numpy.random.randn(*data_shape).astype('float32')
    Dy = -1 + 2 * numpy.random.randn(*data_shape).astype('float32')
    Scale = numpy.random.randn(*param_shape).astype('float32')
    Bias = numpy.random.randn(*param_shape).astype('float32')
    f_gpu(X, Scale, Bias, Dy)
    f_abstract(X, Scale, Bias, Dy)


def test_dnn_batchnorm_train_inplace():
    # test inplace_running_mean and inplace_running_var
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    if cuda.dnn.version() < (5000, 5000):
        raise SkipTest("batch normalization requires cudnn v5+")
    utt.seed_rng()

    x, scale, bias = T.ftensor4('x'), T.ftensor4('scale'), T.ftensor4('bias')
    data_shape = (5, 10, 30, 25)
    param_shape = (1, 10, 30, 25)
    running_mean = shared(
        numpy.random.randn(*param_shape).astype('float32'),
        broadcastable=(True, False, False, False))
    running_var = shared(
        numpy.random.randn(*param_shape).astype('float32'),
        broadcastable=(True, False, False, False))

    # forward pass
    out, x_mean, x_invstd, new_running_mean, new_running_var = \
        dnn.dnn_batch_normalization_train(x, scale, bias, 'per-activation',
                                          epsilon=5e-3, running_average_factor=0.3,
                                          running_mean=running_mean, running_var=running_var)
    # update running averages
    updates = OrderedDict()
    updates[running_mean] = new_running_mean
    updates[running_var] = new_running_var
    # compile
    f = theano.function([x, scale, bias],
                        [out, x_mean, x_invstd],
                        updates=updates,
                        mode=mode_with_gpu)
    # check for the inplace settings
    nodes = [n for n in f.maker.fgraph.toposort()
             if isinstance(n.op, dnn.GpuDnnBatchNorm)]
    assert len(nodes) == 1
    assert nodes[0].op.inplace_running_mean
    assert nodes[0].op.inplace_running_var
    assert nodes[0].op.inplace_output
    # run
    X = 4 + 3 * numpy.random.randn(*data_shape).astype('float32')
    Scale = numpy.random.randn(*param_shape).astype('float32')
    Bias = numpy.random.randn(*param_shape).astype('float32')
    f(X, Scale, Bias)


def test_batchnorm_inference():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    if cuda.dnn.version() < (5000, 5000):
        raise SkipTest("batch normalization requires cudnn v5+")
    utt.seed_rng()

    tensor6 = T.TensorType('float32', (False,) * 6)

    for mode in ('per-activation', 'spatial'):
        for vartype in (tensor6, T.ftensor5, T.ftensor4, T.ftensor3, T.fmatrix, T.fvector):
            x, scale, bias, mean, var = (vartype(n)
                                         for n in ('x', 'scale', 'bias', 'mean', 'var'))
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used

            # forward pass, direct interface
            out_gpu = dnn.dnn_batch_normalization_test(x, scale, bias, mean,
                                                       var, mode, eps)
            # forward pass, abstract interface
            out_abstract = bn.batch_normalization_test(x, scale, bias, mean,
                                                       var, mode, eps)
            # reference forward pass
            if mode == 'per-activation':
                axes = (0,)
            elif mode == 'spatial':
                axes = (0,) + tuple(range(2, ndim))
            scale_ref, bias_ref, mean_ref, var_ref = (T.addbroadcast(t, *axes)
                                                      for t in (scale, bias, mean, var))
            out_ref = (x - mean_ref) * (scale_ref / T.sqrt(var_ref + eps)) + bias_ref
            # backward pass
            dy = vartype('dy')
            grads_gpu = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out_gpu: dy})
            grads_abstract = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out_abstract: dy})
            # reference backward pass
            grads_ref = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out_ref: dy})
            # compile
            f_gpu = theano.function([x, scale, bias, mean, var, dy],
                                    [out_gpu] + grads_gpu, mode=mode_with_gpu)
            f_abstract = theano.function([x, scale, bias, mean, var, dy],
                                         [out_abstract] + grads_abstract, mode=mode_with_gpu)
            f_ref = theano.function([x, scale, bias, mean, var, dy],
                                    [out_ref] + grads_ref)
            # check if the abstract Ops have been replaced
            assert any([isinstance(n.op, dnn.GpuDnnBatchNormInference) for n
                        in f_abstract.maker.fgraph.toposort()])
            assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                              bn.AbstractBatchNormInference,
                                              bn.AbstractBatchNormTrainGrad)) for n
                            in f_abstract.maker.fgraph.toposort()])
            # run
            for data_shape in ((10, 2, 15, 4, 6, 5), (4, 3, 1, 1, 1, 1), (1, 1, 5, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(1 if d in axes else s
                                    for d, s in enumerate(data_shape))
                X = 4 + 3 * numpy.random.randn(*data_shape).astype('float32')
                Dy = -1 + 2 * numpy.random.randn(*data_shape).astype('float32')
                Scale = numpy.random.randn(*param_shape).astype('float32')
                Bias = numpy.random.randn(*param_shape).astype('float32')
                Mean = numpy.random.randn(*param_shape).astype('float32')
                Var = numpy.random.rand(*param_shape).astype('float32')
                outputs_gpu = f_gpu(X, Scale, Bias, Mean, Var, Dy)
                outputs_abstract = f_abstract(X, Scale, Bias, Mean, Var, Dy)
                outputs_ref = f_ref(X, Scale, Bias, Mean, Var, Dy)
                # compare outputs
                utt.assert_allclose(outputs_gpu[0], outputs_ref[0])  # out
                utt.assert_allclose(outputs_abstract[0], outputs_ref[0])  # out
                # compare gradients
                utt.assert_allclose(outputs_gpu[1], outputs_ref[1], atol=4e-5)  # dx
                utt.assert_allclose(outputs_gpu[2], outputs_ref[2], atol=4e-5)  # dscale
                utt.assert_allclose(outputs_gpu[3], outputs_ref[3])  # dbias
                utt.assert_allclose(outputs_gpu[4], outputs_ref[4])  # dmean
                utt.assert_allclose(outputs_gpu[5], outputs_ref[5], rtol=2e-3, atol=4e-5)  # dvar
                utt.assert_allclose(outputs_abstract[1], outputs_ref[1], atol=4e-5)  # dx
                utt.assert_allclose(outputs_abstract[2], outputs_ref[2], atol=4e-5)  # dscale
                utt.assert_allclose(outputs_abstract[3], outputs_ref[3])  # dbias
                utt.assert_allclose(outputs_abstract[4], outputs_ref[4])  # dmean
                utt.assert_allclose(outputs_abstract[5], outputs_ref[5], rtol=2e-3, atol=4e-5)  # dvar


def test_batchnorm_inference_inplace():
    # test inplace
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    if cuda.dnn.version() < (5000, 5000):
        raise SkipTest("batch normalization requires cudnn v5+")
    utt.seed_rng()

    x, scale, bias, mean, var = (T.ftensor4(n) for n in ('x', 'scale', 'bias', 'mean', 'var'))
    data_shape = (5, 10, 30, 25)
    param_shape = (1, 10, 30, 25)

    out = dnn.dnn_batch_normalization_test(x, scale, bias, mean, var)
    f = theano.function([x, scale, bias, mean, var], [out], mode=mode_with_gpu)

    # check for the inplace settings
    nodes = [n for n in f.maker.fgraph.toposort()
             if isinstance(n.op, dnn.GpuDnnBatchNormInference)]
    assert len(nodes) == 1
    assert nodes[0].op.inplace

    # run
    X = 4 + 3 * numpy.random.randn(*data_shape).astype('float32')
    Scale = numpy.random.randn(*param_shape).astype('float32')
    Bias = numpy.random.randn(*param_shape).astype('float32')
    Mean = numpy.random.randn(*param_shape).astype('float32')
    Var = numpy.random.rand(*param_shape).astype('float32')
    f(X, Scale, Bias, Mean, Var)


def test_dnn_batchnorm_valid_and_invalid_axes():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    if cuda.dnn.version() < (5000, 5000):
        raise SkipTest("batch normalization requires cudnn v5+")

    for vartype in (T.ftensor5, T.ftensor4, T.ftensor3, T.fmatrix):
        x, scale, bias, mean, var, dy = (vartype(n)
                                         for n in ('x', 'scale', 'bias', 'mean', 'var', 'dy'))
        ndim = x.ndim

        # supported: per-activation and spatial
        valid_axes_lists = ((0,), (0,) + tuple(range(2, ndim)))
        # not supported: an axes list without 0 and including 1
        invalid_axes_lists = (tuple(range(1, ndim)),)
        for axes in valid_axes_lists + invalid_axes_lists:
            # forward pass, abstract interface
            out_train, x_mean, x_invstd = bn.batch_normalization_train(
                x, scale, bias, axes)
            out_test = bn.batch_normalization_test(
                x, scale, bias, mean, var, axes)
            # backward pass
            dy = vartype('dy')
            grads_train = T.grad(None, wrt=[x, scale, bias], known_grads={out_train: dy})
            grads_test = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out_test: dy})
            # compile
            f = theano.function([x, scale, bias, mean, var, dy],
                                [out_train, x_mean, x_invstd, out_test] +
                                grads_train + grads_test,
                                mode=mode_with_gpu)

            if axes in valid_axes_lists:
                # check if the abstract Ops have been replaced by the cuDNN Ops
                assert any([isinstance(n.op, dnn.GpuDnnBatchNorm) for n
                            in f.maker.fgraph.toposort()])
                assert any([isinstance(n.op, dnn.GpuDnnBatchNormGrad) for n
                            in f.maker.fgraph.toposort()])
                assert any([isinstance(n.op, dnn.GpuDnnBatchNormInference) for n
                            in f.maker.fgraph.toposort()])
                assert not any([isinstance(n.op, (bn.AbstractBatchNormTrain,
                                                  bn.AbstractBatchNormInference,
                                                  bn.AbstractBatchNormTrainGrad)) for n
                                in f.maker.fgraph.toposort()])
            else:
                # check if the abstract Ops have been replaced, but not by the cuDNN Ops
                assert not any([isinstance(n.op, (dnn.GpuDnnBatchNorm,
                                                  dnn.GpuDnnBatchNormGrad,
                                                  bn.AbstractBatchNormTrain,
                                                  bn.AbstractBatchNormInference,
                                                  bn.AbstractBatchNormTrainGrad)) for n
                                in f.maker.fgraph.toposort()])


def test_dnn_tag():
    """
    Test that if cudnn isn't avail we crash and that if it is avail, we use it.
    """
    x = T.ftensor4()
    old = theano.config.on_opt_error
    theano.config.on_opt_error = "raise"

    sio = StringIO()
    handler = logging.StreamHandler(sio)
    logging.getLogger('theano.compile.tests.test_dnn').addHandler(handler)
    # Silence original handler when intentionnally generating warning messages
    logging.getLogger('theano').removeHandler(theano.logging_default_handler)
    raised = False
    try:
        f = theano.function(
            [x],
            pool_2d(x, ws=(2, 2), ignore_border=True),
            mode=mode_with_gpu.including("cudnn"))
    except (AssertionError, RuntimeError):
        assert not cuda.dnn.dnn_available()
        raised = True
    finally:
        theano.config.on_opt_error = old
        logging.getLogger(
            'theano.compile.tests.test_dnn').removeHandler(handler)
        logging.getLogger('theano').addHandler(theano.logging_default_handler)

    if not raised:
        assert cuda.dnn.dnn_available()
        assert any([isinstance(n.op, cuda.dnn.GpuDnnPool)
                    for n in f.maker.fgraph.toposort()])


class TestDnnInferShapes(utt.InferShapeTester):

    def setUp(self):
        super(TestDnnInferShapes, self).setUp()
        self.mode = mode_with_gpu

    def test_softmax(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        t = T.ftensor4('t')
        rand_tensor = numpy.asarray(
            numpy.random.rand(5, 4, 3, 2),
            dtype='float32'
        )
        self._compile_and_check(
            [t],
            [dnn.GpuDnnSoftmax('bc01', 'accurate', 'channel')(t)],
            [rand_tensor],
            dnn.GpuDnnSoftmax
        )

        self._compile_and_check(
            [t],
            [
                T.grad(
                    dnn.GpuDnnSoftmax(
                        'bc01',
                        'accurate',
                        'channel'
                    )(t).mean(),
                    t
                )
            ],
            [rand_tensor],
            dnn.GpuDnnSoftmaxGrad
        )

    def test_conv(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        kerns = T.ftensor4('kerns')
        out = T.ftensor4('out')
        img_val = numpy.asarray(
            numpy.random.rand(10, 2, 6, 4),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(8, 2, 4, 3),
            dtype='float32'
        )

        for params in product(
            ['valid', 'full', 'half'],
            [(1, 1), (2, 2)],
            ['conv', 'cross']
        ):
            out_vals = numpy.zeros(
                dnn.GpuDnnConv.get_out_shape(img_val.shape, kern_vals.shape,
                                             border_mode=params[0],
                                             subsample=params[1]),
                dtype='float32')
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(img.shape, kerns.shape)
            conv = dnn.GpuDnnConv()(img, kerns, out, desc)
            self._compile_and_check(
                [img, kerns, out],
                [conv],
                [img_val, kern_vals, out_vals],
                dnn.GpuDnnConv
            )

    def test_conv3d(self):
        if not (cuda.dnn.dnn_available() and dnn.version() >= (2000, 2000)):
            raise SkipTest('"cuDNN 3D convolution requires cuDNN v2')
        img = T.ftensor5('img')
        kerns = T.ftensor5('kerns')
        out = T.ftensor5('out')
        img_val = numpy.asarray(
            numpy.random.rand(10, 2, 6, 4, 11),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(8, 2, 4, 3, 1),
            dtype='float32'
        )

        for params in product(
            ['valid', 'full', 'half'],
            [(1, 1, 1), (2, 2, 2)],
            ['conv', 'cross']
        ):
            out_vals = numpy.zeros(
                dnn.GpuDnnConv3d.get_out_shape(img_val.shape, kern_vals.shape,
                                               border_mode=params[0],
                                               subsample=params[1]),
                dtype='float32')
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(img.shape, kerns.shape)
            conv = dnn.GpuDnnConv3d()(img, kerns, out, desc)
            self._compile_and_check(
                [img, kerns, out],
                [conv],
                [img_val, kern_vals, out_vals],
                dnn.GpuDnnConv3d
            )

    def _test_conv_gradw(self, img, topgrad, kerns, img_shape, kerns_shape, border_mode, conv_mode, subsample):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)

        topgrad_shape = get_conv_output_shape(img_shape, kerns_shape,
                                              border_mode, subsample)

        img_val = numpy.asarray(
            numpy.random.rand(*img_shape),
            dtype="float32"
        )
        topgrad_vals = numpy.asarray(
            numpy.random.rand(*topgrad_shape),
            dtype="float32"
        )

        kerns_vals = numpy.zeros(kerns_shape, dtype="float32")
        kerns_shape = theano.shared(numpy.asarray(kerns_shape))
        topgrad_shape = theano.shared(numpy.asarray(topgrad_shape))
        desc = dnn.GpuDnnConvDesc(
            border_mode=border_mode,
            subsample=subsample,
            conv_mode=conv_mode
        )(topgrad_shape, kerns_shape)
        conv_grad_w = dnn.GpuDnnConvGradW()(
            img,
            topgrad,
            kerns,
            desc,
        )
        self._compile_and_check(
            [img, topgrad, kerns],
            [conv_grad_w],
            [img_val, topgrad_vals, kerns_vals],
            dnn.GpuDnnConvGradW
        )

    border_modes = ['valid', 'full', 'half']
    conv_modes = ['conv', 'cross']

    @parameterized.expand(product(border_modes, conv_modes), utt.custom_name_func)
    def test_conv_gradw(self, border_mode, conv_mode):
        self._test_conv_gradw(T.ftensor4('img'),
                              T.ftensor4('topgrad'),
                              T.ftensor4('kerns'),
                              (5, 2, 6, 13),
                              (1, 2, 3, 7),
                              border_mode,
                              conv_mode,
                              (1, 1))

    def _test_conv3d_gradw(self, img, topgrad, kerns, img_shape, kerns_shape, border_mode, conv_mode, subsample):
        if not (cuda.dnn.dnn_available() and dnn.version() >= (2000, 2000)):
            raise SkipTest('"cuDNN 3D convolution requires cuDNN v2')

        topgrad_shape = get_conv_output_shape(img_shape, kerns_shape,
                                              border_mode, subsample)

        img_val = numpy.asarray(
            numpy.random.rand(*img_shape),
            dtype="float32"
        )
        topgrad_vals = numpy.asarray(
            numpy.random.rand(*topgrad_shape),
            dtype="float32"
        )

        kerns_vals = numpy.zeros(kerns_shape, dtype="float32")
        kerns_shape = theano.shared(numpy.asarray(kerns_shape))
        topgrad_shape = theano.shared(numpy.asarray(topgrad_shape))
        desc = dnn.GpuDnnConvDesc(
            border_mode=border_mode,
            subsample=subsample,
            conv_mode=conv_mode
        )(topgrad_shape, kerns_shape)
        conv_grad_w = dnn.GpuDnnConv3dGradW()(
            img,
            topgrad,
            kerns,
            desc,
        )
        self._compile_and_check(
            [img, topgrad, kerns],
            [conv_grad_w],
            [img_val, topgrad_vals, kerns_vals],
            dnn.GpuDnnConv3dGradW
        )

    @parameterized.expand(product(border_modes, conv_modes), utt.custom_name_func)
    def test_conv3d_gradw(self, border_mode, conv_mode):
        self._test_conv3d_gradw(T.ftensor5('img'),
                                T.ftensor5('topgrad'),
                                T.ftensor5('kerns'),
                                (5, 2, 6, 13, 21),
                                (1, 2, 3, 7, 9),
                                border_mode,
                                conv_mode,
                                (1, 1, 1))

    def test_conv_gradi(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        kerns = T.ftensor4('kerns')
        out = T.ftensor4('out')
        img_val = numpy.asarray(
            numpy.random.rand(3, 4, 5, 6),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(4, 14, 15, 16),
            dtype='float32'
        )

        for params in product(
            ['valid'],  # Should this work for 'full'?
            [(1, 1)],
            ['conv', 'cross']
        ):
            temp_kerns = kerns.dimshuffle(1, 0, 2, 3)
            shape = (
                img_val.shape[0], kern_vals.shape[1],
                img_val.shape[2] + kern_vals.shape[2] - 1,
                img_val.shape[3] + kern_vals.shape[3] - 1
            )
            out_vals = numpy.zeros(shape, dtype='float32')
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(out.shape, temp_kerns.shape)
            conv_grad_i = dnn.GpuDnnConvGradI()(
                temp_kerns,
                img,
                out,
                desc,
            )
            self._compile_and_check(
                [temp_kerns, img, out],
                [conv_grad_i],
                [kern_vals, img_val, out_vals],
                dnn.GpuDnnConvGradI
            )

    def test_conv3d_gradi(self):
        if not (cuda.dnn.dnn_available() and dnn.version() >= (2000, 2000)):
            raise SkipTest('"cuDNN 3D convolution requires cuDNN v2')
        img = T.ftensor5('img')
        kerns = T.ftensor5('kerns')
        out = T.ftensor5('out')
        img_val = numpy.asarray(
            numpy.random.rand(8, 4, 6, 7, 11),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(9, 4, 5, 1, 2),
            dtype='float32'
        )

        for params in product(
            ['valid', 'full', 'half'],
            [(1, 1, 1), (2, 2, 2)],
            ['conv', 'cross']
        ):
            out_vals = numpy.zeros(
                dnn.GpuDnnConv3d.get_out_shape(img_val.shape, kern_vals.shape,
                                               border_mode=params[0],
                                               subsample=params[1]),
                dtype='float32')

            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(img.shape, kerns.shape)
            conv_grad_i = dnn.GpuDnnConv3dGradI()(
                kerns,
                out,
                img,
                desc,
            )
            self._compile_and_check(
                [kerns, out, img],
                [conv_grad_i],
                [kern_vals, out_vals, img_val],
                dnn.GpuDnnConv3dGradI
            )

    def test_pool(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        img_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5),
            dtype='float32'
        )

        # 'average_exc_pad' is disabled for versions < 4004
        if cuda.dnn.version() < (4004, 4004):
            modes = ['max', 'average_inc_pad']
        else:
            modes = ['max', 'average_inc_pad', 'average_exc_pad']

        for params in product(
            [(1, 1), (2, 2), (3, 3)],
            [(1, 1), (2, 2), (3, 3)],
            modes
        ):
            self._compile_and_check(
                [img],
                [dnn.GpuDnnPool(mode=params[2])
                               (img, params[0], params[1], (0, 0))],
                [img_val],
                dnn.GpuDnnPool
            )

    def test_pool_3d(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor5('img')
        img_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5, 6),
            dtype='float32'
        )

        # 'average_exc_pad' is disabled for versions < 4004
        if cuda.dnn.version() < (4004, 4004):
            modes = ['max', 'average_inc_pad']
        else:
            modes = ['max', 'average_inc_pad', 'average_exc_pad']

        for params in product(
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            modes
        ):
            self._compile_and_check(
                [img],
                [dnn.GpuDnnPool(mode=params[2])(img, params[0], params[1], (0, 0, 0))],
                [img_val],
                dnn.GpuDnnPool
            )

    def test_pool_grad(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        img_grad = T.ftensor4('img_grad')
        out = T.ftensor4('out')
        img_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5),
            dtype='float32'
        )
        img_grad_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5),
            dtype='float32'
        )
        out_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5),
            dtype='float32'
        )

        for params in product(
            [(1, 1), (2, 2), (3, 3)],
            [(1, 1), (2, 2), (3, 3)],
            ['max', 'average_inc_pad']
        ):
            pool_grad = dnn.GpuDnnPoolGrad()(
                img,
                out,
                img_grad,
                params[0],
                params[1],
                (0, 0)
            )
            self._compile_and_check(
                [img, img_grad, out],
                [pool_grad],
                [img_val, img_grad_val, out_val],
                dnn.GpuDnnPoolGrad
            )

    def test_pool_3d_grad(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor5('img')
        img_grad = T.ftensor5('img_grad')
        out = T.ftensor5('out')
        img_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5, 6),
            dtype='float32'
        )
        img_grad_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5, 6),
            dtype='float32'
        )
        out_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5, 6),
            dtype='float32'
        )

        for params in product(
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            [(1, 1, 1), (2, 2, 2), (3, 3, 3)],
            ['max', 'average_inc_pad']
        ):
            pool_grad = dnn.GpuDnnPoolGrad(mode=params[2])(
                img,
                out,
                img_grad,
                params[0],
                params[1],
                (0, 0, 0)
            )
            self._compile_and_check(
                [img, img_grad, out],
                [pool_grad],
                [img_val, img_grad_val, out_val],
                dnn.GpuDnnPoolGrad
            )


# this has been a problem in the past
def test_dnn_conv_border_mode():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    img = T.ftensor4()
    kern = T.ftensor4()

    dnn.dnn_conv(img, kern, border_mode=1)
    dnn.dnn_conv(img, kern, border_mode=(2, 3))
    dnn.dnn_conv(img, kern, border_mode='full')
    dnn.dnn_conv(img, kern, border_mode='valid')
    dnn.dnn_conv(img, kern, border_mode='half')


def test_dnn_conv_alpha_output_merge():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    img = T.ftensor4()
    kern = T.ftensor4()
    out = T.ftensor4()

    b = 1
    c = 4
    f = 3
    ih = 5
    iw = 8
    kh = 2
    kw = 6
    img_val = numpy.random.random((b, c, ih, iw)).astype('float32')
    kern_val = numpy.random.random((f, c, kh, kw)).astype('float32')
    out_val = numpy.random.random((b, f, ih - kh + 1,
                                   iw - kw + 1)).astype('float32')

    conv = dnn.dnn_conv(img, kern)
    gw = theano.grad(conv.sum(), kern)
    gi = theano.grad(conv.sum(), img)

    lr = numpy.asarray(0.05, dtype='float32')

    if cuda.dnn.version() == -1:
        # Can't merge alpha with cudnn v1
        fr = conv + out
        wr = kern + gw
        ir = img + gi
    else:
        fr = lr * (conv + out)
        wr = kern + lr * gw
        ir = img + lr * gi

    f1 = theano.function([img, kern, out], [fr, wr, ir], mode=mode_with_gpu)
    assert isinstance(f1.maker.fgraph.outputs[0].owner.inputs[0].owner.op,
                      dnn.GpuDnnConv)
    assert isinstance(f1.maker.fgraph.outputs[1].owner.inputs[0].owner.op,
                      dnn.GpuDnnConvGradW)
    assert isinstance(f1.maker.fgraph.outputs[2].owner.inputs[0].owner.op,
                      dnn.GpuDnnConvGradI)

    mode = mode_with_gpu
    mode = mode.excluding('local_dnn_conv_alpha_merge')
    mode = mode.excluding('local_dnn_convw_alpha_merge')
    mode = mode.excluding('local_dnn_convi_alpha_merge')
    mode = mode.excluding('local_dnn_conv_output_merge')
    mode = mode.excluding('local_dnn_convw_output_merge')
    mode = mode.excluding('local_dnn_convi_output_merge')

    f2 = theano.function([img, kern, out], [fr, wr, ir], mode=mode)

    assert not isinstance(f2.maker.fgraph.outputs[0].owner.inputs[0].owner.op,
                          dnn.GpuDnnConv)
    assert not isinstance(f2.maker.fgraph.outputs[1].owner.inputs[0].owner.op,
                          dnn.GpuDnnConvGradW)
    assert not isinstance(f2.maker.fgraph.outputs[2].owner.inputs[0].owner.op,
                          dnn.GpuDnnConvGradI)

    out_f1 = f1(img_val, kern_val, out_val)
    out_f2 = f2(img_val, kern_val, out_val)

    assert len(out_f1) == len(out_f2)

    for v1, v2 in zip(out_f1, out_f2):
        utt.assert_allclose(v1, v2)


def test_dnn_conv3d_alpha_output_merge():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    t = T.TensorType(broadcastable=(False, False, False, False, False),
                     dtype='float32')

    img = t()
    kern = t()
    out = t()

    b = 1
    c = 4
    f = 3
    it = 10
    ih = 5
    iw = 8
    kt = 3
    kh = 2
    kw = 6
    img_val = numpy.random.random((b, c, it, ih, iw)).astype('float32')
    kern_val = numpy.random.random((f, c, kt, kh, kw)).astype('float32')
    out_val = numpy.random.random((b, f, it - kt + 1, ih - kh + 1,
                                   iw - kw + 1)).astype('float32')

    conv = dnn.dnn_conv3d(img, kern)
    gw = theano.grad(conv.sum(), kern)
    gi = theano.grad(conv.sum(), img)

    lr = numpy.asarray(0.05, dtype='float32')

    if cuda.dnn.version() == -1:
        # Can't merge alpha with cudnn v1
        fr = conv + out
        wr = kern + gw
        ir = img + gi
    else:
        fr = lr * (conv + out)
        wr = kern + lr * gw
        ir = img + lr * gi

    f1 = theano.function([img, kern, out], [fr, wr, ir], mode=mode_with_gpu)
    assert isinstance(f1.maker.fgraph.outputs[0].owner.inputs[0].owner.op,
                      dnn.GpuDnnConv)
    assert isinstance(f1.maker.fgraph.outputs[1].owner.inputs[0].owner.op,
                      dnn.GpuDnnConvGradW)
    assert isinstance(f1.maker.fgraph.outputs[2].owner.inputs[0].owner.op,
                      dnn.GpuDnnConvGradI)

    mode = mode_with_gpu
    mode = mode.excluding('local_dnn_conv_alpha_merge')
    mode = mode.excluding('local_dnn_convw_alpha_merge')
    mode = mode.excluding('local_dnn_convi_alpha_merge')
    mode = mode.excluding('local_dnn_conv_output_merge')
    mode = mode.excluding('local_dnn_convw_output_merge')
    mode = mode.excluding('local_dnn_convi_output_merge')

    f2 = theano.function([img, kern, out], [fr, wr, ir], mode=mode)

    assert not isinstance(f2.maker.fgraph.outputs[0].owner.inputs[0].owner.op,
                          dnn.GpuDnnConv3d)
    assert not isinstance(f2.maker.fgraph.outputs[1].owner.inputs[0].owner.op,
                          dnn.GpuDnnConv3dGradW)
    assert not isinstance(f2.maker.fgraph.outputs[2].owner.inputs[0].owner.op,
                          dnn.GpuDnnConv3dGradI)

    out_f1 = f1(img_val, kern_val, out_val)
    out_f2 = f2(img_val, kern_val, out_val)

    assert len(out_f1) == len(out_f2)

    for v1, v2 in zip(out_f1, out_f2):
        utt.assert_allclose(v1, v2)


def test_dnn_conv_merge_mouts():
    # make sure it doesn't attempt to output/alpha merge a convolution
    # that has multiple clients.
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    img = T.ftensor4()
    kern = T.ftensor4()
    out = T.ftensor4()

    conv = dnn.dnn_conv(img, kern)

    lr = numpy.asarray(0.05, dtype='float32')

    if cuda.dnn.version() == -1:
        # Can't merge alpha with cudnn v1
        fr = conv + out
    else:
        fr = lr * (conv + out)
    rr = conv * lr

    f = theano.function([img, kern, out], [fr, rr], mode=mode_with_gpu)
    convs = [n for n in f.maker.fgraph.toposort()
             if isinstance(n.op, dnn.GpuDnnConv)]
    assert len(convs) == 1


def test_dnn_conv_merge_broad():
    # Make sure that we don't apply output_merge on broadcasted values.
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    img = T.ftensor4()
    kern = T.ftensor4()

    conv = dnn.dnn_conv(img, kern)

    lr = numpy.asarray(0.05, dtype='float32')

    # this does broadcasting
    fr = conv + lr

    f = theano.function([img, kern], [fr])
    convs = [n for n in f.maker.fgraph.toposort()
             if isinstance(n.op, dnn.GpuDnnConv)]
    assert len(convs) == 1
    conv = convs[0]
    # Assert output was not merged
    assert isinstance(conv.inputs[2].owner.op, GpuAllocEmpty)


def test_dnn_conv_grad():
    if not cuda.dnn.dnn_available() or dnn.version() == -1:
        raise SkipTest('alpha != 1.0 not supported in cudnn v1')

    b = 1
    c = 4
    f = 3
    ih = 2
    iw = 8
    kh = 2
    kw = 2
    img_val = numpy.random.random((b, c, ih, iw)).astype('float32')
    kern_val = numpy.random.random((f, c, kh, kw)).astype('float32')
    out_val = numpy.random.random((b, f, ih - kw + 1,
                                   iw - kw + 1)).astype('float32')

    def dconv(img, kern, out):
        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                                  conv_mode='conv')(img.shape, kern.shape)
        return dnn.GpuDnnConv()(img, kern, out, desc, alpha=0.5, beta=0.75)

    def dconvi(img, kern, out):
        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                                  conv_mode='conv')(img.shape, kern.shape)
        return dnn.GpuDnnConvGradI()(kern, out, img, desc, alpha=-1.0,
                                     beta=0.0)

    def dconvw(img, kern, out):
        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                                  conv_mode='conv')(img.shape, kern.shape)
        return dnn.GpuDnnConvGradW()(img, out, kern, desc, alpha=0.75,
                                     beta=-1.0)

    utt.verify_grad(dconv, [img_val, kern_val, out_val], mode=mode_with_gpu)
    utt.verify_grad(dconvi, [img_val, kern_val, out_val], mode=mode_with_gpu)
    utt.verify_grad(dconvw, [img_val, kern_val, out_val], mode=mode_with_gpu)


def get_conv3d_test_cases():
    # Every element of test_shapes follows the format
    # [input_shape, filter_shape, subsample]
    test_shapes = [[(128, 3, 5, 5, 5), (64, 3, 1, 2, 4), (1, 1, 1)],
                   [(8, 4, 20, 12, 15), (5, 4, 6, 12, 4), (2, 2, 2)],
                   [(8, 1, 20, 12, 15), (5, 1, 6, 12, 4), (3, 3, 3)],
                   [(8, 1, 20, 12, 15), (5, 1, 6, 12, 4), (3, 2, 1)],
                   # Test with 1x1x1 filters
                   [(8, 1, 10, 10, 10), (10, 1, 1, 1, 1), (1, 1, 1)],
                   # Test with dimensions larger than 1024 (thread block dim)
                   [(1025, 1, 2, 3, 4), (5, 1, 1, 2, 3), (1, 1, 1)],
                   [(8, 1, 2, 3, 4), (1025, 1, 1, 2, 3), (1, 1, 1)],
                   [(8, 1025, 2, 3, 4), (5, 1025, 1, 1, 2), (1, 1, 1)],
                   [(8, 1, 1030, 3, 4), (5, 1, 1025, 1, 1), (1, 1, 1)],
                   [(8, 1, 2, 1030, 4), (5, 1, 2, 1025, 1), (1, 1, 1)],
                   [(8, 1, 2, 3, 1030), (5, 1, 1, 2, 1025), (1, 1, 1)],
                   # The equivalent of this caused a crash with conv2d
                   [(1, 1, 1, 44800, 1), (6, 1, 1, 1, 1), (1, 1, 1)]]

    # With border mode 'full', test with kernel bigger than image in some/all
    # dimensions
    test_shapes_full = [[(6, 2, 2, 2, 2), (4, 2, 3, 1, 1), (1, 1, 1)],
                        [(6, 2, 2, 2, 2), (4, 2, 1, 3, 1), (1, 1, 1)],
                        [(6, 2, 2, 2, 2), (4, 2, 1, 1, 3), (1, 1, 1)],
                        [(6, 2, 2, 2, 2), (4, 2, 5, 5, 5), (1, 1, 1)]]
    border_modes = ['valid', 'full', 'half', (1, 2, 3), (3, 2, 1), 1, 2]
    conv_modes = ['conv', 'cross']

    if cuda.dnn.dnn_available() and dnn.version() >= (3000, 3000):
        itt = chain(product(test_shapes, border_modes, conv_modes),
                    product(test_shapes_full, ['full'], conv_modes))
    else:
        # cuDNN, before V3, did not support kernels larger than the inputs,
        # even if the original inputs were padded so they would be larger than
        # the kernels. If using a version older than V3 don't run the tests
        # with kernels larger than the unpadded inputs.
        itt = product(test_shapes, border_modes, conv_modes)

    return itt


def test_conv3d_fwd():

    if not (cuda.dnn.dnn_available() and dnn.version() >= (2000, 2000)):
        raise SkipTest('"cuDNN 3D convolution requires cuDNN v2')

    def run_conv3d_fwd(inputs_shape, filters_shape, subsample,
                       border_mode, conv_mode):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        # Scale down the input values to prevent very large absolute errors
        # due to float rounding
        inputs_val /= 10
        filters_val /= 10

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        # Compile a theano function for the cuDNN implementation
        conv = dnn.dnn_conv3d(img=inputs, kerns=filters,
                              border_mode=border_mode, subsample=subsample,
                              conv_mode=conv_mode)
        f = theano.function([], conv, mode=mode_with_gpu)

        # If conv_mode is 'conv' the reference implementation should use
        # filters filpped according to the width, height and time axis
        if conv_mode == 'conv':
            flipped_filters = filters[:, :, ::-1, ::-1, ::-1]
        else:
            flipped_filters = filters

        # Compile a theano function for the reference implementation
        conv_ref = theano.tensor.nnet.corr3d.Corr3dMM(border_mode=border_mode,
                                                      subsample=subsample
                                                      )(inputs, flipped_filters)
        f_ref = theano.function([], conv_ref, mode="FAST_RUN")

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res)

    test_cases = get_conv3d_test_cases()
    for (i_shape, f_shape, subsample), border_mode, conv_mode in test_cases:
        yield (run_conv3d_fwd, i_shape, f_shape, subsample, border_mode,
               conv_mode)


def test_conv3d_bwd():

    if not (cuda.dnn.dnn_available() and dnn.version() >= (2000, 2000)):
        raise SkipTest('"cuDNN 3D convolution requires cuDNN v2')

    def run_conv3d_bwd(inputs_shape, filters_shape, subsample,
                       border_mode, conv_mode):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = shared(inputs_val)
        filters = shared(filters_val)

        # Compile a theano function for the cuDNN implementation
        conv = dnn.dnn_conv3d(img=inputs, kerns=filters,
                              border_mode=border_mode, subsample=subsample,
                              conv_mode=conv_mode)

        grad_i, grad_w = theano.tensor.grad(conv.sum(), [inputs, filters])

        f = theano.function([], [grad_i, grad_w], mode=mode_with_gpu)

        # If conv_mode is 'conv' the reference implementation should use
        # filters filpped according to the width, height and time axis
        if conv_mode == 'conv':
            flipped_filters = filters[:, :, ::-1, ::-1, ::-1]
        else:
            flipped_filters = filters

        # Compile a theano function for the reference implementation
        conv_ref = theano.tensor.nnet.corr3d.Corr3dMM(border_mode=border_mode,
                                                      subsample=subsample
                                                      )(inputs, flipped_filters)
        (grad_i_ref,
         grad_w_ref) = theano.tensor.grad(conv_ref.sum(),
                                          [inputs, filters])

        f_ref = theano.function([], [grad_i_ref, grad_w_ref], mode="FAST_RUN")

        # Compare the results of the two implementations
        res_ref = f_ref()
        res = f()
        # Needed for big size for some seed
        # raise rtol to make the test pass with more seed.
        utt.assert_allclose(res_ref[0], res[0], rtol=2e-5)
        utt.assert_allclose(res_ref[1], res[1], rtol=2e-5)

    test_cases = get_conv3d_test_cases()
    for (i_shape, f_shape, subsample), border_mode, conv_mode in test_cases:
        yield (run_conv3d_bwd, i_shape, f_shape, subsample, border_mode,
               conv_mode)


def test_version():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    assert isinstance(cuda.dnn.version(), (int, tuple))
