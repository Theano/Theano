from __future__ import absolute_import, print_function, division
import logging

from nose.plugins.skip import SkipTest
from nose_parameterized import parameterized
import numpy
from itertools import product, chain

import theano
from six import StringIO
import theano.tensor as T
import theano.tests.unittest_tools as utt
from theano.tensor.signal.pool import pool_2d, pool_3d
from theano.tensor.signal.pool import Pool, MaxPoolGrad, AveragePoolGrad

from .. import dnn
from ..basic_ops import GpuAllocEmpty
from ..type import gpuarray_shared_constructor

from .config import mode_with_gpu, mode_without_gpu, test_ctx_name
from . import test_nnet
from .rnn_support import Model, GRU, LSTM, WrapperLayer

from theano.configdefaults import SUPPORTED_DNN_CONV_ALGO_FWD


def test_dnn_conv_desc_merge():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    kern_shp = T.as_tensor_variable(
        numpy.asarray([3, 1, 2, 2]).astype('int64'))
    desc1 = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(2, 2),
                               conv_mode='conv')(kern_shp)
    desc2 = dnn.GpuDnnConvDesc(border_mode='full', subsample=(1, 1),
                               conv_mode='cross')(kern_shp)
    # CDataType is not DeepCopyable so this will crash if we don't use
    # borrow=True
    f = theano.function([], [theano.Out(desc1, borrow=True),
                             theano.Out(desc2, borrow=True)])

    d1, d2 = f()

    # This will be the case if they are merged, which would be bad.
    assert d1 != d2


def test_dnn_conv_merge():
    # This test that we merge correctly multiple dnn_conv.
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    img_shp = [2, 5, 6, 8]
    kern_shp = [3, 5, 5, 6]
    img = T.ftensor4('img')
    kern = T.ftensor4('kern')
    out = T.ftensor4('out')
    desc = dnn.GpuDnnConvDesc(
        border_mode='valid')(kern.shape)

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
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    img_shp = [2, 5, 6, 8]
    kern_shp = [3, 5, 5, 6]
    img = T.ftensor4('img')
    kern = T.ftensor4('kern')
    out = T.ftensor4('out')
    desc1 = dnn.GpuDnnConvDesc(border_mode='valid', conv_mode='conv')(
        kern.shape)
    desc2 = dnn.GpuDnnConvDesc(
        border_mode='valid', conv_mode='cross')(kern.shape)

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
    out = GpuAllocEmpty(kern.dtype, test_ctx_name)(*kern.shape)
    o1 = dnn.GpuDnnConvGradW()(img, kern, out, desc1)
    o2 = dnn.GpuDnnConvGradW()(img, kern, out, desc2)
    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradW)]
    assert len(convs) == 2
    assert all([node.op.inplace for node in convs])
    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2

    # Test grad i op
    out = GpuAllocEmpty(img.dtype, test_ctx_name)(*img.shape)
    o1 = dnn.GpuDnnConvGradI()(img, kern, out, desc1)
    o2 = dnn.GpuDnnConvGradI()(img, kern, out, desc2)
    f = theano.function([img, kern], [o1, o2], mode=mode_with_gpu)
    topo = f.maker.fgraph.toposort()
    convs = [n for n in topo if isinstance(n.op, dnn.GpuDnnConvGradI)]
    assert len(convs) == 2
    assert all([node.op.inplace for node in convs])
    assert len([n for n in topo if isinstance(n.op, GpuAllocEmpty)]) == 2


def test_pooling():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)

    # 'average_exc_pad' is disabled for versions < 4004
    if dnn.version(raises=False) < 4004:
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
                              st=(stride, stride),
                              ignore_border=True,
                              padding=pad, mode=mode)
                mode_without_gpu2 = mode_without_gpu.including()
                mode_without_gpu2.check_isfinite = False

                # GPU implementation
                f_gpu = theano.function([x], out, mode=mode_with_gpu)
                assert any([isinstance(node.op, dnn.GpuDnnPool)
                            for node in f_gpu.maker.fgraph.apply_nodes])

                # CPU implementation
                f_cpu = theano.function([x], out, mode=mode_without_gpu2)
                assert not any([isinstance(node.op, dnn.GpuDnnPool)
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
                               padding=pad, mode=mode)
            utt.verify_grad(fn, [data], mode=mode_with_gpu)
            # Confirm that the opt would have inserted it.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
                        for node in fg.maker.fgraph.toposort()])

            # Test the GPU grad + GPU implementation
            def fn(x):
                dnn_op = dnn.dnn_pool(
                    x, ws=(ws, ws),
                    stride=(stride, stride),
                    pad=pad,
                    mode=mode)
                return dnn_op
            utt.verify_grad(fn, [data], mode=mode_with_gpu)
            # Confirm that we get the good op.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
                        for node in fg.maker.fgraph.toposort()])


def test_pooling_with_tensor_vars():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    x = T.ftensor4()
    ws = theano.shared(numpy.array([2, 2], dtype='int32'))
    st = theano.shared(numpy.array([1, 1], dtype='int32'))
    pad = theano.shared(numpy.array([0, 0], dtype='int32'))
    mode = 'max'

    def fn(x):
        dnn_op = dnn.dnn_pool(
            x, ws=ws,
            stride=st,
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
    assert any([isinstance(node.op, dnn.GpuDnnPool)
                for node in f_gpu.maker.fgraph.apply_nodes])

    # CPU implementation
    out_cpu = pool_2d(x, ws, ignore_border=True, st=st, padding=pad, mode=mode)
    f_cpu = theano.function([x], out_cpu, mode=mode_without_gpu2)
    assert not any([isinstance(node.op, dnn.GpuDnnPool)
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


def test_pooling3d():
    # 3d pooling requires version 3 or newer.
    if not dnn.dnn_available(test_ctx_name) or dnn.version(raises=False) < 3000:
        raise SkipTest(dnn.dnn_available.msg)

    # We force the FAST_RUN as we don't want the reference to run in DebugMode.
    mode_without_gpu_ref = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpuarray')

    # 'average_exc_pad' is disabled for versions < 4004
    if dnn.version(raises=False) < 4004:
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
                              st=(stride, stride, stride),
                              ignore_border=True,
                              padding=pad, mode=mode)

                # GPU implementation
                f_gpu = theano.function([x], out, mode=mode_with_gpu)
                assert any([isinstance(node.op, dnn.GpuDnnPool)
                            for node in f_gpu.maker.fgraph.apply_nodes])

                # CPU implementation
                f_cpu = theano.function([x], out, mode=mode_without_gpu_ref)
                assert not any([isinstance(node.op, dnn.GpuDnnPool)
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
                dnn_op = dnn.dnn_pool(
                    x, ws=(ws, ws, ws),
                    stride=(stride, stride, stride),
                    pad=pad,
                    mode=mode)
                return dnn_op
            utt.verify_grad(fn, [data], mode=mode_with_gpu)
            # Confirm that we get the good op.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
                        for node in fg.maker.fgraph.toposort()])


def test_pooling_opt():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)

    # 2D pooling
    x = T.fmatrix()

    f = theano.function(
        [x],
        pool_2d(x, ds=(2, 2), mode='average_inc_pad',
                ignore_border=True),
        mode=mode_with_gpu)

    assert any([isinstance(n.op, dnn.GpuDnnPool)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10), dtype='float32'))

    # gradient of 2D pooling
    f = theano.function(
        [x],
        T.grad(pool_2d(x, ds=(2, 2), mode='average_inc_pad',
                       ignore_border=True).sum(),
               x),
        mode=mode_with_gpu.including("cudnn"))

    assert any([isinstance(n.op, dnn.GpuDnnPoolGrad)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10), dtype='float32'))

    # Test sum pooling
    f = theano.function(
        [x],
        pool_2d(x, ds=(2, 3), mode='sum',
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
        pool_3d(x, ds=(2, 2, 2), mode='average_inc_pad',
                ignore_border=True),
        mode=mode_with_gpu)

    assert any([isinstance(n.op, dnn.GpuDnnPool)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10, 10), dtype='float32'))

    # gradient of 3D pooling
    f = theano.function(
        [x],
        T.grad(pool_3d(x, ds=(2, 2, 2), mode='average_inc_pad',
                       ignore_border=True).sum(),
               x),
        mode=mode_with_gpu.including("cudnn"))

    assert any([isinstance(n.op, dnn.GpuDnnPoolGrad)
                for n in f.maker.fgraph.toposort()])

    f(numpy.zeros((10, 10, 10), dtype='float32'))


def test_pooling_opt_arbitrary_dimensions():
    # test if input with an arbitrary number of non-pooling dimensions
    # is correctly reshaped to run on the GPU

    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)

    # 'average_exc_pad' is disabled for versions < 4004
    if dnn.version(raises=False) < 4004:
        modes = ('max', 'average_inc_pad')
    else:
        modes = ('max', 'average_inc_pad', 'average_exc_pad')

    for n_non_pool_dims in (0, 1, 2, 3):
        for ws in ((2, 2), (3, 3, 3)):
            # create input shape: non-pooling dimensions
            # followed by 2 or 3 pooling dimensions
            shp = tuple(range(2, 2 + n_non_pool_dims)) + tuple(range(5, 5 + len(ws)))
            data = numpy.random.normal(0, 1, shp).astype('float32')
            input = gpuarray_shared_constructor(data)

            for mode in modes:
                out_pool = Pool(ndim=len(ws), mode=mode, ignore_border=True)(input, ws)
                out_pool_grad = T.grad(T.sum(out_pool), wrt=input)
                out = [out_pool, out_pool_grad]

                # run on GPU
                fg = theano.function([], out, mode=mode_with_gpu)
                assert any([isinstance(node.op, dnn.GpuDnnPool)
                           for node in fg.maker.fgraph.toposort()])
                assert any([isinstance(node.op, dnn.GpuDnnPoolGrad)
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
            pool_2d(x, ds=(2, 2), ignore_border=True),
            mode=mode_with_gpu.including("cudnn"))
    except (AssertionError, RuntimeError):
        assert not dnn.dnn_available(test_ctx_name)
        raised = True
    finally:
        theano.config.on_opt_error = old
        logging.getLogger(
            'theano.compile.tests.test_dnn').removeHandler(handler)
        logging.getLogger('theano').addHandler(theano.logging_default_handler)

    if not raised:
        assert dnn.dnn_available(test_ctx_name)
        assert any([isinstance(n.op, dnn.GpuDnnPool)
                    for n in f.maker.fgraph.toposort()])


class TestDnnInferShapes(utt.InferShapeTester):

    border_modes = ['valid', 'full', 'half']
    conv_modes = ['conv', 'cross']

    def setUp(self):
        super(TestDnnInferShapes, self).setUp()
        self.mode = mode_with_gpu

    def test_softmax(self):
        if not dnn.dnn_available(test_ctx_name):
            raise SkipTest(dnn.dnn_available.msg)
        t = T.ftensor4('t')
        rand_tensor = numpy.asarray(
            numpy.random.rand(5, 4, 3, 2),
            dtype='float32'
        )
        self._compile_and_check(
            [t],
            [dnn.GpuDnnSoftmax('accurate', 'channel')(t)],
            [rand_tensor],
            dnn.GpuDnnSoftmax
        )

        self._compile_and_check(
            [t],
            [
                T.grad(
                    dnn.GpuDnnSoftmax(
                        'accurate',
                        'channel'
                    )(t).mean(),
                    t
                )
            ],
            [rand_tensor],
            dnn.GpuDnnSoftmaxGrad
        )

    def _test_conv(self, img, kerns, out, img_val, kern_vals, border_mode, conv_mode, subsamples, algo):
        if not dnn.dnn_available(test_ctx_name):
            raise SkipTest(dnn.dnn_available.msg)

        img_val = numpy.asarray(img_val, dtype='float32')
        kern_vals = numpy.asarray(kern_vals, dtype='float32')

        for subsample in subsamples:
            out_vals = numpy.zeros(
                dnn.GpuDnnConv.get_out_shape(img_val.shape, kern_vals.shape,
                                             border_mode=border_mode,
                                             subsample=subsample),
                dtype='float32')
            desc = dnn.GpuDnnConvDesc(
                border_mode=border_mode,
                subsample=subsample,
                conv_mode=conv_mode
            )(kerns.shape)
            conv = dnn.GpuDnnConv(algo=algo)(img, kerns, out, desc)
            self._compile_and_check(
                [img, kerns, out],
                [conv],
                [img_val, kern_vals, out_vals],
                dnn.GpuDnnConv
            )

    @parameterized.expand(chain(product([SUPPORTED_DNN_CONV_ALGO_FWD[0]],
                                        border_modes,
                                        conv_modes),
                                product(SUPPORTED_DNN_CONV_ALGO_FWD[1:],
                                        [border_modes[0]],
                                        [conv_modes[0]])),
                          testcase_func_name=utt.custom_name_func)
    def test_conv(self, algo, border_mode, conv_mode):
        if algo == 'winograd' and dnn.version(raises=False) < 5000:
            raise SkipTest(dnn.dnn_available.msg)

        self._test_conv(T.ftensor4('img'),
                        T.ftensor4('kerns'),
                        T.ftensor4('out'),
                        numpy.random.rand(7, 2, 8, 4),
                        numpy.random.rand(8, 2, 4, 3),
                        border_mode,
                        conv_mode,
                        [(1, 1), (2, 2)],
                        algo)

    @parameterized.expand(product(border_modes, conv_modes), utt.custom_name_func)
    def test_conv3d_none(self, border_mode, conv_mode):
        self._test_conv(T.ftensor5('img'),
                        T.ftensor5('kerns'),
                        T.ftensor5('out'),
                        numpy.random.rand(10, 2, 6, 4, 11),
                        numpy.random.rand(8, 2, 4, 3, 1),
                        border_mode,
                        conv_mode,
                        [(1, 1, 1), (2, 2, 2)],
                        'none')

    def _test_conv_gradw(self, img, kerns, out, img_val, kern_vals, border_mode, conv_mode, subsample):
        if not dnn.dnn_available(test_ctx_name):
            raise SkipTest(dnn.dnn_available.msg)

        img_val = numpy.asarray(
            img_val,
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            kern_vals,
            dtype='float32'
        )

        temp_img = img.dimshuffle(1, 0, 2, 3)
        temp_kerns = kerns
        if conv_mode == 'conv':
            temp_kerns = temp_kerns[:, :, ::-1, ::-1]
        temp_kerns = temp_kerns.dimshuffle(1, 0, 2, 3)
        shape = (
            kern_vals.shape[1], img_val.shape[1],
            img_val.shape[2] - kern_vals.shape[2] + 1,
            img_val.shape[3] - kern_vals.shape[3] + 1
        )
        out_vals = numpy.zeros(shape, dtype='float32')
        desc = dnn.GpuDnnConvDesc(
            border_mode=border_mode,
            subsample=subsample,
            conv_mode=conv_mode
        )(out.shape)
        conv_grad_w = dnn.GpuDnnConvGradW()(
            temp_img,
            temp_kerns,
            out,
            desc,
        )
        self._compile_and_check(
            [temp_img, temp_kerns, out],
            [conv_grad_w],
            [img_val, kern_vals, out_vals],
            dnn.GpuDnnConvGradW
        )

    @parameterized.expand(product(border_modes, conv_modes), utt.custom_name_func)
    def test_conv_gradw(self, border_mode, conv_mode):
        self._test_conv_gradw(T.ftensor4('img'),
                              T.ftensor4('kerns'),
                              T.ftensor4('out'),
                              numpy.random.rand(2, 5, 6, 8),
                              numpy.random.rand(2, 1, 5, 6),
                              border_mode,
                              conv_mode,
                              (1, 1))

    def test_conv_gradi(self):
        if not dnn.dnn_available(test_ctx_name):
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        kerns = T.ftensor4('kerns')
        out = T.ftensor4('out')
        kern_vals = numpy.asarray(
            numpy.random.rand(13, 14, 15, 16),
            dtype='float32'
        )
        out_vals = numpy.asarray(
            numpy.random.rand(3, 13, 5, 6),
            dtype='float32'
        )

        for params in product(
            ['valid'],  # Should this work for 'full'?
            [(1, 1)],
            ['conv', 'cross']
        ):
            shape = (
                out_vals.shape[0], kern_vals.shape[1],
                out_vals.shape[2] + kern_vals.shape[2] - 1,
                out_vals.shape[3] + kern_vals.shape[3] - 1
            )
            img_vals = numpy.zeros(shape, dtype='float32')
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(kerns.shape)
            conv_grad_i = dnn.GpuDnnConvGradI()(
                kerns,
                out,
                img,
                desc,
            )
            self._compile_and_check(
                [kerns, img, out],
                [conv_grad_i],
                [kern_vals, img_vals, out_vals],
                dnn.GpuDnnConvGradI
            )

    def test_pool(self):
        if not dnn.dnn_available(test_ctx_name):
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        img_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5),
            dtype='float32'
        )

        # 'average_exc_pad' is disabled for versions < 4004
        if dnn.version(raises=False) < 4004:
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
                [dnn.GpuDnnPool(mode=params[2])(img, params[0], params[1], (0, 0))],
                [img_val],
                dnn.GpuDnnPool
            )

    def test_pool_3d(self):
        if not dnn.dnn_available(test_ctx_name):
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor5('img')
        img_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5, 6),
            dtype='float32'
        )

        # 'average_exc_pad' is disabled for versions < 4004
        if dnn.version(raises=False) < 4004:
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
        if not dnn.dnn_available(test_ctx_name):
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
            pool_grad = dnn.GpuDnnPoolGrad(mode=params[2])(
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
        if not dnn.dnn_available(test_ctx_name):
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
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    img = T.ftensor4()
    kern = T.ftensor4()

    dnn.dnn_conv(img, kern, border_mode=1)
    dnn.dnn_conv(img, kern, border_mode=(2, 3))
    dnn.dnn_conv(img, kern, border_mode='full')
    dnn.dnn_conv(img, kern, border_mode='valid')
    dnn.dnn_conv(img, kern, border_mode='half')


def test_dnn_conv_alpha_output_merge():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
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


def test_dnn_conv_grad():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
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
                                  conv_mode='conv')(kern.shape)
        return dnn.GpuDnnConv()(img, kern, out, desc, alpha=0.5, beta=0.75)

    def dconvi(img, kern, out):
        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                                  conv_mode='conv')(kern.shape)
        return dnn.GpuDnnConvGradI()(kern, out, img, desc, alpha=-1.0,
                                     beta=0.0)

    def dconvw(img, kern, out):
        desc = dnn.GpuDnnConvDesc(border_mode='valid', subsample=(1, 1),
                                  conv_mode='conv')(kern.shape)
        return dnn.GpuDnnConvGradW()(img, out, kern, desc, alpha=0.75,
                                     beta=-1.0)

    utt.verify_grad(dconv, [img_val, kern_val, out_val])
    utt.verify_grad(dconvi, [img_val, kern_val, out_val])
    utt.verify_grad(dconvw, [img_val, kern_val, out_val])


def get_conv3d_test_cases():
    # Every element of test_shapes follows the format
    # [input_shape, filter_shape, subsample]
    test_shapes = [[(128, 3, 5, 5, 5), (64, 3, 1, 2, 4), (1, 1, 1)],
                   [(8, 4, 20, 12, 15), (5, 4, 6, 12, 4), (2, 2, 2)],
                   [(8, 1, 20, 12, 15), (5, 1, 6, 12, 4), (3, 3, 3)],
                   [(8, 1, 20, 12, 15), (5, 1, 6, 12, 4), (3, 2, 1)],
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

    itt = chain(product(test_shapes, border_modes, conv_modes),
                product(test_shapes_full, ['full'], conv_modes))

    return itt


def test_conv3d_fwd():

    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)

    def run_conv3d_fwd(inputs_shape, filters_shape, subsample,
                       border_mode, conv_mode):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        # Scale down the input values to prevent very large absolute errors
        # due to float rounding
        inputs_val /= 10
        filters_val /= 10

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)

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

    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)

    def run_conv3d_bwd(inputs_shape, filters_shape, subsample,
                       border_mode, conv_mode):

        inputs_val = numpy.random.random(inputs_shape).astype('float32')
        filters_val = numpy.random.random(filters_shape).astype('float32')

        inputs = theano.shared(inputs_val)
        filters = theano.shared(filters_val)

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
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    assert isinstance(dnn.version(), int)


class test_SoftMax(test_nnet.test_SoftMax):
    gpu_op = dnn.GpuDnnSoftmax
    gpu_grad_op = dnn.GpuDnnSoftmaxGrad
    mode = mode_with_gpu

    def setUp(self):
        if not dnn.dnn_available(test_ctx_name):
            raise SkipTest(dnn.dnn_available.msg)

    def test_softmax_shape_0(self):
        raise SkipTest("Cudnn doesn't support 0 shapes")

    def test_softmax_f16(self):
        x = T.matrix('x', 'float16')
        x_gpu = T.tensor4('x_gpu', 'float16')
        f_z = T.nnet.softmax_op
        f_gpu = dnn.GpuDnnSoftmax(
            'accurate',
            'channel'
        )

        def cmp(n, m, f, f_gpu):
            data = numpy.random.random((n, m)).astype('float16')
            gdata = numpy.asarray(data)[:, :, None, None]

            out = f(data)
            gout = numpy.asarray(f_gpu(gdata))[:, :, 0, 0]
            utt.assert_allclose(out, gout)

        self._test_softmax(x, x_gpu, f_z, f_gpu, cmp)

    def test_softmax_grad(self):
        def cmp(n, m, f, f_gpu):
            data = numpy.arange(n * m, dtype='float32').reshape(n, m)
            gdata = numpy.asarray(data)[:, :, None, None]

            out = f(data)
            gout = numpy.asarray(f_gpu(gdata))[:, :, 0, 0]
            utt.assert_allclose(out, gout)

        x = T.matrix('x', 'float32')
        x_gpu = T.tensor4('x_gpu', 'float32')
        f_z = T.nnet.softmax_op
        f_gpu = dnn.GpuDnnSoftmax(
            'accurate',
            'channel'
        )

        # Verify the grad operation
        dims = (2, 3, 4, 5)
        gdata = numpy.arange(
            numpy.product(dims),
            dtype='float32'
        ).reshape(dims)
        T.verify_grad(f_gpu, [gdata], rng=numpy.random,
                      mode=mode_with_gpu)

        # Verify that the CPU and GPU implementations return the same results
        # up to a tolerance.

        self._test_softmax(
            x,
            x_gpu,
            f_z,
            f_gpu,
            cmp
        )

        self._test_softmax(
            x, x, f_z, f_z, self._cmp
        )

        # Verify that the SoftmaxGrad -> Gpu[Dnn]SoftmaxGrad
        # optimization is applied when cudnn is required
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
                        self.gpu_grad_op)
                    ]) == 1)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.tensor.nnet.SoftmaxGrad)
                    ]) == 0)

        # Verify that the SoftmaxGrad -> Gpu[Dnn]SoftmaxGrad
        # optimization is not applied when cudnn is excluded or not
        # available
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
                        self.gpu_grad_op)
                    ]) == 0)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.tensor.nnet.SoftmaxGrad)
                    ]) == 1)

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
                        self.gpu_grad_op)
                    ]) == 1)
        assert(len([i
                    for i in sorted_f
                    if isinstance(
                        i.op,
                        theano.tensor.nnet.SoftmaxGrad)
                    ]) == 0)

    def test_log_softmax(self):
        # This is a test for an optimization that depends on cuDNN v3 or
        # more recent. Don't test if the cuDNN version is too old.
        if dnn.version(raises=False) < 3000:
            raise SkipTest("Log-softmax is only in cudnn v3+")

        x = T.ftensor4()
        softmax_out = dnn.GpuDnnSoftmax('accurate', 'channel')(x)
        log_out = T.log(T.as_tensor_variable(softmax_out))

        f = theano.function([x], log_out, mode=mode_with_gpu)

        # Ensure that the optimization has been applied
        dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                             isinstance(n.op, dnn.GpuDnnSoftmax)]
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
            expected_out = numpy.log(numpy.exp(input_val) /
                                     numpy.exp(input_val).sum(1)[:, None, :, :])

            utt.assert_allclose(out, expected_out)

    def test_log_softmax2(self):
        # Test that the op LogSoftmax is correctly replaced by the op
        # DnnSoftmax with the 'log' mode.

        # This is a test for an optimization that depends on cuDNN v3 or
        # more recent. Don't test if the cuDNN version is too old.
        if dnn.version(raises=False) < 3000:
            raise SkipTest("Log-softmax is only in cudnn v3+")

        # Compile a reference function, on the CPU, to be used to validate the
        # results of the other function.
        x = T.fmatrix()
        f_ref = theano.function([x], T.nnet.LogSoftmax()(x))

        # Build the first graph and ensure that the optimization is applied
        log_softmax_out = T.nnet.LogSoftmax()(x)
        f = theano.function([x], log_softmax_out, mode=mode_with_gpu)

        dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                             isinstance(n.op, dnn.GpuDnnSoftmax)]
        assert len(dnn_softmax_nodes) == 1
        assert dnn_softmax_nodes[0].op.algo == "log"

        # Compare the output of the function with the reference function
        inp = numpy.random.normal(0, 1, (5, 6)).astype("float32")
        utt.assert_allclose(f(inp), f_ref(inp))

        # Build the first graph and ensure that the optimization is applied
        log_softmax_out = T.log(T.nnet.Softmax()(x))
        f = theano.function([x], log_softmax_out, mode=mode_with_gpu)

        dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                             isinstance(n.op, dnn.GpuDnnSoftmax)]
        assert len(dnn_softmax_nodes) == 1
        assert dnn_softmax_nodes[0].op.algo == "log"

        # Compare the output of the function with the reference function
        inp = numpy.random.normal(0, 1, (5, 6)).astype("float32")
        utt.assert_allclose(f(inp), f_ref(inp))


def test_dnn_batchnorm_train():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    if dnn.version(raises=False) < 5000:
        raise SkipTest("batch normalization requires cudnn v5+")
    utt.seed_rng()

    for mode in ('per-activation', 'spatial'):
        for vartype in (T.ftensor5, T.ftensor4, T.ftensor3, T.fmatrix, T.fvector):
            x, scale, bias = (vartype(n) for n in ('x', 'scale', 'bias'))
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used

            # forward pass
            out, x_mean, x_invstd = dnn.dnn_batch_normalization_train(
                x, scale, bias, mode, eps)
            # reference forward pass
            if mode == 'per-activation':
                axes = (0,)
            elif mode == 'spatial':
                axes = (0,) + tuple(range(2, ndim))
            x_mean2 = x.mean(axis=axes, keepdims=True)
            x_invstd2 = T.inv(T.sqrt(x.var(axis=axes, keepdims=True) + eps))
            scale2 = T.addbroadcast(scale, *axes)
            bias2 = T.addbroadcast(bias, *axes)
            out2 = (x - x_mean2) * (scale2 * x_invstd2) + bias2
            # backward pass
            dy = vartype('dy')
            grads = T.grad(None, wrt=[x, scale, bias], known_grads={out: dy})
            # reference backward pass
            grads2 = T.grad(None, wrt=[x, scale, bias], known_grads={out2: dy})
            # compile
            f = theano.function([x, scale, bias, dy],
                                [out, x_mean, x_invstd, out2, x_mean2, x_invstd2] +
                                grads + grads2, mode=mode_with_gpu)
            # run
            for data_shape in ((5, 10, 30, 40, 10), (4, 3, 1, 1, 1), (1, 1, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(1 if d in axes else s
                                    for d, s in enumerate(data_shape))
                X = 4 + 3 * numpy.random.randn(*data_shape).astype('float32')
                Dy = -1 + 2 * numpy.random.randn(*data_shape).astype('float32')
                Scale = numpy.random.randn(*param_shape).astype('float32')
                Bias = numpy.random.randn(*param_shape).astype('float32')
                outputs = f(X, Scale, Bias, Dy)
                # compare outputs
                utt.assert_allclose(outputs[0], outputs[0 + 3])  # out
                utt.assert_allclose(outputs[1], outputs[1 + 3])  # mean
                utt.assert_allclose(outputs[2], outputs[2 + 3])  # invstd
                # compare gradients
                utt.assert_allclose(outputs[6], outputs[6 + 3], atol=1e-4)  # dx
                utt.assert_allclose(outputs[7], outputs[7 + 3], rtol=2e-4, atol=1e-4)  # dscale
                utt.assert_allclose(outputs[8], outputs[8 + 3])  # dbias


def test_batchnorm_inference():
    if not dnn.dnn_available(test_ctx_name):
        raise SkipTest(dnn.dnn_available.msg)
    if dnn.version(raises=False) < 5000:
        raise SkipTest("batch normalization requires cudnn v5+")
    utt.seed_rng()

    for mode in ('per-activation', 'spatial'):
        for vartype in (T.ftensor5, T.ftensor4, T.ftensor3, T.fmatrix, T.fvector):
            x, scale, bias, mean, var = (vartype(n) for n in ('x', 'scale',
                                                              'bias', 'mean',
                                                              'var'))
            ndim = x.ndim
            eps = 5e-3  # some non-standard value to test if it's used

            # forward pass
            out = dnn.dnn_batch_normalization_test(x, scale, bias, mean,
                                                   var, mode, eps)
            # reference forward pass
            if mode == 'per-activation':
                axes = (0,)
            elif mode == 'spatial':
                axes = (0,) + tuple(range(2, ndim))
            scale2, bias2, mean2, var2 = (T.addbroadcast(t, *axes)
                                          for t in (scale, bias, mean, var))
            out2 = (x - mean2) * (scale2 / T.sqrt(var2 + eps)) + bias2
            # backward pass
            dy = vartype('dy')
            grads = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out: dy})
            # reference backward pass
            grads2 = T.grad(None, wrt=[x, scale, bias, mean, var], known_grads={out2: dy})
            # compile
            f = theano.function([x, scale, bias, mean, var, dy],
                                [out, out2] + grads + grads2, mode=mode_with_gpu)
            # run
            for data_shape in ((10, 20, 30, 40, 10), (4, 3, 1, 1, 1), (1, 1, 5, 5, 5)):
                data_shape = data_shape[:ndim]
                param_shape = tuple(1 if d in axes else s
                                    for d, s in enumerate(data_shape))
                X = 4 + 3 * numpy.random.randn(*data_shape).astype('float32')
                Dy = -1 + 2 * numpy.random.randn(*data_shape).astype('float32')
                Scale = numpy.random.randn(*param_shape).astype('float32')
                Bias = numpy.random.randn(*param_shape).astype('float32')
                Mean = numpy.random.randn(*param_shape).astype('float32')
                Var = numpy.random.rand(*param_shape).astype('float32')
                outputs = f(X, Scale, Bias, Mean, Var, Dy)
                # compare outputs
                utt.assert_allclose(outputs[0], outputs[1])  # out
                # compare gradients
                utt.assert_allclose(outputs[2], outputs[2 + 5], atol=4e-5)  # dx
                utt.assert_allclose(outputs[3], outputs[3 + 5], atol=4e-5)  # dscale
                utt.assert_allclose(outputs[4], outputs[4 + 5])  # dbias
                utt.assert_allclose(outputs[5], outputs[5 + 5])  # dmean
                utt.assert_allclose(outputs[6], outputs[6 + 5], rtol=2e-3, atol=4e-5)  # dvar


def test_dnn_rnn_gru():
    # test params
    input_dim = 32
    hidden_dim = 16
    batch_size = 2
    depth = 3
    timesteps = 5

    # test code
    X = T.tensor3('X')
    Y = T.tensor3('Y')
    h0 = T.tensor3('h0')

    rnnb = dnn.RNNBlock(theano.config.floatX, hidden_dim, depth, 'gru')
    psize = rnnb.get_param_size([batch_size, input_dim])
    params_cudnn = gpuarray_shared_constructor(
        numpy.zeros((psize,), dtype=theano.config.floatX))

    model = Model()
    last_layer = WrapperLayer(X)
    last_dim = input_dim
    for i in range(depth):
        gru = GRU(last_dim, hidden_dim, last_layer, s0=h0[i, :, :])
        model.add_layer(gru)
        last_layer = gru
        last_dim = hidden_dim
        layer_params = gru.get_params()
        dnn_params = rnnb.split_params(params_cudnn, i,
                                       [batch_size, input_dim])
        for j, p in enumerate(dnn_params):
            p[:] = layer_params[j].get_value(borrow=True,
                                             return_internal_type=True)

    def funcs(out, params, hy=None):
        cost = 0
        if out:
            cost += T.mean((Y - out)**2)
        if hy:
            cost += T.mean(hy**2)
        grad = T.grad(cost, [X, h0] + params)
        grad_fn = theano.function([X, Y, h0], grad, mode=mode_with_gpu,
                                  on_unused_input='ignore')
        return grad_fn

    ref_y = last_layer.output()

    # This will grab the hy from the scan implementation
    ref_hy = T.stack([model.layers[0].Y[-1],
                      model.layers[1].Y[-1],
                      model.layers[2].Y[-1]])

    y, hy = rnnb.apply(params_cudnn, X, h0)

    ref_fn = theano.function([X, h0], ref_y, mode=mode_with_gpu)
    cudnn_fn = theano.function([X, h0], y, mode=mode_with_gpu)

    # Test with grad connected to y
    ref_grad_fn = funcs(ref_y, model.get_params())
    cudnn_grad_fn = funcs(y, [params_cudnn])

    # Test with grad connected to both y and hy
    ref2_grad_fn = funcs(ref_y, model.get_params(), ref_hy)
    cudnn2_grad_fn = funcs(y, [params_cudnn], hy)

    # Test with grad connected to hy
    ref3_grad_fn = funcs(None, model.get_params(), ref_hy)
    cudnn3_grad_fn = funcs(None, [params_cudnn], hy)

    ref_grad_fns = [ref_grad_fn, ref2_grad_fn, ref3_grad_fn]
    cudnn_grad_fns = [cudnn_grad_fn, cudnn2_grad_fn, cudnn3_grad_fn]

    x_val = numpy.random.random((timesteps, batch_size, input_dim)).astype(theano.config.floatX)
    y_val = numpy.random.random((timesteps, batch_size, hidden_dim)).astype(theano.config.floatX)
    h0_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)

    ref_out = ref_fn(x_val, h0_val)
    cudnn_out = cudnn_fn(x_val, h0_val)

    utt.assert_allclose(ref_out, cudnn_out)

    for ref_grad_fn, cudnn_grad_fn in zip(ref_grad_fns, cudnn_grad_fns):
        ref_grads = ref_grad_fn(x_val, y_val, h0_val)
        cudnn_grads = cudnn_grad_fn(x_val, y_val, h0_val)

        utt.assert_allclose(ref_grads[0], cudnn_grads[0])
        utt.assert_allclose(ref_grads[1], cudnn_grads[1])

        ref_grad_params = ref_grads[2:]
        cudnn_grad_params = gpuarray_shared_constructor(cudnn_grads[2])

        for i in range(depth):
            cudnn_grad_layer = rnnb.split_params(cudnn_grad_params, i,
                                                 [batch_size, input_dim])
            ref_grad_layer = ref_grad_params[i * len(cudnn_grad_layer):
                                             (i + 1) * len(cudnn_grad_layer)]
            for j, g in enumerate(cudnn_grad_layer):
                utt.assert_allclose(ref_grad_layer[j], g)


def test_dnn_rnn_lstm():
    # test params
    input_dim = 32
    hidden_dim = 16
    batch_size = 2
    depth = 3
    timesteps = 5

    # test code
    X = T.tensor3('X')
    Y = T.tensor3('Y')
    h0 = T.tensor3('h0')
    c0 = T.tensor3('c0')

    rnnb = dnn.RNNBlock(theano.config.floatX, hidden_dim, depth, 'lstm')
    psize = rnnb.get_param_size([batch_size, input_dim])
    params_cudnn = gpuarray_shared_constructor(
        numpy.zeros((psize,), dtype=theano.config.floatX))

    model = Model()
    last_layer = WrapperLayer(X)
    last_dim = input_dim
    for i in range(depth):
        lstm = LSTM(last_dim, hidden_dim, last_layer, s0=h0[i, :, :], c0=c0[i, :, :])
        model.add_layer(lstm)
        last_layer = lstm
        last_dim = hidden_dim
        layer_params = lstm.get_params()
        dnn_params = rnnb.split_params(params_cudnn, i,
                                       [batch_size, input_dim])
        for j, p in enumerate(dnn_params):
            p[:] = layer_params[j].get_value(borrow=True,
                                             return_internal_type=True)

    def funcs(out, params):
        fn = theano.function([X, h0, c0], out, mode=mode_with_gpu)
        cost = T.mean((Y - out)**2)
        grad = T.grad(cost, [X, h0, c0] + params)
        grad_fn = theano.function([X, Y, h0, c0], grad, mode=mode_with_gpu)
        return fn, grad_fn

    ref_fn, ref_grad_fn = funcs(last_layer.output(),
                                model.get_params())
    cudnn_fn, cudnn_grad_fn = funcs(rnnb.apply(params_cudnn, X, h0, c0)[0],
                                    [params_cudnn])

    x_val = numpy.random.random((timesteps, batch_size, input_dim)).astype(theano.config.floatX)
    y_val = numpy.random.random((timesteps, batch_size, hidden_dim)).astype(theano.config.floatX)
    h0_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)
    c0_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)

    ref_out = ref_fn(x_val, h0_val, c0_val)
    cudnn_out = cudnn_fn(x_val, h0_val, c0_val)

    utt.assert_allclose(ref_out, cudnn_out)

    ref_grads = ref_grad_fn(x_val, y_val, h0_val, c0_val)
    cudnn_grads = cudnn_grad_fn(x_val, y_val, h0_val, c0_val)

    utt.assert_allclose(ref_grads[0], cudnn_grads[0])
    utt.assert_allclose(ref_grads[1], cudnn_grads[1])
    utt.assert_allclose(ref_grads[2], cudnn_grads[2])

    ref_grads_params = ref_grads[3:]
    cudnn_grads_params = gpuarray_shared_constructor(cudnn_grads[3])

    for i in range(depth):
        cudnn_grads_layer = rnnb.split_params(cudnn_grads_params, i,
                                              [batch_size, input_dim])
        ref_grads_layer = ref_grads_params[i * len(cudnn_grads_layer):
                                           (i + 1) * len(cudnn_grads_layer)]
        for j, g in enumerate(cudnn_grads_layer):
            utt.assert_allclose(ref_grads_layer[j], g)


def test_dnn_rnn_lstm_grad_c():
    # test params
    input_dim = 32
    hidden_dim = 16
    batch_size = 2
    depth = 3
    timesteps = 5

    # test code
    X = T.tensor3('X')
    CY = T.tensor3('CY')
    h0 = T.tensor3('h0')
    c0 = T.tensor3('c0')

    rnnb = dnn.RNNBlock(theano.config.floatX, hidden_dim, depth, 'lstm')
    psize = rnnb.get_param_size([batch_size, input_dim])
    params_cudnn = gpuarray_shared_constructor(
        numpy.zeros((psize,), dtype=theano.config.floatX))

    model = Model()
    last_layer = WrapperLayer(X)
    last_dim = input_dim
    for i in range(depth):
        lstm = LSTM(last_dim, hidden_dim, last_layer, s0=h0[i, :, :], c0=c0[i, :, :])
        model.add_layer(lstm)
        last_layer = lstm
        last_dim = hidden_dim
        layer_params = lstm.get_params()
        dnn_params = rnnb.split_params(params_cudnn, i,
                                       [batch_size, input_dim])
        for j, p in enumerate(dnn_params):
            p[:] = layer_params[j].get_value(borrow=True,
                                             return_internal_type=True)

    def funcs(out, params):
        cost = T.mean((CY - out)**2)
        grad = T.grad(cost, [X, h0, c0] + params)
        grad_fn = theano.function([X, CY, h0, c0], grad, mode=mode_with_gpu)
        return grad_fn

    _, _, cy = rnnb.apply(params_cudnn, X, h0, c0)
    ref_cy = T.stack([model.layers[0].C[-1],
                      model.layers[1].C[-1],
                      model.layers[2].C[-1]])

    ref_grad_fn = funcs(ref_cy, model.get_params())
    cudnn_grad_fn = funcs(cy, [params_cudnn])

    x_val = numpy.random.random((timesteps, batch_size, input_dim)).astype(theano.config.floatX)
    cy_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)
    h0_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)
    c0_val = numpy.random.random((depth, batch_size, hidden_dim)).astype(theano.config.floatX)

    ref_grads = ref_grad_fn(x_val, cy_val, h0_val, c0_val)
    cudnn_grads = cudnn_grad_fn(x_val, cy_val, h0_val, c0_val)

    utt.assert_allclose(ref_grads[0], cudnn_grads[0])
    utt.assert_allclose(ref_grads[1], cudnn_grads[1])
    utt.assert_allclose(ref_grads[2], cudnn_grads[2])

    ref_grads_params = ref_grads[3:]
    cudnn_grads_params = gpuarray_shared_constructor(cudnn_grads[3])

    for i in range(depth):
        cudnn_grads_layer = rnnb.split_params(cudnn_grads_params, i,
                                              [batch_size, input_dim])
        ref_grads_layer = ref_grads_params[i * len(cudnn_grads_layer):
                                           (i + 1) * len(cudnn_grads_layer)]
        for j, g in enumerate(cudnn_grads_layer):
            utt.assert_allclose(ref_grads_layer[j], g)
