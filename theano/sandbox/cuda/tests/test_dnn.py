import logging

from nose.plugins.skip import SkipTest
import numpy
from itertools import product

import theano
from theano.compat.six import StringIO
from theano.compat.python2x import any
import theano.tensor as T
import theano.tests.unittest_tools as utt
from theano.sandbox.neighbours import images2neibs
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.downsample import DownsampleFactorMaxGrad
import theano.sandbox.cuda.dnn as dnn

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


def pool_2d_i2n(input, ds=(2, 2), strides=None,
                pad=(0, 0),
                pool_function=T.max, mode='ignore_borders'):
    if strides is None:
        strides = ds

    if strides[0] > ds[0] or strides[1] > ds[1]:
        raise RuntimeError(
            "strides should be smaller than or equal to ds,"
            " strides=(%d, %d) and ds=(%d, %d)" %
            (strides + ds))
    shape = input.shape
    if pad != (0, 0):
        assert pool_function is T.max
        pad_x = pad[0]
        pad_y = pad[1]
        a = T.alloc(-numpy.inf, shape[0], shape[1], shape[2] + pad_x * 2,
                    shape[3] + pad_y * 2)
        input = T.set_subtensor(a[:, :,
                                  pad_x:pad_x + shape[2],
                                  pad_y:pad_y + shape[3]],
                                input)
        shape = input.shape

    neibs = images2neibs(input, ds, strides, mode=mode)
    pooled_neibs = pool_function(neibs, axis=1)

    output_width = (shape[2] - ds[0]) // strides[0] + 1
    output_height = (shape[3] - ds[1]) // strides[1] + 1

    pooled_output = pooled_neibs.reshape((shape[0], shape[1],
                                          output_width, output_height))
    return pooled_output


def test_pooling():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

    x = T.ftensor4()
    for func, pad in product((T.max, T.mean),
                             ((0, 0), (1, 0), (1, 0), (2, 3), (3, 2))):
        if pad != (0, 0) and cuda.dnn.version() == -1:
            continue

        if pad != (0, 0) and func is T.mean:
            continue

        for ws in (4, 2, 5):
            for stride in (2, 3):
                if stride > ws:
                    continue
                if func is T.max:
                    if pad[0] > stride or pad[1] > stride:
                        # Not implemented
                        continue
                    # We will check that the opt introduced it.
                    out1 = max_pool_2d(x, (ws, ws),
                                       st=(stride, stride),
                                       ignore_border=True,
                                       padding=pad)
                else:
                    out1 = cuda.dnn.dnn_pool(
                        x, ws=(ws, ws),
                        stride=(stride, stride),
                        pad=pad,
                        mode='max' if func is T.max else "average")
                out2 = pool_2d_i2n(x, ds=(ws, ws), strides=(stride, stride),
                                   pad=pad,
                                   pool_function=func)

                f1 = theano.function([x], out1, mode=mode_with_gpu)
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                            for node in f1.maker.fgraph.apply_nodes])
                f2 = theano.function([x], out2, mode=mode_without_gpu)
                assert not any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                                for node in f2.maker.fgraph.apply_nodes])
                for shp in [(1, 10, 100, 100),
                            (1, 3, 99, 99),
                            (32, 1, 147, 197),
                            ]:
                    data = numpy.random.normal(0, 1, shp).astype("float32")
                    a = f1(data).__array__()

                    b = f2(data).__array__()
                    assert numpy.allclose(a, b,
                                          atol=numpy.finfo(numpy.float32).eps)

        # Test the grad
        for shp in [(1, 1, 2, 2),
                    (1, 1, 3, 3)]:
            data = numpy.random.normal(0, 1, shp).astype("float32") * 10

            ws = 2
            stride = 2
            if pad[0] > stride or pad[1] > stride:
                # Not implemented
                continue

            # This test the CPU grad + opt + GPU implemtentation
            def fn(x):
                return max_pool_2d(x, (ws, ws), ignore_border=True,
                                   padding=pad)
            theano.tests.unittest_tools.verify_grad(fn, [data],
                                                    cast_to_output_type=False,
                                                    mode=mode_with_gpu)
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
                    mode='max' if func is T.max else "average")
                return dnn_op
            theano.tests.unittest_tools.verify_grad(
                fn, [data],
                cast_to_output_type=False,
                mode=mode_with_gpu)
            # Confirm that we get the good op.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                        for node in fg.maker.fgraph.toposort()])
            g_out = fg(data)

            if func is T.max:
                # Compare again the CPU result
                out = max_pool_2d(x, (ws, ws),
                                  padding=pad,
                                  ignore_border=True)
                fc = theano.function([x], theano.grad(out.sum(), x),
                                     mode=mode_without_gpu)
                assert any([isinstance(node.op, DownsampleFactorMaxGrad)
                            for node in fc.maker.fgraph.toposort()])
                c_out = fc(data)
                assert numpy.allclose(c_out, g_out)


def test_pooling_opt():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

    x = T.ftensor4()

    f = theano.function(
        [x],
        max_pool_2d(x, ds=(2, 2), ignore_border=True),
        mode=mode_with_gpu)

    assert any([isinstance(n.op, cuda.dnn.GpuDnnPool)
                for n in f.maker.fgraph.toposort()])

    f = theano.function(
        [x],
        T.grad(max_pool_2d(x, ds=(2, 2), ignore_border=True).sum(), x),
        mode=mode_with_gpu.including("cudnn"))

    assert any([isinstance(n.op, cuda.dnn.GpuDnnPoolGrad)
                for n in f.maker.fgraph.toposort()])


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
            max_pool_2d(x, ds=(2, 2), ignore_border=True),
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
            numpy.random.rand(7, 2, 6, 4),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(8, 2, 4, 3),
            dtype='float32'
        )

        for params in product(
            ['valid', 'full'],
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

    def test_conv_gradw(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        kerns = T.ftensor4('kerns')
        out = T.ftensor4('out')
        img_val = numpy.asarray(
            numpy.random.rand(2, 5, 6, 8),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(2, 1, 5, 6),
            dtype='float32'
        )
        out_vals = numpy.zeros((3, 3, 1, 1), dtype='float32')

        for params in product(
            ['valid', 'full'],
            [(1, 1)],  # strides besides (1, 1)
            ['conv', 'cross']
        ):
            temp_img = img.dimshuffle(1, 0, 2, 3)
            temp_kerns = kerns
            if params[2] == 'conv':
                temp_kerns = temp_kerns[:, :, ::-1, ::-1]
            temp_kerns = temp_kerns.dimshuffle(1, 0, 2, 3)
            shape = (
                kern_vals.shape[1], img_val.shape[1],
                img_val.shape[2] - kern_vals.shape[2] + 1,
                img_val.shape[3] - kern_vals.shape[3] + 1
            )
            out_vals = numpy.zeros(shape, dtype='float32')
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(temp_img.shape, out.shape)
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
            numpy.random.rand(3, 4, 5, 6),
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

    def test_pool(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        img_val = numpy.asarray(
            numpy.random.rand(2, 3, 4, 5),
            dtype='float32'
        )
        for params in product(
            [(1, 1), (2, 2), (3, 3)],
            [(1, 1), (2, 2), (3, 3)],
            ['max', 'average']
        ):
            desc = dnn.GpuDnnPoolDesc(
                ws=params[0],
                stride=params[1],
                mode=params[2]
            )()
            self._compile_and_check(
                [img],
                [dnn.GpuDnnPool()(img, desc)],
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
            ['max', 'average']
        ):
            desc = dnn.GpuDnnPoolDesc(
                ws=params[0],
                stride=params[1],
                mode=params[2]
            )()
            pool_grad = dnn.GpuDnnPoolGrad()(
                img,
                out,
                img_grad,
                desc
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


def test_dnn_conv_merge():
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

    utt.verify_grad(dconv, [img_val, kern_val, out_val])
    utt.verify_grad(dconvi, [img_val, kern_val, out_val])
    utt.verify_grad(dconvw, [img_val, kern_val, out_val])


def test_version():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    assert isinstance(cuda.dnn.version(), (int, tuple))
