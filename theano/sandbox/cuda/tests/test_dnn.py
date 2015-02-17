import logging
import unittest

from nose.plugins.skip import SkipTest
import numpy
from itertools import product

import theano
from theano.compat.six import StringIO
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt
from theano.sandbox.neighbours import images2neibs, neibs2images
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.downsample import DownsampleFactorMaxGrad
import theano.sandbox.cuda.dnn as dnn
from theano.sandbox.cuda.basic_ops import gpu_contiguous

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
                ignore_border=True,
                pool_function=T.max, mode='ignore_borders'):
    if strides is None:
        strides = ds

    if strides[0] > ds[0] or strides[1] > ds[1]:
        raise RuntimeError(
            "strides should be smaller than or equal to ds,"
            " strides=(%d, %d) and ds=(%d, %d)" %
            (strides + ds))

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
    for func, ignore_border in product(
        (T.max, T.mean), (False, True)):
        for ws in (4, 2, 5):
            for stride in (2, 3):
                if stride > ws:
                    continue
                if func is T.max:
                    # We will check that the opt introduced it.
                    out1 = max_pool_2d(x, (ws, ws),
                                       st=(stride, stride),
                                       ignore_border=ignore_border)
                else:
                    out1 = cuda.dnn.dnn_pool(
                        x, ws=(ws, ws),
                        stride=(stride, stride),
                        ignore_border=ignore_border,
                        mode='max' if func is T.max else "average")

                f1 = theano.function([x], out1, mode=mode_with_gpu)
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                            for node in f1.maker.fgraph.apply_nodes])
                f2 = theano.function([x], out1, mode=mode_without_gpu)
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
            data = numpy.random.normal(0, 1, shp).astype("float32")*10

            ws = 2
            strides = 2

            # This test the CPU grad + opt + GPU implemtentation
            def fn(x):
                return max_pool_2d(x, (ws, ws), ignore_border=ignore_border)
            theano.tests.unittest_tools.verify_grad(fn, [data],
                                                    cast_to_output_type=False,
                                                    mode=mode_with_gpu)
            # Confirm that the opt would have inserted it.
            fg = theano.function([x], theano.grad(fn(x).sum(), x),
                                 mode=mode_with_gpu)
            if ignore_border:
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                            for node in fg.maker.fgraph.toposort()])
            else:
                assert not any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                                for node in fg.maker.fgraph.toposort()])

            # Test the GPU grad + GPU implementation
            def fn(x):
                dnn_op = cuda.dnn.dnn_pool(
                    x, ws=(ws, ws),
                    stride=(stride, stride),
                    ignore_border=ignore_border,
                    mode='max' if func is T.max else "average")
                return dnn_op
            try:
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
                assert ignore_border
            except NotImplementedError:
                assert not ignore_border

            if func is T.max and ignore_border:
                # Compare again the CPU result
                out = max_pool_2d(x, (ws, ws), ignore_border=ignore_border)
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
    We test that if cudnn isn't avail we crash and that if it is avail, we use it.
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
    except (AssertionError, RuntimeError), e:
        assert not cuda.dnn.dnn_available()
        raised = True
    finally:
        theano.config.on_opt_error = old
        logging.getLogger('theano.compile.tests.test_dnn').removeHandler(handler)
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
        img_val = numpy.asarray(
            numpy.random.rand(3, 4, 5, 6),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(3, 4, 5, 6),
            dtype='float32'
        )

        for params in product(
            ['valid', 'full'],
            [(1, 1), (2, 2)],
            ['conv', 'cross']
        ):
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(img.shape, kerns.shape)
            conv = dnn.GpuDnnConv()(img_val, kern_vals, desc)
            self._compile_and_check(
                [img, kerns],
                [conv],
                [img_val, kern_vals],
                dnn.GpuDnnConv
            )

    def test_conv_gradw(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        kerns = T.ftensor4('kerns')
        img_val = numpy.asarray(
            numpy.random.rand(3, 4, 5, 6),
            dtype='float32'
        )
        kern_vals = numpy.asarray(
            numpy.random.rand(3, 4, 5, 6),
            dtype='float32'
        )

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
            shape = theano.tensor.stack(
                temp_kerns.shape[1], temp_img.shape[1],
                temp_img.shape[2] - temp_kerns.shape[2] + 1,
                temp_img.shape[3] - temp_kerns.shape[3] + 1
            )
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(temp_img.shape, shape)
            conv_grad_w = dnn.GpuDnnConvGradW()(
                temp_img,
                temp_kerns,
                desc,
                shape[2],
                shape[3]
            )
            self._compile_and_check(
                [temp_img, temp_kerns],
                [conv_grad_w],
                [img_val, kern_vals],
                dnn.GpuDnnConvGradW
            )

    def test_conv_gradi(self):
        if not dnn.dnn_available():
            raise SkipTest(dnn.dnn_available.msg)
        img = T.ftensor4('img')
        kerns = T.ftensor4('kerns')
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
            print params
            temp_kerns = kerns.dimshuffle(1, 0, 2, 3)
            shape = theano.tensor.stack(
                img.shape[0], temp_kerns.shape[1],
                img.shape[2] + temp_kerns.shape[2] - 1,
                img.shape[3] + temp_kerns.shape[3] - 1
            )
            desc = dnn.GpuDnnConvDesc(
                border_mode=params[0],
                subsample=params[1],
                conv_mode=params[2]
            )(shape, temp_kerns.shape)
            conv_grad_i = dnn.GpuDnnConvGradI()(
                temp_kerns,
                img,
                desc,
                shape[2],
                shape[3]
            )
            self._compile_and_check(
                [temp_kerns, img],
                [conv_grad_i],
                [kern_vals, img_val],
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


def test_version():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    assert isinstance(cuda.dnn.version(), (int, tuple))
