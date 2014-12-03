import logging
import unittest

from nose.plugins.skip import SkipTest
import numpy

import theano
from theano.compat.six import StringIO
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt
from theano.sandbox.neighbours import images2neibs, neibs2images
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.downsample import DownsampleFactorMaxGrad


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

    for func in (T.max, T.mean):
        for ws in (2, 4, 5):
            for stride in (2, 3):
                if stride > ws:
                    continue
                if ws == stride and func is T.max:
                    # We will check that the opt introduced it.
                    out1 = max_pool_2d(x, (ws, ws), ignore_border=True)
                else:
                    out1 = cuda.dnn.dnn_pool(
                        x, ws=(ws, ws),
                        stride=(stride, stride),
                        mode='max' if func is T.max else "average")
                out2 = pool_2d_i2n(x, ds=(ws, ws), strides=(stride, stride),
                                   pool_function=func)

                f1 = theano.function([x], out1, mode=mode_with_gpu)
                assert any([isinstance(node.op, cuda.dnn.GpuDnnPool)
                            for node in f1.maker.fgraph.apply_nodes])
                f2 = theano.function([x], out2, mode=mode_with_gpu)
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
                return max_pool_2d(x, (ws, ws), ignore_border=True)
            theano.tests.unittest_tools.verify_grad(fn, [data],
                                                    cast_to_output_type=False,
                                                    mode=mode_with_gpu)
            # Confirm that the opt would have inserted it.
            f = theano.function([x], theano.grad(fn(x).sum(), x),
                                mode=mode_with_gpu)
            assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                        for node in f.maker.fgraph.toposort()])

            # Test the GPU grad + GPU implementation
            def fn(x):
                dnn_op = cuda.dnn.dnn_pool(
                    x, ws=(ws, ws),
                    stride=(stride, stride),
                    mode='max' if func is T.max else "average")
                return dnn_op
            theano.tests.unittest_tools.verify_grad(fn, [data],
                                                    cast_to_output_type=False,
                                                    mode=mode_with_gpu)
            # Confirm that we get the good op.
            f = theano.function([x], theano.grad(fn(x).sum(), x),
                                mode=mode_with_gpu)
            assert any([isinstance(node.op, cuda.dnn.GpuDnnPoolGrad)
                        for node in f.maker.fgraph.toposort()])
            g_out = f(data)

            if func is T.max:
                # Compare again the CPU result
                out = max_pool_2d(x, (ws, ws), ignore_border=True)
                f = theano.function([x], theano.grad(out.sum(), x),
                                    mode=mode_without_gpu)
                assert any([isinstance(node.op, DownsampleFactorMaxGrad)
                            for node in f.maker.fgraph.toposort()])
                c_out = f(data)
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
