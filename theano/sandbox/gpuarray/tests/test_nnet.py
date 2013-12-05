from nose.plugins.skip import SkipTest
import numpy

import theano
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt

from theano.sandbox import gpuarray

# We let that import do the init of the back-end if needed.
from theano.sandbox.gpuarray.tests.test_basic_ops import (mode_with_gpu,
                                                          mode_without_gpu)

from theano.sandbox.gpuarray.nnet import (
    GpuCrossentropySoftmaxArgmax1HotWithBias,
    GpuCrossentropySoftmax1HotWithBiasDx)


def test_GpuCrossentropySoftmaxArgmax1HotWithBias():
    """
    This is basic test for GpuCrossentropySoftmaxArgmax1HotWithBias

    We check that we loop when their is too much threads

    """

    n_in = 1000
    batch_size = 4097
    n_out = 1250

    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
        n_in = 4098
        n_out = 4099

    x = T.fmatrix('x')
    y = T.lvector('y')

    b = T.fvector('b')
    #W = T.fmatrix('W')

    #we precompute the dot with big shape before to allow the test of
    #GpuCrossentropySoftmax1HotWithBiasDx to don't fail with the error
    #(the launch timed out and was terminated) on GPU card not
    #powerful enough. We need the big shape to check for corner
    #case.
    dot_result = T.fmatrix('dot_result')

    # Seed numpy.random with config.unittests.rseed
    utt.seed_rng()

    xx = numpy.asarray(numpy.random.rand(batch_size, n_in),
                       dtype=numpy.float32)
    #?????yy = numpy.ones((batch_size,),dtype='float32')
    yy = numpy.ones((batch_size,), dtype='int32')
    b_values = numpy.zeros((n_out,), dtype='float32')
    W_values = numpy.asarray(numpy.random.rand(n_in, n_out), dtype='float32')

    dot_value = numpy.asarray(numpy.dot(xx, W_values), dtype='float32')
    del W_values
    p_y_given_x = T.nnet.softmax(dot_result + b)
    y_pred = T.argmax(p_y_given_x, axis=-1)
    loss = -T.mean(T.log(p_y_given_x)[T.arange(y.shape[0]), y])
    dW = T.grad(loss, dot_result)
    classify = theano.function(inputs=[y, b, dot_result],
                               outputs=[loss, y_pred, dW],
                               mode=mode_without_gpu)
    classify_gpu = theano.function(inputs=[y, b, dot_result],
                                   outputs=[loss, y_pred, dW],
                                   mode=mode_with_gpu)
    #theano.printing.debugprint(classify)
    #theano.printing.debugprint(classify_gpu)

    assert any([isinstance(node.op,
                           T.nnet.CrossentropySoftmaxArgmax1HotWithBias)
                for node in classify.maker.fgraph.toposort()])
    assert any([isinstance(node.op,
                           GpuCrossentropySoftmaxArgmax1HotWithBias)
                for node in classify_gpu.maker.fgraph.toposort()])

    out = classify(yy, b_values, dot_value)
    gout = classify_gpu(yy, b_values, dot_value)

    assert len(out) == len(gout) == 3
    assert numpy.allclose(out[0], gout[0])
    assert numpy.allclose(out[2], gout[2], atol=3e-6), numpy.absolute(
        gout[2] - out[2]).max()
    assert numpy.allclose(out[1], gout[1]), [(id, out[1][id], gout[1][id], val)
                                             for id, val in enumerate(out[1] -
                                                                      gout[1])
                                             if val != 0]


def test_GpuCrossentropySoftmax1HotWithBiasDx():
    """
    This is basic test for GpuCrossentropySoftmax1HotWithBiasDx

    We check that we loop when their is too much threads

    """
    n_in = 1000
    batch_size = 4097
    n_out = 1250

    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
        n_in = 4098
        n_out = 4099

    # Seed numpy.random with config.unittests.rseed
    utt.seed_rng()

    softmax_output_value = numpy.random.rand(batch_size,
                                             n_out).astype('float32')
    dnll_value = numpy.asarray(numpy.random.rand(batch_size), dtype='float32')
    y_idx_value = numpy.random.randint(low=0, high=5, size=batch_size)

    softmax_output = T.fmatrix()
    softmax_output /= softmax_output.sum(axis=1).reshape(
        softmax_output.shape[1], 1)
    op = theano.tensor.nnet.crossentropy_softmax_1hot_with_bias_dx(
        dnll_value,
        softmax_output,
        y_idx_value)

    cpu_f = theano.function([softmax_output], op, mode=mode_without_gpu)
    gpu_f = theano.function([softmax_output], op, mode=mode_with_gpu)
    #theano.printing.debugprint(cpu_f)
    #theano.printing.debugprint(gpu_f)

    assert any([isinstance(node.op, T.nnet.CrossentropySoftmax1HotWithBiasDx)
                for node in cpu_f.maker.fgraph.toposort()])
    assert any([isinstance(node.op,
                           GpuCrossentropySoftmax1HotWithBiasDx)
                for node in gpu_f.maker.fgraph.toposort()])

    cpu_out = cpu_f(softmax_output_value)
    gpu_out = gpu_f(softmax_output_value)

    rtol = 1e-5
    atol = 1e-6
    if not numpy.allclose(cpu_out, gpu_out, rtol=rtol, atol=atol):
        abs_err, rel_err = T.numeric_grad.abs_rel_err(cpu_out, gpu_out)
        scaled_err = numpy.minimum(abs_err / atol, rel_err / rtol)
        max_i = scaled_err.argmax()

        print 'max err index:', max_i, max_i / batch_size,
        print max_i % batch_size, max_i / n_out, max_i & n_out
        print 'At that index:'
        print 'err:', scaled_err.flatten()[max_i]
        print 'absolute error:', abs_err.flatten()[max_i]
        print 'relative error:', rel_err.flatten()[max_i]
        print 'cpu_out:', cpu_out.flatten()[max_i]
        print 'gpu_out:', gpu_out.flatten()[max_i]
        print 'softmax_output_value:', softmax_output_value.flatten()[max_i]
        print 'dnll_value:', dnll_value[max_i / n_out]
        print 'y_idx_value:', y_idx_value[max_i / n_out]

        assert False, "numpy.allclose(cpu_out, gpu_out, rtol=%s, atol=%s)" % (
            rtol, atol)
