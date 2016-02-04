from __future__ import absolute_import, print_function, division
from nose.plugins.skip import SkipTest
import numpy
import unittest

import theano
import theano.tensor as T
import theano.tests.unittest_tools as utt

# Skip test if cuda_ndarray is not available.
import theano.sandbox.cuda as cuda
if not cuda.cuda_available:
    raise SkipTest('Optional package cuda disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode('FAST_RUN').including('gpu')
    # We should not exclude the 'gpu' tag, as some CPU opt are tagged
    # as GPU to make them run in fast_compile with gpu.

    mode_without_gpu = theano.compile.mode.get_mode('FAST_RUN')
else:
    mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')
    mode_without_gpu = theano.compile.mode.get_default_mode()


def test_GpuCrossentropySoftmaxArgmax1HotWithBias():
    """
    This is basic test for GpuCrossentropySoftmaxArgmax1HotWithBias

    We check that we loop when there are too many threads

    """

    n_in = 1000
    batch_size = 4097
    n_out = 1250

    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
        n_in = 4098
        n_out = 4099

    y = T.lvector('y')

    b = T.fvector('b')

    # we precompute the dot with big shape before to allow the test of
    # GpuCrossentropySoftmax1HotWithBiasDx to don't fail with the error
    # (the launch timed out and was terminated) on GPU card not
    # powerful enough. We need the big shape to check for corner
    # case.
    dot_result = T.fmatrix('dot_result')

    # Seed numpy.random with config.unittests.rseed
    utt.seed_rng()

    xx = numpy.asarray(numpy.random.rand(batch_size, n_in),
                       dtype=numpy.float32)
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
    # theano.printing.debugprint(classify)
    # theano.printing.debugprint(classify_gpu)

    assert any([isinstance(node.op,
                           T.nnet.CrossentropySoftmaxArgmax1HotWithBias)
                for node in classify.maker.fgraph.toposort()])
    assert any([isinstance(node.op,
                           cuda.nnet.GpuCrossentropySoftmaxArgmax1HotWithBias)
                for node in classify_gpu.maker.fgraph.toposort()])

    out = classify(yy, b_values, dot_value)
    gout = classify_gpu(yy, b_values, dot_value)

    assert len(out) == len(gout) == 3
    assert numpy.allclose(out[0], gout[0])
    assert numpy.allclose(out[2], gout[2], atol=3e-6), numpy.absolute(
        gout - out).max()
    assert numpy.allclose(out[1], gout[1]), [(id, out[1][id], gout[1][id], val)
                                             for id, val in enumerate(out[1] -
                                                                      gout[1])
                                             if val != 0]


def test_GpuCrossentropySoftmax1HotWithBiasDx():
    """
    This is basic test for GpuCrossentropySoftmax1HotWithBiasDx

    We check that we loop when there are too many threads

    """
    batch_size = 4097
    n_out = 1250

    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
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
    # theano.printing.debugprint(cpu_f)
    # theano.printing.debugprint(gpu_f)

    assert any([isinstance(node.op, T.nnet.CrossentropySoftmax1HotWithBiasDx)
                for node in cpu_f.maker.fgraph.toposort()])
    assert any([isinstance(node.op,
                           cuda.nnet.GpuCrossentropySoftmax1HotWithBiasDx)
                for node in gpu_f.maker.fgraph.toposort()])

    cpu_out = cpu_f(softmax_output_value)
    gpu_out = gpu_f(softmax_output_value)

    rtol = 1e-5
    atol = 1e-6
    if not numpy.allclose(cpu_out, gpu_out, rtol=rtol, atol=atol):
        abs_err, rel_err = T.numeric_grad.abs_rel_err(cpu_out, gpu_out)
        scaled_err = numpy.minimum(abs_err / atol, rel_err / rtol)
        max_i = scaled_err.argmax()

        print('max err index:', max_i, max_i / batch_size, end=' ')
        print(max_i % batch_size, max_i / n_out, max_i & n_out)
        print('At that index:')
        print('err:', scaled_err.flatten()[max_i])
        print('absolute error:', abs_err.flatten()[max_i])
        print('relative error:', rel_err.flatten()[max_i])
        print('cpu_out:', cpu_out.flatten()[max_i])
        print('gpu_out:', gpu_out.flatten()[max_i])
        print('softmax_output_value:', softmax_output_value.flatten()[max_i])
        print('dnll_value:', dnll_value[max_i / n_out])
        print('y_idx_value:', y_idx_value[max_i / n_out])

        assert False, "numpy.allclose(cpu_out, gpu_out, rtol=%s, atol=%s)" % (
            rtol, atol)


def test_softmax_with_bias():
    """
    This is basic test for GpuSoftmaxWithBias

    We check that we loop when their is too much block

    TODO: check that we loop when there are too many threads.(THIS IS
    NOT IMPLEMENTED)
    """
    x = T.fmatrix('x')
    # We can't use zeros_like(x[0,::]) as this don't allow to test with
    # 0 shape.
    z = T.nnet.softmax_with_bias(x, T.arange(x.shape[1] * 2,
                                             dtype='float32')[::2])

    f = theano.function([x], z, mode=mode_without_gpu)
    f_gpu = theano.function([x], z, mode=mode_with_gpu)
    assert f.maker.fgraph.toposort()[-1].op == T.nnet.softmax_with_bias
    assert isinstance(f_gpu.maker.fgraph.toposort()[-2].op,
                      cuda.nnet.GpuSoftmaxWithBias)

    def cmp(n, m):
        # print "test_softmax",n,m
        data = numpy.arange(n * m, dtype='float32').reshape(n, m)
        out = f(data)
        gout = f_gpu(data)
        assert numpy.allclose(out, gout), numpy.absolute(out - gout)

    cmp(2, 5)
    # we need to test n>32*1024 to check that we make the block loop.
    cmp(2 << 15, 5)
    cmp(4074, 400)
    cmp(0, 10)
    cmp(784, 784)
    cmp(4, 1000)
    cmp(4, 1024)
    cmp(4, 2000)
    cmp(4, 2024)
    # GTX285 don't have enough shared mem for this case.
    cmp(4, 4074)
    # The GTX580, 680 and kepler don't have enough shared memory.
    cmp(2, 10000)
    cmp(128, 16 * 1024)
    cmp(128, 64 * 1024)


class test_SoftMax(unittest.TestCase):
    gpu_op = cuda.nnet.GpuSoftmax
    mode = mode_with_gpu.excluding("cudnn")
    do_big = True
    do_0 = True
    topo_idx = -2

    def _test_softmax(
        self,
        x,
        x_gpu,
        f_z,
        f_gpu_z,
        cmp,
        check_types
    ):
        """
        This is basic test for GpuSoftmax and GpuDnnSoftmax

        We check that we loop when there is too much block
        We use slower code when there isn't enough shared memory
        """
        f_z_out = f_z(x)
        f_gpu_z_out = f_gpu_z(x_gpu)

        f = theano.function([x], f_z_out, mode=mode_without_gpu)
        f_gpu = theano.function([x_gpu], f_gpu_z_out, mode=self.mode)
        check_types(f, f_gpu)

        # we need to test n>32*1024 to check that we make the block loop.
        cmp(1, 5, f, f_gpu)
        cmp(2, 5, f, f_gpu)
        cmp(10, 5, f, f_gpu)
        cmp(100, 5, f, f_gpu)
        cmp(1000, 5, f, f_gpu)
        cmp(10000, 5, f, f_gpu)
        cmp(4074, 400, f, f_gpu)
        cmp(784, 784, f, f_gpu)
        cmp(4, 1000, f, f_gpu)
        cmp(4, 1024, f, f_gpu)
        cmp(4, 2000, f, f_gpu)
        cmp(4, 2024, f, f_gpu)
        # The GTX285 don't have enough shared memory.
        cmp(4, 4074, f, f_gpu)
        # The GTX580, 680 and kepler don't have enough shared memory.
        cmp(2, 10000, f, f_gpu)
        cmp(128, 16 * 1024, f, f_gpu)
        cmp(128, 64 * 1024, f, f_gpu)
        # cudnn permits no more than 2^15 - 1 rows
        cmp((2 << 15) - 1, 5, f, f_gpu)
        cmp(5, 2 << 15, f, f_gpu)

        return f, f_gpu

    def _cmp(self, n, m, f, f_gpu):
        data = numpy.arange(n * m, dtype='float32').reshape(n, m)
        out = f(data)
        gout = f_gpu(data)
        utt.assert_allclose(out, gout)

    def _check_types(self, graph, graph_gpu, f_type, f_gpu_type):
        assert isinstance(graph.maker.fgraph.toposort()[-1].op, f_type)
        assert isinstance(
            graph_gpu.maker.fgraph.toposort()[self.topo_idx].op,
            f_gpu_type
        )

    def test_softmax(self):
        x = T.fmatrix('x')
        z = T.nnet.softmax_op

        def check_types(graph, graph_gpu):
            self._check_types(
                graph,
                graph_gpu,
                type(z),
                self.gpu_op
            )

        f, f_gpu = self._test_softmax(
            x,
            x,
            z,
            z,
            self._cmp,
            check_types
        )

        if self.do_big:
            self._cmp(2 << 15, 5, f, f_gpu)
        if self.do_0:
            self._cmp(0, 10, f, f_gpu)
