from __future__ import absolute_import, print_function, division

import numpy

import theano
from theano import config, function, tensor
from ..multinomial import GPUAMultinomialFromUniform
import theano.tests.unittest_tools as utt
from .config import mode_with_gpu, mode_without_gpu


def get_mode(gpu):
    mode = mode_without_gpu
    if gpu:
        mode = mode_with_gpu
    return mode


def run_with_c(f, gpu=False):
    mode = get_mode(gpu)
    f(mode, gpu)


def test_multinomial_0():
    # This tests the MultinomialFromUniform Op directly, not going through the
    # multinomial() call in GPU random generation.

    p = tensor.fmatrix()
    u = tensor.fvector()

    m = theano.sandbox.multinomial.MultinomialFromUniform('auto')(p, u)

    def body(mode, gpu):
        # the m*2 allows the multinomial to reuse output
        f = function([p, u], m * 2, allow_input_downcast=True, mode=mode)

        if gpu:
            assert any([type(node.op) is GPUAMultinomialFromUniform
                        for node in f.maker.fgraph.toposort()])

        # test that both first and second samples can be drawn
        utt.assert_allclose(f([[1, 0], [0, 1]], [.1, .1]),
                            [[2, 0], [0, 2]])

        # test that both second labels can be drawn
        r = f([[.2, .8], [.3, .7]], [.31, .31])
        utt.assert_allclose(r, [[0, 2], [0, 2]])

        # test that both first labels can be drawn
        r = f([[.2, .8], [.3, .7]], [.21, .21])
        utt.assert_allclose(r, [[0, 2], [2, 0]])

        # change the size to make sure output gets reallocated ok
        # and also make sure that the GPU version doesn't screw up the
        # transposed-ness
        r = f([[.2, .8]], [.25])
        utt.assert_allclose(r, [[0, 2]])

    run_with_c(body)
    run_with_c(body, True)


# TODO: check a bigger example (make sure blocking on GPU is handled correctly)
def test_multinomial_large():
    # DEBUG_MODE will test this on GPU
    def body(mode, gpu):
        p = tensor.fmatrix()
        u = tensor.fvector()
        m = theano.sandbox.multinomial.MultinomialFromUniform('auto')(p, u)
        f = function([p, u], m * 2, allow_input_downcast=True, mode=mode)
        if gpu:
            assert any([type(node.op) is GPUAMultinomialFromUniform
                        for node in f.maker.fgraph.toposort()])

        pval = numpy.arange(10000 * 4,
                            dtype='float32').reshape((10000, 4)) + 0.1
        pval = pval / pval.sum(axis=1)[:, None]
        uval = numpy.ones_like(pval[:, 0]) * 0.5
        mval = f(pval, uval)

        assert mval.shape == pval.shape
        if config.cast_policy == 'custom':
            assert mval.dtype == pval.dtype
        elif config.cast_policy == 'numpy+floatX':
            assert mval.dtype == config.floatX
        elif config.cast_policy == 'numpy':
            assert mval.dtype == 'float64'
        else:
            raise NotImplementedError(config.cast_policy)
        utt.assert_allclose(mval.sum(axis=1), 2)
        asdf = numpy.asarray([0, 0, 2, 0]) + 0 * pval
        utt.assert_allclose(mval, asdf)  # broadcast over all rows
    run_with_c(body)
    run_with_c(body, True)


def test_gpu_opt():
    # Does have some overlap with test_multinomial_0

    # We test the case where we put the op on the gpu when the output
    # is moved to the gpu.
    p = tensor.fmatrix()
    u = tensor.fvector()
    m = theano.sandbox.multinomial.MultinomialFromUniform('auto')(p, u)
    assert m.dtype == 'float32', m.dtype

    f = function([p, u], m, allow_input_downcast=True, mode=get_mode(True))
    assert any([type(node.op) is GPUAMultinomialFromUniform
                for node in f.maker.fgraph.toposort()])
    pval = numpy.arange(10000 * 4, dtype='float32').reshape((10000, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    f(pval, uval)

    # Test with a row, it was failing in the past.
    r = tensor.frow()
    m = theano.sandbox.multinomial.MultinomialFromUniform('auto')(r, u)
    assert m.dtype == 'float32', m.dtype

    f = function([r, u], m, allow_input_downcast=True, mode=get_mode(True))
    assert any([type(node.op) is GPUAMultinomialFromUniform
                for node in f.maker.fgraph.toposort()])
    pval = numpy.arange(1 * 4, dtype='float32').reshape((1, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    f(pval, uval)
