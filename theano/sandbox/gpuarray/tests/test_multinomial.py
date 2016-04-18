from __future__ import absolute_import, print_function, division

import copy

import numpy

import theano
from theano import config, function, tensor
from ..multinomial import GPUAMultinomialFromUniform
from .config import mode_with_gpu
from theano.compile.mode import get_default_mode, predefined_linkers
import theano.tests.unittest_tools as utt
from .. import pygpu_activated


def get_mode(gpu):
    mode = get_default_mode()
    mode = copy.copy(mode)
    if gpu:
        mode = mode.including('gpuarray', 'gpu_local_optimizations',
                              'local_cut_gpu_host_gpu')
    if isinstance(mode.linker, theano.gof.PerformLinker):
        mode.linker = predefined_linkers['c|py']
    if hasattr(mode.linker, 'c_thunks'):
        mode.linker.c_thunks = True
    return mode


def run_with_c(f, gpu=False):
    mode = get_mode(gpu)
    f(mode, gpu)


def test_multinomial0():
    # This tests the MultinomialFromUniform Op directly, not going through the
    # multinomial() call in GPU random generation.

    p = tensor.fmatrix()
    u = tensor.fvector()

    m = GPUAMultinomialFromUniform('auto')(p, u)

    f = theano.function([p, u], m, mode=mode_with_gpu)
    theano.printing.debugprint(f)
    ret = f(numpy.array([[0.1, 0.2, 0.3, 0.4],
                         [0.1, 0.2, 0.3, 0.4]], dtype='float32'),
            numpy.array([0.05, 0.05], dtype='float32'))
    print(numpy.asarray(ret))


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
    if pygpu_activated:
        run_with_c(body, True)
