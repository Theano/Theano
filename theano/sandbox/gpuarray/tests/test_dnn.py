import logging

from nose.plugins.skip import SkipTest
import numpy
from itertools import product

import theano
from theano.compat.six import StringIO
import theano.tensor as T
import theano.tests.unittest_tools as utt
from theano.sandbox.neighbours import images2neibs
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.downsample import DownsampleFactorMaxGrad
from  theano.sandbox import gpuarray
import theano.sandbox.gpuarray.dnn as dnn

# Skip test if pygpu is not available.
if not gpuarray.pygpu_activated:
    raise SkipTest('Optional package pygpu disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').including('gpuarray')
    mode_without_gpu = theano.compile.mode.get_mode(
        'FAST_RUN').excluding('gpuarray')
else:
    default_mode = theano.compile.mode.get_default_mode()
    mode_with_gpu = default_mode.including('gpuarray')
    mode_without_gpu = default_mode.excluding('gpuarray')


def test_log_softmax():
    if not gpuarray.dnn.dnn_available():
        raise SkipTest(gpuarray.dnn.dnn_available.msg)

    x = T.ftensor4()
    softmax_out = dnn.GpuDnnSoftmax('bc01', 'accurate', 'channel')(x)
    log_out = T.log(T.as_tensor_variable(softmax_out))

    f = theano.function([x], log_out, mode=mode_with_gpu)

    # Ensure that the output of the function is valid
    input_shapes = [(3, 4, 5, 6),
                    (1025, 2, 3, 4),
                    (2, 1025, 3, 4),
                    (2, 3, 1025, 4),
                    (2, 3, 4, 1025),
                    (66000, 2, 3, 4),
                    (2, 66000, 3, 4),
                    (2, 3, 66000, 4),
                    (2, 3, 4, 66000),]

    for inp_shape in input_shapes:
        input_val = numpy.random.normal(0, 1, inp_shape).astype("float32")

        out = f(input_val)
        expected_out = numpy.log(numpy.exp(input_val) /
                                 numpy.exp(input_val).sum(1)[:, None, :, :])

        utt.assert_allclose(out, expected_out)


def test_log_softmax_opt():
    if not gpuarray.dnn.dnn_available():
        raise SkipTest(gpuarray.dnn.dnn_available.msg)

    x = T.ftensor4()
    softmax_out = dnn.GpuDnnSoftmax('bc01', 'accurate', 'channel')(x)
    log_out = T.log(T.as_tensor_variable(softmax_out))

    f = theano.function([x], log_out, mode=mode_with_gpu)

    dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                         isinstance(n.op, gpuarray.dnn.GpuDnnSoftmax)]

    # Ensure that the optimization has been applied
    assert len(dnn_softmax_nodes) == 1
    assert dnn_softmax_nodes[0].op.algo == "log"


def test_version():
    if not gpuarray.dnn.dnn_available():
        raise SkipTest(gpuarray.dnn.dnn_available.msg)
    assert isinstance(gpuarray.dnn.version(), (int, tuple))
