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


def test_log_softmax():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

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
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)

    x = T.ftensor4()
    softmax_out = dnn.GpuDnnSoftmax('bc01', 'accurate', 'channel')(x)
    log_out = T.log(T.as_tensor_variable(softmax_out))

    f = theano.function([x], log_out, mode=mode_with_gpu)

    dnn_softmax_nodes = [n for n in f.maker.fgraph.toposort() if
                         isinstance(n.op, cuda.dnn.GpuDnnSoftmax)]

    # Ensure that the optimization has been applied
    assert len(dnn_softmax_nodes) == 1
    assert dnn_softmax_nodes[0].op.algo == "log"


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


def test_version():
    if not cuda.dnn.dnn_available():
        raise SkipTest(cuda.dnn.dnn_available.msg)
    assert isinstance(cuda.dnn.version(), (int, tuple))
