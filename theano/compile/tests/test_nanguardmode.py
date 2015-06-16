"""
This test is for testing the NanGuardMode.
"""
from theano.compile.nanguardmode import NanGuardMode
import numpy
import theano
import theano.tensor as T


def test_NanGuardMode():
    """
    Tests if NanGuardMode is working by feeding in numpy.inf and numpy.nans
    intentionally. A working implementation should be able to capture all
    the abnormalties.
    """
    x = T.matrix()
    w = theano.shared(numpy.random.randn(5, 7).astype(theano.config.floatX))
    y = T.dot(x, w)

    fun = theano.function(
        [x], y,
        mode=NanGuardMode(nan_is_error=True, inf_is_error=True)
    )
    a = numpy.random.randn(3, 5).astype(theano.config.floatX)
    infa = numpy.tile(
        (numpy.asarray(100.) ** 1000000).astype(theano.config.floatX), (3, 5))
    nana = numpy.tile(
        numpy.asarray(numpy.nan).astype(theano.config.floatX), (3, 5))
    biga = numpy.tile(
        numpy.asarray(1e20).astype(theano.config.floatX), (3, 5))

    work = [False, False, False]

    fun(a)  # normal values
    try:
        fun(infa)  # INFs
    except AssertionError:
        work[0] = True
    try:
        fun(nana)  # NANs
    except AssertionError:
        work[1] = True
    try:
        fun(biga)  # big values
    except AssertionError:
        work[2] = True

    if not (work[0] and work[1] and work[2]):
        raise AssertionError("NanGuardMode not working.")
