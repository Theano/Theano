"""
Test of memory profiling

"""
import StringIO

import numpy

import theano
import theano.tensor as T
from theano.ifelse import ifelse


def test_profiling():

    config1 = theano.config.profile
    config2 = theano.config.profile_memory
    config3 = theano.config.profiling.min_peak_memory
    try:
        theano.config.profile = True
        theano.config.profile_memory = True
        theano.config.profiling.min_peak_memory = True

        x = [T.fvector("val%i" % i) for i in range(3)]

        z = []
        z += [T.outer(x[i], x[i + 1]).sum(axis=1) for i in range(len(x) - 1)]
        z += [x[i] + x[i + 1] for i in range(len(x) - 1)]

        p = theano.ProfileStats(False)

        if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            m = "FAST_RUN"
        else:
            m = None

        f = theano.function(x, z, profile=p, name="test_profiling",
                            mode=m)

        inp = [numpy.arange(1024, dtype='float32') + 1 for i in range(len(x))]
        output = f(*inp)

        buf = StringIO.StringIO()
        f.profile.summary(buf)

        # regression testing for future algo speed up
        the_string = buf.getvalue()
        lines1 = [l for l in the_string.split("\n") if "Max if linker" in l]
        lines2 = [l for l in the_string.split("\n") if "Minimum peak" in l]
        if theano.config.device == 'cpu':
            assert "Max if linker=cvm(default): 4112KB (8204KB)" in the_string, (
                lines1, lines2)
            assert "Minimum peak from all valid apply node order is 4104KB" in the_string, (
                lines1, lines2)
        else:
            assert "Max if linker=cvm(default): 8220KB (8220KB)" in the_string, (
                lines1, lines2)
            assert "Minimum peak from all valid apply node order is 4116KB" in the_string, (
                lines1, lines2)

    finally:
        theano.config.profile = config1
        theano.config.profile_memory = config2
        theano.config.profiling.min_peak_memory = config3


def test_ifelse():
    config1 = theano.config.profile
    config2 = theano.config.profile_memory

    try:
        theano.config.profile = True
        theano.config.profile_memory = True

        a, b = T.scalars('a', 'b')
        x, y = T.scalars('x', 'y')

        z = ifelse(T.lt(a, b), x * 2, y * 2)

        p = theano.ProfileStats(False)

        if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            m = "FAST_RUN"
        else:
            m = None

        f_ifelse = theano.function([a, b, x, y], z, profile=p, name="test_ifelse",
                                   mode=m)

        val1 = 0.
        val2 = 1.
        big_mat1 = 10
        big_mat2 = 11

        out = f_ifelse(val1, val2, big_mat1, big_mat2)

    finally:
        theano.config.profile = config1
        theano.config.profile_memory = config2


if __name__ == '__main__':
    test_profiling()
    test_ifelse()
