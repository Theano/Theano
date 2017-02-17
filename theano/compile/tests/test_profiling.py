"""
Test of memory profiling

"""
from __future__ import absolute_import, print_function, division

import unittest

import numpy as np

import theano
from six.moves import StringIO
import theano.tensor as T
from theano.ifelse import ifelse


class Test_profiling(unittest.TestCase):
    """
    Test of Theano profiling with min_peak_memory=True
    """

    def test_profiling(self):

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

            inp = [np.arange(1024, dtype='float32') + 1 for i in range(len(x))]
            f(*inp)

            buf = StringIO()
            f.profile.summary(buf)

            # regression testing for future algo speed up
            the_string = buf.getvalue()
            lines1 = [l for l in the_string.split("\n") if "Max if linker" in l]
            lines2 = [l for l in the_string.split("\n") if "Minimum peak" in l]
            if theano.config.device == 'cpu':
                assert "CPU: 4112KB (8204KB)" in the_string, (lines1, lines2)
                assert "CPU: 8204KB (12296KB)" in the_string, (lines1, lines2)
                assert "CPU: 8208KB" in the_string, (lines1, lines2)
                assert "Minimum peak from all valid apply node order is 4104KB" in the_string, (
                    lines1, lines2)
            else:
                assert "CPU: 16KB (16KB)" in the_string, (lines1, lines2)
                assert "GPU: 8204KB (8204KB)" in the_string, (lines1, lines2)
                assert "GPU: 12300KB (12300KB)" in the_string, (lines1, lines2)
                assert "GPU: 8212KB" in the_string, (lines1, lines2)
                assert "Minimum peak from all valid apply node order is 4116KB" in the_string, (
                    lines1, lines2)

        finally:
            theano.config.profile = config1
            theano.config.profile_memory = config2
            theano.config.profiling.min_peak_memory = config3

    def test_ifelse(self):
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

            f_ifelse(val1, val2, big_mat1, big_mat2)

        finally:
            theano.config.profile = config1
            theano.config.profile_memory = config2


if __name__ == '__main__':
    unittest.main()
