"""
Test of memory profiling

"""
import StringIO

import numpy

import theano
import theano.tensor as T


def test_profiling():

    old1 = theano.config.profile
    old2 = theano.config.profile_memory
    try:
        theano.config.profile = True
        theano.config.profile_memory = True

        x = [T.dvector("val%i" % i) for i in range(3)]

        z = [x[i] + x[i+1] for i in range(len(x)-1)]
        z += [T.outer(x[i], x[i+1]).sum() for i in range(len(x)-1)]

        p = theano.ProfileStats(False)

        if theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            m = "FAST_RUN"
        else:
            m = None

        f = theano.function(x, z, profile=p, name="test_profiling",
                            mode=m)

        inp = [numpy.arange(1024) + 1 for i in range(len(x))]
        output = f(*inp)

        buf = StringIO.StringIO()
        f.profile.summary(buf)

    finally:
        theano.config.profile = old1
        theano.config.profile_memory = old2


if __name__ == '__main__':
    test_profiling()
