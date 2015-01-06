import StringIO

import numpy

import theano
import theano.tensor as T
from theano.ifelse import ifelse


def test_reduce():

    config1 = theano.config.profile
    config2 = theano.config.profile_memory
    try:
        theano.config.profile = True
        theano.config.profile_memory = True
        theano.config.profiling.min_peak_memory = True

        x = T.scalar('x')
        y = T.scalar('y')

        z = 5*y + x**2 + y**3 - 4*x

        p = theano.ProfileStats(False)

        if theano.config.mode in ["DebugMode", "DEBUG_MODE", "FAST_COMPILE"]:
            m = "FAST_RUN"
        else:
            m = None

        m=theano.compile.get_mode(m).excluding('fusion', 'inplace')

        f = theano.function([x, y], z, profile=p, name="test_profiling",
                            mode=m)

        out = f(1, 2)

    finally:
        theano.config.profile = config1
        theano.config.profile_memory = config2

if __name__ == "__main__":
    test_reduce()
