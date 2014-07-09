"""
Test of memory profiling

"""
import theano
import theano.tensor as T
import StringIO


def test_profiling():

    old1 = theano.config.profile
    old2 = theano.config.profile_memory
    try:
        theano.config.profile = True
        theano.config.profile_memory = True

        val1 = T.dvector("val1")
        val2 = T.dvector("val2")
        val3 = T.dvector("val3")
        val4 = T.dvector("val4")
        val5 = T.dvector("val5")
        val6 = T.dvector("val6")

        x = [val1, val2, val3, val4, val5, val6]

        z = [x[i] + x[i+1] for i in range(len(x)-1)] + [T.outer(x[i], x[i+1]).sum() for i in range(len(x)-1)]

        p = theano.ProfileStats(False)

        if theano.config.mode in ["DebugMode", "DEBUG_MODE"]:
            m = "FAST_RUN"
        else:
            m = None

        f = theano.function([val1, val2, val3, val4, val5, val6], z, profile=p, name="test_profiling",
                            mode=m)

        output = f([0, 1, 2, 3, 4], [1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9])

        buf = StringIO.StringIO()
        f.profile.summary(buf)

    finally:
        theano.config.profile = old1
        theano.config.profile_memory = old2


if __name__ == '__main__':
    test_profiling()
