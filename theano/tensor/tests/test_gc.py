from __future__ import absolute_import, print_function, division
import numpy
import six.moves.cPickle as pickle
from six.moves import xrange
import theano
from theano import tensor as T
import time


def test_no_reuse():
    x = T.lvector()
    y = T.lvector()
    f = theano.function([x, y], x + y)

    # provide both inputs in the first call
    f(numpy.ones(10, dtype='int64'), numpy.ones(10, dtype='int64'))

    try:
        f(numpy.ones(10))
    except TypeError:
        return
    assert not 'should not get here'


def test_gc_never_pickles_temporaries():
    x = T.dvector()

    for i in xrange(2):  # TODO: 30 causes like LONG compilation due to MERGE
        if i:
            r = r + r/10
        else:
            r = x

    optimizer = None
    optimizer = 'fast_run'

    for f_linker, g_linker in [
            (theano.PerformLinker(allow_gc=True),
             theano.PerformLinker(allow_gc=False)),
            (theano.OpWiseCLinker(allow_gc=True),
             theano.OpWiseCLinker(allow_gc=False))]:

        # f_linker has garbage collection

        # g_linker has no garbage collection

        f = theano.function([x], r, mode=theano.Mode(optimizer=optimizer,
                                                     linker=f_linker))
        g = theano.function([x], r, mode=theano.Mode(optimizer=optimizer,
                                                     linker=g_linker))

        pre_f = pickle.dumps(f)
        pre_g = pickle.dumps(g)
        len_pre_f = len(pre_f)
        len_pre_g = len(pre_g)

        # We can't compare the content or the length of the string
        # between f and g. 2 reason, we store some timming information
        # in float. They won't be the same each time. Different float
        # can have different lenght when printed.

        def a(fn):
            return len(pickle.dumps(fn.maker))
        assert a(f) == a(f)  # some sanity checks on the pickling mechanism
        assert a(g) == a(g)  # some sanity checks on the pickling mechanism

        def b(fn):
            return len(
                pickle.dumps(
                    theano.compile.function_module._pickle_Function(
                        fn)))
        assert b(f) == b(f)  # some sanity checks on the pickling mechanism

        def c(fn):
            return len(pickle.dumps(fn))
        assert c(f) == c(f)  # some sanity checks on the pickling mechanism
        assert c(g) == c(g)  # some sanity checks on the pickling mechanism

        # now run the function once to create temporaries within the no-gc
        # linker
        f(numpy.ones(100, dtype='float64'))
        g(numpy.ones(100, dtype='float64'))

        # serialize the functions again
        post_f = pickle.dumps(f)
        post_g = pickle.dumps(g)
        len_post_f = len(post_f)
        len_post_g = len(post_g)

        # assert that f() didn't cause the function to grow
        # allow_gc should leave the function un-changed by calling
        assert len_pre_f == len_post_f, (len_pre_f, len_post_f)

        # assert that g() didn't cause g to grow because temporaries
        # that weren't collected shouldn't be pickled anyway
        # Allow for a couple of bytes of difference, since timing info,
        # for instance, can be represented as text of varying size.
        assert abs(len_post_f - len_post_g) < 256, (
            f_linker, len_post_f, len_post_g)


def test_merge_opt_runtime():
    """In the original merge optimization, the following graph took
    like caused the MERGE optimizer to exhibit really bad performance
    (quadratic? exponential?)

    Ironically, there is actually no merging to do in this graph.

    """
    x = T.dvector()
    for i in xrange(50):
        if i:
            r = r + r/10
        else:
            r = x
    t = time.time()
    f = theano.function([x], r, mode='FAST_COMPILE')
    # FAST_RUN does in-place optimizer which requires a lot of
    # toposorting, which is actually pretty slow at the moment.  This
    # test was designed to test MergeOptimizer... so I'm leaving
    # toposort optimizations for a later date.
    dt = time.time() - t

    # it should never take longer than 5 seconds to compile this graph
    assert dt < 5.0, dt
