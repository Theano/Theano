import cPickle
import sys
import numpy
import theano
from theano import tensor as T
import time


def test_no_reuse():
    x = T.lvector()
    y = T.lvector()
    f = theano.function([x, y], x + y)

    #provide both inputs in the first call
    f(numpy.ones(10, dtype='int64'), numpy.ones(10, dtype='int64'))

    try:
        f(numpy.ones(10))
    except TypeError:
        return
    assert not 'should not get here'

def test_gc():
    x = T.dvector()

    #print >> sys.stderr, 'BUILDING GRAPH'
    for i in xrange(2): #TODO: 30 causes like LONG compilation due to MERGE
        if i :
            r = r + r/10
        else:
            r = x

    optimizer=None
    optimizer='fast_run'
    for f_linker, g_linker in [
            (theano.PerformLinker(allow_gc = True), theano.PerformLinker(allow_gc=False)), 
            (theano.OpWiseCLinker(allow_gc = True), theano.OpWiseCLinker(allow_gc=False))]:

        #print >> sys.stderr, 'COMPILING'
        f = theano.function([x], r,mode=theano.Mode(optimizer=optimizer, linker=f_linker))

        g = theano.function([x], r,mode=theano.Mode(optimizer=optimizer, linker=f_linker))

        pre_f = cPickle.dumps(f)
        pre_g = cPickle.dumps(g)

        #print >> sys.stderr, 'RUNNING'
        f(numpy.ones(100, dtype='float64'))
        g(numpy.ones(100, dtype='float64'))

        post_f = cPickle.dumps(f)
        post_g = cPickle.dumps(g)

        #because allow_gc should leave the function un-changed by calling
        assert len(pre_f) == len(post_f)

        #because temporaries that weren't collected shouldn't be pickled anyway
        len_post_f = len(post_f)
        len_post_g = len(post_g)
        assert len_post_f == len_post_g


def test_merge_opt_runtime():
    """In the original merge optimization, the following graph took like caused the MERGE
    optimizer to exhibit really bad performance (quadratic? exponential?)

    Ironically, there is actually no merging to do in this graph.
    """
    x = T.dvector()
    for i in xrange(50):
        if i :
            r = r + r/10
        else:
            r = x
    t = time.time()
    f = theano.function([x], r, mode='FAST_COMPILE')
    # FAST_RUN does in-place optimizer which requires a lot of toposorting, which is actually
    # pretty slow at the moment.  This test was designed to test MergeOptimizer... so I'm
    # leaving toposort optimizations for a later date.
    dt = time.time() - t

    assert dt < 5.0 #it should never take longer than 5 seconds to compile this graph
