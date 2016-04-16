from __future__ import absolute_import, print_function, division
import os
import numpy
import theano
import theano.tensor as T

floatX = 'float32'


def test_graph_opt_caching():
    opt_db_file = theano.config.compiledir + '/optimized_graphs.pkl'
    os.system('rm %s' % opt_db_file)

    mode = theano.config.mode
    if mode in ["DEBUG_MODE", "DebugMode"]:
        mode = "FAST_RUN"
    default = theano.config.cache_optimizations
    try:
        theano.config.cache_optimizations = True
        a = T.fmatrix('a')
        b = T.fmatrix('b')
        c = theano.shared(numpy.ones((10, 10), dtype=floatX))
        d = theano.shared(numpy.ones((10, 10), dtype=floatX))
        e = T.sum(T.sum(T.sum(a ** 2 + b) + c) + d)
        f1 = theano.function([a, b], e, mode=mode)

        m = T.fmatrix('x1')
        n = T.fmatrix('x2')
        p = theano.shared(numpy.ones((10, 10), dtype=floatX))
        q = theano.shared(numpy.ones((10, 10), dtype=floatX))
        j = T.sum(T.sum(T.sum(m ** 2 + n) + p) + q)
        f2 = theano.function([m, n], j, mode=mode)

        in1 = numpy.ones((10, 10), dtype=floatX)
        in2 = numpy.ones((10, 10), dtype=floatX)
        assert f1(in1, in2) == f2(in1, in2)
    finally:
        theano.config.cache_optimizations = default

if __name__ == '__main__':
    test_graph_opt_caching()
