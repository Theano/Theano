from __future__ import absolute_import, print_function, division

__author__ = "deepworx"
__contact__ = "deepworx@GitHub"

import unittest
import theano, numpy

class RaiseOnFloat64(unittest.TestCase):
    s = theano.tensor.lscalar()
    a = theano.tensor.alloc(numpy.asarray(5, dtype='float32'), s, s)
    orig = theano.config.warn_float64
    theano.config.warn_float64 = "raise"
    try:
        f = theano.function([s], a.sum())
        theano.printing.debugprint(f)
        f(5)
    finally:
        theano.config.warn_float64 = orig