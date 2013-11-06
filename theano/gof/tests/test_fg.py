import unittest

import theano
from theano.gof import CachedConstantError, FunctionGraph


class TFunctionGraph(unittest.TestCase):
    def test_constant_cache_error(self):
        v = theano.tensor.constant(1)
        assert v.cached
        self.assertRaises(CachedConstantError, FunctionGraph, [], [v + 1],
                          clone=False)

    def test_clone(self):
        v = theano.tensor.constant(1)
        assert v.cached
        FunctionGraph([], [v + 1])
