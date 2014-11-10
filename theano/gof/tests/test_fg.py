import pickle
import unittest

import theano
from theano.gof import CachedConstantError, FunctionGraph
from theano import tensor as tt


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

    def test_pickle(self):
        v = tt.vector()
        func = theano.gof.FunctionGraph([v], [v + 1])

        s = pickle.dumps(func)
        func2 = pickle.loads(s)
