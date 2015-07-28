import os
import pickle
import sys
import unittest

from nose.plugins.skip import SkipTest

import theano
from theano.compat import PY3
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
        pickle.loads(s)

    def test_node_outputs_not_used(self):
        """In the past, we where removing some not used variable from
        fgraph.variables event if the apply had other output used in
        the graph. This caused a crash.

        This test run the pickle that reproduce this case.
        """
        if sys.version_info[:2] < (2, 7):
            raise SkipTest("This test need python 2.7 or more recent.")
        with open(os.path.join(os.path.dirname(__file__),
                               'test_fg_old_crash.pkl'),
                  'rb') as f:
            from theano.misc.pkl_utils import CompatUnpickler
            if PY3:
                u = CompatUnpickler(f, encoding="latin1")
            else:
                u = CompatUnpickler(f)
            d = u.load()
        f = theano.function(**d)
