import numpy as N
import unittest
from theano.tests import unittest_tools as utt
import theano
import theano.tensor as T

class Test_incsubtensor(unittest.TestCase):
    """Partial testing.

    What could be tested:
    - increment vs set
    - thing incremented: scalar, vector, matrix, 
    - increment/set: constant, scalar, vector, matrix
    - indices: scalar vs slice, constant vs variable, out of bound, ...
    - inplace
    """
    def setUp(self):
        utt.seed_rng()

    def test_simple_ok(self):
        """Increments or sets part of a tensor by a scalar using full slice and
        a partial slice depending on a scalar.
        """
        a = T.dmatrix()
        increment = T.dscalar()
        sl1 = slice(None)
        sl2_end = T.lscalar()
        sl2 = slice(sl2_end)

        for do_set in [False,True]:
            a_incremented = T.incsubtensor(a, increment, [sl1, sl2], set_instead_of_inc=do_set)
            f = theano.function([a, increment, sl2_end], a_incremented)

            val_a = N.ones((5,5))
            val_inc = 2.3
            val_sl2_end = 2

            result = f(val_a, val_inc, val_sl2_end)

            expected_result = N.copy(val_a)
            if do_set:
                expected_result[:,:val_sl2_end] = val_inc
            else:
                expected_result[:,:val_sl2_end] += val_inc
           
            self.failUnless(N.array_equal(result, expected_result))
        return

