import unittest

import numpy as np
from numpy.testing import assert_allclose

from theano import config, function, tensor
from theano.masked.basic import MaskedTensorVariable


class TestAddition(unittest.TestCase):
    def test_add_constant(self):
        x = MaskedTensorVariable(tensor.matrix)
        f = function([x], x + 1)
        data_val = np.random.rand(3, 3).astype(config.floatX)
        mask_val = np.random.randint(0, 2, (3, 3))
        x_val = np.ma.array(data_val, mask=mask_val)
        assert_allclose(np.ma.filled(f(x_val), 0), np.ma.filled(x_val + 1), 0)
