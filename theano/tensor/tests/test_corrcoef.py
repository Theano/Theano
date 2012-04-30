from numpy.testing import assert_allclose

import theano
import theano.tensor as T

import numpy as np


def test_corrcoef_basic():
    data = np.random.randn(10, 4)
    X = T.matrix()
    gt = np.corrcoef(data)
    f = theano.function([X], T.corrcoef(X))
    gv = f(data)
    assert_allclose(gt, gv)
