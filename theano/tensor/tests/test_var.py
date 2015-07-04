import numpy as np

import theano
import theano.tensor as tt
import theano.tests.unittest_tools as utt


def test_numpy_method():
    # This type of code is used frequently by PyMC3 users
    x = tt.dmatrix('x')
    data = np.random.rand(5, 5)
    for fct in [np.exp]:
        print fct
        y = fct(x)
        f = theano.function([x], y)
        utt.assert_allclose(f(data), fct(data))
