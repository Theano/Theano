from __future__ import absolute_import, print_function, division
import numpy as np
from numpy.testing import assert_equal, assert_string_equal

import theano
import theano.tensor as tt
import theano.tests.unittest_tools as utt


def test_numpy_method():
    # This type of code is used frequently by PyMC3 users
    x = tt.dmatrix('x')
    data = np.random.rand(5, 5)
    x.tag.test_value = data
    for fct in [np.arccos, np.arccosh, np.arcsin, np.arcsinh,
                np.arctan, np.arctanh, np.ceil, np.cos, np.cosh, np.deg2rad,
                np.exp, np.exp2, np.expm1, np.floor, np.log,
                np.log10, np.log1p, np.log2, np.rad2deg,
                np.sin, np.sinh, np.sqrt, np.tan, np.tanh, np.trunc]:
        y = fct(x)
        f = theano.function([x], y)
        utt.assert_allclose(np.nan_to_num(f(data)),
                            np.nan_to_num(fct(data)))


def test_copy():
    x = tt.dmatrix('x')
    data = np.random.rand(5, 5)
    y = x.copy(name='y')
    f = theano.function([x], y)
    assert_equal(f(data), data)
    assert_string_equal(y.name, 'y')
