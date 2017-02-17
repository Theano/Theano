from __future__ import absolute_import, print_function, division
import numpy as np
from numpy.testing import assert_equal, assert_string_equal

import theano
import theano.tensor as tt
import theano.tests.unittest_tools as utt
from theano.tensor import (Subtensor, AdvancedSubtensor, AdvancedSubtensor1,
                           IncSubtensor, AdvancedIncSubtensor,
                           AdvancedIncSubtensor1)


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


def test_empty_list_indexing():
    ynp = np.zeros((2, 2))[:, []]
    znp = np.zeros((2, 2))[:, ()]
    data = [[0, 0], [0, 0]]
    x = tt.dmatrix('x')
    y = x[:, []]
    z = x[:, ()]
    fy = theano.function([x], y)
    fz = theano.function([x], z)
    assert_equal(fy(data).shape, ynp.shape)
    assert_equal(fz(data).shape, znp.shape)


def test_copy():
    x = tt.dmatrix('x')
    data = np.random.rand(5, 5)
    y = x.copy(name='y')
    f = theano.function([x], y)
    assert_equal(f(data), data)
    assert_string_equal(y.name, 'y')


def test_None_dimShuffle_replace():
    # tests replacing None usage in subtensor with dimshuffle
    #
    # tests whenever None is used in subtensor to reshape a variable, it is
    # replaced by dimshuffle. If the replacement is done properly, Subtensor op
    # (or any of its variants) should not be used anymore.

    x = tt.dmatrix('x')
    y = x[:, None, :]
    f = theano.function([x], y)
    for elem in f.maker.fgraph.toposort():
        assert type(elem.op) not in [Subtensor, AdvancedSubtensor,
                                     AdvancedSubtensor1, IncSubtensor,
                                     AdvancedIncSubtensor,
                                     AdvancedIncSubtensor1]

    x = tt.tensor3('x')
    y1 = x[:, :, None, :]
    y2 = x[None, :, :, None, :]
    y3 = x[:, :, None, :, None, None]
    f = theano.function([x], [y1, y2, y3])
    for elem in f.maker.fgraph.toposort():
        assert type(elem.op) not in [Subtensor, AdvancedSubtensor,
                                     AdvancedSubtensor1, IncSubtensor,
                                     AdvancedIncSubtensor,
                                     AdvancedIncSubtensor1]
