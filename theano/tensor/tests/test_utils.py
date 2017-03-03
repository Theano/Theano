from __future__ import absolute_import, print_function, division
import unittest

import numpy as np

import theano
from theano.tensor.utils import (hash_from_ndarray, shape_of_variables)


def test_hash_from_ndarray():
    hashs = []
    rng = np.random.rand(5, 5)

    for data in [-2, -1, 0, 1, 2, np.zeros((1, 5)), np.zeros((1, 6)),
                 # Data buffer empty but different shapes
                 np.zeros((1, 0)), np.zeros((2, 0)),
                 # Same data buffer and shapes but different strides
                 np.arange(25).reshape(5, 5),
                 np.arange(25).reshape(5, 5).T,
                 # Same data buffer, shapes and strides but different dtypes
                 np.zeros((5, 5), dtype="uint32"),
                 np.zeros((5, 5), dtype="int32"),

                 # Test slice
                 rng, rng[1:], rng[:4], rng[1:3], rng[::2], rng[::-1]
                 ]:
        data = np.asarray(data)
        hashs.append(hash_from_ndarray(data))

    assert len(set(hashs)) == len(hashs)

    # test that different type of views and their copy give the same hash
    assert hash_from_ndarray(rng[1:]) == hash_from_ndarray(rng[1:].copy())
    assert hash_from_ndarray(rng[1:3]) == hash_from_ndarray(rng[1:3].copy())
    assert hash_from_ndarray(rng[:4]) == hash_from_ndarray(rng[:4].copy())
    assert hash_from_ndarray(rng[::2]) == hash_from_ndarray(rng[::2].copy())
    assert hash_from_ndarray(rng[::-1]) == hash_from_ndarray(rng[::-1].copy())


class Tshape_of_variables(unittest.TestCase):
    def test_simple(self):
        x = theano.tensor.matrix('x')
        y = x+x
        fgraph = theano.FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 5)})
        assert shapes == {x: (5, 5), y: (5, 5)}

        x = theano.tensor.matrix('x')
        y = theano.tensor.dot(x, x.T)
        fgraph = theano.FunctionGraph([x], [y], clone=False)
        shapes = shape_of_variables(fgraph, {x: (5, 1)})
        assert shapes[x] == (5, 1)
        assert shapes[y] == (5, 5)

    def test_subtensor(self):
        x = theano.tensor.matrix('x')
        subx = x[1:]
        fgraph = theano.FunctionGraph([x], [subx], clone=False)
        shapes = shape_of_variables(fgraph, {x: (10, 10)})
        assert shapes[subx] == (9, 10)

    def test_err(self):
        x = theano.tensor.matrix('x')
        subx = x[1:]
        fgraph = theano.FunctionGraph([x], [subx])
        self.assertRaises(ValueError, shape_of_variables, fgraph, {x: (10, 10)})
