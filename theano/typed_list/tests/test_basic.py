import unittest

from nose.plugins.skip import SkipTest
from nose.plugins.attrib import attr
import numpy
from numpy.testing import dec, assert_array_equal, assert_allclose
from numpy.testing.noseclasses import KnownFailureTest

import theano
import theano.typed_list
from theano import tensor as T
from theano.tensor.type_other import SliceType
from theano.typed_list.type import TypedListType
from theano.typed_list.basic import (get_item, append, extend)
from theano.tests import unittest_tools as utt


#took from tensors/tests/test_basic.py
def rand_ranged_matrix(minimum, maximum, shape):
    return numpy.asarray(numpy.random.rand(*shape) * (maximum - minimum)
                         + minimum, dtype=theano.config.floatX)


class test_get_item(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def test_sanity_check_slice(self):

        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()

        mySymbolicSlice = SliceType()()

        z = get_item()(mySymbolicMatricesList, mySymbolicSlice)

        self.assertFalse(isinstance(z, T.TensorVariable))

        f = theano.function([mySymbolicMatricesList, mySymbolicSlice],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 100])

        self.assertTrue(numpy.array_equal(f([x], slice(0, 1, 1)), [x]))

    def test_sanity_check_single(self):

        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()

        mySymbolicScalar = T.scalar()

        z = get_item()(mySymbolicMatricesList, mySymbolicScalar)

        f = theano.function([mySymbolicMatricesList, mySymbolicScalar],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 100])

        self.assertTrue(numpy.array_equal(f([x], numpy.asarray(0)), x))

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        mySymbolicScalar = T.scalar()

        z = mySymbolicMatricesList[mySymbolicScalar]

        f = theano.function([mySymbolicMatricesList, mySymbolicScalar],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 100])

        self.assertTrue(numpy.array_equal(f([x], numpy.asarray(0)), x))

    def test_wrong_input(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        mySymbolicMatrix = T.matrix()

        self.assertRaises(TypeError, get_item(), mySymbolicMatricesList,
                          mySymbolicMatrix)


class test_append(unittest.TestCase):

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = append()(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 100])

        y = rand_ranged_matrix(-1000, 1000, [100, 100])

        self.assertTrue(numpy.array_equal(f([x], y), [x, y]))


class test_extend(unittest.TestCase):

    def test_sanity_check(self):
        mySymbolicMatricesList1 = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        mySymbolicMatricesList2 = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()

        z = extend()(mySymbolicMatricesList1, mySymbolicMatricesList2)

        f = theano.function([mySymbolicMatricesList1, mySymbolicMatricesList2],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 100])

        y = rand_ranged_matrix(-1000, 1000, [100, 100])

        self.assertTrue(numpy.array_equal(f([x], [y]), [x, y]))
