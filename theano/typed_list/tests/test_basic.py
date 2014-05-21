import unittest

import numpy

import theano
import theano.typed_list
from theano import tensor as T
from theano.tensor.type_other import SliceType
from theano.typed_list.type import TypedListType
from theano.typed_list.basic import (GetItem, AppendInplace, ExtendInplace,
                                      Append, Extend)
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

        z = GetItem()(mySymbolicMatricesList, mySymbolicSlice)

        self.assertFalse(isinstance(z, T.TensorVariable))

        f = theano.function([mySymbolicMatricesList, mySymbolicSlice],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], slice(0, 1, 1)), [x]))

    def test_sanity_check_single(self):

        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()

        mySymbolicScalar = T.scalar()

        z = GetItem()(mySymbolicMatricesList, mySymbolicScalar)

        f = theano.function([mySymbolicMatricesList, mySymbolicScalar],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], numpy.asarray(0)), x))

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        mySymbolicScalar = T.scalar()

        z = mySymbolicMatricesList[mySymbolicScalar]

        f = theano.function([mySymbolicMatricesList, mySymbolicScalar],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], numpy.asarray(0)), x))

        z = mySymbolicMatricesList[0: 1: 1]

        f = theano.function([mySymbolicMatricesList],
                            z)

        self.assertTrue(numpy.array_equal(f([x]), [x]))

    def test_wrong_input(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        mySymbolicMatrix = T.matrix()

        self.assertRaises(TypeError, GetItem(), mySymbolicMatricesList,
                          mySymbolicMatrix)

    def test_constant_input(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()

        z = GetItem()(mySymbolicMatricesList, 0)

        f = theano.function([mySymbolicMatricesList],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x]), x))

        z = GetItem()(mySymbolicMatricesList, slice(0, 1, 1))

        f = theano.function([mySymbolicMatricesList],
                            z)

        self.assertTrue(numpy.array_equal(f([x]), [x]))


class test_append(unittest.TestCase):

    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = AppendInplace()(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z,
                            accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], y), [x, y]))

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = Append()(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], y), [x, y]))


class test_extend(unittest.TestCase):

    def test_inplace(self):
        mySymbolicMatricesList1 = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        mySymbolicMatricesList2 = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()

        z = ExtendInplace()(mySymbolicMatricesList1, mySymbolicMatricesList2)

        f = theano.function([mySymbolicMatricesList1, mySymbolicMatricesList2],
                            z, accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], [y]), [x, y]))

    def test_sanity_check(self):
        mySymbolicMatricesList1 = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()
        mySymbolicMatricesList2 = TypedListType(T.TensorType(
                            theano.config.floatX, (False, False)))()

        z = Extend()(mySymbolicMatricesList1, mySymbolicMatricesList2)

        f = theano.function([mySymbolicMatricesList1, mySymbolicMatricesList2],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], [y]), [x, y]))
