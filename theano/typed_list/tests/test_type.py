from __future__ import absolute_import, print_function, division
import unittest

import numpy

import theano
import theano.typed_list
from theano import tensor as T
from theano.typed_list.type import TypedListType
from theano.tests import unittest_tools as utt


# took from tensors/tests/test_basic.py
def rand_ranged_matrix(minimum, maximum, shape):
    return numpy.asarray(numpy.random.rand(*shape) * (maximum - minimum) +
                         minimum, dtype=theano.config.floatX)


class test_typed_list_type(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def test_wrong_input_on_creation(self):
        """
        Typed list type should raises an
        error if the argument passed for
        type is not a valid theano type
        """

        self.assertRaises(TypeError, TypedListType, None)

    def test_wrong_input_on_filter(self):
        """
        Typed list type should raises an
        error if the argument given to filter
        isn't of the same type as the one
        specified on creation
        """

        # list of matrices
        myType = TypedListType(T.TensorType(theano.config.floatX,
                                            (False, False)))

        self.assertRaises(TypeError, myType.filter, [4])

    def test_not_a_list_on_filter(self):
        """
        Typed List Value should raises an error
        if no iterable variable is given on input
        """

        # list of matrices
        myType = TypedListType(T.TensorType(theano.config.floatX,
                                            (False, False)))

        self.assertRaises(TypeError, myType.filter, 4)

    def test_type_equality(self):
        """
        Typed list types should only be equal
        when they contains the same theano
        variables
        """
        # list of matrices
        myType1 = TypedListType(T.TensorType(theano.config.floatX,
                                             (False, False)))
        # list of matrices
        myType2 = TypedListType(T.TensorType(theano.config.floatX,
                                             (False, False)))
        # list of scalars
        myType3 = TypedListType(T.TensorType(theano.config.floatX,
                                             ()))

        self.assertTrue(myType2 == myType1)
        self.assertFalse(myType3 == myType1)

    def test_filter_sanity_check(self):
        """
        Simple test on typed list type filter
        """
        myType = TypedListType(T.TensorType(theano.config.floatX,
                                            (False, False)))

        x = rand_ranged_matrix(-1000, 1000, [100, 100])

        self.assertTrue(numpy.array_equal(myType.filter([x]), [x]))

    def test_intern_filter(self):
        """
        Test checking if values contained are themselves
        filtered. If they weren't this code would raise
        an exception.
        """
        myType = TypedListType(T.TensorType('float64',
                                            (False, False)))

        x = numpy.asarray([[4, 5], [4, 5]], dtype='float32')

        self.assertTrue(numpy.array_equal(myType.filter([x]), [x]))

    # Will fail for unknown reasons
    # under search
    """
    def test_load(self):
        myType = TypedListType(T.TensorType(theano.config.floatX,
                                            (False, False)))

        x = rand_ranged_matrix(-1000, 1000, [100, 100])
        testList = []
        for i in range(10000):
            testList.append(x)

        self.assertTrue(numpy.array_equal(myType.filter(testList), testList))
    """

    def test_basic_nested_list(self):
        """
        Testing nested list with one level of depth
        """
        myNestedType = TypedListType(T.TensorType(theano.config.floatX,
                                                  (False, False)))

        myType = TypedListType(myNestedType)

        x = rand_ranged_matrix(-1000, 1000, [100, 100])

        self.assertTrue(numpy.array_equal(myType.filter([[x]]), [[x]]))

    def test_comparison_different_depth(self):
        """
        Nested list with different depth aren't the same
        """
        myNestedType = TypedListType(T.TensorType(theano.config.floatX,
                                                  (False, False)))

        myNestedType2 = TypedListType(myNestedType)

        myNestedType3 = TypedListType(myNestedType2)

        self.assertFalse(myNestedType2 == myNestedType3)

    def test_nested_list_arg(self):
        """
        test for the 'depth' optionnal argument
        """
        myNestedType = TypedListType(T.TensorType(theano.config.floatX,
                                                  (False, False)), 3)

        myType = TypedListType(T.TensorType(theano.config.floatX,
                                            (False, False)))

        myManualNestedType = TypedListType(TypedListType(
                                           TypedListType(myType)))

        self.assertTrue(myNestedType == myManualNestedType)

    def test_get_depth(self):
        """
        test case for get_depth utilitary function
        """
        myType = TypedListType(T.TensorType(theano.config.floatX,
                                            (False, False)))

        myManualNestedType = TypedListType(TypedListType(
                                           TypedListType(myType)))

        self.assertTrue(myManualNestedType.get_depth() == 3)

    def test_comparison_uneven_nested(self):
        """
        test for comparison between uneven nested list
        """

        myType = TypedListType(T.TensorType(theano.config.floatX,
                                            (False, False)))

        myManualNestedType1 = TypedListType(TypedListType(
                                            TypedListType(myType)))

        myManualNestedType2 = TypedListType(TypedListType(myType))

        self.assertFalse(myManualNestedType1 == myManualNestedType2)
        self.assertFalse(myManualNestedType2 == myManualNestedType1)

    def test_variable_is_Typed_List_variable(self):
        mySymbolicVariable = TypedListType(T.TensorType(theano.config.floatX,
                                           (False, False)))()

        self.assertTrue(isinstance(mySymbolicVariable,
                                   theano.typed_list.TypedListVariable))
