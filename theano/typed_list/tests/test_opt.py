import unittest

import numpy

import theano
import theano.typed_list
from theano import tensor as T
from theano.tensor.type_other import SliceType
from theano.typed_list.type import TypedListType
from theano.typed_list.basic import (GetItem, Insert,
                                      Append, Extend, Remove, Reverse,
                                      Index, Count)
from theano import In


#took from tensors/tests/test_basic.py
def rand_ranged_matrix(minimum, maximum, shape):
    return numpy.asarray(numpy.random.rand(*shape) * (maximum - minimum)
                         + minimum, dtype=theano.config.floatX)


class test_inplace(unittest.TestCase):

    def test_reverse_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                theano.config.floatX, (False, False)))()

        z = Reverse()(mySymbolicMatricesList)

        f = theano.function([In(mySymbolicMatricesList, borrow=True,
                        mutable=True)], z, accept_inplace=True)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)
        
        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x, y]), [y, x]))

    def test_append_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                theano.config.floatX, (False, False)))()
        mySymbolicMatrix = T.matrix()
        z = Append()(mySymbolicMatricesList, mySymbolicMatrix)

        f = theano.function([In(mySymbolicMatricesList, borrow=True,
                        mutable=True), In(mySymbolicMatrix, borrow=True,
                        mutable=True)], z, accept_inplace=True)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)
        
         x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], y), [x, y]))

    def test_extend_inplace(self):
        mySymbolicMatricesList1 = TypedListType(T.TensorType(
                                theano.config.floatX, (False, False)))()

        mySymbolicMatricesList2 = TypedListType(T.TensorType(
                                theano.config.floatX, (False, False)))()

        z = Extend()(mySymbolicMatricesList1, mySymbolicMatricesList2)

        f = theano.function([In(mySymbolicMatricesList1, borrow=True,
                    mutable=True), mySymbolicMatricesList2],
                            z)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], [y]), [x, y]))

    def test_insert_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                theano.config.floatX, (False, False)))()
        mySymbolicIndex = T.scalar()
        mySymbolicMatrix = T.matrix()

        z = Insert()(mySymbolicMatricesList, mySymbolicIndex, mySymbolicMatrix)

        f = theano.function([In(mySymbolicMatricesList, borrow=True,
                        mutable=True), mySymbolicIndex, mySymbolicMatrix],
                        z, accept_inplace=True)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(numpy.array_equal(f([x], numpy.asarray(1,
                                dtype=theano.config.floatX), y), [x, y]))
