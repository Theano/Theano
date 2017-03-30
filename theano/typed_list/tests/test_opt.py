from __future__ import absolute_import, print_function, division
import unittest

import numpy as np

import theano
import theano.typed_list
from theano import tensor as T
from theano.typed_list.type import TypedListType
from theano.typed_list.basic import (Insert,
                                     Append, Extend, Remove, Reverse)
from theano import In


# took from tensors/tests/test_basic.py
def rand_ranged_matrix(minimum, maximum, shape):
    return np.asarray(np.random.rand(*shape) * (maximum - minimum) +
                      minimum, dtype=theano.config.floatX)


class test_inplace(unittest.TestCase):

    def test_reverse_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                               theano.config.floatX, (False, False)))()

        z = Reverse()(mySymbolicMatricesList)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function([In(mySymbolicMatricesList, borrow=True,
                                mutable=True)], z, accept_inplace=True, mode=m)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y]), [y, x]))

    def test_append_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                               theano.config.floatX, (False, False)))()
        mySymbolicMatrix = T.matrix()
        z = Append()(mySymbolicMatricesList, mySymbolicMatrix)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function([In(mySymbolicMatricesList, borrow=True,
                                mutable=True),
                            In(mySymbolicMatrix, borrow=True,
                               mutable=True)], z, accept_inplace=True, mode=m)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], y), [x, y]))

    def test_extend_inplace(self):
        mySymbolicMatricesList1 = TypedListType(T.TensorType(
                                                theano.config.floatX, (False, False)))()

        mySymbolicMatricesList2 = TypedListType(T.TensorType(
                                                theano.config.floatX, (False, False)))()

        z = Extend()(mySymbolicMatricesList1, mySymbolicMatricesList2)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function([In(mySymbolicMatricesList1, borrow=True,
                             mutable=True), mySymbolicMatricesList2],
                            z, mode=m)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], [y]), [x, y]))

    def test_insert_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                               theano.config.floatX, (False, False)))()
        mySymbolicIndex = T.scalar(dtype='int64')
        mySymbolicMatrix = T.matrix()

        z = Insert()(mySymbolicMatricesList, mySymbolicIndex, mySymbolicMatrix)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")

        f = theano.function([In(mySymbolicMatricesList, borrow=True,
                             mutable=True), mySymbolicIndex, mySymbolicMatrix],
                            z, accept_inplace=True, mode=m)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], np.asarray(1,
                        dtype='int64'), y), [x, y]))

    def test_remove_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
                                               theano.config.floatX, (False, False)))()
        mySymbolicMatrix = T.matrix()
        z = Remove()(mySymbolicMatricesList, mySymbolicMatrix)
        m = theano.compile.mode.get_default_mode().including("typed_list_inplace_opt")
        f = theano.function([In(mySymbolicMatricesList, borrow=True,
                            mutable=True), In(mySymbolicMatrix, borrow=True,
                            mutable=True)], z, accept_inplace=True, mode=m)
        self.assertTrue(f.maker.fgraph.toposort()[0].op.inplace)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y], y), [x]))


def test_constant_folding():
    m = theano.tensor.ones((1,), dtype='int8')
    l = theano.typed_list.make_list([m, m])
    f = theano.function([], l)
    topo = f.maker.fgraph.toposort()
    assert len(topo)
    assert isinstance(topo[0].op, theano.compile.ops.DeepCopyOp)
    assert f() == [1, 1]
