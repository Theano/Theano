from __future__ import absolute_import, print_function, division
import unittest

from nose.plugins.skip import SkipTest
import numpy as np

import theano
import theano.typed_list
from theano import tensor as T
from theano.tensor.type_other import SliceType
from theano.typed_list.type import TypedListType
from theano.typed_list.basic import (GetItem, Insert,
                                     Append, Extend, Remove, Reverse,
                                     Index, Count, Length, make_list)
from theano import sparse
from theano.tests import unittest_tools as utt
# TODO, handle the case where scipy isn't installed.
try:
    import scipy.sparse as sp
    scipy_imported = True
except ImportError:
    scipy_imported = False


# took from tensors/tests/test_basic.py
def rand_ranged_matrix(minimum, maximum, shape):
    return np.asarray(np.random.rand(*shape) * (maximum - minimum) +
                      minimum, dtype=theano.config.floatX)


# took from sparse/tests/test_basic.py
def random_lil(shape, dtype, nnz):
    rval = sp.lil_matrix(shape, dtype=dtype)
    huge = 2 ** 30
    for k in range(nnz):
        # set non-zeros in random locations (row x, col y)
        idx = np.random.randint(1, huge + 1, size=2) % shape
        value = np.random.rand()
        # if dtype *int*, value will always be zeros!
        if dtype in theano.tensor.integer_dtypes:
            value = int(value * 100)
        # The call to tuple is needed as scipy 0.13.1 do not support
        # ndarray with length 2 as idx tuple.
        rval.__setitem__(
            tuple(idx),
            value)
    return rval


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

        self.assertTrue(np.array_equal(f([x], slice(0, 1, 1)), [x]))

    def test_sanity_check_single(self):

        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        mySymbolicScalar = T.scalar(dtype='int64')

        z = GetItem()(mySymbolicMatricesList, mySymbolicScalar)

        f = theano.function([mySymbolicMatricesList, mySymbolicScalar],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x],
                                         np.asarray(0, dtype='int64')),
                                       x))

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        mySymbolicScalar = T.scalar(dtype='int64')

        z = mySymbolicMatricesList[mySymbolicScalar]

        f = theano.function([mySymbolicMatricesList, mySymbolicScalar],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x],
                                         np.asarray(0, dtype='int64')),
                                       x))

        z = mySymbolicMatricesList[0]

        f = theano.function([mySymbolicMatricesList],
                            z)

        self.assertTrue(np.array_equal(f([x]), x))

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

        self.assertTrue(np.array_equal(f([x]), x))

        z = GetItem()(mySymbolicMatricesList, slice(0, 1, 1))

        f = theano.function([mySymbolicMatricesList],
                            z)

        self.assertTrue(np.array_equal(f([x]), [x]))


class test_append(unittest.TestCase):

    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = Append(True)(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z,
                            accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], y), [x, y]))

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = Append()(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], y), [x, y]))

    def test_interfaces(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = mySymbolicMatricesList.append(myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], y), [x, y]))


class test_extend(unittest.TestCase):

    def test_inplace(self):
        mySymbolicMatricesList1 = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        mySymbolicMatricesList2 = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = Extend(True)(mySymbolicMatricesList1, mySymbolicMatricesList2)

        f = theano.function([mySymbolicMatricesList1, mySymbolicMatricesList2],
                            z, accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], [y]), [x, y]))

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

        self.assertTrue(np.array_equal(f([x], [y]), [x, y]))

    def test_interface(self):
        mySymbolicMatricesList1 = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        mySymbolicMatricesList2 = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = mySymbolicMatricesList1.extend(mySymbolicMatricesList2)

        f = theano.function([mySymbolicMatricesList1, mySymbolicMatricesList2],
                            z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], [y]), [x, y]))


class test_insert(unittest.TestCase):

    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()
        myScalar = T.scalar(dtype='int64')

        z = Insert(True)(mySymbolicMatricesList, myScalar, myMatrix)

        f = theano.function([mySymbolicMatricesList, myScalar, myMatrix], z,
                            accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x],
                                         np.asarray(1, dtype='int64'),
                                         y),
                                       [x, y]))

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()
        myScalar = T.scalar(dtype='int64')

        z = Insert()(mySymbolicMatricesList, myScalar, myMatrix)

        f = theano.function([mySymbolicMatricesList, myScalar, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x], np.asarray(1,
                        dtype='int64'), y), [x, y]))

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()
        myScalar = T.scalar(dtype='int64')

        z = mySymbolicMatricesList.insert(myScalar, myMatrix)

        f = theano.function([mySymbolicMatricesList, myScalar, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x],
                                         np.asarray(1, dtype='int64'),
                                         y),
                                       [x, y]))


class test_remove(unittest.TestCase):

    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = Remove(True)(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z,
                            accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y], y), [x]))

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = Remove()(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y], y), [x]))

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = mySymbolicMatricesList.remove(myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y], y), [x]))


class test_reverse(unittest.TestCase):

    def test_inplace(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = Reverse(True)(mySymbolicMatricesList)

        f = theano.function([mySymbolicMatricesList], z,
                            accept_inplace=True)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y]), [y, x]))

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = Reverse()(mySymbolicMatricesList)

        f = theano.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y]), [y, x]))

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = mySymbolicMatricesList.reverse()

        f = theano.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(np.array_equal(f([x, y]), [y, x]))


class test_index(unittest.TestCase):

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = Index()(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([x, y], y) == 1)

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = mySymbolicMatricesList.ind(myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([x, y], y) == 1)

    def test_non_tensor_type(self):
        mySymbolicNestedMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)), 1)()
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = Index()(mySymbolicNestedMatricesList, mySymbolicMatricesList)

        f = theano.function([mySymbolicNestedMatricesList,
                             mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([[x, y], [x, y, y]], [x, y]) == 0)

    def test_sparse(self):
        if not scipy_imported:
            raise SkipTest('Optional package SciPy not installed')
        mySymbolicSparseList = TypedListType(
            sparse.SparseType('csr', theano.config.floatX))()
        mySymbolicSparse = sparse.csr_matrix()

        z = Index()(mySymbolicSparseList, mySymbolicSparse)

        f = theano.function([mySymbolicSparseList, mySymbolicSparse], z)

        x = sp.csr_matrix(random_lil((10, 40), theano.config.floatX, 3))
        y = sp.csr_matrix(random_lil((10, 40), theano.config.floatX, 3))

        self.assertTrue(f([x, y], y) == 1)


class test_count(unittest.TestCase):

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = Count()(mySymbolicMatricesList, myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([y, y, x, y], y) == 3)

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        myMatrix = T.matrix()

        z = mySymbolicMatricesList.count(myMatrix)

        f = theano.function([mySymbolicMatricesList, myMatrix], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([x, y], y) == 1)

    def test_non_tensor_type(self):
        mySymbolicNestedMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)), 1)()
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = Count()(mySymbolicNestedMatricesList, mySymbolicMatricesList)

        f = theano.function([mySymbolicNestedMatricesList,
                             mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        y = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([[x, y], [x, y, y]], [x, y]) == 1)

    def test_sparse(self):
        if not scipy_imported:
            raise SkipTest('Optional package SciPy not installed')
        mySymbolicSparseList = TypedListType(
            sparse.SparseType('csr', theano.config.floatX))()
        mySymbolicSparse = sparse.csr_matrix()

        z = Count()(mySymbolicSparseList, mySymbolicSparse)

        f = theano.function([mySymbolicSparseList, mySymbolicSparse], z)

        x = sp.csr_matrix(random_lil((10, 40), theano.config.floatX, 3))
        y = sp.csr_matrix(random_lil((10, 40), theano.config.floatX, 3))

        self.assertTrue(f([x, y, y], y) == 2)


class test_length(unittest.TestCase):

    def test_sanity_check(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()

        z = Length()(mySymbolicMatricesList)

        f = theano.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([x, x, x, x]) == 4)

    def test_interface(self):
        mySymbolicMatricesList = TypedListType(T.TensorType(
            theano.config.floatX, (False, False)))()
        z = mySymbolicMatricesList.__len__()

        f = theano.function([mySymbolicMatricesList], z)

        x = rand_ranged_matrix(-1000, 1000, [100, 101])

        self.assertTrue(f([x, x]) == 2)


class TestMakeList(unittest.TestCase):

    def test_wrong_shape(self):
        a = T.vector()
        b = T.matrix()

        self.assertRaises(TypeError, make_list, (a, b))

    def test_correct_answer(self):
        a = T.matrix()
        b = T.matrix()

        x = T.tensor3()
        y = T.tensor3()

        A = np.cast[theano.config.floatX](np.random.rand(5, 3))
        B = np.cast[theano.config.floatX](np.random.rand(7, 2))
        X = np.cast[theano.config.floatX](np.random.rand(5, 6, 1))
        Y = np.cast[theano.config.floatX](np.random.rand(1, 9, 3))

        make_list((3., 4.))
        c = make_list((a, b))
        z = make_list((x, y))
        fc = theano.function([a, b], c)
        fz = theano.function([x, y], z)
        self.assertTrue((m == n).all() for m, n in zip(fc(A, B), [A, B]))
        self.assertTrue((m == n).all() for m, n in zip(fz(X, Y), [X, Y]))
