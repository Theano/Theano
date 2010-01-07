import numpy
import unittest
import copy
import theano
from theano.tensor import Tensor, TensorType

from theano.compile.sharedvalue import *

class Test_SharedVariable(unittest.TestCase):

    def test_ctors(self):

        if 0: #when using an implementation that handles scalars with Scalar type
            assert shared(7).type == Scalar('int64')
            assert shared(7.0).type == Scalar('float64')
            assert shared(7, dtype='float64').type == Scalar('float64')

        else:
            assert shared(7).type == theano.tensor.lscalar
            assert shared(7.0).type == theano.tensor.dscalar
            assert shared(7, dtype='float64').type == theano.tensor.dscalar

        # test tensor constructor
        b = shared(numpy.zeros((5,5), dtype='int32'))
        assert b.type == TensorType('int32', broadcastable=[False,False])
        b = shared(numpy.random.rand(4,5))
        assert b.type == TensorType('float64', broadcastable=[False,False])
        b = shared(numpy.random.rand(5,1,2))
        assert b.type == TensorType('float64', broadcastable=[False,False,False])

        assert shared([]).type == generic
        def badfunc():
            shared(7, bad_kw=False)
        self.failUnlessRaises(TypeError, badfunc)

    def test_strict_generic(self):

        #this should work, because
        # generic can hold anything even when strict=True

        u = shared('asdf', strict=False)
        v = shared('asdf', strict=True) 

        u.value = 88
        v.value = 88

    def test_create_numpy_strict_false(self):

        # here the value is perfect, and we're not strict about it,
        # so creation should work
        SharedVariable(
                name='u',
                type=Tensor(broadcastable=[False], dtype='float64'),
                value=numpy.asarray([1., 2.]),
                strict=False)

        # here the value is castable, and we're not strict about it,
        # so creation should work
        SharedVariable(
                name='u',
                type=Tensor(broadcastable=[False], dtype='float64'),
                value=[1., 2.],
                strict=False)

        # here the value is castable, and we're not strict about it,
        # so creation should work
        SharedVariable(
                name='u',
                type=Tensor(broadcastable=[False], dtype='float64'),
                value=[1, 2], #different dtype and not a numpy array
                strict=False)

        # here the value is not castable, and we're not strict about it,
        # this is beyond strictness, it must fail
        try:
            SharedVariable(
                    name='u',
                    type=Tensor(broadcastable=[False], dtype='float64'),
                    value=dict(), #not an array by any stretch
                    strict=False)
            assert 0
        except TypeError:
            pass

    def test_use_numpy_strict_false(self):

        # here the value is perfect, and we're not strict about it,
        # so creation should work
        u = SharedVariable(
                name='u',
                type=Tensor(broadcastable=[False], dtype='float64'),
                value=numpy.asarray([1., 2.]),
                strict=False)

        # check that assignments to value are casted properly
        u.value = [3,4]
        assert type(u.value) is numpy.ndarray
        assert str(u.value.dtype) == 'float64'
        assert numpy.all(u.value == [3,4])

        # check that assignments of nonsense fail
        try: 
            u.value = 'adsf'
            assert 0
        except ValueError:
            pass

        # check that an assignment of a perfect value results in no copying
        uval = numpy.asarray([5,6,7,8], dtype='float64')
        u.value = uval
        assert u.value is uval

    def test_strict(self):
        def f(var, val): var.value = val

        b = shared(numpy.int64(7), strict=True)
        #assert b.type == Scalar('int64')
        assert b.type == theano.tensor.lscalar
        self.failUnlessRaises(TypeError, f, b, 8.23)
        b = shared(numpy.float64(7.234), strict=True)
        #assert b.type == Scalar('float64')
        assert b.type == theano.tensor.dscalar
        self.failUnlessRaises(TypeError, f, b, 8)

        c = shared(numpy.zeros((5,5), dtype='float32'))
        self.failUnlessRaises(TypeError, f, b, numpy.random.rand(5,5))



