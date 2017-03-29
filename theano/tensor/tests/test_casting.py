from __future__ import absolute_import, print_function, division

import unittest
from theano import function
from theano.tensor.basic import (_convert_to_int32, _convert_to_int8,
                                 _convert_to_int16, _convert_to_int64,
                                 _convert_to_float32, _convert_to_float64)
from theano.tensor import *


class test_casting(unittest.TestCase):
    def test_0(self):
        for op_fn in [_convert_to_int32, _convert_to_float32,
                      _convert_to_float64]:
            for type_fn in bvector, ivector, fvector, dvector:
                x = type_fn()
                f = function([x], op_fn(x))

                xval = theano._asarray(np.random.rand(10) * 10,
                                       dtype=type_fn.dtype)
                yval = f(xval)
                assert (str(yval.dtype) ==
                        op_fn.scalar_op.output_types_preference.spec[0].dtype)

    def test_illegal(self):
        try:
            x = zmatrix()
            function([x], cast(x, 'float64'))(np.ones((2, 3),
                                                         dtype='complex128'))
        except TypeError:
            return
        assert 0

    def test_basic(self):
        for type1 in ['uint8', 'uint16', 'uint32', 'uint64',
                      'int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            x = TensorType(dtype=type1,
                           broadcastable=(False, ))()
            for type2, converter in zip(['int8', 'int16', 'int32', 'int64',
                                         'float32', 'float64'],
                                        [_convert_to_int8, _convert_to_int16,
                                         _convert_to_int32, _convert_to_int64,
                                         _convert_to_float32,
                                         _convert_to_float64]):
                y = converter(x)
                f = function([compile.In(x, strict=True)], y)
                a = np.arange(10, dtype=type1)
                b = f(a)
                self.assertTrue(np.all(b == np.arange(10, dtype=type2)))

    def test_convert_to_complex(self):
        val64 = np.ones(3, dtype='complex64') + 0.5j
        val128 = np.ones(3, dtype='complex128') + 0.5j

        vec64 = TensorType('complex64', (False, ))()
        vec128 = TensorType('complex128', (False, ))()

        f = function([vec64], basic._convert_to_complex128(vec64))
        # we need to compare with the same type.
        assert vec64.type.values_eq_approx(val128, f(val64))

        f = function([vec128], basic._convert_to_complex128(vec128))
        assert vec64.type.values_eq_approx(val128, f(val128))

        f = function([vec64], basic._convert_to_complex64(vec64))
        assert vec64.type.values_eq_approx(val64, f(val64))

        f = function([vec128], basic._convert_to_complex64(vec128))
        assert vec128.type.values_eq_approx(val64, f(val128))

        # upcasting to complex128
        for t in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            a = theano.shared(np.ones(3, dtype=t))
            b = theano.shared(np.ones(3, dtype='complex128'))
            f = function([], basic._convert_to_complex128(a))
            assert a.type.values_eq_approx(b.get_value(), f())

        # upcasting to complex64
        for t in ['int8', 'int16', 'int32', 'int64', 'float32']:
            a = theano.shared(np.ones(3, dtype=t))
            b = theano.shared(np.ones(3, dtype='complex64'))
            f = function([], basic._convert_to_complex64(a))
            assert a.type.values_eq_approx(b.get_value(), f())

        # downcast to complex64
        for t in ['float64']:
            a = theano.shared(np.ones(3, dtype=t))
            b = theano.shared(np.ones(3, dtype='complex64'))
            f = function([], basic._convert_to_complex64(a))
            assert a.type.values_eq_approx(b.get_value(), f())

    def test_bug_complext_10_august_09(self):
        v0 = dmatrix()
        v1 = basic._convert_to_complex128(v0)

        inputs = [v0]
        outputs = [v1]
        f = function(inputs, outputs)
        i = np.zeros((2, 2))
        assert (f(i) == np.zeros((2, 2))).all()
