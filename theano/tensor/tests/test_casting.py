
import unittest
from theano import function
from theano.tensor.basic import (_convert_to_int32, _convert_to_int8, _convert_to_int16,
        _convert_to_int64, _convert_to_float32, _convert_to_float64)
from theano.tensor import *


class test_casting(unittest.TestCase):
    def test_0(self):
        for op_fn in _convert_to_int32, _convert_to_float32, _convert_to_float64:
            for type_fn in bvector, ivector, fvector, dvector:
                x = type_fn()
                f = function([x], op_fn(x))

                xval = theano._asarray(numpy.random.rand(10)*10, dtype=type_fn.dtype)
                yval = f(xval)
                assert str(yval.dtype) == op_fn.scalar_op.output_types_preference.spec[0].dtype

    def test_illegal(self):
        try:
            x = zmatrix()
            function([x], cast(x, 'float64'))(numpy.ones((2,3), dtype='complex128'))
        except TypeError:
            return
        assert 0

    def test_basic(self):
        for type1 in ['uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            x = TensorType(dtype = type1, broadcastable = (False, )).make_variable()
            for type2, converter in zip(['int8', 'int16', 'int32', 'int64', 'float32', 'float64'],
                                        [_convert_to_int8, _convert_to_int16,
                                            _convert_to_int32, _convert_to_int64,
                                         _convert_to_float32, _convert_to_float64]):
                y = converter(x)
                f = function([compile.In(x, strict = True)], y)
                a = numpy.arange(10, dtype = type1)
                b = f(a)
                self.assertTrue(numpy.all(b == numpy.arange(10, dtype = type2)))

    def test_convert_to_complex(self):
        a = value(numpy.ones(3, dtype='complex64')+0.5j)
        b = value(numpy.ones(3, dtype='complex128')+0.5j)

        f = function([a],basic._convert_to_complex128(a))
        #we need to compare with the same type.
        assert a.type.values_eq_approx(b.data, f(a.data))

        f = function([b],basic._convert_to_complex128(b))
        assert b.type.values_eq_approx(b.data, f(b.data))

        f = function([a],basic._convert_to_complex64(a))
        assert a.type.values_eq_approx(a.data, f(a.data))

        f = function([b],basic._convert_to_complex64(b))
        assert b.type.values_eq_approx(a.data, f(b.data))

        for nbits in (64, 128):
            # upcasting to complex128
            for t in ['int8','int16','int32','int64','float32','float64']:
                a = value(numpy.ones(3, dtype=t))
                b = value(numpy.ones(3, dtype='complex128'))
                f = function([a],basic._convert_to_complex128(a))
                assert a.type.values_eq_approx(b.data, f(a.data))

            # upcasting to complex64
            for t in ['int8','int16','int32','int64','float32']:
                a = value(numpy.ones(3, dtype=t))
                b = value(numpy.ones(3, dtype='complex64'))
                f = function([a],basic._convert_to_complex64(a))
                assert a.type.values_eq_approx(b.data, f(a.data))

            # downcast to complex64
            for t in ['float64']:
                a = value(numpy.ones(3, dtype=t))
                b = value(numpy.ones(3, dtype='complex64'))
                f = function([a],basic._convert_to_complex64(a))
                assert a.type.values_eq_approx(b.data, f(a.data))


    def test_bug_complext_10_august_09(self):
        v0 = dmatrix()
        v1 = basic._convert_to_complex128(v0)

        inputs = [v0]
        outputs = [v1]
        f = function(inputs, outputs)
        i = numpy.zeros((2,2))
        assert (f(i)==numpy.zeros((2,2))).all()
