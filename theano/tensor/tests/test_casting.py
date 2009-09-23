
import unittest
from theano import function
from theano.tensor import *

class test_casting(unittest.TestCase):
    def test_0(self):
        for op_fn in convert_to_int32, convert_to_float32, convert_to_float64:
            for type_fn in bvector, ivector, fvector, dvector:
                x = type_fn()
                f = function([x], op_fn(x))

                xval = numpy.asarray(numpy.random.rand(10)*10, dtype=type_fn.dtype)
                yval = f(xval)
                assert str(yval.dtype) == op_fn.scalar_op.output_types_preference.spec[0].dtype

    def test_illegal(self):
        try:
            x = zmatrix()
            function([x], convert_to_float64(x))(numpy.ones((2,3), dtype='complex128'))
        except TypeError:
            return
        assert 0


