from __future__ import absolute_import, print_function, division
import theano
import numpy
from unittest import TestCase
from theano.gof import Op, Apply
from theano import Generic
from theano.tensor import TensorType
from theano.common import Wrapper, Wrap
from theano import config
from theano import tensor
from theano.tests import unittest_tools as utt

dtype = config.floatX
ScalarType = TensorType(dtype, tuple())


# A test op to compute `y = a*x^2 + bx + c` for any tensor x,
# such that a, b, c are parameters of that op.
class QuadraticFunction(Op):
    __props__ = ('a', 'b', 'c')
    params_type = Wrapper(a=ScalarType, b=ScalarType, c=ScalarType)

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def get_params(self, node):
        return Wrap(a=self.a, b=self.b, c=self.c)

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, coefficients):
        x = inputs[0]
        y = output_storage[0]
        y[0] = coefficients.a * (x**2) + coefficients.b * x + coefficients.c

    def c_code_cache_version(self):
        return (1, 1)

    def c_support_code_apply(self, node, name):
        float_type = node.inputs[0].type.dtype_specs()[1]
        return """
        /* Computes: x = a*x*x + b*x + c for x in matrix. */
        int quadratic_%(float_type)s(PyArrayObject* matrix, %(float_type)s a, %(float_type)s b, %(float_type)s c) {
            NpyIter* iterator = NpyIter_New(matrix,
                NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
                NPY_KEEPORDER, NPY_NO_CASTING, NULL);
            if(iterator == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to iterate over a matrix for an elemwise operation.");
                return -1;
            }
            NpyIter_IterNextFunc* get_next = NpyIter_GetIterNext(iterator, NULL);
            char** data_ptr = NpyIter_GetDataPtrArray(iterator);
            npy_intp* stride_ptr = NpyIter_GetInnerStrideArray(iterator);
            npy_intp* innersize_ptr = NpyIter_GetInnerLoopSizePtr(iterator);
            do {
                char* data = *data_ptr;
                npy_intp stride = *stride_ptr;
                npy_intp count = *innersize_ptr;
                while(count) {
                    %(float_type)s x = *((%(float_type)s*)data);
                    *((%(float_type)s*)data) = a*x*x + b*x + c;
                    data += stride;
                    --count;
                }
            } while(get_next(iterator));
            NpyIter_Deallocate(iterator);
            return 0;
        }
        """ % {'float_type': float_type}

    def c_code(self, node, name, inputs, outputs, sub):
        X = inputs[0]
        Y = outputs[0]
        coeff = sub['params']
        fail = sub['fail']
        float_type = node.inputs[0].type.dtype_specs()[1]
        float_typenum = numpy.dtype(node.inputs[0].type.dtype).num
        coeff_type = 'npy_' + numpy.dtype(dtype).name
        return """
        PyArrayObject* o_a = %(coeff)s.a;
        PyArrayObject* o_b = %(coeff)s.b;
        PyArrayObject* o_c = %(coeff)s.c;
        %(float_type)s a = (%(float_type)s) (*(%(coeff_type)s*) PyArray_GETPTR1(o_a, 0));
        %(float_type)s b = (%(float_type)s) (*(%(coeff_type)s*) PyArray_GETPTR1(o_b, 0));
        %(float_type)s c = (%(float_type)s) (*(%(coeff_type)s*) PyArray_GETPTR1(o_c, 0));
        Py_XDECREF(%(Y)s);
        %(Y)s = (PyArrayObject*)PyArray_EMPTY(PyArray_NDIM(%(X)s), PyArray_DIMS(%(X)s), %(float_typenum)s, PyArray_IS_F_CONTIGUOUS(%(X)s));
        if (PyArray_CopyInto(%(Y)s, %(X)s) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to copy input into output.");
            %(fail)s
        };
        if (quadratic_%(float_type)s(%(Y)s, a, b, c) != 0) {
            PyErr_SetString(PyExc_RuntimeError, "Unable to compute quadratic function.");
            %(fail)s
        }
        """ % locals()


class TestWrapper(TestCase):

    def test_wrap_instances(self):
        w1 = Wrap(a=1, b='test string', array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        w2 = Wrap(a=1, b='test string', array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 == w2
        assert hash(w1) == hash(w2)
        assert all(hasattr(w1, key) for key in ('a', 'b', 'array', 'floatting', 'npy_scalar'))
        # Changing attributes names only.
        w2 = Wrap(other_name=1, b='test string', array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 != w2
        # Changing attributes types only.
        w2 = Wrap(a=1, b='test string', array=[1, 2, 4, 5, 7], floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 != w2
        # Changing attributes values only.
        w2 = Wrap(a=1, b='string', array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 != w2
        # Changing NumPy array values.
        w2 = Wrap(a=1, b='test string', array=numpy.asarray([1, 2, 4, -5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 != w2

    def test_wrapper_instances(self):
        w1 = Wrapper(a1=TensorType('int64', (False, False)),
                     a2=TensorType('int64', (False, True, False, False, True)),
                     a3=Generic())
        w2 = Wrapper(a1=TensorType('int64', (False, False)),
                     a2=TensorType('int64', (False, True, False, False, True)),
                     a3=Generic())
        assert w1 == w2
        assert hash(w1) == hash(w2)
        # Changing attributes names only.
        w2 = Wrapper(a1=TensorType('int64', (False, False)),
                     other_name=TensorType('int64', (False, True, False, False, True)),
                     a3=Generic())
        assert w1 != w2
        # Changing attributes types only.
        w2 = Wrapper(a1=TensorType('int64', (False, False)),
                     a2=Generic(),  # changing class
                     a3=Generic())
        assert w1 != w2
        # Changing attributes types characteristics only.
        w2 = Wrapper(a1=TensorType('int64', (False, True)),  # changing broadcasting
                     a2=TensorType('int64', (False, True, False, False, True)),
                     a3=Generic())
        assert w1 != w2

    def test_wrapper_filtering(self):
        shape_tensor5 = (1, 2, 2, 3, 2)
        size_tensor5 = reduce(lambda x, y: x * y, shape_tensor5, 1)
        random_tensor = numpy.random.normal(size=size_tensor5).astype('float64').reshape(shape_tensor5)

        # With a wrapper that does not match the value.
        w = Wrapper(a1=TensorType('int64', (False, False)),
                    a2=TensorType('float32', (False, False, False, False, False)),
                    a3=Generic())
        o = Wrap(a1=numpy.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype('int64'),
                 a2=random_tensor,
                 a3=2000)
        # should fail (a2 is not float32)
        self.assertRaises(TypeError, w.filter, o, True)
        # should fail (a2 is float64, but downcast to float32 is disallowed)
        self.assertRaises(TypeError, w.filter, o, False, False)
        # Should pass.
        w.filter(o, strict=False, allow_downcast=True)

        # With a wrapper that matches the value.
        w = Wrapper(a1=TensorType('int64', (False, False)),
                    a2=TensorType('float64', (False, False, False, False, False)),
                    a3=Generic())
        # All should pass.
        w.filter(o, strict=True)
        w.filter(o, strict=False, allow_downcast=False)
        w.filter(o, strict=False, allow_downcast=True)

        # Check value_eq and value_eq_approx.
        o2 = Wrap(a1=numpy.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype('int64'),
                  a2=random_tensor,
                  a3=2000)
        assert w.values_eq(o, o2)
        assert w.values_eq_approx(o, o2)

        # Check value_eq_approx.
        o3 = Wrap(a1=numpy.asarray([[1, 2.0, 3.000, 4, 5.0, 6], [7, 8, 9, 10, 11, 12]]).astype('int32'),
                  a2=random_tensor.astype('float32'),
                  a3=2000.0)

        assert w.values_eq_approx(o, o3)

    def test_wrapper(self):
        a, b, c = 2, 3, -7
        x = tensor.matrix()
        y = QuadraticFunction(a, b, c)(x)
        f = theano.function([x], y)
        shape = (100, 100)
        # The for-loop is here just to force profiling print something interesting.
        # When running this test without this loop, profiling does not print neither list of classes nor list of ops
        # (maybe because the function is extremely fast ?).
        for i in range(50):
            vx = numpy.random.normal(size=shape[0] * shape[1]).astype(dtype).reshape(*shape)
            vy = f(vx)
            ref = a * (vx**2) + b * vx + c
            utt.assert_allclose(ref, vy)
