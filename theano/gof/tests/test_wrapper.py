from __future__ import absolute_import, print_function, division
import theano
import numpy
from unittest import TestCase
from theano.gof import Op, COp, Apply
from theano import Generic
from theano.scalar import Scalar
from theano.tensor import TensorType
from theano.gof.wrapper import Wrapper, Wrap
from theano import config
from theano import tensor
from theano.tests import unittest_tools as utt

dtype = config.floatX
tensor_type_0d = TensorType(dtype, tuple())
scalar_type = Scalar(dtype)
generic_type = Generic()


# A test op to compute `y = a*x^2 + bx + c` for any tensor x, with a, b, c as op params.
class QuadraticOpFunc(Op):
    __props__ = ('a', 'b', 'c')
    params_type = Wrapper(a=tensor_type_0d,
                          b=scalar_type,
                          c=generic_type)

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, coefficients):
        x = inputs[0]
        y = output_storage[0]
        y[0] = coefficients.a * (x**2) + coefficients.b * x + coefficients.c

    def c_code_cache_version(self):
        return (1, 3)

    def c_support_code_apply(self, node, name):
        float_type = node.inputs[0].type.dtype_specs()[1]
        return """
        /* Computes: x = a*x*x + b*x + c for x in tensor. */
        int quadratic_%(float_type)s(PyArrayObject* tensor, %(float_type)s a, %(float_type)s b, %(float_type)s c) {
            NpyIter* iterator = NpyIter_New(tensor,
                NPY_ITER_READWRITE | NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK,
                NPY_KEEPORDER, NPY_NO_CASTING, NULL);
            if(iterator == NULL) {
                PyErr_SetString(PyExc_RuntimeError, "Unable to iterate over a tensor for an elemwise operation.");
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
        %(float_type)s a = (%(float_type)s) (*(%(coeff_type)s*) PyArray_GETPTR1(%(coeff)s->a, 0)); // 0-D TensorType.
        %(float_type)s b =                                                      %(coeff)s->b;      // Scalar.
        %(float_type)s c =                     (%(float_type)s)PyFloat_AsDouble(%(coeff)s->c);     // Generic.
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


# Same op as above, but implemented as a COp (with C code in an external file).
class QuadraticCOpFunc(COp):
    __props__ = ('a', 'b', 'c')
    params_type = Wrapper(a=tensor_type_0d,
                          b=scalar_type,
                          c=generic_type)

    def get_op_params(self):
        return [('QUADRATIC_WRAPPER', self.params_type.name),
                ('COEFF_TYPE', 'npy_' + numpy.dtype(dtype).name)]

    def __init__(self, a, b, c):
        super(QuadraticCOpFunc, self).__init__('test_quadratic_function.c',
                                               'APPLY_SPECIFIC(compute_quadratic)')
        self.a = a
        self.b = b
        self.c = c

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage, coefficients):
        x = inputs[0]
        y = output_storage[0]
        y[0] = coefficients.a * (x**2) + coefficients.b * x + coefficients.c


class TestWrapper(TestCase):

    def test_hash_and_eq_wrap(self):
        wp1 = Wrapper(a=Generic(), array=TensorType('int64', (False,)), floatting=Scalar('float64'),
                      npy_scalar=TensorType('float64', tuple()))
        wp2 = Wrapper(a=Generic(), array=TensorType('int64', (False,)), floatting=Scalar('float64'),
                      npy_scalar=TensorType('float64', tuple()))
        w1 = Wrap(wp1, a=1, array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        w2 = Wrap(wp2, a=1, array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 == w2
        assert not (w1 != w2)
        assert hash(w1) == hash(w2)
        # Changing attributes names only (a -> other_name).
        wp2_other = Wrapper(other_name=Generic(), array=TensorType('int64', (False,)), floatting=Scalar('float64'),
                            npy_scalar=TensorType('float64', tuple()))
        w2 = Wrap(wp2_other, other_name=1, array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 != w2
        # Changing attributes values only (now a=2).
        w2 = Wrap(wp2, a=2, array=numpy.asarray([1, 2, 4, 5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 != w2
        # Changing NumPy array values (5 -> -5).
        w2 = Wrap(wp2, a=1, array=numpy.asarray([1, 2, 4, -5, 7]), floatting=-4.5, npy_scalar=numpy.asarray(12))
        assert w1 != w2

    def test_hash_and_eq_wrapper(self):
        w1 = Wrapper(a1=TensorType('int64', (False, False)),
                     a2=TensorType('int64', (False, True, False, False, True)),
                     a3=Generic())
        w2 = Wrapper(a1=TensorType('int64', (False, False)),
                     a2=TensorType('int64', (False, True, False, False, True)),
                     a3=Generic())
        assert w1 == w2
        assert not (w1 != w2)
        assert hash(w1) == hash(w2)
        assert w1.name == w2.name
        # Changing attributes names only.
        w2 = Wrapper(a1=TensorType('int64', (False, False)),
                     other_name=TensorType('int64', (False, True, False, False, True)),  # a2 -> other_name
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
        size_tensor5 = shape_tensor5[0] * shape_tensor5[1] * shape_tensor5[2] * shape_tensor5[3] * shape_tensor5[4]
        random_tensor = numpy.random.normal(size=size_tensor5).reshape(shape_tensor5)

        w = Wrapper(a1=TensorType('int32', (False, False)),
                    a2=TensorType('float64', (False, False, False, False, False)),
                    a3=Generic())

        # With a value that does not match the wrapper.
        o = Wrap(w,
                 a1=numpy.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype('int64'),
                 a2=random_tensor.astype('float32'),
                 a3=2000)
        # should fail (o.a1 is not int32, o.a2 is not float64)
        self.assertRaises(TypeError, w.filter, o, True)
        # should fail (o.a1 is not int32, o.a2 is not float64, and downcast is disallowed)
        self.assertRaises(TypeError, w.filter, o, False, False)
        # Should pass.
        w.filter(o, strict=False, allow_downcast=True)

        # With a value that matches the wrapper.
        o1 = Wrap(w,
                  a1=numpy.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype('int32'),
                  a2=random_tensor.astype('float64'),
                  a3=2000)
        # All should pass.
        w.filter(o1, strict=True)
        w.filter(o1, strict=False, allow_downcast=False)
        w.filter(o1, strict=False, allow_downcast=True)

        # Check values_eq and values_eq_approx.
        o2 = Wrap(w,
                  a1=numpy.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype('int32'),
                  a2=random_tensor.astype('float64'),
                  a3=2000)
        assert w.values_eq(o1, o2)
        assert w.values_eq_approx(o1, o2)

        # Check value_eq_approx.
        # NB: I don't know exactly which kind of differences is rejected by values_eq but accepted by values_eq_approx.
        # So, I just play a little with float values.
        o3 = Wrap(w,
                  a1=numpy.asarray([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]).astype('int32'),
                  a2=(random_tensor.astype('float32') * 10 / 2.2 * 2.19999999999 / 10).astype('float64'),
                  a3=2000.0 - 0.00000000000000001)
        assert w.values_eq_approx(o1, o3)

    def test_op_params(self):
        a, b, c = 2, 3, -7
        x = tensor.matrix()
        y1 = QuadraticOpFunc(a, b, c)(x)
        y2 = QuadraticCOpFunc(a, b, c)(x)
        f1 = theano.function([x], y1)
        f2 = theano.function([x], y2)
        shape = (100, 100)
        # The for-loop is here just to force profiling print something interesting.
        # When running this test without this loop, profiling does not print neither list of classes nor list of ops
        # (maybe because the function is extremely fast ?).
        for i in range(50):
            vx = numpy.random.normal(size=shape[0] * shape[1]).astype(dtype).reshape(*shape)
            vy1 = f1(vx)
            vy2 = f2(vx)
            ref = a * (vx**2) + b * vx + c
            utt.assert_allclose(vy1, vy2)
            utt.assert_allclose(ref, vy1)
