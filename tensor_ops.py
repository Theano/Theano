
from gof import Op, utils, Destroyer, Viewer
import gof.op

from tensor import *



# # TensorOp is a convenient base class, permitting to factor the code for the
# # Ops in this file.
# # It is not necessary to inherit from TensorOp to make an Op that manipulates
# # Tensors.
# <<<<<<< /u/breuleuo/hg/new_theano/tensor_ops.py
# class TensorOp(Op):

#     nin = -1
#     nout = 1

#     cast_method = lambda self, *args: _upcast(*args)
    
#     def __init__(self, *inputs):

#         inputs = map(_wrap_as_tensor, inputs)
        
#         if self.nin >= 0:
#             if len(inputs) != self.nin:
#                 raise TypeError("Wrong number of inputs for %s (got %i, expected %i)") \
#                     % (self, len(inputs), self.nin)

#         i_broadcastables = [getattr(input, 'broadcastable', None) for input in inputs]
#         i_dtypes = [getattr(input, 'dtype', None) for input in inputs]

#         o_broadcastables = utils.from_return_values(self.propagate_broadcastable(*i_broadcastables))
#         o_dtypes = utils.from_return_values(self.propagate_dtype(*i_dtypes))

#         self.inputs = inputs
#         self.outputs = [Tensor(dtype, broadcastable) for broadcastable, dtype in zip(o_broadcastables, o_dtypes)]

#     def propagate_broadcastable(self, *inputs):
#         raise AbstractFunctionError()
    
#     def propagate_dtype(self, *i_dtypes):
#         for dtype in i_dtypes:
#             if dtype is None:
#                 raise TypeError("Expected a Tensor.")
#         return self.cast_method(*i_dtypes)
    
#     def impl(self, *inputs):
#         raise AbstractFunctionError()
    
#     def perform(self):
#         res = self.impl(*[input.data for input in self.inputs])
#         if self.nout == 1:
#             self.outputs[0].data = res
#         else:
#             for output, value in zip(self.outputs, res):
#                 output.data = value
    
#     def c_var_names(self):
#         (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
#         inames = utils.from_return_values(inames)
#         onames = utils.from_return_values(onames)
#         return [inames, onames]
    
#     def c_code(self):
#         return self.c_impl(self.inputs, self.outputs)

#     def c_impl(self, inputs, outputs):
#         raise AbstractFunctionError()



# class UnaryTensorOp(TensorOp):
#     nin = 1

# class BinaryTensorOp(TensorOp):
#     nin = 2


# # class Transpose(UnaryTensorOp):

# #     def propagate_broadcastable(self, x):
# #         x2 = copy(x)
# #         x2.reverse()
# #         return [x2]

# #     def impl(self, x):
# #         return x.T

# #     def c_impl(self, x, z):
# #         return """
# #         PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
# #         //if (PyArray_REFCOUNT(transposed) == 1) {
# #         //    printf("lala\\n");
# #         //}
# #         //if (%(z)s) {
# #         //    Py_XDECREF(%(z)s);
# #         //}
# #         %(z)s = transposed;
# #         Py_XINCREF(%(z)s);
# #         """




# def scalar_switch(normal_f, scalar_f, scalar_f_reverse = None):
#     def f(x, y):
#         x, y = _wrap_as_tensor(x), _wrap_as_tensor(y)
#         if 0 not in y.broadcastable:
#             return scalar_f(x, y)
#         if 0 not in x.broadcastable:
#             if scalar_f_reverse:
#                 return scalar_f_reverse(y, x)
#             else:
#                 raise TypeError("You cannot do this operation on a scalar.")
#         return normal_f(x, y)
#     return f

# # Wrapper to ensure that all inputs to the function impl have the same size (foils numpy's broadcasting)
# def assert_same_shapes(x, *rest):
#     shape = x.shape
#     for other in rest:
#         if other.shape != shape:
#             raise ValueError("The dimensions of the inputs do not match.")

# # Wrapper to ensure that the last input to impl is a scalar
# def assert_tensor_scalar(x, a):
#     if numpy.product(a.shape) != 1:
#         raise ValueError("The second argument must be a scalar.")



# class Elemwise(TensorOp):

#     @staticmethod
#     def extract_name(name):
#         if name.endswith("_i"):
#             return name[:-2]
#         else:
#             return name
    
#     @staticmethod
#     def is_loop_var(name):
#         return name.endswith("_i")
    
#     def c_var_names(self):
#         cls = self.__class__
#         (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_foreach)
#         spec = ([cls.extract_name(name) for name in inames],
#                 [cls.extract_name(name) for name in onames])
#         return spec

#     def loop_variables(self):
#         cls = self.__class__
#         (self, inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_foreach)
#         return ([cls.extract_name(name) for name in inames if cls.is_loop_var(name)],
#                 [cls.extract_name(name) for name in onames if cls.is_loop_var(name)])
    
#     def propagate_broadcastable(self, *inputs):
#         inames, onames = self.c_var_names()
#         iloop, oloop = self.loop_variables()
#         if oloop != onames:
#             raise Exception("Cannot infer broadcastable for non-loop variable(s) %s" % set(onames).difference(oloop))
#         all_bcast = [broadcastable for broadcastable, iname in zip(inputs, inames) if iname in iloop]
#         ret = []
#         for arr in zip(*all_bcast):
#             if 0 in arr:
#                 ret.append(0)
#             else:
#                 ret.append(1)
#         return [ret] * self.nout

#     @classmethod
#     def inplace_version(cls):
#         class Ret(cls, Destroyer):
#             def destroy_list(self):
#                 return self.inputs[0]
#         return Ret

#     def c_init(self, inputs, outputs):
#         pass

#     def c_foreach(self, inputs, outputs):
#         pass

#     def c_finalize(self, inputs, outputs):
#         pass



# class TensorScalarOp(Elemwise):
#     def c_var_names(self):
#         return (['x', '_a'], ['z', ])
#     def loop_variables(self):
#         return (['x', ], ['z', ])
#     def c_init((x, _a), (z, )):
#         return """
#         if (PyArray_SIZE(_a) != 1) {
#             PyErr_SetString(PyExc_ValueError, \"The size of the scalar argument is not 1.\");
#         }
#         _a_dtype a = ((_a_dtype*)PyArray_DATA(_a))[0];
#         """
#     def _c_foreach(self):
#         return "z_i = %s;" % self.c_expr
# =======
# >>>>>>> /tmp/tensor_ops.py~other.fNA50a


###########################
#### Binary Operations ####
###########################

#########
## Dot ##
#########

class Dot(TensorOp):
    @staticmethod
    def _output_shape(xshape, yshape):
        # This describes the logic to calculate numpy.dot(x, y).shape
        # given x.shape and y.shape
        if len(xshape) == 0: # x is a scalar
            shape = yshape
        else:
            if len(yshape) >= 2: #y is a matrix or tensor
                assert xshape[-1] == yshape[-2]
                shape = tuple(xshape[:-1]+ yshape[:-2]+yshape[-1:])
            elif len(yshape)==1: #y is vector
                assert xshape[-1] == yshape[-1]
                shape = tuple(xshape[:-1])
            else:                #y is a scalar
                shape = xshape
        return shape

    def impl(self, x, y):
        return numpy.dot(x, y)
    def grad(self, (x, y), gz):
        return dot(gz, transpose(y)), dot(transpose(x), gz)
    def propagate_broadcastable(self, x, y):
        assert len(x) == 2 and len(x) == len(y)
        return [(x[0], y[1])]
    def c_support_code(self):
        return blas.cblas_header_text()
    def c_libs(self):
        return blas.ldflags()
    def c_impl(self, (_x, _y), (_z, )):
        return blas.gemm_code('', '1.0', '0.0')






############
## Others ##
############




class Neg(Elemwise):
    def impl(self, x):
        return -x
    def grad(self, x, gz):
        return -gz
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = -x_i;"

class NegInplace(Neg.inplace_version()):
    def impl(self, x):
        x *= -1
        return x


class InvElemwise(Elemwise):
    def impl(self, x):
        return 1 / x
    def grad(self, x, gz):
        return -gz / (x * x)
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = 1 / x_i;"

class InvElemwiseInplace(InvElemwise.inplace_version()):
    def impl(self, x):
        x[:] = 1 / x
        return x


class Exp(Elemwise):
    def impl(self, x): return numpy.exp(x)
    def grad(self, x, gz): return gz * exp(x)
    def c_foreach(self, (x_i, ), (z_i, )): return "z_i = exp(x_i);"
    
class Log(Elemwise):
    def impl(self, x): return numpy.log(x)
    def grad(self, x, gz): return gz / x
    def c_foreach(self, (x_i, ), (z_i, )): return "z_i = log(x_i);"

class Log2(Elemwise):
    def impl(self, x): return numpy.log2(x)
    def grad(self, x, gz): return gz / (x * numpy.log(2))
    def c_foreach(self, (x_i, ), (z_i, )): return "z_i = log2(x_i);"


class Twice(Elemwise):
    def impl(self, x):
        return 2.0 * x
    def grad(self, x, gz):
        return scale(gz, 2.0)
    def c_foreach(self, (x_i, ), (z_i, )):
        "z_i = x_i + x_i;"

class TwiceInplace(Twice.inplace_version()):
    def impl(self, x):
        x *= 2.0
        return x


class Sqr(Elemwise):
    def impl(self, x):
        return x * x
    def grad(self, x, gz):
        return scale(mul_elemwise(x, gz), 2.0)
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = x_i * x_i;"

class SqrInplace(Sqr.inplace_version()):
    def impl(x):
        x *= x
        return x


class Sqrt(Elemwise):
    def impl(self, x):
        return numpy.sqrt(x)
    def grad(self, x, gz):
        return scale(div(gz, sqrt(x)), 0.5)
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = pow(x_i, 0.5);"

class SqrtInplace(Sqrt.inplace_version()):
    def impl(self, x):
        x **= 0.5
        return x


class OnesLike(Elemwise):
    def impl(self, x):
        return numpy.ones_like(x)
    def grad(self, x, gz):
        return None

class ZerosLike(Elemwise):
    def impl(self, x):
        return numpy.zeros_like(x)
    def grad(self, x, gz):
        return None


class Min:
    pass

class Max:
    pass

class Argmin:
    pass

class Argmax:
    pass

class MinMax:
    pass
#     nout = 2
#     def impl(x):
#         return x.min, x.max
#     def specs(x):
#         return [(numpy.ndarray, x[1], ())] * 2
# #     def alloc((x, ), (_min, _max)):
# #         _min.data = numpy.ndarray((), x.dtype)
# #         _max.data = numpy.ndarray((), x.dtype)
#     def c_init((x, ), (_min, _max)):
#         raise NotImplementedError
#         return """
#         _x_dtype min = _x[0];
#         _x_dtype max = _x[0];
#         """
#     def c_foreach((x, ), (_min, _max)):
#         return """
#         if (x < min) min = x;
#         if (x > max) max = x;
#         """
#     def c_finalize((x, ), (_min, _max)):
#         return """
#         _min[0] = min;
#         _max[0] = max;
#         """




# class Transpose(UnaryTensorOp):

#     def propagate_broadcastable(self, x):
#         x2 = copy(x)
#         x2.reverse()
#         return [x2]

#     def impl(self, x):
#         return x.T

#     def c_impl(self, x, z):
#         return """
#         PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
#         //if (PyArray_REFCOUNT(transposed) == 1) {
#         //    printf("lala\\n");
#         //}
#         //if (%(z)s) {
#         //    Py_XDECREF(%(z)s);
#         //}
#         %(z)s = transposed;
#         Py_XINCREF(%(z)s);
#         """




