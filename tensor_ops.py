
from gof import Op, utils, Destroyer, Viewer
import gof.op

from tensor import *



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
