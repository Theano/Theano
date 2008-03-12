
from gof import Op, utils, Destroyer, Viewer
import gof.op

import gradient
from tensor import *


def _upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return str(z.dtype)

def _wrap_as_tensor(x):
    if isinstance(x,Op):
        return _wrap_as_tensor(x.out)
    elif isinstance(x, Tensor):
        return x
    else:
        return Tensor(data=x, constant=True)

# TensorOp is a convenient base class, permitting to factor the code for the
# Ops in this file.
# It is not necessary to inherit from TensorOp to make an Op that manipulates
# Tensors.
class TensorOp(Op, gradient.SelfGrad):

    nin = -1
    nout = 1

    cast_method = lambda self, *args: _upcast(*args)
    
    def __init__(self, *inputs):

        inputs = map(_wrap_as_tensor, inputs)
        
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)") \
                    % (self, len(inputs), self.nin)

        i_broadcastables = [getattr(input, 'broadcastable', None) for input in inputs]
        i_dtypes = [getattr(input, 'dtype', None) for input in inputs]

        o_broadcastables = utils.from_return_values(self.propagate_broadcastable(*i_broadcastables))
        o_dtypes = utils.from_return_values(self.propagate_dtype(*i_dtypes))

        self.inputs = inputs
        self.outputs = [Tensor(dtype, broadcastable) for broadcastable, dtype in zip(o_broadcastables, o_dtypes)]

    def propagate_broadcastable(self, *inputs):
        raise AbstractFunctionError()
    
    def propagate_dtype(self, *i_dtypes):
        for dtype in i_dtypes:
            if dtype is None:
                raise TypeError("Expected a Tensor.")
        return self.cast_method(*i_dtypes)
    
    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def perform(self):
        self.outputs[0].data = self.impl(*[input.data for input in self.inputs])
    
    def c_var_names(self):
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
        inames = utils.from_return_values(inames)
        onames = utils.from_return_values(onames)
        return [inames, onames]
    
    def c_code(self):
        return self.c_impl(self.inputs, self.outputs)

    def c_impl(self, inputs, outputs):
        raise AbstractFunctionError()



class UnaryTensorOp(TensorOp):
    nin = 1

class BinaryTensorOp(TensorOp):
    nin = 2


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




def scalar_switch(normal_f, scalar_f, scalar_f_reverse = None):
    def f(x, y):
        x, y = _wrap_as_tensor(x), _wrap_as_tensor(y)
        if 0 not in y.broadcastable:
            return scalar_f(x, y)
        if 0 not in x.broadcastable:
            if scalar_f_reverse:
                return scalar_f_reverse(y, x)
            else:
                raise TypeError("You cannot do this operation on a scalar.")
        return normal_f(x, y)
    return f

# Wrapper to ensure that all inputs to the function impl have the same size (foils numpy's broadcasting)
def assert_same_shapes(x, *rest):
    shape = x.shape
    for other in rest:
        if other.shape != shape:
            raise ValueError("The dimensions of the inputs do not match.")

# Wrapper to ensure that the last input to impl is a scalar
def assert_tensor_scalar(x, a):
    if numpy.product(a.shape) != 1:
        raise ValueError("The second argument must be a scalar.")



class Elemwise(TensorOp):

    @staticmethod
    def extract_name(name):
        if name.endswith("_i"):
            return name[:-2]
        else:
            return name
    
    @staticmethod
    def is_loop_var(name):
        return name.endswith("_i")
    
    def c_var_names(self):
        cls = self.__class__
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_foreach)
        spec = ([cls.extract_name(name) for name in inames],
                [cls.extract_name(name) for name in onames])
        return spec

    def loop_variables(self):
        cls = self.__class__
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_foreach)
        return ([cls.extract_name(name) for name in inames if cls.is_loop_var(name)],
                [cls.extract_name(name) for name in onames if cls.is_loop_var(name)])
    
    def propagate_broadcastable(self, *inputs):
        inames, onames = self.c_var_names()
        iloop, oloop = self.loop_variables()
        if oloop != onames:
            raise Exception("Cannot infer broadcastable for non-loop variable(s) %s" % set(onames).difference(oloop))
        all_bcast = [broadcastable for broadcastable, iname in zip(inputs, inames) if iname in iloop]
        ret = []
        for arr in zip(*all_bcast):
            if 0 in arr:
                ret.append(0)
            else:
                ret.append(1)
        return [ret] * self.nout

    @classmethod
    def inplace_version(cls):
        class Ret(cls, Destroyer):
            def destroy_list(self):
                return self.inputs[0]
        return Ret

    def c_init(self, inputs, outputs):
        pass

    def c_foreach(self, inputs, outputs):
        pass

    def c_finalize(self, inputs, outputs):
        pass



class TensorScalarOp(Elemwise):
    def c_var_names(self):
        return (['x', '_a'], ['z', ])
    def loop_variables(self):
        return (['x', ], ['z', ])
    def c_init((x, _a), (z, )):
        return """
        if (PyArray_SIZE(_a) != 1) {
            PyErr_SetString(PyExc_ValueError, \"The size of the scalar argument is not 1.\");
        }
        _a_dtype a = ((_a_dtype*)PyArray_DATA(_a))[0];
        """
    def _c_foreach(self):
        return "z_i = %s;" % self.c_expr


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


#########
## Add ##
#########

# Elemwise #
class AddElemwise(Elemwise):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        return x + y
    def grad(self, (x, y), gz):
        return gz, gz
    def c_foreach(self, (x_i, y_i), (z_i, )):
        return "z_i = x_i + y_i;"

class AddElemwiseInplace(AddElemwise.inplace_version()):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        x += y
        return x

# Scalar #
class AddScalar(TensorScalarOp):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        return x + a
    def grad(self, (x, a), gz):
        return gz, sum(gz)
    c_expr = "x_i + a"

class AddScalarInplace(AddScalar.inplace_version()):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        x += a
        return x



#########
## Sub ##
#########

# Elemwise #
class SubElemwise(Elemwise):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        return x - y
    def grad(self, (x, y), gz):
        return gz, -gz
    def c_foreach(self, (x_i, y_i), (z_i, )):
        return "z_i = x_i - y_i;"

class SubElemwiseInplace(SubElemwise.inplace_version()):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        x -= y
        return x

# Scalar #
def sub_scalar_r(x, a):
    return add_scalar(x, -a)

def sub_scalar_l(x, a):
    return add_scalar(-x, a)

def sub_scalar_rinplace(x, a):
    return add_scalar_inplace(x, -a)



#########
## Mul ##
#########

# Elemwise #
class MulElemwise(Elemwise):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        return x * y
    def grad(self, (x, y), gz):
        return mul(y, gz), mul(x, gz)
    def c_foreach(self, (x_i, y_i), (z_i, )):
        return "z_i = x_i * y_i;"

class MulElemwiseInplace(MulElemwise.inplace_version()):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        x *= y
        return x

# Scalar #
class Scale(TensorScalarOp):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        return x * a
    def grad(self, (x, a), gz):
        return scale(a, gz), sum(mul_elemwise(x, gz))
    c_expr = "x_i * a"

class ScaleInplace(Scale.inplace_version()):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        x *= a
        return x



#########
## Div ##
#########

# Elemwise #
class DivElemwise(Elemwise):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        return x / y
    def grad(self, (x, y), gz):
        return div(gz, y), -div(mul(x, gz), sqr(y))
    def c_foreach(self, (x_i, y_i), (z_i, )):
        return "z_i = x_i / y_i;"

class DivElemwiseInplace(DivElemwise.inplace_version()):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        x /= y
        return x

# Scalar #
def div_scalar_r(x, a):
    return scale(x, inv_elemwise(a))

def div_scalar_l(x, a):
    return scale(inv_elemwise(x), a)

def div_scalar_rinplace(x, a):
    return scale_inplace(x, inv_elemwise(a))



#########
## Pow ##
#########

# Elemwise #

class PowElemwise(Elemwise):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        return x ** y
    def grad(self, (x, s), gz):
        gx = gz * s * (pow_elemwise(x, s-1.0))
        gs = gz * log(x) * pow_elemwise(x, s)
        return gx, gs
    def c_foreach(self, (x_i, s_i), (z_i, )):
        return "z_i = pow(x_i, s_i)"

class PowElemwiseInplace(PowElemwise.inplace_version()):
    def impl(self, x, y):
        assert_same_shapes(x, y)
        x **= y
        return x

# Scalar #
class PowScalarL(TensorScalarOp):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        return a ** x
    def grad(self, (x, s), gz):
        gx = sum(gz * s * pow_scalar_l(add_scalar(s,-1.0), x))
        gs = scale(mul(gz, pow_scalar_l(s, x)), log(x))
        return gx, gs
    c_expr = "pow(a, x_i)"

class PowScalarR(TensorScalarOp):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        return x ** a
    def grad(self, (x, s), gz):
        gx = scale(mul_elemwise(gz,pow_scalar_r(x, add_scalar(s,-1.0))), s)
        gs = sum(mul_elemwise(mul_elemwise(gz, pow_scalar_r(x,s)), log(x)))
        return gx, gs
    c_expr = "pow(x_i, a)"

class PowScalarRInplace(PowScalarR.inplace_version()):
    def impl(self, x, a):
        assert_tensor_scalar(x, a)
        x **= a
        return x



############
## Others ##
############

class Fill(Elemwise):
    def impl(self, model, value):
        return (model * 0) + value
    def grad(self, (model, value), gz):
        return None, sum(gz)
    def c_init(self, (model, value), (z, )):
        return "value_dtype value0 = ((value_dtype*)PyArray_DATA(value))[0];"
    def c_foreach(self, (model_i, value), (z_i, )):
        return "z_i = value0;"



##########################
#### Unary Operations ####
##########################

class Transpose(TensorOp, Viewer):
    def view_map(self):
        return {self.out: [self.inputs[0]]}
    def impl(self, x):
        return x.T
    def grad(self, x, gz):
        return transpose_copy(gz)
    def propagate_broadcastable(self, x):
        rval = list(x)
        rval.reverse()
        return [rval]
    
    def c_impl(self, x, z):
        return """
        PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
        if (%(z)s) {
            Py_XDECREF(%(z)s);
        }
        %(z)s = transposed;
        """

#     def c_impl(self, (x, ), (xt, )):
#         return """
#         const int l = x->nd;
#         // The user must ensure that all references to
#         //xt->data go through xt, or there's going to be trouble..
#         int refcheck = 0;

#           if (x == xt)
#             {
#               return -1;
#             }
#           if (refcheck)
#             {
#               int refcnt =  PyArray_REFCOUNT(xt);
#                 if ((refcnt > 2)  // you might think this should be 1.. but this works
#                     //|| (xt->base != NULL)
#                     || (xt->weakreflist != NULL))
#                   {
#                     PyErr_SetString(PyExc_ValueError,
#                                         "cannot resize an array that has "\\
#                                         "been referenced or is referencing\\n"\\
#                                         "another array in this way.  Use the "\\
#                                         "resize function");
#                     return -2;
#                   }
#             }

#         if (xt->nd != x->nd)
#         {
#             // this technique comes from PyArray_Resize()
#             npy_intp * dimptr = (npy_intp*)PyDimMem_RENEW(xt->dimensions, 2 * x->nd);
#             if (!dimptr)
#             {
#                   PyErr_NoMemory();
#                   return 1;
#             }
#             xt->nd = x->nd;
#             xt->dimensions = dimptr;
#             xt->strides = dimptr + x->nd;
#         }
#         //copy x's dimensions and strides
#         for (int i = 0; i < l; ++i)
#         {
#             xt->dimensions[i] = x->dimensions[l-i-1];
#             xt->strides[i] = x->strides[l-i-1];
#         }

#         // point directly at b's type descriptor
#         Py_INCREF(x->descr);
#         Py_DECREF(xt->descr);
#         xt->descr = x->descr;

#         // name x as a base of xt, increment its refcount
#         if ( xt->base != (PyObject*)x)
#         {
#           Py_INCREF(x);
#           if ((xt->base) && (xt->base != Py_None)) 
#             {
#               Py_DECREF(xt->base);
#             }
#           xt->base = (PyObject*)x;
#         }
    
#         // mark xt as not owning its data
#         if (PyArray_CHKFLAGS(xt, NPY_OWNDATA))
#           {
#             PyDataMem_FREE(xt->data);
#             xt->flags &= ~NPY_OWNDATA;
#           }
#         xt->data = x->data;

#         // this function is described in 
#         // ~/zzz.NOBACKUP/pub/src/numpy-1.0.3.1/numpy/core/src/arrayobject.c:1890
#         PyArray_UpdateFlags(xt, NPY_CONTIGUOUS|NPY_FORTRAN|NPY_ALIGNED|NPY_WRITEABLE); 

#         /*
#           TODO
#           What should be done with the weakreflist ?
#         */
#     """

def transpose_copy(x):
    return array_copy(transpose(x))


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


class Sum(Elemwise):
    def impl(self, x):
        return numpy.sum(x)
    def grad(self, x, gz):
        return fill(x, gz)
    def propagate_broadcastable(self, *inputs):
        return [()]
    def c_init(self, (x, ), (sum, )):
        return "sum_dtype* sump = ((sum_dtype*)PyArray_DATA(sum)); sump[0] = 0;"
    def c_foreach(self, (x_i, ), (sum, )):
        return "sump[0] += x_i;"


class ArrayCopy(Elemwise):
    def impl(self, x):
        return numpy.array(x)
    def grad(self, x, gz):
        return gz
    def c_foreach(self, (x_i, ), (z_i, )):
        return "z_i = x_i;"

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



# ## Others ##

# class minmax(elemwise):
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


# ## Array slicing ##

# class get_slice(omega_op):
#     def view_map(self): return {self.out: [self.inputs[0]]}
#     def impl(x, item): 
#         rval = x.__getitem__(item)
#         #print 'get_slice running', rval
#         return rval
#     def grad(x, gz): raise NotImplemented
#     def refresh_shape(self): 
#         x,item = self.inputs
#         rval = x.data.__getitem__(item.data).shape 
#         #print 'refresh_shape', rval
#         return rval
#     def refresh_dtype(self):
#         return self.inputs[0].data.dtype



from gof import modes
modes.make_constructors(globals())


add = scalar_switch(add_elemwise, add_scalar, add_scalar)
add_inplace = scalar_switch(add_elemwise_inplace, add_scalar_inplace)

sub = scalar_switch(sub_elemwise, sub_scalar_r, sub_scalar_l)
sub_inplace = scalar_switch(sub_elemwise_inplace, sub_scalar_rinplace)

mul = scalar_switch(mul_elemwise, scale, scale)
mul_inplace = scalar_switch(mul_elemwise_inplace, scale_inplace)

div = scalar_switch(div_elemwise, div_scalar_r, div_scalar_l)
div_inplace = scalar_switch(div_elemwise_inplace, div_scalar_rinplace)

pow = scalar_switch(pow_elemwise, pow_scalar_r, pow_scalar_l)
pow_inplace = scalar_switch(pow_elemwise_inplace, pow_scalar_rinplace)

Tensor.__add__ = add
Tensor.__sub__ = sub
Tensor.__mul__ = mul
Tensor.__iadd__ = add_inplace
Tensor.__isub__ = sub_inplace
Tensor.__imul__ = mul_inplace
Tensor.__pow__ = pow
Tensor.__ipow__ = pow_inplace
Tensor.T = property(transpose)


