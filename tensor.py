"""A ResultBase to store numpy.ndarray with basic accompanying Ops"""
import sys # for sys.maxint
import inspect

import numpy

from gof import ResultBase, Op, utils, Destroyer, Viewer, AbstractFunctionError
import gof.result
import gof.op

from base_tensor import BaseTensor, BaseTensorOp
from elemwise import Elemwise
import blas # for gemm, dot

import elemwise2 as s2t
import scalar_ops as scal


class Tensor(BaseTensor):
    """
    This subclass of BaseTensor provides operator overloading using implementations
    of Tensor operations contained in this file.
    
    Operators:
    - most numeric operators are overloaded (to return Ops that perform the
      corresponding calculation)
    """

    #UNARY
    def __abs__(self): return Abs(self).out
    def __neg__(self): return Neg(self).out

    #CASTS
    def __int__(self): return AsInt(self).out
    def __float__(self): return AsInt(self).out
    def __complex__(self): return AsComplex(self).out

    #COMPARISONS
    def __lt__(self,other): return lt(self, other)
    def __le__(self,other): return le(self, other)
    def __gt__(self,other): return gt(self, other)
    def __ge__(self,other): return ge(self, other)

    #ARITHMETIC - NORMAL
    def __add__(self,other): return add(self,other)
    def __sub__(self,other): return sub(self,other)
    def __mul__(self,other): return mul(self,other)
    def __div__(self,other): return div(self,other)
    def __pow__(self,other): return pow(self,other)

    #ARITHMETIC - INPLACE
    def __iadd__(self,other): return add_inplace(self,other)
    def __isub__(self,other): return sub_inplace(self,other)
    def __imul__(self,other): return mul_inplace(self,other)
    def __idiv__(self,other): return div_inplace(self,other)
    def __ipow__(self,other): return pow_inplace(self,other)

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div(other,self)
    def __rpow__(self,other): return pow(other,self)

    #TRANSPOSE
    T = property(lambda self: transpose(self))

    #SLICING
    def __getitem__(self, item): return subtensor(self, item)
    def __getslice__(self, *args): return subtensor(self, slice(*args))
s2t.Tensor = Tensor

    
# alternate Tensor constructor
def astensor(data, broadcastable=None, role=None, name=None):
    """Return a Tensor containing given data"""
    if isinstance(data, Tensor) and broadcastable is None and role is None and name is None:
        return data
    data = numpy.asarray(data)
    if broadcastable is None:
        broadcastable = [s==1 for s in data.shape]
    elif broadcastable in [0, 1]:
        broadcastable = [broadcastable] *  len(data.shape)
    rval = Tensor(data.dtype, broadcastable, role, name)
    rval.data = data # will raise if broadcastable was mis-specified
    return rval
s2t.astensor = astensor


############################
# Supporting Ops
############################

def _scalar_switch(normal_f, scalar_f, scalar_f_reverse = None):
    """a decorator for operators before broadcasting works properly"""
    def f(x, y):
        def as_tensor(obj):
            if isinstance(obj, Tensor):
                return obj
            else:
                return astensor(obj)
        x, y = as_tensor(x), as_tensor(y)
        if 0 not in y.broadcastable:
            return scalar_f(x, y)
        if 0 not in x.broadcastable:
            if scalar_f_reverse:
                return scalar_f_reverse(y, x)
            else:
                raise TypeError("You cannot do this operation on a scalar.")
        return normal_f(x, y)
    return f

def _assert_same_shapes(x, *rest):
    """Ensure that all inputs to the function impl have the same size (foils numpy's broadcasting)"""
    shape = x.shape
    for other in rest:
        if other.shape != shape:
            raise ValueError(_assert_same_shapes.E_shape, shape, other.shape)
_assert_same_shapes.E_shape = "The dimensions of the inputs do not match."

def _assert_tensor_scalar(x, a):
    """ensure that the second input is a scalar"""
    if numpy.product(a.shape) != 1:
        raise ValueError("The second argument must be a scalar.")

# this has a different name, because _as_tensor is the function which ops use
# to upcast their arguments... this internal-use function is a good place to put debugging stuff, better than the global astensor.
_as_tensor = astensor

class _Op(BaseTensorOp):
    """A convenient base for the ops in this file"""

    out_tensor_class = Tensor

    @classmethod
    def input_wrapper(cls, obj):
        return _as_tensor(obj)
    
    def c_var_names(self):
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
        inames = utils.from_return_values(inames)
        onames = utils.from_return_values(onames)
        return [inames, onames]
    
    def c_code(self, input_names, output_names, sub):
        sub = dict(sub)
        icvn, ocvn = self.c_var_names()
        for real, tosub in zip(input_names + output_names, icvn + ocvn):
            sub[tosub] = real
        return self.c_impl(self.inputs, self.outputs) % sub

    def c_impl(self, inputs, outputs):
        raise AbstractFunctionError()

class _Unary:
    nin = 1

class _Binary:
    nin = 2


class _Elemwise(Elemwise, _Op):

    @staticmethod
    def extract_name(name):
        if name.endswith("_i"):
            return name[:-2]
        else:
            return name
    
    @staticmethod
    def is_loop_var(name):
        return name.endswith("_i")

    def var_desc(self):
        cls = self.__class__
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(cls.c_foreach)
        return ([(cls.extract_name(name), cls.is_loop_var(name)) for name in inames],
                [(cls.extract_name(name), cls.is_loop_var(name)) for name in onames])

    def propagate_broadcastable(self, *inputs):
        idesc, odesc = self.var_desc()
        nonloop_o = [o[0] for o in odesc if not o[1]]
        if nonloop_o:
            raise Exception("Cannot infer broadcastable for non-loop variable(s) %s" % nonloop_o)
        all_bcast = [broadcastable for broadcastable, i in zip(inputs, idesc) if i[1]]
        if reduce(lambda x, y: x is not False and x == y and y, [len(x) for x in all_bcast]) is False:
            raise TypeError(_Elemwise.propagate_broadcastable.E_ndim, self.__class__)
        ret = []
        for arr in zip(*all_bcast):
            if 0 in arr:
                ret.append(0)
            else:
                ret.append(1)
        return [ret] * self.nout
    propagate_broadcastable.E_ndim \
            = "Inputs that are loop variables do not all have the same number of dimensions."

    def c_init(self, inputs, outputs):
        raise AbstractFunctionError()        

    def c_foreach(self, inputs, outputs):
        raise AbstractFunctionError()

    def c_finalize(self, inputs, outputs):
        raise AbstractFunctionError()

    def c_code_init(self):
        return self.c_init(self.inputs, self.outputs)

    def c_code_foreach(self):
        return self.c_foreach(self.inputs, self.outputs)

    def c_code_finalize(self):
        return self.c_finalize(self.inputs, self.outputs)


class TensorScalarOp(_Elemwise):
    def var_desc(self):
        return [('x', 1), ('a', 0)], [('z', 1)]
    def c_code_init(self):
        return """
        dtype_%(a)s _%(a)s;
        if (PyArray_SIZE(%(a)s) != 1) {
            PyErr_SetString(PyExc_ValueError, \"The size of the scalar argument is not 1.\");
            %(fail)s
        }
        _%(a)s = ((dtype_%(a)s*)PyArray_DATA(%(a)s))[0];
        """
    def c_code_foreach(self):
        return "%%(z)s_i = %s;" % self.c_expr


##########################
# Unary Operations
##########################

# class Abs(_Elemwise):
#     def impl(self, x):
#         return numpy.abs(x)
#     def grad(self, x, gz):
#         return gz * Sgn(x).out #TODO: handle the corner case (get it? pun?) (there's a special place in hell for people like you)
#     def c_foreach(self, (x_i, ), (z_i, )):
#         return "%(z)s_i = abs(%(x)s_i);"
# #Constructor not necessary because builtin abs() does this


Abs = s2t.make_broadcast(scal.Abs)
AbsInplace = s2t.make_broadcast(scal.Abs, {0:0})

#Constructor not necessary because builtin abs() does this
abs_inplace = gof.op.constructor(s2t.wrap_broadcast(AbsInplace))

    
class Argmax(Op):
    nin=2 # tensor, axis
    nout=2 # max val, max idx
    E_axis = 'invalid axis'
    debug = 0
    def __init__(self, x, axis=None):
        x = _as_tensor(x)
        if axis is None:
            axis = len(x.broadcastable) -1
        axis = _as_tensor(axis)
        self.inputs = [x, axis]
        broadcastable = [0] * (len(x.broadcastable) - 1)
        self.outputs = [Tensor(x.dtype, broadcastable), 
                Tensor(axis.dtype, broadcastable)]
    def perform(self): 
        axis = self.inputs[1].data
        x = self.inputs[0].data
        self.outputs[0].data = numpy.max(x, axis)
        self.outputs[1].data = numpy.argmax(x,axis)
argmax = gof.op.constructor(Argmax)

def max(x, axis=None):
    """Return maximum elements obtained by iterating over given axis

    Default axis is the last one.
    """
    # In python (using Argmax.perform()) this leads to an wasteful
    # implementation that goes through the data twice instead of once
    # but when Argmax.c_impl() is in place, it should be fine.
    return argmax(x,axis)[0]

# class Exp(_Elemwise):
#     def impl(self, x): return numpy.exp(x)
#     def grad(self, x, gz): return gz * exp(x)
#     def c_foreach(self, (x_i, ), (z_i, )): return "z_i = exp(x_i);"
# exp = gof.op.constructor(Exp)

Exp = s2t.make_broadcast(scal.Exp)
ExpInplace = s2t.make_broadcast(scal.Exp, {0:0})
exp = gof.op.constructor(s2t.wrap_broadcast(Exp))
exp_inplace = gof.op.constructor(s2t.wrap_broadcast(ExpInplace))


# class Neg(_Elemwise):
#     def impl(self, x):
#         return -x
#     def grad(self, x, gz):
#         return -gz
#     def c_foreach(self, (x_i, ), (z_i, )):
#         return "%(z)s_i = -%(x)s_i;"
# #Constructor not necessary because unary '-' does this

Neg = s2t.make_broadcast(scal.Neg)
NegInplace = s2t.make_broadcast(scal.Neg, {0:0})
neg = gof.op.constructor(s2t.wrap_broadcast(Neg))
neg_inplace = gof.op.constructor(s2t.wrap_broadcast(NegInplace))


# class Log(_Elemwise):
#     def impl(self, x): return numpy.log(x)
#     def grad(self, x, gz): return gz / x
#     def c_foreach(self, (x_i, ), (z_i, )): return "z_i = log(x_i);"
# log = gof.op.constructor(Log)

Log = s2t.make_broadcast(scal.Log)
LogInplace = s2t.make_broadcast(scal.Log, {0:0})
log = gof.op.constructor(s2t.wrap_broadcast(Log))
log_inplace = gof.op.constructor(s2t.wrap_broadcast(LogInplace))

# class Log2(_Elemwise):
#     def impl(self, x): return numpy.log2(x)
#     def grad(self, x, gz): return gz / (x * numpy.log(2.0))
#     def c_foreach(self, (x_i, ), (z_i, )): return "%(z)s_i = log2(%(x)s_i);"
# log2 = gof.op.constructor(Log2)

Log2 = s2t.make_broadcast(scal.Log2)
Log2Inplace = s2t.make_broadcast(scal.Log2, {0:0})
log2 = gof.op.constructor(s2t.wrap_broadcast(Log2))
log2_inplace = gof.op.constructor(s2t.wrap_broadcast(Log2Inplace))

# class Sgn(_Elemwise):
#     def impl(self, x):
#         return numpy.abs(x) / x
#     def grad(self, x, gz):
#         return [None]
#     def c_foreach(self, (x_i, ), (z_i, )):
#         return "%(z)s_i = %(x)s_i/abs(%(x)s_i);" # TODO: C use copysign
# sgn = gof.op.constructor(Sgn)

Sgn = s2t.make_broadcast(scal.Sgn)
SgnInplace = s2t.make_broadcast(scal.Sgn, {0:0})
sgn = gof.op.constructor(s2t.wrap_broadcast(Sgn))
sgn_inplace = gof.op.constructor(s2t.wrap_broadcast(SgnInplace))

# class Sqr(_Elemwise):
#     def impl(self, x): return x * x
#     def grad(self, x, gz): return 2.0 * x * gz
#     def c_foreach(self, (x_i, ), (z_i, )): return "%(z)s_i = %(x)s_i * %(x)s_i;"
# sqr = gof.op.constructor(Sqr)

Sqr = s2t.make_broadcast(scal.Sqr)
SqrInplace = s2t.make_broadcast(scal.Sqr, {0:0})
sqr = gof.op.constructor(s2t.wrap_broadcast(Sqr))
sqr_inplace = gof.op.constructor(s2t.wrap_broadcast(SqrInplace))

# class Sqrt(_Elemwise):
#     def impl(self, x): return numpy.sqrt(x)
#     def grad(self, x, gz): return 0.5 * gz / sqrt(x) 
#     def c_foreach(self, (x_i, ), (z_i, )): return "%(z)s_i = sqrt(%(x)s_i);"
# sqrt = gof.op.constructor(Sqrt)

Sqrt = s2t.make_broadcast(scal.Sqrt)
SqrtInplace = s2t.make_broadcast(scal.Sqrt, {0:0})
sqrt = gof.op.constructor(s2t.wrap_broadcast(Sqrt))
sqrt_inplace = gof.op.constructor(s2t.wrap_broadcast(SqrtInplace))

# class Sum(_Elemwise):
#     def impl(self, x):
#         return numpy.sum(x)
#     def grad(self, (x, ), (gz, )):
#         return fill(x, gz),
#     def propagate_broadcastable(self, *inputs):
#         return [()]
#     def c_init(self, (x, ), (sum, )):
#         return "dtype_%(sum)s* %(sum)sp = ((dtype_%(sum)s*)PyArray_DATA(%(sum)s)); %(sum)sp[0] = 0;"
#     def c_foreach(self, (x_i, ), (sum, )):
#         return "%(sum)sp[0] += %(x)s_i;"
# sum0 = gof.op.constructor(Sum)

Sum = s2t.Sum
sum = gof.op.constructor(Sum)

# class Fill(_Elemwise):
#     def impl(self, model, value):
#         return (model * 0) + value #TODO: we can probably do better than this
#     def grad(self, (model, value), (gz, )):
#         return None, sum(gz)
#     def c_init(self, (model, value), (z, )):
#         return "dtype_%(value)s %(value)s0 = ((dtype_%(value)s*)PyArray_DATA(%(value)s))[0];"
#     def c_foreach(self, (model_i, value), (z_i, )):
#         return "%(z)s_i = %(value)s0;"
# fill = gof.op.constructor(Fill)


def broadcast_package(scalar_opclass, name, inplace_versions = True):
    C = s2t.make_broadcast(scalar_opclass, name = name)
    c = gof.op.constructor(s2t.wrap_broadcast(C))
    if inplace_versions:
        CInplace = s2t.make_broadcast(scalar_opclass, name = name+"Inplace")
        c_inplace = gof.op.constructor(s2t.wrap_broadcast(CInplace))
        return C, c, CInplace, c_inplace
    else:
        return C, c
    

# Fill = s2t.make_broadcast(scal.Second)
# FillInplace = s2t.make_broadcast(scal.Second, {0:0})
# fill = gof.op.constructor(s2t.wrap_broadcast(Fill))
# fill_inplace = gof.op.constructor(s2t.wrap_broadcast(FillInplace))

Fill, fill, FillInplace, fill_inplace = broadcast_package(scal.Second, 'Fill')

def ones_like(model):
    return fill(model, 1.0)
def zeros_like(model):
    return fill(model, 0.0)


# class TensorCopy(_Elemwise):
#     def impl(self, x):
#         return numpy.array(x)
#     def grad(self, x, gz):
#         return gz
#     def c_foreach(self, (x_i, ), (z_i, )):
#         return "%(z)s_i = %(x)s_i;"

TensorCopy = s2t.make_broadcast(scal.Identity)
tensor_copy = gof.op.constructor(TensorCopy)

##########################
# View Operations
##########################

class TransposeInplace(_Op, Viewer):
    def view_map(self):
        return {self.out: [self.inputs[0]]}
    def propagate_broadcastable(self, x):
        rval = list(x)
        rval.reverse()
        return [rval]
    def impl(self, x):
        return x.T #numpy's transpose
    def grad(self, x, gz):
        return transpose(gz)
    
    def c_impl(self, x, z):
        return """
        PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
        if (%(z)s) {
            Py_XDECREF(%(z)s);
        }
        %(z)s = transposed;
        """
transpose_inplace = gof.op.constructor(TransposeInplace)
def transpose(x, **kwargs):
    return transpose_inplace(tensor_copy(x), **kwargs)

class Subtensor(Op, Viewer):
    nin = 2
    nout = 1
    e_invalid = 'invalid index'
    debug = 0
    def __init__(self, *args,**kwargs):
        def as_tuple_result(obj):
            if isinstance(obj, ResultBase):
                return obj
            r = gof.result.PythonResult(None)
            if isinstance(obj, tuple):
                r.data = obj
            else:
                r.data = (obj,)
            return r
        def pad(tplR, N):
            l = list(tplR.data)
            for i in range(len(l), N):
                l.append(slice(0,sys.maxint,1))
            tplR.data = tuple(l)

        if Subtensor.debug:
            print 'Subtensor.__init__', args, kwargs
        #Olivier says not to call this
        #Op.__init__(self,  *args,**kwargs) 
        #Viewer.__init__(self, *args,**kwargs)
        t, coord = args
        t = _as_tensor(t)
        coord = as_tuple_result(coord)
        if len(coord.data) > len(t.broadcastable):
            raise ValueError(Subtensor.e_invalid)
        # add the implicit extra unbounded slices 
        # e.g. n[0] on a 3d tensor pads to n[0,:,:]
        pad(coord, len(t.broadcastable))
        broadcastable = [0 for c in coord.data if isinstance(c, slice)]
        if Subtensor.debug:
            print 'brdcstble', broadcastable
            print 't', t.data
            print 'coord', coord.data
        self.inputs = [t, coord]
        self.outputs = [Tensor(t.dtype, broadcastable)]
    def view_map(self): 
        return {self.out: [self.inputs[0]]}
    def perform(self):
        x = self.inputs[0].data
        c = self.inputs[1].data
        if Subtensor.debug:
            print 'perform: x', x
            print 'perform: c', c
        if len(c) == 1:
            self.outputs[0].data = x.__getitem__(c[0])
        else:
            self.outputs[0].data = x.__getitem__(c)
    def grad(x, gz):
        # - option: allocate a potentially large matrix of zeros, and fill in
        # the appropriate elements from gz
        # - option: return a sparse matrix
        # - option: return gz, but think about how to include a special addition
        # function that works on a corresponding view of the original data
        raise NotImplementedError() 
subtensor = gof.op.constructor(Subtensor)


##########################
# Arithmetic : Add
##########################

# # Elemwise #
# class AddElemwise(_Elemwise):
#     def impl(self, x, y):
#         try:
#             _assert_same_shapes(x, y)
#         except Exception, e:
#             print '------ ERROR HERE'
#             raise
#         return x + y
#     def grad(self, (x, y), gz):
#         return gz, gz
#     def c_foreach(self, (x_i, y_i), (z_i, )):
#         return "%(z)s_i = %(x)s_i + %(y)s_i;"
# add_elemwise = gof.op.constructor(AddElemwise)

# class AddElemwiseInplace(AddElemwise.inplace_version()):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         x += y
#         return x
# add_elemwise_inplace = gof.op.constructor(AddElemwiseInplace)

# # Scalar #
# class AddScalar(TensorScalarOp):
#     def impl(self, x, a):
#         _assert_tensor_scalar(x, a)
#         return x + a
#     def grad(self, (x, a), gz):
#         return gz, sum(gz)
#     c_expr = "x_i + a"
# add_scalar = gof.op.constructor(AddScalar)

# class AddScalarInplace(AddScalar.inplace_version()):
#     def impl(self, x, a):
#         _assert_tensor_scalar(x, a)
#         x += a
#         return x
# add_scalar_inplace = gof.op.constructor(AddScalarInplace)

# add = _scalar_switch(add_elemwise, add_scalar, add_scalar)
# add_inplace = _scalar_switch(add_elemwise_inplace, add_scalar_inplace)

Add = s2t.make_broadcast(scal.Add)
AddInplace = s2t.make_broadcast(scal.Add, {0:0})

add = gof.op.constructor(s2t.wrap_broadcast(Add))
add_inplace = gof.op.constructor(s2t.wrap_broadcast(AddInplace))


##########################
# Arithmetic : Sub
##########################

# # Elemwise #
# class SubElemwise(_Elemwise):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         return x - y
#     def grad(self, (x, y), gz):
#         return gz, -gz
#     def c_foreach(self, (x_i, y_i), (z_i, )):
#         return "%(z)s_i = %(x)s_i - %(y)s_i;"
# sub_elemwise = gof.op.constructor(SubElemwise)

# class SubElemwiseInplace(SubElemwise.inplace_version()):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         x -= y
#         return x
# sub_elemwise_inplace = gof.op.constructor(SubElemwiseInplace)

# # Scalar #
# def sub_scalar_r(x, a):
#     return add_scalar(x, -a)

# def sub_scalar_l(x, a):
#     return add_scalar(-x, a)

# def sub_scalar_rinplace(x, a):
#     return add_scalar_inplace(x, -a)

# sub = _scalar_switch(sub_elemwise, sub_scalar_r, sub_scalar_l)
# sub_inplace = _scalar_switch(sub_elemwise_inplace, sub_scalar_rinplace)

Sub = s2t.make_broadcast(scal.Sub)
SubInplace = s2t.make_broadcast(scal.Sub, {0:0})

sub = gof.op.constructor(s2t.wrap_broadcast(Sub))
sub_inplace = gof.op.constructor(s2t.wrap_broadcast(SubInplace))


##########################
# Arithmetic : Mul
##########################

# # Elemwise #
# class MulElemwise(_Elemwise):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         return x * y
#     def grad(self, (x, y), gz):
#         return mul(y, gz), mul(x, gz)
#     def c_foreach(self, (x_i, y_i), (z_i, )):
#         return "%(z)s_i = %(x)s_i * %(y)s_i;"
# mul_elemwise = gof.op.constructor(MulElemwise)

# class MulElemwiseInplace(MulElemwise.inplace_version()):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         x *= y
#         return x
# mul_elemwise_inplace = gof.op.constructor(MulElemwiseInplace)

# # Scalar #
# class Scale(TensorScalarOp):
#     def impl(self, x, a):
#         _assert_tensor_scalar(x, a)
#         return x * a
#     def grad(self, (x, a), gz):
#         return scale(a, gz), sum(mul_elemwise(x, gz))
#     c_expr = "%(x)s_i * _%(a)s"
# scale = gof.op.constructor(Scale)

# class ScaleInplace(Scale.inplace_version()):
#     def impl(self, x, a):
#         _assert_tensor_scalar(x, a)
#         x *= a
#         return x
# scale_inplace = gof.op.constructor(ScaleInplace)

# mul = _scalar_switch(mul_elemwise, scale, scale)
# mul_inplace = _scalar_switch(mul_elemwise_inplace, scale_inplace)


Mul = s2t.make_broadcast(scal.Mul)
MulInplace = s2t.make_broadcast(scal.Mul, {0:0})

mul = gof.op.constructor(s2t.wrap_broadcast(Mul))
mul_inplace = gof.op.constructor(s2t.wrap_broadcast(MulInplace))


##########################
# Arithmetic : Div
##########################

# # Elemwise #
# class DivElemwise(_Elemwise):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         return x / y
#     def grad(self, (x, y), gz):
#         return div(gz, y), -div(mul(x, gz), (y*y))
#     def c_foreach(self, (x_i, y_i), (z_i, )):
#         return "%(z)s_i = %(x)s_i / %(y)s_i;"
# div_elemwise = gof.op.constructor(DivElemwise)

# class DivElemwiseInplace(DivElemwise.inplace_version()):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         x /= y
#         return x
# div_elemwise_inplace = gof.op.constructor(DivElemwiseInplace)

# class InvElemwise(_Elemwise):
#     def impl(self, x):
#         return 1.0/x
#     def grad(self, x, gz):
#         ix = inv(x)
#         return -gz * (ix * ix)
#     def c_foreach(self, (x_i, ), (z_i, )):
#         return "%(z)s_i = 1.0 / %(x)s_i;" #TODO: cast 1.0 to the dtype of x
# inv_elemwise = gof.op.constructor(InvElemwise)

# # Scalar #
# def div_scalar_r(x, a):
#     return scale(x, inv_elemwise(a))

# def div_scalar_l(x, a):
#     return scale(inv_elemwise(x), a)

# def div_scalar_rinplace(x, a):
#     return scale_inplace(x, inv_elemwise(a))

# div = _scalar_switch(div_elemwise, div_scalar_r, div_scalar_l)
# div_inplace = _scalar_switch(div_elemwise_inplace, div_scalar_rinplace)

Div = s2t.make_broadcast(scal.Div)
DivInplace = s2t.make_broadcast(scal.Div, {0:0})

div = gof.op.constructor(s2t.wrap_broadcast(Div))
div_inplace = gof.op.constructor(s2t.wrap_broadcast(DivInplace))





##########################
# Arithmetic : Pow
##########################

# # Elemwise #

# class PowElemwise(_Elemwise):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         return x ** y
#     def grad(self, (x, y), gz):
#         gx = gz * y * (pow_elemwise(x, y-1.0))
#         gy = gz * log(x) * pow_elemwise(x, y)
#         return gx, gy
#     def c_foreach(self, (x_i, y_i), (z_i, )):
#         return "%(z)s_i = pow(%(x)s_i, %(y)s_i);"
# pow_elemwise = gof.op.constructor(PowElemwise)

# class PowElemwiseInplace(PowElemwise.inplace_version()):
#     def impl(self, x, y):
#         _assert_same_shapes(x, y)
#         x **= y
#         return x
# pow_elemwise_inplace = gof.op.constructor(PowElemwiseInplace)

# # Scalar #
# class PowScalarL(TensorScalarOp):
#     def impl(self, y, x):
#         _assert_tensor_scalar(y, x)
#         return x ** y
#     def grad(self, (y, x), gz):
#         gx = sum(gz * y * x ** (y-1.0))
#         gy = gz * log(x) * x ** y
#         return gy, gx
#     c_expr = "pow(%(a)s, %(x)s_i)"
# pow_scalar_l = gof.op.constructor(PowScalarL)

# class PowScalarR(TensorScalarOp):
#     def impl(self, x, a):
#         _assert_tensor_scalar(x, a)
#         return x ** a
#     def grad(self, (x, s), gz):
#         gx = scale(mul_elemwise(gz,pow_scalar_r(x, add_scalar(s,-1.0))), s)
#         gs = sum(mul_elemwise(mul_elemwise(gz, pow_scalar_r(x,s)), log(x)))
#         return gx, gs
#     c_expr = "pow(%(x)s_i, _%(a)s)"
# pow_scalar_r = gof.op.constructor(PowScalarR)

# class PowScalarRInplace(PowScalarR.inplace_version()):
#     def impl(self, x, a):
#         _assert_tensor_scalar(x, a)
#         x **= a
#         return x
# pow_scalar_r_inplace = gof.op.constructor(PowScalarRInplace)

# pow = _scalar_switch(pow_elemwise, pow_scalar_r, pow_scalar_l)
# pow_inplace = _scalar_switch(pow_elemwise_inplace, pow_scalar_r_inplace)


Pow = s2t.make_broadcast(scal.Pow)
PowInplace = s2t.make_broadcast(scal.Pow, {0:0})

pow = gof.op.constructor(s2t.wrap_broadcast(Pow))
pow_inplace = gof.op.constructor(s2t.wrap_broadcast(PowInplace))



#########################
# Linalg : Dot
#########################

class Dot(_Op):
    nin=2
    nout=1
    @staticmethod 
    def broadcastable_rule(bx,by):
        if len(bx) == 0:     # x is a scalar
            rval = by
        else:
            if len(by) >= 2: #y is a matrix or tensor
                rval = bx[:-1] + by[:-2] + by[-1:]
            elif len(by)==1: #y is vector
                rval = bx[:-1]
            else:            #y is a scalar
                rval = bx
        return rval
    def propagate_broadcastable(self, bx, by):
        return [self.broadcastable_rule(bx,by)]
    def impl(self, x, y):
        return numpy.dot(x, y)
    def grad(self, (x, y), gz):
        return dot(gz, y.T), dot(x.T, gz)
    if 0:
        def c_support_code(self):
            return blas.cblas_header_text()
        def c_libs(self):
            return blas.ldflags()
        def c_impl(self, (_x, _y), (_z, )):
            return blas.gemm_code('', '1.0', '0.0')
dot = gof.op.constructor(Dot)

class Gemm(_Op):
    nin=5
    nout=1
    E_rank = 'gemm only works for rank 2'
    E_scalar = 'gemm requires scalar argument'
    def destroy_map(self):
        return {self.out:[self.inputs[0]]}
    def propagate_broadcastable(self, bz, ba, bx, by, bb):
        if len(bz) != 2: raise ValueError(Gemm.E_rank, len(bz))
        if len(bx) != 2: raise ValueError(Gemm.E_rank, len(bx))
        if len(by) != 2: raise ValueError(Gemm.E_rank, len(by))
        if len(ba): raise ValueError(Gemm.E_scalar, ba)
        if len(bb): raise ValueError(Gemm.E_scalar, bb)

        return [bz]
    def impl(self, z, a, x, y, b):
        assert a.shape == ()
        assert b.shape == ()
        if z.shape == ():
            z.itemset(z*a + b*numpy.dot(x,y))
            return z
        else:
            if b == 0.0:
                if a == 1.0:
                    z[:] = numpy.dot(x,y)
                elif a == -1.0:
                    z[:] = -numpy.dot(x,y)
                else:
                    z[:] = a * numpy.dot(x,y)
            elif b == 1.0:
                if a == 1.0:
                    z += numpy.dot(x,y)
                elif a == -1.0:
                    z -= numpy.dot(x,y)
                else:
                    z += a * numpy.dot(x,y)
            else:
                z *= b
                z += a * numpy.dot(x,y)
            return z
    def grad(self, (z, a, x, y, b), gz):
        raise NotImplementedError()

    def c_support_code(self):
        return blas.cblas_header_text()
    def c_libraries(self):
        return blas.ldflags()
    def c_var_names(self):
        return [['_z', '_a', '_x', '_y', '_b'], ['_zout']]
    def c_validate_update(self, (_z, _a, _x, _y, _b), (_zout, ), sub):
        return """
        if (%(_zout)s)
        {
            Py_DECREF(%(_zout)s);
        }
        if (%(_zout)s != %(_z)s)
        {
            %(_zout)s = %(_z)s;
            Py_INCREF(%(_zout)s);
        }
        """ % locals()
    def c_validate_update_cleanup(self, ignore, _ignore, __ignore):
        return ""
    def c_code(self, (_z, _a, _x, _y, _b), (_zout, ), sub):
        return """
        int unit = 0;

        int type_num = %(_x)s->descr->type_num;
        int type_size = %(_x)s->descr->elsize; // in bytes

        npy_intp* Nx = %(_x)s->dimensions;
        npy_intp* Ny = %(_y)s->dimensions;
        npy_intp* Nz = %(_z)s->dimensions;

        npy_intp* Sx = %(_x)s->strides;
        npy_intp* Sy = %(_y)s->strides;
        npy_intp* Sz = %(_z)s->strides;

        size_t sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;

        if (%(_x)s->nd != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(_y)s->nd != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(_z)s->nd != 2)
        {PyErr_SetString(PyExc_NotImplementedError, "rank(z) != 2"); %(fail)s;}

        if ((%(_a)s->descr->type_num != PyArray_DOUBLE)
            && (%(_a)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        if ((%(_b)s->descr->type_num != PyArray_DOUBLE)
            && (%(_b)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(b) is not double or float"); %(fail)s;}

        if ((%(_x)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_x)s->descr->type_num != PyArray_FLOAT))
            %(fail)s;

        if ((%(_y)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_y)s->descr->type_num != PyArray_FLOAT))
            %(fail)s;

        if ((%(_y)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_y)s->descr->type_num != PyArray_FLOAT))
            %(fail)s;

        if ((%(_x)s->descr->type_num != %(_y)s->descr->type_num)
            ||(%(_x)s->descr->type_num != %(_z)s->descr->type_num))
            %(fail)s;

        if ((Nx[0] != Nz[0]) || (Nx[1] != Ny[0]) || (Ny[1] != Nz[1]))
        {
            PyErr_SetString(PyExc_ValueError, "Input dimensions do not agree");
            %(fail)s;
        }
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
           || (Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
           || (Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size))
        {
            PyErr_SetString(PyExc_ValueError, "gemm cant run on these inputs");
            %(fail)s;
        }


        /*
        encode the stride structure of _x,_y,_z into a single integer
        */
        unit |= ((Sx[1] == type_size) ? 0x0 : (Sx[0] == type_size) ? 0x1 : 0x2) << 0;
        unit |= ((Sy[1] == type_size) ? 0x0 : (Sy[0] == type_size) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size) ? 0x0 : (Sz[0] == type_size) ? 0x1 : 0x2) << 8;

        /* create appropriate strides for malformed matrices that are row or column
         * vectors
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : Nx[1];
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : Nx[0];
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : Ny[1];
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : Ny[0];
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : Nz[1];
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : Nz[0];

        switch (type_num)
        {
            case PyArray_FLOAT:
            {
                #define REAL float
                float a = (%(_a)s->descr->type_num == PyArray_FLOAT) 
                ? (REAL)(((float*)%(_a)s->data)[0])
                : (REAL)(((double*)%(_a)s->data)[0]);
                float b = (%(_b)s->descr->type_num == PyArray_FLOAT) ?
                (REAL)(((float*)%(_b)s->data)[0])
                : (REAL)(((double*)%(_b)s->data)[0]);

                float* x = (float*)PyArray_DATA(%(_x)s);
                float* y = (float*)PyArray_DATA(%(_y)s);
                float* z = (float*)PyArray_DATA(%(_z)s);

                switch(unit)
                {
                    case 0x000: cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_0); break;
                    case 0x001: cblas_sgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_0); break;
                    case 0x010: cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_0); break;
                    case 0x011: cblas_sgemm(CblasRowMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_0); break;
                    case 0x100: cblas_sgemm(CblasColMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_1); break;
                    case 0x101: cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_1); break;
                    case 0x110: cblas_sgemm(CblasColMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_1); break;
                    case 0x111: cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_1); break;
                    default: %(fail)s;
                };
                #undef REAL
            }
            break;
            case PyArray_DOUBLE:
            {
                #define REAL double

                double a = (%(_a)s->descr->type_num == PyArray_FLOAT) 
                ? (REAL)(((float*)%(_a)s->data)[0])
                : (REAL)(((double*)%(_a)s->data)[0]);
                double b = (%(_b)s->descr->type_num == PyArray_FLOAT) ?
                (REAL)(((float*)%(_b)s->data)[0])
                : (REAL)(((double*)%(_b)s->data)[0]);
                double* x = (double*)PyArray_DATA(%(_x)s);
                double* y = (double*)PyArray_DATA(%(_y)s);
                double* z = (double*)PyArray_DATA(%(_z)s);
                switch(unit)
                {
                    case 0x000: cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_0); break;
                    case 0x001: cblas_dgemm(CblasRowMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_0); break;
                    case 0x010: cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_0); break;
                    case 0x011: cblas_dgemm(CblasRowMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_0); break;
                    case 0x100: cblas_dgemm(CblasColMajor, CblasTrans,   CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_0, b, z, sz_1); break;
                    case 0x101: cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,   Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_0, b, z, sz_1); break;
                    case 0x110: cblas_dgemm(CblasColMajor, CblasTrans,   CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_0, y, sy_1, b, z, sz_1); break;
                    case 0x111: cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nz[0], Nz[1], Nx[1], a, x, sx_1, y, sy_1, b, z, sz_1); break;
                    default: %(fail)s;
                };
                #undef REAL
            }
            break;
        }

        """ % dict(locals(), **sub)
gemm = gof.op.constructor(Gemm)


if 0:
    ##########################
    # Comparisons 
    ##########################

    # Less-than
    class lt_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class lt_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    # Less-than or equal
    class le_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class le_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    # Greater-than or equal
    class gt_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class gt_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    # Greater-than or equal
    class ge_elemwise(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()
    class ge_scalar_r(_Elemwise):
        def __init__(self, *args):
            raise NotImplementedError()




if 0:
    def _broadcastable_pattern(pattern):
        def factory(data = None, name = None, dtype=None):
            if data: 
                assert len(data.shape) == len(pattern)
                if dtype is not None:
                    assert dtype is data.dtype
                dtype = data.dtype
                rval = Tensor(dtype, pattern, name)
                rval.data = data
            else:
                rval = Tensor(dtype, pattern, name)
            return  rval
        return factory

    row = _broadcastable_pattern([1, 0])
    col = _broadcastable_pattern([0, 1])
    matrix = _broadcastable_pattern([0, 0])

if 0: #old __init__ code
    """Create a Tensor

    If data is given:
        - constant defaults to True
        - if dtype is given, it must match data.dtype
            - otherwise: default is data.dtype
        - if broadcastable is given, len(broadcastable) must match len(data.shape)
            - otherwise: if it is constant, it defaults to 1 where shape[i]==1
            - if it is not constant, it defaults to 0s

    If data is not given:
        - constant defaults to False
    """
    if dtype is None or broadcastable is None:
        if data is None:
            raise TypeError("Provide non-None data to complete the dtype and broadcastable flags.")
        data = numpy.asarray(data)
        if constant is None:
            constant = True
        dtype = data.dtype
        if constant:
            broadcastable = [1*(x == 1) for x in data.shape]
        else:
            broadcastable = [0] * len(data.shape)

if 0:
    def tensor__new__(cls, *args, **kwargs):
        """__new__ is overloaded to handle the special form Tensor(x) when x is
        a Tensor or an Op whose default output is a Tensor.  In these cases, the
        argument x is returned, and a new Tensor is not created.
        """
        if len(args) == 1:
            a = args[0]

        t = super(Tensor, cls).__new__(cls, *args, **kwargs)
        t.__init__(*args, **kwargs)
        return t


#         def upcast(dtype, *dtypes):
#             z = numpy.zeros((), dtype = dtype)
#             for dtype in dtypes:
#                 z = z + numpy.zeros((), dtype = dtype)
#             return str(z.dtype)
#         for dtype in i_dtypes:
#             if dtype is None:
#                 raise TypeError("Expected a Tensor.")
#         upcasted = upcast(*i_dtypes)
#         return [upcasted] * self.nout
# #         try:
# #             dmap = self.destroy_map()
# #         except AttributeError:
# #             dmap = {}
# #         rval = []
# #         for i in xrange(self.nout):
# #             if i in dmap:
# #                 destroyed = dmap[output]
# #                 if len(destroyed) != 1:
# #                     raise TypeError("Cannot infer dtype of output %s because it destroys more than one input." % output)
# #                 rval.append(destroyed[0])
# #             else:
# #                 rval.append(upcasted)
# #         return rval
    
