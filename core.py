
import gof
from gof import current_mode, set_mode, build_mode, eval_mode, build_eval_mode, pop_mode, UNCOMPUTED, UNDEFINED, PythonR

import numpy

from copy import copy as pycopy

# __all__ = ['set_mode', 'get_mode', 'NumpyR', 'NumpyOp']


def build(f, *args, **kwargs):
    build_mode()
    r = f(*args, **kwargs)
    pop_mode()
    return r


class Proxy(object):

    __slots__ = ['_obj']
    
    def __init__(self, obj = None):
        self._obj = obj

    def __getattribute__(self, attr):
        if attr in ['__class__', '_obj']:
            return object.__getattribute__(self, attr)
        else:
            return getattr(object.__getattribute__(self, '_obj'), attr)

    def __setattr__(self, attr, value):
        if attr in ['_obj']:
            object.__setattr__(self, attr, value)
        else:
            setattr(self._obj, attr, value)

    def __delattr__(self, attr):
        delattr(self._obj, attr)


def as_string(*rs):
    s = gof.graph.as_string(gof.graph.inputs(rs), rs)
    if len(rs) == 1:
        return s[1:-1]
    else:
        return s
#    return str(gof.Env(gof.graph.inputs([r]), [r]))[1:-1]

def print_graph(*rs):
    print as_string(*rs)


def input(x):
    if isinstance(x, numpy.ndarray):
        return NumpyR(x)
    elif isinstance(x, (int, float)):
        return NumpyR(numpy.array(x))
    elif isinstance(x, gof.Result):
        raise TypeError("%s is already a result." % x)
    else:
        return PythonR(x)

def wrap(x):
    if isinstance(x, NumpyR):
        return x
    elif isinstance(x, PythonR):
        return x
    elif isinstance(x, omega_op):
        return x.out
    elif isinstance(x, Proxy):
        return wrap(x._obj)
    else:
        return literal(x)
#     elif isinstance(x, numpy.ndarray):
#         return NumpyR(x)
#     elif isinstance(x, (int, float)):
#         return NumpyR(numpy.array(x))
#     else:
#         return PythonR(x)

def literal(x):
    try:
        present = x in gof.literals_db
        hashable = True
    except TypeError: # x is unhashable
        present = False
        hashable = False

    if present:
        return gof.literals_db.get(x)
    elif isinstance(x, numpy.ndarray):
        ret = NumpyR(x, constant = True)
    elif isinstance(x, (int, float)):
        ret = NumpyR(numpy.array(x), constant = True)
    elif isinstance(x, gof.Result):
        raise TypeError("%s is already a result." % x)
    else:
        return PythonR(x, constant = True)

    if hashable:
        gof.literals_db[x] = ret

    return ret



inplace = gof.Destroyer
view = gof.Viewer

class omega_op(gof.PythonOp):

    forbid_broadcast = False

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make grad a static method
        grad = cls.grad
        if hasattr(grad, 'im_func'):
            grad = grad.im_func
        cls.grad = staticmethod(grad)

#         # adjust impl
#         if cls.forbid_broadcast:
#             cls.impl = assert_same_shapes(cls.impl)

        # make impl a static method
        gof.PythonOp.__clsinit__(cls, name, bases, dct)
    
    def __new__(cls, *inputs):
        inputs = [wrap(input) for input in inputs]
        return gof.PythonOp.__new__(cls, *inputs)

    def gen_outputs(self):
        return [NumpyR() for i in xrange(self.nout)]
    
    def update_gradient(self, grad_d):
        inputgs = self.grad(*(self.inputs + [grad_d[output] for output in self.outputs]))
        if not isinstance(inputgs, (list, tuple)):
            inputgs = [inputgs] * len(self.inputs)
        for input, inputg in zip(self.inputs, inputgs):
            grad_d.add(input, inputg)

    def grad(*args):
        return UNDEFINED


def scalar_switch(normal_f, scalar_f, scalar_f_reverse = None):
    def f(x, y):
        x, y = wrap(x), wrap(y)
        if y.constant and not y.data.shape:
            return scalar_f(x, y)
        if x.constant and not x.data.shape:
            if scalar_f_reverse:
                return scalar_f_reverse(y, x)
            else:
                raise TypeError("You cannot do this operation on a scalar.")
        return normal_f(x, y)
    return f


class NumpyR(gof.PythonR):

    def set_value(self, value):
        if value is None or value is UNCOMPUTED:
            self.data = UNCOMPUTED
        elif isinstance(value, numpy.ndarray):
            self.data = value
        else:
            self.data = numpy.array(value)
        self.up_to_date = True

    def  __add__(self, y): return add(self, y)
    def __radd__(self, x): return add(x, self)
    def __iadd__(self, y): return iadd(self, y)
    
    def  __sub__(self, y): return sub(self, y)
    def __rsub__(self, x): return sub(x, self)
    def __isub__(self, y): return isub(self, y)
    
    def  __mul__(self, y): return mul(self, y)
    def __rmul__(self, x): return mul(x, self)
    def __imul__(self, y): return imul(self, y)
 
    def  __div__(self, y): return div(self, y)
    def __rdiv__(self, x): return div(x, self)
    def __idiv__(self, y): return idiv(self, y)
        
    def  __pow__(self, y): return pow(self, y)
    def __rpow__(self, x): return pow(x, self)
    def __ipow__(self, y): return ipow(self, y)

    def __neg__(self):     return neg(self)

    T  = property(lambda self: transpose(self))
    Tc = property(lambda self: transpose_copy(self))

    def __copy__(self):    return array_copy(self)

    
def wrap_producer(f):
    class producer(omega_op):
        impl = f
    producer.__name__ = f.__name__
    def ret(dim, dtype = 'float', order = 'C'):
        return producer(dim, dtype, order)
    return ret

ndarray = wrap_producer(numpy.ndarray)
array = wrap_producer(numpy.array)
zeros = wrap_producer(numpy.zeros)
ones = wrap_producer(numpy.ones)


# Wrapper to ensure that all inputs to the function impl have the same size (foils numpy's broadcasting)
def assert_same_shapes(impl):
    def ret(x, *rest):
        shape = x.shape
        for other in rest:
            if other.shape != shape:
                raise ValueError("The dimensions of the inputs do not match.")
        return impl(x, *rest)
    return ret

# Wrapper to ensure that the last input to impl is a scalar
def tensor_scalar_op(impl):
    def ret(x, a):
        if a.shape:
            raise ValueError("The second argument to %s must be a scalar." % impl)
        return impl(x, a)
    return ret


## Addition ##

class proto_add_elemwise(omega_op):
    def grad(x, y, gz):
        return gz

class add_elemwise(proto_add_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__add__)

class iadd_elemwise(proto_add_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__iadd__)

class proto_add_scalar(omega_op):
    def grad(x, a, gz):
        return gz, sum(gz)

class add_scalar(proto_add_scalar):
    impl = tensor_scalar_op(numpy.ndarray.__add__)

class iadd_scalar(proto_add_scalar, inplace):
    impl = tensor_scalar_op(numpy.ndarray.__iadd__)


class proto_twice(omega_op):
    def grad(x, gz):
        return scale(gz, 2.0)

class twice(proto_twice):
    def impl(x):
        return x + x

class itwice(proto_twice, inplace):
    def impl(x):
        x += x
        return x


## Subtraction ##

class proto_sub_elemwise(omega_op):
    def grad(x, y, gz):
        return gz, -gz

class sub_elemwise(proto_sub_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__sub__)

class isub_elemwise(proto_sub_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__isub__)

def sub_scalar_r(x, a):
    return add_scalar(x, -a)

def sub_scalar_l(x, a):
    return add_scalar(-x, a)

def isub_scalar_r(x, a):
    return iadd_scalar(x, -a)

# def isub_scalar_l(x, a):
#     return iadd_scalar(ineg(x), a)


## Element-wise multiplication ##

class proto_mul_elemwise(omega_op):
    def grad(x, y, gz):
        return mul(y, gz), mul(x, gz)

class mul_elemwise(proto_mul_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__mul__)

class imul_elemwise(proto_mul_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__imul__)


class proto_scale(omega_op):
    def grad(x, a, gz):
        return scale(a, gz), sum(mul_elemwise(x, gz))

class scale(proto_scale):
    impl = tensor_scalar_op(numpy.ndarray.__mul__)

class iscale(proto_scale, inplace):
    impl = tensor_scalar_op(numpy.ndarray.__imul__)


class proto_sqr(omega_op):
    def grad(x, gz):
        return scale(mul_elemwise(x, gz), 2.0)

class sqr(proto_sqr):
    impl = lambda x: numpy.multiply(x, x)

class isqr(proto_sqr, inplace):
    impl = lambda x: x.__imul__(x)


class proto_sqrt(omega_op):
    def grad(x, gz):
        return scale(div(gz, sqrt(x)), 0.5)

class sqrt(proto_sqrt):
    impl = numpy.sqrt

class isqrt(proto_sqrt, inplace):
    impl = lambda x: x.__ipow__(0.5)


## Exponentiation ##

class exp(omega_op):
    impl = numpy.exp
    

## Element-wise division ##

class proto_div_elemwise(omega_op):
    def grad(x, y, gz):
        return div(gz, y), -div(mul(x, gz), sqr(y))

class div_elemwise(proto_div_elemwise):
    impl = assert_same_shapes(numpy.ndarray.__div__)

class idiv_elemwise(proto_div_elemwise, inplace):
    impl = assert_same_shapes(numpy.ndarray.__idiv__)

def div_scalar_r(x, a):
    return scale(x, inv_elemwise(a))

def div_scalar_l(x, a):
    return scale(inv_elemwise(x), a)

def idiv_scalar_r(x, a):
    return iscale(x, inv_elemwise(a))

# def idiv_scalar_l(x, a):
#     return iscale(inv_elemwise(x), a)



## Scaling ##

    
class proto_neg(omega_op):
    def grad(x, gz):
        return -gz

class neg(proto_neg):
    impl = numpy.ndarray.__neg__

class ineg(proto_neg, inplace):
    impl = lambda x: x.__imul__(-1)


class proto_inv_elemwise(omega_op):
    def grad(x, gz):
        raise NotImplemented

class inv_elemwise(omega_op):
    impl = lambda x: 1 / x

class iinv_elemwise(omega_op, inplace):
    def impl(x):
        x[:] = 1 / x


## Dot product ##

class dot(omega_op):
    impl = numpy.dot
    def grad(x, y, gz):
        return dot(gz, transpose(y)), dot(transpose(x), gz)


## Transposition ##

class transpose(omega_op, view):
    impl = numpy.transpose
    def grad(x, gz):
        return transpose_copy(gz)

def transpose_copy(x):
    return array_copy(transpose(x))


## Copy ##

class array_copy(omega_op):
    impl = numpy.array
    grad = lambda x, gz: gz


## Power ##

class proto_pow(omega_op):
    pass

class pow_elemwise(proto_pow):
    impl = assert_same_shapes(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        return gz * s * (pow_elemwise(x, s-1.0))

class pow_scalar_l(proto_pow):
    impl = tensor_scalar_op(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        return gz * x * (pow_scalar_l(s,x-1.0))

class pow_scalar_r(proto_pow):
    impl = tensor_scalar_op(numpy.ndarray.__pow__)
    def grad(x, s, gz):
        return gz * s * (pow_scalar_r(x,s-1.0))

class proto_ipow(omega_op):
    pass

class ipow_elemwise(proto_ipow):
    def __init__(self, *args, **kwargs):
        omega_op.__init__(self, *args, **kwargs)
        raise NotImplementedError()

class ipow_scalar_l(proto_ipow):
    def __init__(self, *args, **kwargs):
        omega_op.__init__(self, *args, **kwargs)
        raise NotImplementedError()

class ipow_scalar_r(proto_ipow):
    def __init__(self, *args, **kwargs):
        omega_op.__init__(self, *args, **kwargs)
        raise NotImplementedError()

## Others ##

class minmax(omega_op):
    nout = 2
    def impl(x):
        return x.min, x.max

class fill(omega_op):
    impl = lambda model, value: (model * 0) + value

class sum(omega_op):
    impl = numpy.sum
    def grad(x, gz):
        return fill(x, gz)

    
# array_copy = wrapper("copy",
#                      numpy.array,
#                      lambda x, gz: gz)


# array slicing




add = scalar_switch(add_elemwise, add_scalar, add_scalar)
iadd = scalar_switch(iadd_elemwise, iadd_scalar)

sub = scalar_switch(sub_elemwise, sub_scalar_r, sub_scalar_l)
isub = scalar_switch(isub_elemwise, isub_scalar_r)

mul = scalar_switch(mul_elemwise, scale, scale)
imul = scalar_switch(imul_elemwise, iscale)

div = scalar_switch(div_elemwise, div_scalar_r, div_scalar_l)
idiv = scalar_switch(idiv_elemwise, idiv_scalar_r)

pow = scalar_switch(pow_elemwise, pow_scalar_r, pow_scalar_l)
ipow = scalar_switch(ipow_elemwise, ipow_scalar_r, ipow_scalar_l)



