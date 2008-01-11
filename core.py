
import gof
from gof import current_mode, set_mode, build_mode, eval_mode, pop_mode, UNCOMPUTED, UNDEFINED, PythonR

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

def assert_same_shapes(impl):
    def ret(x, *rest):
        shape = x.shape
        for other in rest:
            if other.shape != shape:
                raise TypeError("The dimensions of the inputs do not match.")
        return impl(x, *rest)
    return ret

class omega_op(gof.PythonOp):

    broadcast_op = False

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make grad a static method
        grad = cls.grad
        if hasattr(grad, 'im_func'):
            grad = grad.im_func
        cls.grad = staticmethod(grad)

        # adjust impl
        if cls.broadcast_op:
            cls.impl = assert_same_shapes(cls.impl)

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


def scalar_switch(x, y, normal_f, scalar_f):
    x, y = wrap(x), wrap(y)
    if x.constant and not x.data.shape:
        return scalar_f(y, x)
    if y.constant and not y.data.shape:
        return scalar_f(x, y)
    return normal_f(x, y)


class NumpyR(gof.PythonR):

    def set_value(self, value):
        if value is None or value is UNCOMPUTED:
            self.data = UNCOMPUTED
        elif isinstance(value, numpy.ndarray):
            self.data = value
        else:
            self.data = numpy.array(value)

    def  __add__(self, y): return scalar_switch(self, y, add, add_scalar)
    def __radd__(self, x): return scalar_switch(x, self, add, add_scalar)
    def __iadd__(self, y): return scalar_switch(self, y, iadd, iadd_scalar)
    
    def  __sub__(self, y): return scalar_switch(self, y, sub, sub_scalar)
    def __rsub__(self, x): return scalar_switch(x, self, sub, sub_scalar)
    def __isub__(self, y): return scalar_switch(self, y, isub, isub_scalar)
    
    def  __mul__(self, y): return scalar_switch(self, y, mul, scale)
    def __rmul__(self, x): return scalar_switch(x, self, mul, scale)
    def __imul__(self, y): return scalar_switch(self, y, imul, iscale)
 
    def  __div__(self, y): return scalar_switch(self, y, div, inv_scale)
    def __rdiv__(self, x): return scalar_switch(x, self, div, inv_scale)
    def __idiv__(self, y): return scalar_switch(self, y, idiv, iinv_scale)
        
    def  __pow__(self, y): return scalar_switch(self, y, pow_elemwise, pow)
    def __rpow__(self, x): return scalar_switch(x, self, pow_elemwise, pow)
    def __ipow__(self, y): return scalar_switch(self, y, ipow_elemwise, ipow)

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


## Addition ##

class proto_add(omega_op):
    broadcast_op = True
    def grad(x, y, gz):
        return gz

class add(proto_add):
    impl = numpy.ndarray.__add__

class iadd(proto_add, inplace):
    impl = numpy.ndarray.__iadd__

class add_scalar(omega_op):
    impl = numpy.ndarray.__add__

class iadd_scalar(omega_op):
    impl = numpy.ndarray.__iadd__


class proto_twice(omega_op):
    def grad(x, gz):
        return scal(gz, 2.0)

class twice(proto_twice):
    def impl(x):
        return x + x

class itwice(proto_twice, inplace):
    def impl(x):
        x += x
        return x


## Subtraction ##

class proto_sub(omega_op):
    def grad(x, y, gz):
        return gz, -gz

class sub(proto_sub):
    impl = numpy.ndarray.__sub__

class isub(proto_sub, inplace):
    impl = numpy.ndarray.__isub__

class sub_scalar(omega_op):
    impl = numpy.ndarray.__sub__

class isub_scalar(omega_op, inplace):
    impl = numpy.ndarray.__isub__


## Element-wise multiplication ##

class proto_mul(omega_op):
    def grad(x, y, gz):
        return mul(y, gz), mul(x, gz)

class mul(proto_mul):
    impl = numpy.ndarray.__mul__

class imul(proto_mul, inplace):
    impl = numpy.ndarray.__imul__


class proto_sqr(omega_op):
    def grad(x, gz):
        return scale(mul(x, gz), 2.0)

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

class proto_div(omega_op):
    def grad(x, y, gz):
        return div(gz, y), -div(mul(x, gz), sqr(y))

class div(proto_div):
    impl = numpy.ndarray.__div__

class idiv(proto_div, inplace):
    impl = numpy.ndarray.__idiv__

class inv_scale(omega_op):
    impl = numpy.ndarray.__div__

class iinv_scale(omega_op, inplace):
    impl = numpy.ndarray.__idiv__


## Scaling ##

class proto_scale(omega_op):
    def grad(x, a, gz):
        return scale(a, gz), sum(mul(x, gz))

class scale(omega_op):
    impl = numpy.ndarray.__mul__

class iscale(omega_op, inplace):
    impl = numpy.ndarray.__imul__

    
class proto_neg(omega_op):
    def grad(x, gz):
        return -gz

class neg(omega_op):
    impl = numpy.ndarray.__neg__

class ineg(omega_op, inplace):
    impl = lambda x: x.__imul__(-1)


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
    def grad(x, y, gz):
        pass

class pow(proto_pow):
    impl = numpy.ndarray.__pow__

class ipow(proto_pow, inplace):
    impl = numpy.ndarray.__ipow__


class proto_pow_elemwise(omega_op):
    def grad(x, y, gz):
        pass

class pow_elemwise(proto_pow_elemwise):
    impl = numpy.ndarray.__pow__

class ipow_elemwise(proto_pow_elemwise, inplace):
    impl = numpy.ndarray.__ipow__


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



