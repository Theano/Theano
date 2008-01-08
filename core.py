
import gof
import numpy

from copy import copy as pycopy

# __all__ = ['set_mode', 'get_mode', 'NumpyR', 'NumpyOp']


_mode = ['eval']

# def set_mode(mode):
#     _mode.append(mode)

# def current_mode():
#     return _mode[-1]

# def build_mode():
#     set_mode('build')

# def eval_mode():
#     set_mode('eval')

# def pop_mode():
#     if len(_mode) == 1:
#         raise Exception("There's only one mode left on the stack.")
#     else:
#         _mode.pop()

# def end_eval():
#     set_mode('eval')

current_mode, set_mode, build_mode, eval_mode, pop_mode = gof.current_mode, gof.set_mode, gof.build_mode, gof.eval_mode, gof.pop_mode

def build(f, *args, **kwargs):
    build_mode()
    r = f(*args, **kwargs)
    pop_mode()
    return r


# class Keyword:

#     def __init__(self, name, nonzero=True):
#         self.name = name
#         self.nonzero = nonzero

#     def __nonzero__(self):
#         return self.nonzero

#     def __str__(self):
#         return "<%s>" % self.name

#     def __repr__(self):
#         return str(self)

UNCOMPUTED = gof.UNCOMPUTED
UNDEFINED = gof.UNDEFINED

# UNCOMPUTED = Keyword("UNCOMPUTED", False)
# UNDEFINED = Keyword("UNDEFINED", False)



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


# class IViewer(gof.ext.Viewer):
#     _v_map = {}

#     def view_map(self):
#         rval = {}
#         for output, inputs in self._v_map.items():
#             if isinstance(inputs, (list, tuple)):
#                 rval[self.outputs[output]] = [self.inputs[i] for i in inputs]
#             else:
#                 rval[self.outputs[output]] = self.inputs[inputs]
#         return rval


# class IDestroyer(gof.ext.Destroyer):
#     _d_map = {}

#     def destroy_map(self):
#         rval = {}
#         for output, inputs in self._d_map.items():
#             if isinstance(inputs, (list, tuple)):
#                 rval[self.outputs[output]] = [self.inputs[i] for i in inputs]
#             else:
#                 rval[self.outputs[output]] = self.inputs[inputs]
#         return rval


# class PythonR(gof.HolderResult):

#     def __init__(self, a = None):
#         if a is None:
#             self.storage = UNCOMPUTED
#         else:
#             self.storage = a

#     def set_value(self, value):
#         self.storage = value

#     def __str__(self):
#         return str(self.storage)

#     def __repr__(self):
#         return repr(self.storage)
# gof.PythonR = PythonR


class NumpyR(gof.PythonR):

#     def __init__(self, a = None):
#         self.set_value(a)

    def set_value(self, value):
        if value is None or value is UNCOMPUTED:
            self.data = UNCOMPUTED
        elif isinstance(value, numpy.ndarray):
            self.data = value
        else:
            self.data = numpy.array(value)
        
    def __add__(self, y):
        return add(self, y)

    def __radd__(self, x):
        return add(x, self)

    def __iadd__(self, y):
        return iadd(self, y)
    
    def __sub__(self, y):
        return sub(self, y)

    def __rsub__(self, x):
        return sub(x, self)

    def __isub__(self, y):
        return isub(self, y)
    
    def __mul__(self, y):
        return dot(self, y)

    def __rmul__(self, x):
        return dot(x, self)

    def __imul__(self, y):
        return imul(self, y)
 
    def __div__(self, y):
        return div(self, y)

    def __rdiv__(self, x):
        return div(x, self)

    def __idiv__(self, y):
        return idiv(self, y)
    
    def __mod__(self, y):
        return mod(self, y)

    def __rmod__(self, x):
        return mod(x, self)

    def __pow__(self, y):
        return pow(self, y)

    def __rpow__(self, x):
        return pow(x, self)

    def __ipow__(self, y):
        return ipow(self, y)

    def __neg__(self):
        return neg(self)

    T = property(lambda self: transpose(self))
    Tc = property(lambda self: transpose_copy(self))

    def __copy__(self):
        return array_copy(self)
    

#[iadd(iadd(iadd(iadd(<UNCOMPUTED>, itwice(<UNCOMPUTED>)), <UNCOMPUTED>), 1.0), dot(<UNCOMPUTED>, <UNCOMPUTED>))]
#[iadd(iadd(iadd(iadd(<UNCOMPUTED>, itwice(<UNCOMPUTED>)), <UNCOMPUTED>), 1.0), dot(<UNCOMPUTED>, <UNCOMPUTED>))]


def wrap(x):
#     try:
#         return to_numpyr(x)
#     except TypeError:
#         if isinstance(x, PythonR):
#             return x
#         else:
#             return PythonR(x)


# def to_numpyr(x):
    
    if isinstance(x, NumpyR):
        return x
    elif isinstance(x, gof.PythonR):
        return x
    elif isinstance(x, omega_op):
        return x.out
    elif isinstance(x, Proxy):
        return wrap(x._obj)
    elif isinstance(x, numpy.ndarray):
        return NumpyR(x)
    else:
        return gof.PythonR(x)
#     else:
#         raise TypeError("%s cannot be converted to or encapsulated in a NumpyR instance." % x)


# class NumpyOp(gof.Op, gof.ext.BuildableFromInputs):

#     nout = 1
    
#     def __init__(self, *args):
#         inputs = [wrap(arg) for arg in args]
#         outputs = [NumpyR() for i in xrange(self.nout)]
#         gof.Op.__init__(self, inputs, outputs)

#     @classmethod
#     def from_inputs(cls, *inputs):
#         return cls(*inputs)

#     def gen_outputs(self):
#         return [NumpyR() for i in xrange(self.nout)]



# class wrapper:

#     __slots__ = ['f', 'opclass']
    
#     def __init__(self, name, f, grad, vmap = None, dmap = None, optype = NumpyOp):
#         self.f = f

#         if not callable(f):
#             raise TypeError("Can only wrap a callable.")

#         bases = [optype]
#         if vmap: bases.append(IViewer)
#         if dmap: bases.append(IDestroyer)
        
#         Wrapper = type(name, tuple(bases), {})
#         if vmap: Wrapper._v_map = vmap
#         if dmap: Wrapper._d_map = dmap
        
#         def thunk(self):
#             def ret():
#                 self.outputs[0].set_value(f(*[input.storage for input in self.inputs]))
#             return ret
#         Wrapper.thunk = thunk
        
#         if grad is UNDEFINED:
#             grad = lambda *_: UNDEFINED
#         Wrapper.grad = staticmethod(grad)

#         self.opclass = Wrapper

    
#     def __call__(self, *args):
#         op = self.opclass(*args)
#         if current_mode() == 'eval':
#             op.thunk()()
#         outputs = pycopy(op.outputs)
# #        outputs = [Proxy(output) for output in op.outputs]
#         if op.nout == 1:
#             return outputs[0]
#         else:
#             return outputs


# def wrap_producer(f):
#     def ret(*args, **kwargs):
#         result = f(*args, **kwargs)
#         if not isinstance(result, numpy.ndarray):
#             result = numpy.array(result)
#         return NumpyR(result)
#     return ret


# def wrap_producer(f):
#     wrapped_f = wrapper(f.__name__, f, UNDEFINED)
#     def ret(dim, dtype = 'float', order = 'C'):
#         return wrapped_f(dim, dtype, order)
#     return ret


inplace = gof.ext.Destroyer
view = gof.ext.Viewer


class omega_op_metaclass(type):
    
    def __init__(cls, name, bases, dct):
        type.__init__(cls, name, bases, dct)
        cls.__clsinit__(name, bases, dct)



class omega_op(gof.PythonOp):  #(gof.Op, gof.ext.BuildableFromInputs):

##    __metaclass__ = omega_op_metaclass
    
##    nout = 1

    @staticmethod
    def __clsinit__(cls, name, bases, dct):
        # make grad a static method
        grad = cls.grad
        if hasattr(grad, 'im_func'):
            grad = grad.im_func
        cls.grad = staticmethod(grad)

        # make impl a static method
        gof.PythonOp.__clsinit__(cls, name, bases, dct)
    
    def __new__(cls, *inputs):
        inputs = [wrap(input) for input in inputs]
        return gof.PythonOp.__new__(cls, *inputs)
#         op = gof.Op.__new__(cls)
#         op.__init__(*[wrap(input) for input in inputs])
#         if cls.current_mode() == 'eval':
#             op.thunk()()
#         if op.nout == 1:
#             return op.out
#         else:
#             return op.outputs
    
#     def __init__(self, *inputs):
#         for input in inputs:
#             assert isinstance(input, gof.HolderResult)
#         gof.Op.__init__(self, inputs, self.gen_outputs())

#     @classmethod
#     def from_inputs(cls, *inputs):
#         build_mode()
#         r = cls(*inputs)
#         pop_mode()
#         return r.owner

    def gen_outputs(self):
        return [NumpyR() for i in xrange(self.nout)]

#     def thunk(self):
#         def ret():
#             results = self.impl(*[input.storage for input in self.inputs])
#             if self.nout == 1:
#                 self.out.set_value(results)
#             else:
#                 assert self.nout == len(results)
#                 for result, output in zip(results, self.outputs):
#                     output.set_value(result)
#         return ret
    
    def update_gradient(self, grad_d):
        inputgs = self.grad(*(self.inputs + [grad_d[output] for output in self.outputs]))
        if not isinstance(inputgs, (list, tuple)):
            inputgs = [inputgs] * len(self.inputs)
        for input, inputg in zip(self.inputs, inputgs):
            grad_d.add(input, inputg)

    def grad(*args):
        return UNDEFINED

#     def impl(*args):
#         raise NotImplementedError("This op has no implementation.")

    
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
    def grad(x, y, gz):
        return gz

class add(proto_add):
    impl = numpy.ndarray.__add__

class iadd(proto_add, inplace):
    impl = numpy.ndarray.__iadd__


class proto_twice(omega_op):
    def grad(x, gz):
        return scal(gz, 2.0)

class twice(proto_twice):
    def impl(x):
#        print x
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
        return scal(mul(x, gz), 2.0)

class sqr(proto_sqr):
    impl = lambda x: numpy.multiply(x, x)

class isqr(proto_sqr, inplace):
    impl = lambda x: x.__imul__(x),


class proto_sqrt(omega_op):
    def grad(x, gz):
        return scal(div(gz, sqrt(x)), 0.5)

class sqrt(proto_sqrt):
    impl = numpy.sqrt

class isqrt(proto_sqrt, inplace):
    impl = lambda x: x.__ipow__(0.5)


## Exponentiation ##

class exp(omega_op):
    impl = numpy.exp
    

# ## Element-wise division ##

# def div_grad(x, y, gz):
#     return div(gz, y), -div(mul(x, gz), sqr(y))

# div = wrapper("div",
#               numpy.ndarray.__div__,
#               div_grad)

# idiv = wrapper("idiv",
#                numpy.ndarray.__idiv__,
#                div_grad,
#                dmap = {0: 0})


# ## Scaling ##

# def scal_grad(x, a, gz):
#     return scal(a, gz), sum(mul(x, gz))

# scal = wrapper("scal",
#                numpy.ndarray.__mul__,
#                scal_grad)

# iscal = wrapper("iscal",
#                 numpy.ndarray.__imul__,
#                 scal_grad,
#                 dmap = {0: 0})

# neg = wrapper("neg",
#               numpy.ndarray.__neg__,
#               lambda x, gz: -gz)

# ineg = wrapper("ineg",
#                lambda x: x.__imul__(-1),
#                lambda x, gz: -gz,
#                dmap = {0: 0})


# ## Dot product ##

# dot = wrapper("dot",
#               numpy.dot,
#               lambda x, y, gz: (dot(gz, transpose(y)),
#                                 dot(transpose(x), gz)))


## Transposition ##

class transpose(omega_op, view):
    impl = numpy.transpose
    def grad(x, gz):
        return transpose_copy(gz)

# transpose = wrapper("transpose",
#                     numpy.transpose,
#                     lambda x, z, gz: transpose_copy(gz),
#                     vmap = {0: 0})

def transpose_copy(x):
    return array_copy(transpose(x))


## Copy ##

class array_copy(omega_op):
    impl = numpy.array
    grad = lambda x, gz: gz


## Others ##

class minmax(omega_op):
    nout = 2
    def impl(x):
        return x.min, x.max

    
# array_copy = wrapper("copy",
#                      numpy.array,
#                      lambda x, gz: gz)


# array slicing



