"""A `Type` and `Op` classes to work with numpy.ndarrays symbolically."""

__docformat__ = "restructuredtext en"

import sys # for sys.maxint
import inspect
import functools

import numpy

from copy import copy

from gof import Result, Op, utils, AbstractFunctionError, Type, Constant, Apply, Value
import gof

import blas # for gemm, dot
import gradient

import elemwise
import scalar as scal
from gof.python25 import partial


### set up the external interface
from elemwise import Elemwise, DimShuffle, CAReduce, Sum
import tensor_random as random


_constructor_list = []
"""List of functions to be listed as op constructors in the oplist (`gen_oplist`, doc/oplist.txt)."""
def constructor(f):
    """Make `f` appear as a constructor in the oplist (`gen_oplist`, doc/oplist.txt)."""
    _constructor_list.append(f)
    return f


def as_tensor(x, name = None):
    """Return `x`, transformed into a `Tensor`

    This function is often used by `make_node` methods of `Op` subclasses to
    turn ndarrays, numbers, `Scalar` instances, `Apply` instances and `Tensor`
    instances into valid input list elemnts.

    :Parameters:
     - `x`: Apply instance, Result instance, numpy.ndarray, or number
       This thing will be transformed into a `Result` in a sensible way.  An
       ndarray argument will not be copied, but a list of numbers will be copied
       to make an ndarray.
     - `name`: str or None
       If a new `Result` instance is created, it will be named with this string.

    :Exceptions:
     - `ValueError`: raised if an `Apply` with no default output is fetched
     - `TypeError`: raised if `x` cannot be converted to a Tensor Result

    """
    if isinstance(x, gof.Apply):
        #TODO: use Apply's default output mechanism
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a multi-output Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, Result):
        if isinstance(x.type, scal.Scalar):
            return tensor_from_scalar(x)
        if not isinstance(x.type, Tensor):
            raise TypeError("Result type field must be a Tensor.", x, x.type)
        return x
    try:
        return constant(x)
    except TypeError:
        raise TypeError("Cannot convert %s to Tensor" % x, type(x))

# this has a different name, because _as_tensor is the function which ops use
# to upcast their arguments... this internal-use function is a good place to put debugging stuff, better than the global astensor.
_as_tensor = as_tensor


def constant(x):
    """Return a symbolic `Constant` with value `x`
    
    :Exceptions:
     - `TypeError`: `x` could not be converted to a numpy.ndarray
    """
    if isinstance(x, numpy.ndarray):
        x_ = x
    else:
        x_ = numpy.asarray(x)
    try:
        return TensorConstant(Tensor(dtype = x_.dtype,
                                     broadcastable = [d == 1 for d in x_.shape]), x_)
    except:
        raise TypeError("Could not convert %s to Tensor" % x, type(x))

def value(x):
    """Return a symbolic `Value` with default value `x`
    
    :Exceptions:
     - `TypeError`: `x` could not be converted to a numpy.ndarray
    """
    if isinstance(x, numpy.ndarray):
        x_ = x
    else:
        x_ = numpy.asarray(x)
    try:
        return TensorValue(Tensor(dtype = x_.dtype,
                                  broadcastable = [d == 1 for d in x_.shape]), x_)
    except:
        raise TypeError("Could not convert %s to Tensor" % x, type(x))



class Tensor(Type):
    """Symbolic `Type` representing a numpy.ndarray value."""

    def __init__(self, dtype, broadcastable):
        """Initialize self.dtype and self.broadcastable.

        :Parameters:
         - `dtype`: str corresponding to numpy dtype (e.g., 'int64')
           The value (ndarray) associated to a `Result` of this `Type` will have
           this dtype.
         - `broadcastable`: tuple, list, or array of boolean values
           This argument serves two purposes.  First, the True elements of this
           list indicate the dimensions where the shape of an associated value
           must be 1.  Secondly, the length of this list is the number of
           dimensions that an associated value must have.  See
           :doc:`broadcasting` for an explanation of how this list is used.

        """
        self.dtype = str(dtype)
        self.broadcastable = tuple(broadcastable)
        self.dtype_specs() # error checking is done there
    
    def filter(self, data, strict = False):
        """Convert `data` to something which can be associated to a `TensorResult`.

        This function is not meant to be called in user code.  It is for
        `Linker` instances to use when running a compiled graph.
        """
        _data = data
        if strict:
            if not isinstance(data, numpy.ndarray):
                raise TypeError("%s expected a ndarray object.", data, type(data))
            if not str(data.dtype) == self.dtype:
                raise TypeError("%s expected a ndarray object with dtype = %s (got %s)." % (self, self.dtype, data.dtype))
            if not data.ndim == self.ndim:
                raise TypeError("%s expected a ndarray object with %s dimensions (got %s)." % (self, self.ndim, data.ndim))
            return data
        else:
            data = numpy.asarray(data, dtype = self.dtype)
        if not self.ndim == data.ndim:
            raise TypeError("Wrong number of dimensions: expected %s, got %s." % (self.ndim, data.ndim), _data)
        if any(b and d != 1 for d, b in zip(data.shape, self.broadcastable)):
            raise TypeError("Non-unit value on shape on a broadcastable dimension.", data.shape, self.broadcastable)
        return data

    def dtype_specs(self):
        """Return a tuple (python type, c type, numpy typenum) that corresponds to
        self.dtype.
        
        This function is used internally as part of C code generation.
        """
        #TODO: add more type correspondances for e.g. int32, int64, float32,
        #complex64, etc.
        try:
            return {'float32': (float, 'npy_float32', 'NPY_FLOAT32'),
                    'float64': (float, 'npy_float64', 'NPY_FLOAT64'),
                    'uint8': (int, 'npy_uint8', 'NPY_UINT8'),
                    'int8': (int, 'npy_int8', 'NPY_INT8'),
                    'uint16': (int, 'npy_uint16', 'NPY_UINT16'),
                    'int16': (int, 'npy_int16', 'NPY_INT16'),
                    'uint32': (int, 'npy_uint32', 'NPY_UINT32'),
                    'int32': (int, 'npy_int32', 'NPY_INT32'),
                    'uint64': (int, 'npy_uint64', 'NPY_UINT64'),
                    'int64': (int, 'npy_int64', 'NPY_INT64'),
                    'complex128': (complex, 'theano_complex128', 'NPY_COMPLEX128'),
                    'complex64': (complex, 'theano_complex64', 'NPY_COMPLEX64')}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (self.__class__.__name__, self.dtype))

    def __eq__(self, other):
        """Compare True iff other is the same kind of Tensor"""
        return type(self) == type(other) and other.dtype == self.dtype and other.broadcastable == self.broadcastable

    def __hash__(self):
        """Hash equal for same kinds of Tensor"""
        return hash(self.dtype) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable), doc = "number of dimensions")
    """Number of dimensions

    This read-only property is the preferred way to get the number of dimensions
    of a `Tensor`.
    
    """

    def make_result(self, name = None):
        """Return a `TensorResult` of this type

        :Parameters:
         - `name`: str
           A pretty name to identify this `Result` when printing and debugging

       """
        return TensorResult(self, name = name)

    def __str__(self):
        return "%s(%s)" % (str(self.dtype), str(self.broadcastable))

    def __repr__(self):
        return "Tensor{%s, %s}" % (str(self.dtype), str(self.broadcastable))

    def c_declare(self, name, sub):
        """Override `CLinkerOp.c_declare` """
        return """
        PyArrayObject* %(name)s;
        int type_num_%(name)s;
        typedef %(dtype)s dtype_%(name)s;
        """ % dict(sub, name = name, dtype = self.dtype_specs()[1])

    def c_init(self, name, sub):
        """Override `CLinkerOp.c_init` """
        return """
        %(name)s = NULL;
        type_num_%(name)s = %(type_num)s;
        """ % dict(sub, name = name, type_num = self.dtype_specs()[2])

    def c_extract(self, name, sub):
        """Override `CLinkerOp.c_extract` """
        return """
        %(name)s = NULL;
        type_num_%(name)s = %(type_num)s;
        if (py_%(name)s == Py_None) {
            // We can either fail here or set %(name)s to NULL and rely on Ops using
            // tensors to handle the NULL case, but if they fail to do so they'll end up
            // with nasty segfaults, so this is public service.
            PyErr_SetString(PyExc_ValueError, "expected an ndarray, not None");
            %(fail)s
        }
        else if (!PyArray_Check(py_%(name)s)) {
            PyErr_SetString(PyExc_ValueError, "expected an ndarray");
            %(fail)s
        }
        else if (((PyArrayObject*)py_%(name)s)->descr->type_num != %(type_num)s) {
            PyErr_SetString(PyExc_ValueError, "expected %(type_num)s");
            %(fail)s
        }
        else {
            %(name)s = (PyArrayObject*)(py_%(name)s);
            Py_XINCREF(%(name)s);
        }
        """ % dict(sub, name = name, type_num = self.dtype_specs()[2])

    def c_cleanup(self, name, sub):
        """Override `CLinkerOp.c_cleanup` """
        return """
        if (%(name)s) {
            Py_XDECREF(%(name)s);
        }
        """ % locals()

    def c_sync(self, name, sub):
        """Override `CLinkerOp.c_sync` """
        return """
        Py_XDECREF(py_%(name)s);
        if (!%(name)s) {
            py_%(name)s = Py_None;
        }
        else if ((void*)py_%(name)s != (void*)%(name)s) {
            py_%(name)s = (PyObject*)%(name)s;
        }
        Py_XINCREF(py_%(name)s);
        """ % locals()

    def c_headers(self):
        """Override `CLinkerOp.c_headers` """
        return []

    def c_libraries(self):
        return []

    def c_support_code(cls):
        """Override `CLinkerOp.c_support_code` """
        template = """
        struct theano_complex%(nbits)s : public npy_complex%(nbits)s
        {
            typedef theano_complex%(nbits)s complex_type;
            typedef npy_float%(half_nbits)s scalar_type;

            complex_type operator +(complex_type y) {
                complex_type ret;
                ret.real = this->real + y.real;
                ret.imag = this->imag + y.imag;
                return ret;
            }
            complex_type operator -(complex_type y) {
                complex_type ret;
                ret.real = this->real - y.real;
                ret.imag = this->imag - y.imag;
                return ret;
            }
            complex_type operator *(complex_type y) {
                complex_type ret;
                ret.real = this->real * y.real - this->imag * y.imag;
                ret.imag = this->real * y.imag + this->imag * y.real;
                return ret;
            }
            complex_type operator /(complex_type y) {
                complex_type ret;
                scalar_type y_norm_square = y.real * y.real + y.imag * y.imag;
                ret.real = (this->real * y.real + this->imag * y.imag) / y_norm_square;
                ret.imag = (this->imag * y.real - this->real * y.imag) / y_norm_square;
                return ret;
            }
        };
        """
        return template % dict(nbits = 64, half_nbits = 32) + template % dict(nbits = 128, half_nbits = 64)
        # todo: use C templating


# Easy constructors

def tensor(*args, **kwargs):
    return Tensor(*args, **kwargs).make_result()

def _multi(*fns):
    def f2(f, names):
        if isinstance(names, int):
            if names == 1:
                return f()
            else:
                return [f() for i in xrange(names)]
        if len(names) == 1:
            return f(names)
        else:
            return [f(name) for name in names]
    if len(fns) == 1:
        return partial(f2, fns)
    else:
        return [partial(f2, f) for f in fns]

fscalar = Tensor('float32', ())
dscalar = Tensor('float64', ())
bscalar = Tensor('int8', ())
iscalar = Tensor('int32', ())
lscalar = Tensor('int64', ())
def scalar(name = None, dtype = 'float64'):
    type = Tensor(dtype, ())
    return type(name)
scalars, fscalars, dscalars, iscalars, lscalars = _multi(scalar, fscalar, dscalar, iscalar, lscalar)

fvector = Tensor('float32', (False, ))
dvector = Tensor('float64', (False, ))
bvector = Tensor('int8', (False,))
ivector = Tensor('int32', (False, ))
lvector = Tensor('int64', (False, ))
def vector(name = None, dtype = 'float64'):
    type = Tensor(dtype, (False, ))
    return type(name)
vectors, fvectors, dvectors, ivectors, lvectors = _multi(vector, fvector, dvector, ivector, lvector)

fmatrix = Tensor('float32', (False, False))
dmatrix = Tensor('float64', (False, False))
bmatrix = Tensor('int8', (False, False))
imatrix = Tensor('int32', (False, False))
lmatrix = Tensor('int64', (False, False))
def matrix(name = None, dtype = 'float64'):
    type = Tensor(dtype, (False, False))
    return type(name)
matrices, fmatrices, dmatrices, imatrices, lmatrices = _multi(matrix, fmatrix, dmatrix, imatrix, lmatrix)

frow = Tensor('float32', (True, False))
drow = Tensor('float64', (True, False))
brow = Tensor('int8', (True, False))
irow = Tensor('int32', (True, False))
lrow = Tensor('int64', (True, False))
def row(name = None, dtype = 'float64'):
    type = Tensor(dtype, (True, False))
    return type(name)
rows, frows, drows, irows, lrows = _multi(row, frow, drow, irow, lrow)

fcol = Tensor('float32', (False, True))
dcol = Tensor('float64', (False, True))
bcol = Tensor('int8', (False, True))
icol = Tensor('int32', (False, True))
lcol = Tensor('int64', (False, True))
def col(name = None, dtype = 'float64'):
    type = Tensor(dtype, (False, True))
    return type(name)
cols, fcols, dcols, icols, lcols = _multi(col, fcol, dcol, icol, lcol)


class _tensor_py_operators:
    #UNARY
    def __abs__(self): return _abs(self)
    def __neg__(self): return neg(self)

    #CASTS
    def __int__(self): return AsInt(self).out
    def __float__(self): return AsInt(self).out
    def __complex__(self): return AsComplex(self).out

    #COMPARISONS
    def __lt__(self,other): return lt(self, other)
    def __le__(self,other): return le(self, other)
    def __gt__(self,other): return gt(self, other)
    def __ge__(self,other): return ge(self, other)

    #BITWISE
    def __invert__(self): return invert(self) 
    def __and__(self,other): return and_(self, other)
    def __or__(self,other): return or_(self, other)
    def __xor__(self,other): return xor(self, other)
    def __rand__(self,other): return and_(other,self)
    def __ror__(self,other): return or_(other, self)
    def __rxor__(self,other): return xor(other, self)
    def __iand__(self, other): return _and_inplace(self, other)
    def __ior__(self, other): return _or_inplace(self, other)
    def __ixor__(self, other): return _xor_inplace(self, other)

    #ARITHMETIC - NORMAL
    def __add__(self,other): return add(self,other)
    def __sub__(self,other): return sub(self,other)
    def __mul__(self,other): return mul(self,other)
    def __div__(self,other): return div(self,other)
    def __pow__(self,other): return pow(self,other)
    def __mod__(self,other): return mod(self,other)

    #ARITHMETIC - INPLACE
    def __iadd__(self,other): return _add_inplace(self,other)
    def __isub__(self,other): return _sub_inplace(self,other)
    def __imul__(self,other): return _mul_inplace(self,other)
    def __idiv__(self,other): return _div_inplace(self,other)
    def __ipow__(self,other): return _pow_inplace(self,other)

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div(other,self)
    def __rmod__(self,other): return mod(other,self)
    def __rpow__(self,other): return pow(other,self)

    #TRANSPOSE
    T = property(lambda self: transpose(self))

    #SLICING
#     def __getitem__(self, args): return Subtensor.from_idxs(self,
#             args).outputs[0]
#     def __getslice__(self, *args): return Subtensor.from_idxs(self,
#             (slice(*args),)).outputs[0]
    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = args,
        return Subtensor(args)(self, *Subtensor.collapse(args, lambda entry: isinstance(entry, Result)))
    def __getslice__(self, *args):
        args = slice(*args),
        return Subtensor(args)(self, *Subtensor.collapse(args, lambda entry: isinstance(entry, Result)))
    
    #COPYING
    def copy(self): return tensor_copy(self)

    def __iter__(self): 
        # This prevents accidental iteration via builtin.sum(self)
        raise TypeError('Tensor does not support iteration. '
        'Maybe you are using builtin.sum instead of theano.tensor.sum? (Maybe .max?)')
        
    

class TensorResult(Result, _tensor_py_operators):
    pass

class TensorConstant(Constant, _tensor_py_operators):
    pass

class TensorValue(Value, _tensor_py_operators):
    pass

#QUESTION: why are we doing this!?
elemwise.as_tensor = as_tensor    
elemwise.Tensor = Tensor
elemwise.TensorResult = TensorResult
elemwise.TensorConstant = TensorConstant
elemwise.TensorValue = TensorValue



#########################
# Utilities
#########################

def _elemwise(scalar_op, name, doc_prefix=''):
    straight = elemwise.Elemwise(scalar_op, name = name)
    inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
    inplace = elemwise.Elemwise(inplace_scalar_op, {0: 0}, name = '_'+name+"_inplace")

    # don't add the inplace versions, they aren't supposed to be part of the user interface
    _constructor_list.append(straight) 
    
    # This is here so that gen_oplist can detect which module declared these variables.

    straight.__module__ = 'tensor'
    inplace.__module__ = 'tensor'

    if doc_prefix:
        straight.__doc__ = doc_prefix + '\n' + straight.__doc__

    return straight, inplace

def _epydoc_cheat(real_symbol_value):
    """Replace the value associated with a function symbol"""
    def decorator(f):
        return real_symbol_value
    return decorator

def _scal_elemwise(symbol):
    """Replace a symbol definition with an elementwise version of the corresponding scalar Op"""
    symbolname = symbol.__name__
    inplace = symbolname.endswith('_inplace')

    if inplace:
        scalar_op = getattr(scal, symbolname[1:-len('_inplace')])
        inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
        rval = elemwise.Elemwise(inplace_scalar_op, {0: 0}, name=symbolname)
    else:
        scalar_op = getattr(scal, symbolname)
        rval = elemwise.Elemwise(scalar_op, name=symbolname)

    if getattr(symbol, '__doc__', False):
        rval.__doc__ = symbol.__doc__ + '\n' + rval.__doc__

    #for the meaning of this see the ./epydoc script
    # it makes epydoc display rval as if it were a function, not an object
    rval.__epydoc_asRoutine = True

    return rval


#########################
# Casting Operations
#########################

class TensorFromScalar(Op):
    def make_node(self, s):
        assert isinstance(s.type, scal.Scalar)
        return Apply(self,
                     [s],
                     [tensor(dtype = s.type.dtype,
                             broadcastable = ())])
    def perform(self, node, (s, ), (out, )):
        out[0] = numpy.asarray(s)
    def grad(self, (s,), (dt,)):
        return [ScalarFromTensor(dt)]
tensor_from_scalar = TensorFromScalar()

class ScalarFromTensor(Op):
    def make_node(self, t):
        assert isinstance(t.type, Tensor)
        assert t.type.broadcastable == ()
        return Apply(self,
                     [t],
                     [scal.Scalar(dtype = t.type.dtype).make_result()])
    def perform(self, node, (s, ), (out, )):
        out[0] = s.flatten()[0]
    def grad(self, (s,), (dt,)):
        return [TensorFromScalar(dt)]
scalar_from_tensor = ScalarFromTensor()


@constructor
def cast(t, dtype):
    mapping = {'int8': convert_to_int8,
               'int16': convert_to_int16,
               'int32': convert_to_int32,
               'int64': convert_to_int64,
               'float32': convert_to_float32,
               'float64': convert_to_float64,
               'complex64': convert_to_complex64,
               'complex128': convert_to_complex128}
    return mapping[dtype](t)

def _conversion(f):
    f.__module__ = 'tensor'
    return f

convert_to_int8  = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int8))))
"""Cast to 8-bit integer"""

convert_to_int16 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int16))))
"""Cast to 16-bit integer"""

convert_to_int32 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int32))))
"""Cast to 32-bit integer"""

convert_to_int64 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int64))))
"""Cast to 64-bit integer"""

convert_to_float32 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.float32))))
"""Cast to single-precision floating point"""

convert_to_float64 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.float64))))
"""Cast to double-precision floating point"""

convert_to_complex64  = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.complex64))))
"""Cast to single-precision complex"""

convert_to_complex128 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.complex128))))
"""Cast to double-precision complex"""



##########################
# Unary Operations
##########################

class Shape(Op):
    """
    L{Op} to return the shape of a matrix.

    @note: Non-differentiable.
    """
    def make_node(self, x):
        x = as_tensor(x)
        return Apply(self, [x], [lvector()])
    def perform(self, node, (x, ), (out, )):
        out[0] = numpy.asarray(x.shape)
    def grad(self, (x,), (gz,)):
        return [None]
shape = Shape()

class MaxAndArgmax(Op):
    """Calculate the max and argmax over a given axis"""
    nin=2 # tensor, axis
    nout=2 # max val, max idx
    E_axis = 'invalid axis'
    
    def make_node(self, x, axis=None):
        x = _as_tensor(x)
        if axis is None:
            axis = x.type.ndim - 1
        axis = _as_tensor(axis)
        inputs = [x, axis]
        broadcastable = [False] * (x.type.ndim - 1)
        outputs = [tensor(x.type.dtype, broadcastable), 
                   tensor(axis.type.dtype, broadcastable)]
        return Apply(self, inputs, outputs)
    def perform(self, node, (x, axis), (max, max_idx)):
        max[0] = numpy.max(x, axis)
        max_idx[0] = numpy.argmax(x, axis)
    def grad(self, (x, axis), (g_max, g_max_idx)):
        # @warning: This only works if axis is 0, else the max is
        # broadcasted wrong in the call to eq.
        # @note: This function should work correctly for L{vector}s.
#        (x, y), (gz, gw)
#        gz*dz/dx + gw*dw/dx, gz*dz/dy + gw*dw/dy
#        gMax * dMax/dx + gArgMax * dArgMax/dx, gMax * dMax/daxis + gArgMax * dArgMax/daxis
#       g_max has one less dimension than x, so you need to complete g_max to x's shape
#        when axis=0 the broadcasting mechanism does it automatically
        assert axis.data == 0
        g_x = eq(max(x, axis), x) * g_max
        return g_x, None
max_and_argmax = MaxAndArgmax()



@constructor
def max(x, axis=None):
    """Return indexes of maximum elements obtained by iterating over given axis

    Default axis is the last one.
    """
    # In python (using MaxAndArgmax.perform()) this leads to an wasteful
    # implementation that goes through the data twice instead of once
    # but when Argmax.c_impl() is in place, it should be fine.
    return max_and_argmax(x,axis)[0]

@constructor
def argmax(x, axis=None):
    """Return maximum elements obtained by iterating over given axis

    Default axis is the last one.
    """
    # In python (using MaxAndArgmax.perform()) this leads to an wasteful
    # implementation that goes through the data twice instead of once
    # but when Argmax.c_impl() is in place, it should be fine.
    return max_and_argmax(x,axis)[1]


##########################
# Comparison
##########################

def _elemwise_macro(scalar_op, *args):
    straight = elemwise.Elemwise(scalar_op)
    return straight(*args)

def _elemwise_macro_inplace(scalar_op, *args):
    #construct an inplace version of the scalar op
    inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
    inplace = elemwise.Elemwise(inplace_scalar_op, {0: 0})
    return inplace(*args)

@_scal_elemwise
def lt(a, b):
    """a < b"""
    return _elemwise_macro(scal.lt, a, b)

@_scal_elemwise
def _lt_inplace(a,b):
    """a < b (inplace on a)"""
    return _elemwise_macro_inplace(scal.lt, a, b)

@_scal_elemwise
def gt(a, b):
    """a > b"""

@_scal_elemwise
def _gt_inplace(a,b):
    """a > b (inplace on a)"""

@_scal_elemwise
def le(a, b):
    """a <= b"""

@_scal_elemwise
def _le_inplace(a,b):
    """a <= b (inplace on a)"""

@_scal_elemwise
def ge(a, b):
    """a >= b"""

@_scal_elemwise
def _ge_inplace(a,b):
    """a >= b (inplace on a)"""

@_scal_elemwise
def eq(a, b):
    """a == b"""

@_scal_elemwise
def _eq_inplace(a,b):
    """a == b (inplace on a)"""

@_scal_elemwise
def neq(a, b):
    """a != b"""

@_scal_elemwise
def _neq_inplace(a,b):
    """a != b (inplace on a)"""


##########################
# Bit-wise
##########################

@_scal_elemwise
def and_(a,b):
    """bitwise a & b"""

@_scal_elemwise
def _and__inplace(a,b):
    """bitwise a & b (inplace on a)"""

@_scal_elemwise
def or_(a,b):
    """bitwise a | b"""

@_scal_elemwise
def _or__inplace(a,b):
    """bitwise a | b (inplace on a)"""

@_scal_elemwise
def xor(a,b):
    """bitwise a ^ b"""

@_scal_elemwise
def _xor_inplace(a,b):
    """bitwise a ^ b (inplace on a)"""

@_scal_elemwise
def invert(a):
    """bitwise ~a"""

@_scal_elemwise
def _invert_inplace(a):
    """bitwise ~a (inplace on a)"""

##########################
# Math
##########################

@_scal_elemwise
def _abs(*a):
    """|a|

    _abs has a leading underscore because abs() is a builtin.  TensorResult overloads the
    __abs__ operator so that this function is called when you type abs(a).

    """

@_scal_elemwise
def __abs_inplace(a):
    """|a| (inplace on a)"""

exp, _exp_inplace = _elemwise(scal.exp, 'exp',
    """exponential (elemwise)""")

neg, _neg_inplace = _elemwise(scal.neg, 'neg',
    """negative (elemwise)""")

inv, _inv_inplace = _elemwise(scal.inv, 'inv',
    """multiplicative inverse (elemwise)""")

log, _log_inplace = _elemwise(scal.log, 'log',
    """logarithm base-e (elemwise)""")

log2, _log2_inplace = _elemwise(scal.log2, 'log2',
    """logarithm base-2 (elemwise)""")

sgn, _sgn_inplace = _elemwise(scal.sgn, 'sgn',
    """sign (elemwise)""")

sqr, _sqr_inplace = _elemwise(scal.sqr, 'sqr',
    """square (elemwise)""")

sqrt, _sqrt_inplace = _elemwise(scal.sqrt, 'sqrt',
    """square root (elemwise)""")

cos, _cos_inplace = _elemwise(scal.cos, 'cos',
    """cosine (elemwise)""")

sin, _sin_inplace = _elemwise(scal.sin, 'sin',
    """sine (elemwise)""")

tan, _tan_inplace = _elemwise(scal.tan, 'tan',
    """tan = sin/cos (elemwise)""")

cosh, _cosh_inplace = _elemwise(scal.cosh, 'cosh',
    """hyperbolic cosine (elemwise)""")

sinh, _sinh_inplace = _elemwise(scal.sinh, 'sinh',
    """hyperbolic sine (elemwise)""")

tanh, _tanh_inplace = _elemwise(scal.tanh, 'tanh',
    """hyperbolic tangent (elemwise)""")


##########################
# Misc
##########################

fill, _fill_inplace = _elemwise(scal.second, 'fill',
    """fill WRITEME (elemwise)""")

@constructor
def ones_like(model):
    """WRITEME"""
    #return Ones(model.type.ndim)(shape(model))
    return fill(model, 1.0)

@constructor
def zeros_like(model):
    """WRITEME"""
    #return Zeros(model.type.ndim)(shape(model))
    return fill(model, 0.0)

class Filler(gof.Op):
    """WRITEME"""
    def __init__(self, value, ndim, dtype = 'float64'):
        self.value = value
        self.ndim = ndim
        self.dtype = dtype
        self.type = Tensor(dtype = dtype,
                           broadcastable = (False,)*ndim)

    def make_node(self, dims):
        dims = as_tensor(dims)
        return gof.Apply(self, [dims], [self.type()])

    def perform(self, node, (dims,), (out,)):
        if out[0] is not None:
            out[0].resize(dims, refcheck = 0)
            out[0].fill(self.value)
        else:
            if self.value == 0:
                out[0] = numpy.zeros(dims, dtype = self.dtype)
            elif self.value == 1:
                out[0] = numpy.ones(dims, dtype = self.dtype)
            else:
                out[0] = numpy.ones(dims, dtype = self.dtype) * self.value

    def grad(self, (dims,), (gout,)):
        return None,

    def __eq__(self, other):
        return type(self) == type(other) and self.ndim == other.ndim and self.dtype == other.dtype

    def __hash__(self):
        return hash(self.ndim) ^ hash(self.dtype)

Zeros = functools.partial(Filler, 0)
"""WRITEME"""

Ones = functools.partial(Filler, 1)
"""WRITEME"""

@constructor
def zero():
    """
    Return a scalar zero, e.g. for initializing sums.
    """
    return Zeros(0)([])

@constructor
def one():
    """WRITEME"""
    return Ones(0)([])


tensor_copy = elemwise.Elemwise(scal.identity)
"""Copy a tensor"""

identity = elemwise.Elemwise(scal.identity, inplace_pattern = {0: [0]})
"""Do nothing (elemwise)"""

@constructor
def sum(input, axis = None):
    """WRITEME"""
    return elemwise.Sum(axis)(input)

@constructor
def mean(input, axis = None):
    """WRITEME"""
    s = sum(input, axis)
    shp = shape(input)
    if axis is None:
        axis = range(input.type.ndim)
    elif isinstance(axis, int):
        axis = [axis]
    for i in axis:
        s = s / shp[i]
    return s


class Repeat(gof.Op):

    def make_node(self, input, repeats, axis):
        assert isinstance(input.type, Tensor)
        assert repeats.type == iscalar
        assert axis.type == iscalar
        type = Tensor(dtype = input.type.dtype,
                      broadcastable = [False if i==axis else x for i, x in enumerate(input.broadcastable)])
        return gof.Apply(self, [inputs, repeats, axis], [type()])

    def perform(self, node, (input, repeats, axis), (out, )):
        out[0] = numpy.repeat(input, repeats, axis)

    def grad(self, (input, repeats, axis), (gout, )):
        return add.grad((input, gout), (gout,))[:1]

repeat = Repeat()



##########################
# Arithmetics
##########################

add, _add_inplace = _elemwise(scal.add, 'add', 
    """addition (element-wise)""")

sub, _sub_inplace = _elemwise(scal.sub, 'sub',
    """subtraction (elemwise)""")

mul, _mul_inplace = _elemwise(scal.mul, 'mul',
    """multiplication (elemwise)""")

div, _div_inplace = _elemwise(scal.div, 'div',
    """division (elemwise)""")

mod, _mod_inplace = _elemwise(scal.mod, 'mod',
    """modulo (elemwise)""")

pow, _pow_inplace = _elemwise(scal.pow, 'pow',
    """raise to given exponent (elemwise)""")


##########################
# View Operations
##########################

class TransposeInplace(Op):
    view_map = {0: [0]}
    
    def make_node(self, input):
        return Apply(self, [input], [tensor(dtype = input.type.dtype,
                                            broadcastable = reversed(input.type.broadcastable))])
    
    def perform(self, node, (x, ), (z, )):
        z[0] = x.T
    
    def grad(self, (x,), (gz,)):
        return transpose(gz),
    
    def c_code(self, node, name, (x, ), (z, ), sub):
        return """
        PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
        if (%(z)s) {
            Py_XDECREF(%(z)s);
        }
        %(z)s = transposed;
        """ % locals()

    def __str__(self):
        return "TransposeView"

_transpose_inplace = TransposeInplace()
"""WRITEME"""

def transpose(x, **kwargs):
    """WRITEME"""
    return _transpose_inplace(tensor_copy(x), **kwargs)




class Subtensor(Op):
    """Return a subtensor view

    This class uses a relatively complex internal representation of the inputs
    to remember how the input tensor x should be sliced.  The instance variable
    idxlist is a list whose elements are either integers, or slices.  The
    integers are indexes into the inputs array, and the start/stop/step members
    of each slice are also integer indexes into the inputs array (or None).  The
    inputs array is the tensor x, followed by scalar integer results.
    
    @todo: add support for advanced tensor indexing (in Subtensor_dx too).
    """
    e_invalid = 'The index list is longer than the number of dimensions of the tensor.'
    e_subslice = 'nested slicing is not supported'
    e_indextype = "Invalid index type or slice for Subtensor"
    debug = 0

    view_map = {0: [0]}

    @staticmethod
    def collapse(idxs, cond):
        ret = []
        def helper(entry):
            if cond(entry):
                ret.append(entry)
            elif isinstance(entry, slice):
                helper(entry.start)
                helper(entry.stop)
                helper(entry.step)
        for idx in idxs:
            helper(idx)
        return ret

    def __init__(self, idx_list):
        def convert(entry, slice_ok=True):
            scal_types =[scal.int64, scal.int32, scal.int16, scal.int8]
            tensor_types = [bscalar, iscalar, lscalar]
            if isinstance(entry, gof.Result) and entry.type in scal_types:
                return entry.type
            elif isinstance(entry, gof.Type) and entry in scal_types:
                return entry
            if isinstance(entry, gof.Result) and entry.type in tensor_types:
                return scal.Scalar(entry.type.dtype)
            elif isinstance(entry, gof.Type) and entry in tensor_types:
                return scal.Scalar(entry.dtype)
            elif slice_ok and isinstance(entry, slice):
                a = entry.start
                b = entry.stop
                c = entry.step
                return slice(convert(a, False) if a is not None else None,
                             convert(b, False) if b is not None else None,
                             convert(c, False) if c is not None else None)
            elif isinstance(entry, int):
                return entry
            else:
                raise TypeError(Subtensor.e_indextype, entry)
        self.idx_list = map(convert, idx_list)

    def make_node(self, x, *inputs):
        x = as_tensor(x)
        def my_as_scalar(a):
            if isinstance(a, gof.Result) and isinstance(a.type, Tensor):
                return scalar_from_tensor(a)
            else:
                return scal.as_scalar(a)
        inputs = tuple(my_as_scalar(a) for a in inputs)
        
        idx_list = list(self.idx_list)
        if len(idx_list) > x.type.ndim:
            raise ValueError(Subtensor.e_invalid,
                             (len(idx_list), x.type.ndim))

        #infer the broadcasting pattern
        padded = idx_list + [slice(0,sys.maxint,1)] * (x.type.ndim - len(idx_list))
        broadcastable = [bc for p, bc in zip(padded, x.type.broadcastable) if isinstance(p, slice)]

        input_types = Subtensor.collapse(idx_list, lambda entry: isinstance(entry, gof.Type))
        if len(inputs) != len(input_types):
            raise IndexError("Not enough inputs to fill in the Subtensor template.", inputs, idx_list)
        for input, expected_type in zip(inputs, input_types):
            if input.type != expected_type:
                raise TypeError("Wrong type for the Subtensor template. Expected %s, got %s." % (input.type, expected_type))

        return gof.Apply(self,
                         (x, ) + inputs,
                         [tensor(dtype = x.type.dtype,
                                 broadcastable = broadcastable)])

    def perform(self, node, inputs, (out, )):
        x = inputs[0]
        indices = list(reversed(inputs[1:]))

        def convert(entry):
            if isinstance(entry, gof.Type):
                return indices.pop()
            elif isinstance(entry, slice):
                return slice(convert(entry.start),
                             convert(entry.stop),
                             convert(entry.step))
            else:
                return entry

        cdata = tuple(map(convert, self.idx_list))
        if len(cdata) == 1:
            cdata = cdata[0]
        out[0] = numpy.asarray(x.__getitem__(cdata))

    def grad(self, inputs, (gz,)):
        x = inputs[0]
        rest = inputs[1:]
        return [SetSubtensor(self.idx_list)(zeros_like(x), gz, *rest)] + [None] * len(rest)

    def __eq__(self, other):
        return type(self) == type(other) and self.idx_list == other.idx_list

    def __hash__(self):
        # FIXME: this doesn't work if there are slices in the list because for some mysterious reason slice is unhashable
        return hash(tuple(self.idx_list))

    def __str__(self):
        indices = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                indices.append(":".join("" if x is None else str(x) for x in [entry.start, entry.stop, entry.step]))
            else:
                indices.append(str(entry))
        return "%s{%s}" % (self.__class__.__name__, ", ".join(indices))


class SetSubtensor(Subtensor):
    """WRITEME"""
    view_map = {}
    destroy_map = {0: [0]}

    def make_node(self, x, y, *inputs):
        x, y = map(as_tensor, [x, y])
        inputs = tuple(map(scal.as_scalar, inputs))
        
        idx_list = list(self.idx_list)
        if len(idx_list) > x.type.ndim:
            raise ValueError(Subtensor.e_invalid,
                             (len(idx_list), x.type.ndim))

        #infer the broadcasting pattern
        padded = idx_list + [slice(0,sys.maxint,1)] * (x.type.ndim - len(idx_list))
        broadcastable = [bc for p, bc in zip(padded, x.type.broadcastable) if isinstance(p, slice)]

        if y.type.broadcastable != tuple(broadcastable):
            raise TypeError("Invalid broadcastable pattern for y in SetSubtensor.make_node")

        input_types = Subtensor.collapse(idx_list, lambda entry: isinstance(entry, gof.Type))
        if len(inputs) != len(input_types):
            raise IndexError("Not enough inputs to fill in the Subtensor template.", inputs, idx_list)
        for input, expected_type in zip(inputs, input_types):
            if input.type != expected_type:
                raise TypeError("Wrong type for the Subtensor template. Expected %s, got %s." % (input.type, expected_type))

        return gof.Apply(self,
                         (x, y) + inputs,
                         [x.type()])

    def perform(self, node, inputs, (out, )):
        x, y = inputs[:2]
        indices = list(reversed(inputs[2:]))

        def convert(entry):
            if isinstance(entry, gof.Type):
                return indices.pop()
            elif isinstance(entry, slice):
                return slice(convert(entry.start),
                             convert(entry.stop),
                             convert(entry.step))
            else:
                return entry

        cdata = tuple(map(convert, self.idx_list))
        if len(cdata) == 1:
            cdata = cdata[0]
        x.__setitem__(cdata, y)
        out[0] = x


class MakeVector(Op):
    """WRITEME"""
    def __init__(self, stype):
        self.stype = stype
    def make_node(self, *inputs):
        assert all(a.type == self.stype for a in inputs)
        return Apply(self, inputs, [Tensor(broadcastable = (False,),
                                           dtype = self.stype.dtype)()])
    def perform(self, inputs, (out,)):
        return numpy.asarray([i[0] for i in inputs])
    def grad(self, inputs, (gout,)):
        return [None]*len(inputs)

make_lvector = MakeVector(lscalar)
"""WRITEME"""


class VerticalStack(Op):
    """
    Vertically stack two L{Tensor}s.
    Stack two L{Tensor}s along the first axis (row wise). These
    L{Tensor}s must have the same shape along all dimensions but the
    first.

    @attention: Because we use vstack as the implementation, if the
    inputs have 1-dimension, the output will have 2-dimensions.
    """
    def make_node(self, x, y):
        x = as_tensor(x)
        y = as_tensor(y)
        assert x.type.dtype == y.type.dtype
        if x.type.broadcastable[1:] != y.type.broadcastable[1:]:
            raise NotImplementedError
        inputs = [x, y]
        bcastable = (False, ) + x.type.broadcastable[1:]
        outputs = [tensor(dtype = x.type.dtype,
                          broadcastable = bcastable)]
        return Apply(self, inputs, outputs)
    def perform(self, node, (x, y), (out, )):
        assert x.ndim == y.ndim
        # Make sure every dimension (save the first) is the same
        for i in range(x.ndim): assert i == 0 or x.shape[i] == y.shape[i]
        out[0] = numpy.vstack([x, y])
    def grad(self, (x, y), (gz,)):
        """
        @todo: Make VSplit (or this grad implementation) its own L{Op},
        that way we can do more sanity-checking::
            assert x.ndim == y.ndim
            # Make sure every dimension (save the first) is the same
            for i in range(x.data.ndim): assert i == 0 or x.data.shape[i] == y.shape[i]
            etc...
        """
        xs = shape(x)
        ys = shape(y)
        return gz[:xs[0]], gz[xs[0]:]
vertical_stack = VerticalStack()

def horizontal_stack(x, y):
    """
    Horizontally stack two L{Tensor}s.
    Stack two L{Tensor}s along the second axis (column wise). These
    L{Tensor}s must have the same shape along all dimensions but the
    second.

    @note: Unlike VerticalStack, we assume that the L{Tensor}s have
    two dimensions.
    """
    assert x.type.ndim == 2
    assert y.type.ndim == 2
    return transpose(vertical_stack(x.T, y.T))


#########################
# Linalg : Dot
#########################

class Dot(Op):
    """Compute matrix-matrix, matrix-vector products and vector inner-products.

    """
    def make_node(self, *inputs):
        inputs = map(as_tensor, inputs)

        numpy_semantics = 0
        if numpy_semantics:
            #numpy defines dot for tensor pairs with any rank
            if len(inputs) != 2:
                raise TypeError("Wrong number of inputs for %s (got %i, expected 2)" % self)
            i_broadcastables = [input.type.broadcastable for input in inputs]
            bx, by = i_broadcastables
            if len(bx) == 0:     # x is a scalar
                bz = by
            else:
                if len(by) >= 2: #y is a matrix or tensor
                    bz = bx[:-1] + by[:-2] + by[-1:]
                elif len(by)==1: #y is vector
                    bz = bx[:-1]
                else:            #y is a scalar
                    bz = bx
        else:
            x, y = inputs
            nx = x.type.ndim
            ny = y.type.ndim

            if nx not in (1,2): raise TypeError('not matrix or vector', x)
            if ny not in (1,2): raise TypeError('not matrix or vector', y)

            if nx == 2 and ny == 2:
                bz = [x.type.broadcastable[0], y.type.broadcastable[1]]
            elif nx == 1 and ny == 2:
                bz = [y.type.broadcastable[1]]
            elif nx == 2 and ny == 1:
                bz = [x.type.broadcastable[0]]
            else:
                bz = []

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [tensor(scal.upcast(*i_dtypes), bz)]
        return Apply(self, inputs, outputs)

    def perform(self, node, (x, y), (z, )):
        z[0] = numpy.asarray(numpy.dot(x, y))
    def grad(self, (x, y), (gz,)):
        if gz.type.ndim == 0:
            return gz * y, gz * x
        if x.type.ndim == 1 and y.type.ndim > 1:
            return dot(gz, y.T), outer(x.T, gz)
        if x.type.ndim > 1 and y.type.ndim == 1:
            return outer(gz, y.T), dot(x.T, gz)
        return dot(gz, y.T), dot(x.T, gz)
    def __str__(self):
        return "dot"
dot = Dot()

class Outer(Op):
    """ Compute vector-vector outer product
    """
    def make_node(self, *inputs):
        inputs = map(as_tensor, inputs)

        x, y = inputs
        nx = x.type.ndim
        ny = y.type.ndim

        if nx != 1: raise TypeError('not vector', x)
        if ny != 1: raise TypeError('not vector', y)
        
        bz = [x.type.broadcastable[0], y.type.broadcastable[0]]

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [tensor(scal.upcast(*i_dtypes), bz)]
        return Apply(self, inputs, outputs)

    def perform(self, node, (x, y), (z, )):
        z[0] = numpy.outer(x, y)
    def grad(self, (x, y), (gz,)):
        return dot(gz, y), dot(x, gz) #no transposing necessary
    def __str__(self):
        return "outer"
outer = Outer()

class Gemm(Op):
    """In-place version of matrix-matrix multiplication (with accumulation):

    When a and b are scalars and x, y, and z are matrices, then

        gemm(z,a,x,y,b) 

    is similar to 

        b*z + a*dot(x,y) 

    The difference between the two is that the top form is destructive on z,
    whereas the bottom form is not.  Gemm works in-place on the storage
    associated with z, and the L{Result} returned by Gemm has a storage that
    will be aliased to the storage of the z argument. Because of this in-place
    computation, an L{Apply} of this op will destroy the L{Result} z on
    which it operates.  (See L{DestructiveOps} for an explanation of what
    destroying means in the context of theano graphs. See L{BlasLapackSupport} for
    more optimized linear algebra operations.)

    """
    E_rank = 'gemm only works for rank 2'
    E_scalar = 'gemm requires scalar argument'
    E_z_uniq = 'argument z aliased to x or y'
    destroy_map = {0: [0]}
    def make_node(self, *inputs):
        inputs = map(as_tensor, inputs)
        if len(inputs) != 5:
            raise TypeError("Wrong number of inputs for %s (expected 5, got %s)" % (self, len(inputs)))
        z, a, x, y, b = inputs
        zr, xr, yr = [set(gof.view_roots(i)) for i in z,x,y]
        if zr.intersection(xr):
            raise ValueError(Gemm.E_z_uniq, (z, x))
        if zr.intersection(yr):
            raise ValueError(Gemm.E_z_uniq, (z, y))
        bz, ba, bx, by, bb = [r.type.broadcastable for r in inputs]
        if len(bz) != 2: raise ValueError(Gemm.E_rank, len(bz))
        if len(bx) != 2: raise ValueError(Gemm.E_rank, len(bx))
        if len(by) != 2: raise ValueError(Gemm.E_rank, len(by))
        if len(ba): raise ValueError(Gemm.E_scalar, ba)
        if len(bb): raise ValueError(Gemm.E_scalar, bb)
        output = z.type()
        return Apply(self, inputs, [output])
    def perform(self, node, (z, a, x, y, b), (zout, )):
        assert a.shape == ()
        assert b.shape == ()
        if z.shape == ():
            z.itemset(z*a + b*numpy.dot(x,y))
            zout[0] = z
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
            zout[0] = z
    def grad(self, (z, a, x, y, b), (gz,)):
        raise NotImplementedError()

    def c_support_code(self):
        #return blas.cblas_header_text()
        mod_str = """
        #ifndef MOD
        #define MOD %
        #endif
        """
        return blas.blas_proto() + mod_str
    def c_headers(self):
        return ['<iostream>']
    def c_libraries(self):
        return blas.ldflags()
    def c_code(self, node, name, (_z, _a, _x, _y, _b), (_zout, ), sub):
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

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;

        if (%(_zout)s != %(_z)s)
        {
            if (%(_zout)s)
            {
                Py_DECREF(%(_zout)s);
            }
            %(_zout)s = %(_z)s;
            Py_INCREF(%(_zout)s);
        }

        if (%(_x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(_y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(_z)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(z) != 2"); %(fail)s;}

        if ((%(_a)s->descr->type_num != PyArray_DOUBLE)
            && (%(_a)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        if ((%(_b)s->descr->type_num != PyArray_DOUBLE)
            && (%(_b)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(b) is not double or float"); %(fail)s;}

        if ((%(_x)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_x)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((%(_y)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_y)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((%(_z)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_z)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((%(_x)s->descr->type_num != %(_y)s->descr->type_num)
            ||(%(_x)s->descr->type_num != %(_z)s->descr->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(z), type(y), type(z) are not all the same"); %(fail)s; }

        if ((Nx[0] != Nz[0]) || (Nx[1] != Ny[0]) || (Ny[1] != Nz[1]))
        {
            PyErr_SetString(PyExc_ValueError, "Input dimensions do not agree");
            %(fail)s;
        }
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
           || (Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
           || (Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size))
        {
            PyErr_SetString(PyExc_ValueError, "stride is not multiple of element size"); %(fail)s;
        }

        /*
        encode the stride structure of _x,_y,_z into a single integer
        */
        unit |= ((Sx[1] == type_size) ? 0x0 : (Sx[0] == type_size) ? 0x1 : 0x2) << 8;
        unit |= ((Sy[1] == type_size) ? 0x0 : (Sy[0] == type_size) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size) ? 0x0 : (Sz[0] == type_size) ? 0x1 : 0x2) << 0;

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
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                switch(unit)
                {
                    case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
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
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                switch(unit)
                {
                    case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
                #undef REAL
            }
            break;
        }

        """ % dict(locals(), **sub)
gemm = Gemm()



#########################
# Gradient
#########################

def grad(cost, wrt, g_cost=None):
    """
    @type cost: L{Result}
    @type wrt: L{Result} or list of L{Result}s.
    @type g_cost: L{Result} broadcastable to size of I{cost}, or None
    @param g_cost: an expression for the gradient through cost.  The default is
        {{{ones_like(cost)}}}

    @rtype: L{Result} or list of L{Result}s (depending upon I{wrt})
    @return: symbolic expression of gradient of I{cost} with respect to I{wrt}.
    If I{wrt} is a list, then return a list containing the gradient of I{cost} wrt
    each element of the list.  If an element of I{wrt} is not differentiable
    with respect to the output, then a L{TensorConstant} with an appropriate
    kind of zero is returned.

    """
    if not isinstance(cost, TensorResult):
        raise TypeError('In tensor.grad(), cost argument should be a TensorResult.', cost)

    if g_cost is None:
        g_cost = ones_like(cost)
    inputs = gof.graph.inputs([cost])
    gmap = gradient.grad_sources_inputs([(cost, g_cost)], inputs)

    def zero(p):
        return TensorConstant(
                Tensor(dtype = p.type.dtype, broadcastable = []),
                numpy.asarray(0, dtype=p.type.dtype))

    if isinstance(wrt, list):
        return [gmap.get(p, zero(p)) for p in wrt]
    else:
        return gmap.get(wrt, zero(wrt))

