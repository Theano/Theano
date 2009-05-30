"""A `Type` and `Op` classes to work with numpy.ndarrays symbolically."""

__docformat__ = "restructuredtext en"

import __builtin__
import sys # for sys.maxint
import traceback #for overriding Op.__call__
import functools

import numpy
from copy import copy

from .. import gof
from ..gof import Variable, Op, utils, Type, Constant, Apply, Value

from .. import gradient

import elemwise
from .. import scalar as scal
from ..gof.python25 import partial

from .. import compile, printing
from ..printing import pprint, Print

### set up the external interface
from elemwise import Elemwise, DimShuffle, CAReduce, Sum



def check_equal_numpy(x, y):
    """
    Returns True iff x and y are equal (checks the dtype and
    shape if x and y are numpy.ndarray instances).
    """
    if isinstance(x, numpy.ndarray) and isinstance(y, numpy.ndarray):
        return x.dtype == y.dtype and x.shape == y.shape and numpy.any(abs(x - y) < 1e-10)
    elif isinstance(x, numpy.random.RandomState) and isinstance(y, numpy.random.RandomState):
        return all(numpy.all(a==b) for a, b in zip(x.__getstate__(), y.__getstate__()))
    else:
        return x == y

compile.register_checker(check_equal_numpy)



__oplist_constructor_list = []
"""List of functions to be listed as op constructors in the oplist (`gen_oplist`, doc/oplist.txt)."""
def constructor(f):
    """Add `f` to :doc:`oplist`.
    
    Make `f` appear as a constructor in the oplist (`gen_oplist`, doc/oplist.txt).
    """
    __oplist_constructor_list.append(f)
    return f
def __oplist_tag(thing, tag):
    tags = getattr(thing, '__oplist_tags', [])
    tags.append(tag)
    thing.__oplist_tags = tags


def as_tensor_variable(x, name = None, ndim=None):
    """Return `x`, transformed into a `TensorType`

    This function is often used by `make_node` methods of `Op` subclasses to
    turn ndarrays, numbers, `Scalar` instances, `Apply` instances and `TensorType`
    instances into valid input list elemnts.

    :Parameters:
     - `x`: Apply instance, Variable instance, numpy.ndarray, or number
       This thing will be transformed into a `Variable` in a sensible way.  An
       ndarray argument will not be copied, but a list of numbers will be copied
       to make an ndarray.
     - `name`: str or None
       If a new `Variable` instance is created, it will be named with this string.
     - `ndim`: None or integer
       Return a Variable with this many dimensions.  Raise TypeError if it's not possible.

    :Exceptions:
     - `ValueError`: raised if an `Apply` with no default output is fetched
     - `TypeError`: raised if `x` cannot be converted to a TensorType Variable

    """

    if isinstance(x, gof.Apply):
        #TODO: use Apply's default output mechanism
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a multi-output Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, Variable):
        if isinstance(x.type, scal.Scalar):
            x = tensor_from_scalar(x)

        if not isinstance(x.type, TensorType):
            raise TypeError("Variable type field must be a TensorType.", x, x.type)

        if ndim is None:
            return x
        else:
            if (x.type.ndim > ndim):
                #TODO: strip off leading broadcastable dimensions
                raise ValueError('TensorType could not be cast to have %i dimensions' % ndim, x.type)
            elif (x.type.ndim < ndim):
                return shape_padleft(x, n_ones=(ndim - x.type.ndim))
            else:
                return x
    if isinstance(x, (tuple, list)) and any(isinstance(xi, Variable) for xi in x):
        try:
            return stack(*x)
        except (TypeError, ValueError):
            pass

    try:
        return constant(x, name=name, ndim=ndim)
    except TypeError:
        try:
            str_x = str(x)
        except:
            str_x = repr(x)
        raise TypeError("Cannot convert %s to TensorType" % str_x, type(x))

# this has a different name, because _as_tensor_variable is the function which ops use
# to upcast their arguments... this internal-use function is a good place to put debugging stuff, better than the global astensor.
_as_tensor_variable = as_tensor_variable

as_tensor = as_tensor_variable


def constant_or_value(x, rtype, name=None, ndim=None):
    """Return a symbolic `Constant` with value `x`
    
    :Exceptions:
     - `TypeError`: `x` could not be converted to a numpy.ndarray
     - `ValueError`: `x` could not be expanded to have ndim dimensions

    """
    if isinstance(x, numpy.ndarray):
        x_ = x
    else:
        x_ = numpy.asarray(x)

    bcastable = [d == 1 for d in x_.shape]
    if ndim is not None:
        if len(bcastable) < ndim:
            bcastable = [True] * (ndim - len(bcastable)) + bcastable
        elif len(bcastable) > ndim:
            #TODO: strip off dimensions of size 1
            raise ValueError('ndarray could not be cast to constant with %i dimensions' % ndim)
        assert len(bcastable) == ndim

    try:
        return rtype(TensorType(dtype = x_.dtype, broadcastable = bcastable), x_, name=name)
    except:
        raise TypeError("Could not convert %s to TensorType" % x, type(x))

def constant(x, name=None, ndim=None):
    return constant_or_value(x, rtype=TensorConstant, name=name, ndim=ndim)

def value(x, name=None, ndim=None):
    return constant_or_value(x, rtype=TensorValue, name=name, ndim=ndim)

def _obj_is_wrappable_as_tensor(x):
    try:
        constant(x)
        return True
    except TypeError:
        return False
def _wrap_tensor_into_member(x):
    return compile.module.Member(constant(x))
compile.module.register_wrapper(_obj_is_wrappable_as_tensor, _wrap_tensor_into_member)


class TensorType(Type):
    """Symbolic `Type` representing a numpy.ndarray value."""

    filter_checks_isfinite = False
    """
    When this is True, strict filtering rejects data containing NaN or Inf entries. (Used in `DebugMode`)
    """

    def __init__(self, dtype, broadcastable, name = None):
        """Initialize self.dtype and self.broadcastable.

        :Parameters:
         - `dtype`: str corresponding to numpy dtype (e.g., 'int64')
           The value (ndarray) associated to a `Variable` of this `Type` will have
           this dtype.
         - `broadcastable`: tuple, list, or array of boolean values
           This argument serves two purposes.  First, the True elements of this
           list indicate the dimensions where the shape of an associated value
           must be 1.  Secondly, the length of this list is the number of
           dimensions that an associated value must have.  See
           :doc:`broadcasting` for an explanation of how this list is used.
         - `name`: str
           Optional name for this type.
        """
        self.dtype = str(dtype)
        self.broadcastable = tuple(broadcastable)
        self.dtype_specs() # error checking is done there
        self.name = name
    
    def filter(self, data, strict = False):
        """Convert `data` to something which can be associated to a `TensorVariable`.

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
            if self.filter_checks_isfinite and (not numpy.all(numpy.isfinite(data))):
                raise TypeError("non-finite elements not allowed")
            return data
        else:
            data = numpy.asarray(data, dtype = self.dtype)
        if not self.ndim == data.ndim:
            raise TypeError("Wrong number of dimensions: expected %s, got %s with shape %s." % (self.ndim, data.ndim, data.shape), data)
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

    def to_scalar_type(self):
        return scal.Scalar(dtype = self.dtype)

    def __eq__(self, other):
        """Compare True iff other is the same kind of TensorType"""
        return type(self) == type(other) and other.dtype == self.dtype and other.broadcastable == self.broadcastable

    def values_eq_approx(self, a, b):
        if type(a) is numpy.ndarray and type(b) is numpy.ndarray:
            if a.shape != b.shape:
                return False
            if a.dtype != b.dtype:
                return False
            if 'int' in str(a.dtype):
                return numpy.all(a==b)
            elif a.shape == (): #for comparing scalars, use broadcasting.
                # Note: according to James B, there was a reason for the
                # following two lines, that may seem weird at first glance.
                # If someone can figure out what it is, please say it here!
                ones = numpy.ones(2)
                return numpy.allclose(ones * a, ones*b)
            #elif str(a.dtype).startswith('complex'):
            #    print >> sys.stderr, 'WARNING: skipping comparison of complex'
            #    return True
            else:
                cmp = numpy.allclose(a,b)
                if cmp:
                    # Numpy claims they are close, this is good enough for us.
                    return True
                # Numpy is unhappy, but it does not necessarily mean that a and
                # b are different. Indeed, Numpy does not like missing values
                # and will return False whenever some are found in a or b.
                # The proper way would be to use the MaskArray stuff available
                # in Numpy. However, it looks like it has been added to Numpy's
                # core recently, so it may not be available to everyone. Thus,
                # for now we use a home-made recipe, that should probably be
                # revisited in the future.
                a_missing = numpy.isnan(a)
                if not a_missing.any():
                    # There are no missing values in a, thus this is not the
                    # reason why numpy.allclose(a, b) returned False.
                    return False
                # The following line is what numpy.allclose bases its decision
                # upon, according to its documentation.
                rtol = 1.0000000000000001e-05
                atol = 1e-8
                cmp_elemwise = (numpy.absolute(a - b) <=
                        (atol + rtol * numpy.absolute(b)))
                # Find places where both a and b have missing values.
                both_missing = a_missing * numpy.isnan(b)
                # Combine all information.
                return (cmp_elemwise + both_missing).all()

        return False

    def __hash__(self):
        """Hash equal for same kinds of TensorType"""
        return hash(type(self)) ^ hash(self.dtype) ^ hash(self.broadcastable)

    ndim = property(lambda self: len(self.broadcastable), doc = "number of dimensions")
    """Number of dimensions

    This read-only property is the preferred way to get the number of dimensions
    of a `TensorType`.
    
    """

    def make_variable(self, name = None):
        """Return a `TensorVariable` of this type

        :Parameters:
         - `name`: str
           A pretty name to identify this `Variable` when printing and debugging

        """
        return TensorVariable(self, name = name)

    def __str__(self):
        if self.name:
            return self.name
        else:
            b = self.broadcastable
            #bcast = str(self.broadcastable)
            bcast = {(): 'scalar',
                     (False,): 'vector',
                     (False, True): 'col',
                     (True, False): 'row',
                     (False, False): 'matrix'}.get(b, "%iD" % len(b) if not any(b) else str(b))
            return "TensorType(%s, %s)" % (str(self.dtype), bcast)

    def __repr__(self):
        return str(self)
        #"TensorType{%s, %s}" % (str(self.dtype), str(self.broadcastable))

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
            Py_XINCREF(Py_None);
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
    return TensorType(*args, **kwargs).make_variable()

def _multi(*fns):
    def f2(f, *names):
        if names and isinstance(names[0], int):
            if names == 1:
                return f()
            else:
                return [f() for i in xrange(names[0])]
        if isinstance(names, tuple):
            if len(names) == 1:
                names = names[0]
        if len(names) == 1:
            return f(names)
        else:
            return [f(name) for name in names]
    if len(fns) == 1:
        return partial(f2, fns)
    else:
        return [partial(f2, f) for f in fns]

cscalar = TensorType('complex64', ())
zscalar = TensorType('complex128', ())
fscalar = TensorType('float32', ())
dscalar = TensorType('float64', ())
bscalar = TensorType('int8', ())
wscalar = TensorType('int16', ())
iscalar = TensorType('int32', ())
lscalar = TensorType('int64', ())
def scalar(name = None, dtype = 'float64'):
    type = TensorType(dtype, ())
    return type(name)
scalars, fscalars, dscalars, iscalars, lscalars = _multi(scalar, fscalar, dscalar, iscalar, lscalar)

int_types = bscalar, wscalar, iscalar, lscalar
float_types = fscalar, dscalar
complex_types = cscalar, zscalar
int_scalar_types = int_types
float_scalar_types = float_types
complex_scalar_types = complex_types

cvector = TensorType('complex64', (False, ))
zvector = TensorType('complex128', (False, ))
fvector = TensorType('float32', (False, ))
dvector = TensorType('float64', (False, ))
bvector = TensorType('int8', (False,))
wvector = TensorType('int16', (False,))
ivector = TensorType('int32', (False, ))
lvector = TensorType('int64', (False, ))
def vector(name = None, dtype = 'float64'):
    type = TensorType(dtype, (False, ))
    return type(name)
vectors, fvectors, dvectors, ivectors, lvectors = _multi(vector, fvector, dvector, ivector, lvector)

int_vector_types = bvector, wvector, ivector, lvector
float_vector_types = fvector, dvector
complex_vector_types = cvector, zvector

cmatrix = TensorType('complex64', (False, False))
zmatrix = TensorType('complex128', (False, False))
fmatrix = TensorType('float32', (False, False))
dmatrix = TensorType('float64', (False, False))
bmatrix = TensorType('int8', (False, False))
wmatrix = TensorType('int16', (False, False))
imatrix = TensorType('int32', (False, False))
lmatrix = TensorType('int64', (False, False))
def matrix(name = None, dtype = 'float64'):
    type = TensorType(dtype, (False, False))
    return type(name)
matrices, fmatrices, dmatrices, imatrices, lmatrices = _multi(matrix, fmatrix, dmatrix, imatrix, lmatrix)

int_matrix_types = bmatrix, wmatrix, imatrix, lmatrix
float_matrix_types = fmatrix, dmatrix
complex_matrix_types = cmatrix, zmatrix

crow = TensorType('complex64', (True, False))
zrow = TensorType('complex128', (True, False))
frow = TensorType('float32', (True, False))
drow = TensorType('float64', (True, False))
brow = TensorType('int8', (True, False))
wrow = TensorType('int16', (True, False))
irow = TensorType('int32', (True, False))
lrow = TensorType('int64', (True, False))
def row(name = None, dtype = 'float64'):
    type = TensorType(dtype, (True, False))
    return type(name)
rows, frows, drows, irows, lrows = _multi(row, frow, drow, irow, lrow)

ccol = TensorType('complex64', (False, True))
zcol = TensorType('complex128', (False, True))
fcol = TensorType('float32', (False, True))
dcol = TensorType('float64', (False, True))
bcol = TensorType('int8', (False, True))
wcol = TensorType('int16', (False, True))
icol = TensorType('int32', (False, True))
lcol = TensorType('int64', (False, True))
def col(name = None, dtype = 'float64'):
    type = TensorType(dtype, (False, True))
    return type(name)
cols, fcols, dcols, icols, lcols = _multi(col, fcol, dcol, icol, lcol)


class _tensor_py_operators:
    #UNARY
    def __abs__(self): return abs_(self)
    def __neg__(self): return neg(self)

    #CASTS 
    #### REMOVED THESE BECAUSE PYTHON appears to require __int__ to return an int. -JB 20081112
    #def __int__(self): return convert_to_int32(self)
    #def __float__(self): return convert_to_float64(self)
    #def __complex__(self): return convert_to_complex128(self)

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
#     def __iand__(self, other): return _and_inplace(self, other)
#     def __ior__(self, other): return _or_inplace(self, other)
#     def __ixor__(self, other): return _xor_inplace(self, other)

    #ARITHMETIC - NORMAL
    def __add__(self,other): 
        try:
            return add(self,other)
        except Exception, e:
            return NotImplemented
    def __sub__(self,other): 
        try:
            return sub(self,other)
        except Exception, e:
            return NotImplemented
    def __mul__(self,other): 
        try: 
            return mul(self,other)
        except Exception, e:
            return NotImplemented
    def __div__(self,other): 
        try: 
            return div_proxy(self,other)
        except Exception, e:
            return NotImplemented
    def __pow__(self,other): 
        try:
            return pow(self,other)
        except Exception, e:
            return NotImplemented
    def __mod__(self,other):
        try:
            return mod(self,other)
        except Exception, e:
            return NotImplemented

#     ##### DON"T USE THESE BECAUSE INPLACE OPS SHOULD BE INSERTED BY OPTIMIZATION ONLY
#     #ARITHMETIC - INPLACE
#     def __iadd__(self,other): return _add_inplace(self,other)
#     def __isub__(self,other): return _sub_inplace(self,other)
#     def __imul__(self,other): return _mul_inplace(self,other)
#     def __idiv__(self,other): return _div_inplace(self,other)
#     def __ipow__(self,other): return _pow_inplace(self,other)

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div_proxy(other,self)
    def __rmod__(self,other): return mod(other,self)
    def __rpow__(self,other): return pow(other,self)

    #TRANSPOSE
    T = property(lambda self: transpose(self))

    shape = property(lambda self: shape(self))
    def reshape(self, shape, ndim=None):
        return reshape(self, shape, ndim=ndim)

    #SLICING
#     def __getitem__(self, args): return Subtensor.from_idxs(self,
#             args).outputs[0]
#     def __getslice__(self, *args): return Subtensor.from_idxs(self,
#             (slice(*args),)).outputs[0]
    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = args,
        return Subtensor(args)(self, *Subtensor.collapse(args, lambda entry: isinstance(entry, Variable)))
    def __getslice__(self, *args):
        args = slice(*args),
        return Subtensor(args)(self, *Subtensor.collapse(args, lambda entry: isinstance(entry, Variable)))
    
    #COPYING
    def copy(self): return tensor_copy(self)

    def __iter__(self): 
        try:
            for i in xrange(get_vector_length(self)):
                yield self[i]
        except:
            # This prevents accidental iteration via builtin.sum(self)
            raise TypeError('TensorType does not support iteration. '
            'Maybe you are using builtin.sum instead of theano.tensor.sum? (Maybe .max?)')
        

    # CONVENIENT ACCESS TO TYPE PROPERTIES
    ndim = property(lambda self: self.type.ndim)
    """The rank of this tensor."""
    broadcastable = property(lambda self: self.type.broadcastable)
    """The broadcastable signature of this tensor.

    See :doc:`broadcasting` for details.
    
    """
    dtype = property(lambda self: self.type.dtype)
    """ The dtype of this tensor.  """


    #extra pseudo-operator symbols
    def __dot__(left, right): return dot(left, right)
    def __rdot__(right, left): return dot(left, right)

    def sum(self, axis=None):
        return elemwise.Sum(axis)(self)

    def norm(self, L, axis=None):
        if L==0:
            raise NotImplementedError()
        if L==float('inf'):
            raise NotImplementedError()
        #optimizations will/should catch cases like L=1, L=2
        return pow(pow(abs_(self), L).sum(axis=axis), 1.0/L)


    #TO TRUMP NUMPY OPERATORS
    __array_priority__ = 1000
    

class TensorVariable(Variable, _tensor_py_operators):
    """Subclass to add the tensor operators to the basic `Variable` class."""

class TensorConstantSignature(tuple):
    def __eq__(self, other):
        (a, b), (x,y) = self, other
        #N.B. compare shape to ensure no broadcasting in ==
        return (x == a) and (b.shape == y.shape) and (numpy.all(b == y)) 
    def __hash__(self):
        a, b = self
        return hash(type(self)) ^ hash(a) ^ hash(b.shape)

class TensorConstant(Constant, _tensor_py_operators):
    """Subclass to add the tensor operators to the basic `Constant` class.
    
    To create a TensorConstant, use the `constant` function in this module.
    """
    def signature(self):
        return TensorConstantSignature((self.type, self.data))

class TensorValue(Value, _tensor_py_operators):
    """Subclass to add the tensor operators to the basic `Value` class.
    
    To create a TensorValue, use the `value` function in this module.
    """


Tensor = TensorType
TensorVariable = TensorVariable
TensorConstant = TensorConstant
TensorValue = TensorValue


#QUESTION: why are we doing this!?
elemwise.as_tensor_variable = as_tensor_variable    
elemwise.TensorType = TensorType
elemwise.TensorVariable = TensorVariable
elemwise.TensorConstant = TensorConstant
elemwise.TensorValue = TensorValue



#########################
# Utilities
#########################

def _elemwise(scalar_op, name, doc_prefix=''):
    straight = elemwise.Elemwise(scalar_op, name = name)
    inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
    inplace = elemwise.Elemwise(inplace_scalar_op, {0: 0}, name = name+"_inplace")

    # don't add the inplace versions, they aren't supposed to be part of the user interface
    _constructor_list.append(straight) 
    
    # This is here so that gen_oplist can detect which module declared these variables.

    straight.__module__ = 'tensor'
    inplace.__module__ = 'tensor'

    if doc_prefix:
        straight.__doc__ = doc_prefix + '\n' + straight.__doc__

    return straight, inplace

def _redefine(real_symbol_value, module='tensor'):
    """Replace the value associated with a function symbol.
    
    This is useful to trick epydoc into doing what we want.  It's a hack.
    """
    real_symbol_value.__module__ = 'tensor.basic'
    def decorator(f):
        return real_symbol_value
    return decorator

def _redefine_asRoutine(real_symbol_value):
    real_symbol_value.__epydoc_asRoutine = True
    def decorator(f):
        return real_symbol_value
    return decorator

def _scal_elemwise(symbol):
    """Replace a symbol definition with an elementwise version of the corresponding scalar Op"""
    symbolname = symbol.__name__
    inplace = symbolname.endswith('_inplace')

    if inplace:
        scalar_op = getattr(scal, symbolname[:-len('_inplace')])
        inplace_scalar_op = scalar_op.__class__(scal.transfer_type(0))
        rval = elemwise.Elemwise(inplace_scalar_op, {0: 0}, name=symbolname)
    else:
        scalar_op = getattr(scal, symbolname)
        rval = elemwise.Elemwise(scalar_op, name=symbolname)

    if getattr(symbol, '__doc__', False):
        rval.__doc__ = symbol.__doc__ + '\n' + rval.__doc__

    #for the meaning of this see the ./epydoc script
    # it makes epydoc display rval as if it were a function, not an object
    rval.__epydoc_asRoutine = symbol
    rval.__module__ = 'tensor'

    pprint.assign(rval, printing.FunctionPrinter(symbolname))

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
        assert isinstance(t.type, TensorType)
        assert t.type.broadcastable == ()
        return Apply(self,
                     [t],
                     [scal.Scalar(dtype = t.type.dtype).make_variable()])
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

#to be removed as we get the epydoc routine-documenting thing going -JB 20080924
def _conversion(real_value, name):
    __oplist_tag(real_value, 'casting')
    real_value.__module__='tensor.basic'
    pprint.assign(real_value, printing.FunctionPrinter(name))
    return real_value

convert_to_int8  = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int8))), 'int8')
"""Cast to 8-bit integer"""
    
convert_to_int16 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int16))), 'int16')
"""Cast to 16-bit integer"""

convert_to_int32 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int32))), 'int32')
"""Cast to 32-bit integer"""

convert_to_int64 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.int64))), 'int64')
"""Cast to 64-bit integer"""

convert_to_float32 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.float32))), 'float32')
"""Cast to single-precision floating point"""

convert_to_float64 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.float64))), 'float64')
"""Cast to double-precision floating point"""

convert_to_complex64  = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.complex64))), 'complex64')
"""Cast to single-precision complex"""

convert_to_complex128 = _conversion(elemwise.Elemwise(scal.Identity(scal.specific_out(scal.complex128))), 'complex128')
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
        x = as_tensor_variable(x)
        return Apply(self, [x], [lvector()])
    def perform(self, node, (x, ), (out, )):
        out[0] = numpy.asarray(x.shape, dtype = 'int64')
    def grad(self, (x,), (gz,)):
        return [None]
@_redefine_asRoutine(Shape())
def shape(a):
    pass

pprint.assign(shape, printing.MemberPrinter('shape'))


class MaxAndArgmax(Op):
    """Calculate the max and argmax over a given axis"""
    nin=2 # tensor, axis
    nout=2 # max val, max idx
    E_axis = 'invalid axis'
    
    def make_node(self, x, axis=None):
        x = _as_tensor_variable(x)
        if axis is None:
            axis = x.type.ndim - 1
        axis = _as_tensor_variable(axis)
        inputs = [x, axis]
        broadcastable = [False] * (x.type.ndim - 1) #TODO: be less conservative
        outputs = [tensor(x.type.dtype, broadcastable), 
                   tensor(axis.type.dtype, broadcastable)]
        return Apply(self, inputs, outputs)
    def perform(self, node, (x, axis), (max, max_idx)):
        max[0] = numpy.asarray(numpy.max(x, axis))
        max_idx[0] = numpy.asarray(numpy.argmax(x, axis))
    def grad(self, (x, axis), (g_max, g_max_idx)):
        # @warning: This only works if axis is 0, else the max is
        # broadcasted wrong in the call to eq.
        # @note: This function should work correctly for L{vector}s.
#        (x, y), (gz, gw)
#        gz*dz/dx + gw*dw/dx, gz*dz/dy + gw*dw/dy
#        gMax * dMax/dx + gArgMax * dArgMax/dx, gMax * dMax/daxis + gArgMax * dArgMax/daxis
#       g_max has one less dimension than x, so you need to complete g_max to x's shape
#        when axis=0 the broadcasting mechanism does it automatically
        
        if not ( axis.data == 0 or axis.data == x.ndim-1):
            raise NotImplementedError('MaxAndArgmax gradient with axis corresponding to internal dimension')
        g_max_pad = shape_padleft(g_max) if axis.data==0 else \
                    shape_padright(g_max)
        xmax = max(x, axis)
        xmax_pad = shape_padleft(xmax) if axis.data==0 else \
                   shape_padright(xmax)
        g_x = eq(xmax_pad, x) * g_max_pad
        return g_x, None

_max_and_argmax = MaxAndArgmax()
@_redefine_asRoutine(_max_and_argmax)
def max_and_argmax(a):
    pass


@constructor
def max(x, axis=None):
    """
    Return maximum elements obtained by iterating over given axis

    Default axis is the last one.
    """
    # In python (using MaxAndArgmax.perform()) this leads to an wasteful
    # implementation that goes through the data twice instead of once
    # but when Argmax.c_impl() is in place, it should be fine.
    return max_and_argmax(x,axis)[0]

@constructor
def argmax(x, axis=None):
    """
    Return indexes of maximum elements obtained by iterating over given axis

    Default axis is the last one.
    """
    # In python (using MaxAndArgmax.perform()) this leads to an wasteful
    # implementation that goes through the data twice instead of once
    # but when Argmax.c_impl() is in place, it should be fine.
    return max_and_argmax(x,axis)[1]

@constructor
def min(x, axis=None):
    str_x_type = str(x.dtype)
    if str_x_type.startswith('float') or str_x_type.startswith('int'):
        return -max(-x, axis=axis)
    else:
        #Be careful about unsigned integers, complex
        raise NotImplementedError()

@constructor
def argmin(x, axis=None):
    str_x_type = str(x.dtype)
    if str_x_type.startswith('float') or str_x_type.startswith('int'):
        return argmax(-x, axis=axis)
    else:
        #Be careful about unsigned integers, complex
        raise NotImplementedError()

@constructor
def smallest(*args):
    """Return the [elementwise] smallest of a variable number of arguments (like python's min)."""
    return min(stack(*args), axis=0)

@constructor
def largest(*args):
    """Return the [elementwise] largest of a variable number of arguments (like python's max)."""
    return max(stack(*args), axis=0)


##########################
# Comparison
##########################

@_scal_elemwise
def lt(a, b):
    """a < b"""

@_scal_elemwise
def gt(a, b):
    """a > b"""

@_scal_elemwise
def le(a, b):
    """a <= b"""

@_scal_elemwise
def ge(a, b):
    """a >= b"""

@_scal_elemwise
def eq(a, b):
    """a == b"""

@_scal_elemwise
def neq(a, b):
    """a != b"""


##########################
# Condition
##########################

@_scal_elemwise
def switch(cond, ift, iff):
    """if cond then ift else iff"""


##########################
# Bit-wise
##########################

@_scal_elemwise
def and_(a,b):
    """bitwise a & b"""

@_scal_elemwise
def or_(a,b):
    """bitwise a | b"""

@_scal_elemwise
def xor(a,b):
    """bitwise a ^ b"""

@_scal_elemwise
def invert(a):
    """bitwise ~a"""

##########################
# Math
##########################

@_scal_elemwise
def abs_(a):
    """|`a`|

    TensorVariable overloads the `TensorVariable.__abs__` operator so that
    this function is called when you type abs(a).

    """

pprint.assign(abs_, printing.PatternPrinter(('|%(0)s|', -1000)))


@_scal_elemwise
def exp(a):
    """e^`a`"""

@_scal_elemwise
def neg(a):
    """-a"""

@_scal_elemwise
def inv(a):
    """1.0/a (inplace on a)"""

@_scal_elemwise
def log(a):
    """base e logarithm of a"""

@_scal_elemwise
def log2(a):
    """base 2 logarithm of a"""

@_scal_elemwise
def log10(a):
    """base 10 logarithm of a"""

@_scal_elemwise
def sgn(a):
    """sign of a"""

@_scal_elemwise
def iround(a):
    """int(round(a))"""

@_scal_elemwise
def sqr(a):
    """square of a"""

@_scal_elemwise
def sqrt(a):
    """square root of a"""

@_scal_elemwise
def cos(a):
    """cosine of a"""

@_scal_elemwise
def sin(a):
    """sine of a"""

@_scal_elemwise
def tan(a):
    """tangent of a"""

@_scal_elemwise
def cosh(a):
    """hyperbolic cosine of a"""

@_scal_elemwise
def sinh(a):
    """hyperbolic sine of a"""

@_scal_elemwise
def tanh(a):
    """hyperbolic tangent of a"""


##########################
# Misc
##########################

#fill, _fill_inplace = _elemwise(scal.second, 'fill',
    #"""fill WRITEME (elemwise)""")
@_scal_elemwise
def second(a, b):
    """Create a matrix by filling the shape of a with b"""

fill = second
pprint.assign(fill, printing.FunctionPrinter('fill'))


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
        self.type = TensorType(dtype = dtype,
                           broadcastable = (False,)*ndim)

    def make_node(self, dims):
        dims = as_tensor_variable(dims)
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

pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, Filler) and r.owner.op.value == 0, printing.FunctionPrinter('zeros'))
pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, Filler) and r.owner.op.value == 1, printing.FunctionPrinter('ones'))


@_redefine(elemwise.Elemwise(scal.identity))
def tensor_copy(a):
    """Create a duplicate of `a` (with duplicated storage)"""
pprint.assign(tensor_copy, printing.IgnorePrinter())


@_redefine(elemwise.Elemwise(scal.identity, inplace_pattern = {0: [0]}))
def view(a):
    """Create a duplicate of `a` (with shared storage)"""

@constructor
def sum(input, axis = None):
    """WRITEME"""
    return elemwise.Sum(axis)(input)

pprint.assign(Sum(), printing.FunctionPrinter('sum'))


@constructor
def mean(input, axis = None):
    """Compute the mean value along the given axis of a tensor `input`

    :param axis: compute the mean along this axis of the tensor.  None means all axes (like
    numpy).
    :type axis: None or int or (list of int) (see `Sum`)
    
    """
    if str(input.dtype).startswith('int'):
        # we need to cast eventually anyway, and this helps
        # to prevents overflow
        input = convert_to_float64(input)
    s = sum(input, axis)
    shp = shape(input)
    if axis is None:
        axis = range(input.type.ndim)
    elif isinstance(axis, int):
        axis = [axis]
    for i in axis:
        s = s / shp[i]
    return s

@constructor
def var(input, axis = None):
    """Compute the variance along the given axis of a tensor `input`

    :param axis: compute the variance along this axis of the tensor.  None means trailing axis.
    :type axis: None or int or (list of int) (see `Sum`)

    """
    input_ndim = input.type.ndim
    if axis == None:
        axis = range(input_ndim)
    if isinstance(axis, int):
        axis = [axis]

    #make a pattern that will undo the reduction of dimensions caused by mean
    pattern = []
    next_dim = 0
    for i in range(input_ndim):
        if i in axis:
            pattern.append('x')
        else:
            pattern.append(next_dim)
            next_dim += 1

    #compute the axis-wise mean
    mean_input_reduced = mean(input, axis)

    #broadcast that back out to match input
    mean_input = DimShuffle(
            list(mean_input_reduced.type.broadcastable),
            pattern)(mean_input_reduced)

    #center the input
    centered_input = input - mean_input

    #return the mean sqr
    return mean(centered_input**2, axis)

class Repeat(gof.Op):

    def make_node(self, input, repeats, axis):
        assert isinstance(input.type, TensorType)
        assert repeats.type == iscalar
        assert axis.type == iscalar
        type = TensorType(dtype = input.type.dtype,
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

def div_proxy(x, y):
    """Proxy for either true_div or int_div, depending on types of x, y.
    """
    if as_tensor_variable(x).type.dtype.startswith('int') and as_tensor_variable(y).type.dtype.startswith('int'):
        return int_div(x, y)
    else:
        return true_div(x, y)

@_scal_elemwise
def add(a, b):
    """elementwise addition"""

@_scal_elemwise
def sub(a, b):
    """elementwise subtraction"""

@_scal_elemwise
def mul(a, b):
    """elementwise multiplication"""

@_scal_elemwise
def true_div(a, b):
    """elementwise [true] division (inverse of multiplication)"""

@_scal_elemwise
def int_div(a, b):
    """elementwise integer-division"""

@_scal_elemwise
def mod(a, b):
    """elementwise modulo"""

@_scal_elemwise
def pow(a, b):
    """elementwise power"""

@_scal_elemwise
def clip(x, min, max):
    """clip x to be between min and max"""

pprint.assign(add, printing.OperatorPrinter('+', -2, 'either'))
pprint.assign(mul, printing.OperatorPrinter('*', -1, 'either'))
pprint.assign(sub, printing.OperatorPrinter('-', -2, 'left'))
pprint.assign(neg, printing.OperatorPrinter('-',  0, 'either'))
pprint.assign(true_div, printing.OperatorPrinter('/', -1, 'left'))
pprint.assign(int_div, printing.OperatorPrinter('//', -1, 'left'))
pprint.assign(pow, printing.OperatorPrinter('**', 1, 'right'))



##########################
# View Operations
##########################

def transpose(x, **kwargs):
    dims = range(x.ndim-1, -1, -1)
    return DimShuffle(x.broadcastable, dims, inplace=True)(tensor_copy(x))


class Subtensor(Op):
    """Return a subtensor view

    This class uses a relatively complex internal representation of the inputs
    to remember how the input tensor x should be sliced.  The instance variable
    idxlist is a list whose elements are either integers, or slices.  The
    integers are indexes into the inputs array, and the start/stop/step members
    of each slice are also integer indexes into the inputs array (or None).  The
    inputs array is the tensor x, followed by scalar integer variables.
    
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

    @staticmethod
    def convert(entry, slice_ok=True):
        scal_types = [scal.int64, scal.int32, scal.int16, scal.int8]
        tensor_types = [bscalar, iscalar, lscalar]
        if isinstance(entry, gof.Variable) and entry.type in scal_types:
            return entry.type
        elif isinstance(entry, gof.Type) and entry in scal_types:
            return entry
        if isinstance(entry, gof.Variable) and entry.type in tensor_types:
            return scal.Scalar(entry.type.dtype)
        elif isinstance(entry, gof.Type) and entry in tensor_types:
            return scal.Scalar(entry.dtype)
        elif slice_ok and isinstance(entry, slice):
            a = entry.start
            b = entry.stop
            c = entry.step
            return slice(Subtensor.convert(a, False) if a is not None else None,
                         Subtensor.convert(b, False) if b is not None else None,
                         Subtensor.convert(c, False) if c is not None else None)
        elif isinstance(entry, int):
            return entry
        else:
            raise TypeError(Subtensor.e_indextype, entry)

    def __init__(self, idx_list):
        self.idx_list = map(self.convert, idx_list)

    def make_node(self, x, *inputs):
        x = as_tensor_variable(x)
        def my_as_scalar(a):
            if isinstance(a, gof.Variable) and isinstance(a.type, TensorType):
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
        idx_list = tuple((entry.start, entry.stop, entry.step)
                         if isinstance(entry, slice)
                         else entry
                         for entry in self.idx_list)
        return hash(idx_list)

    def __str__(self):
        indices = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                indices.append(":".join("" if x is None else str(x) for x in [entry.start, entry.stop, entry.step]))
            else:
                indices.append(str(entry))
        return "%s{%s}" % (self.__class__.__name__, ", ".join(indices))



class SubtensorPrinter:

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print Subtensor.")
        elif isinstance(r.owner.op, Subtensor):
            idxs = r.owner.op.idx_list
            inputs = list(r.owner.inputs)
            input = inputs.pop()
            sidxs = []
            inbrack_pstate = pstate.clone(precedence = -1000)
            for entry in idxs:
                if isinstance(entry, int):
                    sidxs.append(str(entry))
                elif isinstance(entry, scal.Scalar):
                    sidxs.append(inbrack_pstate.pprinter.process(inputs.pop()))
                elif isinstance(entry, slice):
                    sidxs.append("%s:%s%s" % ("" if entry.start is None or entry.start == 0 else entry.start,
                                              "" if entry.stop is None or entry.stop == sys.maxint else entry.stop,
                                              "" if entry.step is None else ":%s" % entry.step))
            return "%s[%s]" % (pstate.pprinter.process(input, pstate.clone(precedence = 1000)),
                               ", ".join(sidxs))
        else:
            raise TypeError("Can only print Subtensor.")

pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, Subtensor), SubtensorPrinter())



class SetSubtensor(Op):
    """Set just some elements of a larger TensorType.

    This is like numpy's 

        z[i,j,k] = <something> 
    
    """

    def __init__(self, idx_list, inplace=False):
        self.idx_list = map(Subtensor.convert, idx_list)
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}

    def __eq__(self, other):
        return type(self) == type(other) \
                and self.idx_list == other.idx_list \
                and self.inplace == other.inplace

    def __hash__(self):
        idx_list = tuple((entry.start, entry.stop, entry.step)
                         if isinstance(entry, slice)
                         else entry
                         for entry in self.idx_list)
        return hash(type(self)) ^ hash(idx_list) ^ hash(self.inplace)

    def __str__(self):
        indices = []
        for entry in self.idx_list:
            if isinstance(entry, slice):
                indices.append(":".join("" if x is None else str(x) for x in [entry.start, entry.stop, entry.step]))
            else:
                indices.append(str(entry))
        return "%s%s{%s}" % ('Inplace' if self.inplace else '',
                self.__class__.__name__, ", ".join(indices))

    def make_node(self, x, y, *inputs):
        x, y = map(as_tensor_variable, [x, y])
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
        if not self.inplace:
            x = x.copy()
        x.__setitem__(cdata, y)
        out[0] = x

def split(x, splits_size, n_splits, axis=0):
    the_split = Split(n_splits)
    return the_split(x, axis, splits_size)

class Split(Op):
    """Partition a `TensorVariable` along some axis.

    .. python::
        
        x = vector()
        splits = lvector()
        # you have to declare right away how many split_points there will be.
        ra, rb, rc = split(x, splits, n_splits = 3, axis = 0)  

        f = function([x, splits], [ra, rb, rc])

        a, b, c = f([0,1,2,3,4,5,6], [3, 2, 1])

        #a == [0,1,2]
        #b == [3, 4]
        #c == [5]

    """

    len_splits = None
    """A Split instance will have this many outputs, and require that the splits argument to
    `perform` have exactly this many elements.
    """

    def __init__(self, len_splits):
        self.len_splits = int(len_splits)

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.len_splits == other.len_splits)

    def __hash__(self):
        return hash(Split) ^ self.len_splits

    def __call__(self, *inputs, **kwargs):
        """Override Op.__call__ to suppress unpacking of output list

        """
        node = self.make_node(*inputs, **kwargs)
        node.tag.trace = traceback.extract_stack()[:-1]
        return node.outputs
 
    def make_node(self, x, axis, splits):
        """WRITEME"""
        x = as_tensor_variable(x)
        axis = as_tensor_variable(axis)
        splits = as_tensor_variable(splits)

        if splits.type not in int_vector_types: 
            raise TypeError('splits must have type tensor.lvector', splits.type)
        if axis.type not in int_types: 
            raise TypeError('axis must have type lscalar', axis.type)

        inputs = [x, axis, splits]
        outputs = [x.type() for i in xrange(self.len_splits)]

        return Apply(self, inputs, outputs)


    def perform(self, node, (x, axis, splits), outputs):
        """WRITEME"""
        try:
            len_along_axis = x.shape[axis]
        except :
            raise ValueError('Split.perform() with axis=(%s) is invalid for x.shape==(%s)'
                    %(axis, x.shape))
        if len(splits) != self.len_splits:
            raise ValueError('In Split.perform(), len(splits) != len_splits.', 
                    (len(splits), self.len_splits))
         
        # Checking is done, let's roll the splitting algorithm!
        # Basically we step along the given axis of x, extracting subtensors of size splits[i]
        # as we go along.

        general_key = [slice(None, None, None) for s in x.shape]
        lower_idx = 0
        for i in xrange(self.len_splits):
            upper_idx = lower_idx + splits[i]
            general_key[axis] = slice(lower_idx, upper_idx, None)
            outputs[i][0] = x.__getitem__(general_key).copy()
            lower_idx = upper_idx

    def grad(self, (x, axis, splits), g_outputs):
        """Join the gradients along the axis that was used to split x."""
        return [join(axis, *g_outputs), None, None]

class Join(Op):
    """
    Concatenate multiple `TensorVariable`s along some axis.

    The axis must be given as first argument. All tensors must have the same
    shape along all dimensions other than this axis.
    Of course, TensorVariable instances do not have a shape, so this error
    cannot be caught until runtime.  See `perform()`.

    For joins involving scalar values, see @stack.

    .. python::
        
        x, y, z = tensor.matrix(), tensor.matrix(), tensor.matrix()
        u = tensor.vector()

        r = join(0, x, y, z)
        c = join(1, x, y, z)
        join(2, x, y, z)    # WRONG: the axis has to be an index into the shape
        join(0, x, u)       # WRONG: joined tensors must have the same rank
    """

    def make_node(self, *axis_and_tensors):
        """
        :param axis: an Int or integer-valued Variable

        :param tensors: a variable number (but not zero) of tensors to concatenate along the
        specified axis.  These tensors must have the same shape along all dimensions other than this axis.

        :returns: a symbolic Variable.  It has the same ndim as the input tensors, and the most
        inclusive dtype.

        """
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        if not tensors:
            raise ValueError('Cannot join an empty list of tensors')
        as_tensor_variable_args= [as_tensor_variable(x) for x in tensors]
        dtypes = [x.type.dtype for x in as_tensor_variable_args]
        out_dtype = scal.upcast(*dtypes)

        if not all(targs.type.ndim for targs in as_tensor_variable_args):
            raise TypeError('Join cannot handle arguments of dimension 0. For joining scalar values, see @stack');

        # When the axis may vary, no dimension can be guaranteed to be
        # broadcastable.
        bcastable = [False] * len(as_tensor_variable_args[0].type.broadcastable)

        # When the axis is fixed, the broadcastable dimensions remain, except
        # for the axis dimension.
        # All concatenated elements must also have the same broadcastable
        # dimensions.
        if isinstance(axis, int):
            bcasts = [x.type.broadcastable[0:axis] + \
                      x.type.broadcastable[axis + 1:] for x in as_tensor_variable_args]
            if not all([bcasts[0] == bc for bc in bcasts[1:]]):
                raise ValueError('Dimensions other than the given axis must'
                    ' match', tensors)
            bcastable[:] = as_tensor_variable_args[0].type.broadcastable
            try:
                bcastable[axis] = False
            except IndexError, e:
                raise ValueError('Join argument "axis" is out of range (given input dimensions)')

        inputs = [as_tensor_variable(axis)] + as_tensor_variable_args
        if inputs[0].type not in int_types: 
            raise TypeError('Axis could not be cast to an integer type', axis, inputs[0].type, int_types)

        outputs = [tensor(dtype = out_dtype,
                          broadcastable = bcastable)]
        return Apply(self, inputs, outputs)

    def perform(self, node, axis_and_tensors, (out, )):
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        out[0] = numpy.asarray(numpy.concatenate(tensors, axis = axis),
                dtype=node.outputs[0].type.dtype)

    def grad(self, axis_and_tensors, (gz,)):
        """ The gradient wrt a join op is a `Split`, used to partition the gradient along the
        `axis` which was used for joining.
        """
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        if 'float' in tensors[0].dtype or 'complex' in tensors[0].dtype:
            # assume that this is differentiable
            split = Split(len(tensors))
            split_gz = split(gz, axis, stack(*[shape(x)[axis] for x in tensors]))
            return [None] + split_gz
        else:
            # assume that this isn't differentiable
            return [None] * (1 + len(tensors)) 

    def _native_grad(self, axis_and_tensors, (gz,)):
        """WRITEME"""
        axis, tensors = axis_and_tensors[0], axis_and_tensors[1:]
        sizes_along_axis = [shape(x)[axis] for x in tensors]
        n_dims = len(shape(tensors[0]))
        idx = [0]
        for s in sizes_along_axis:
            idx.append(idx[-1] + s)
        # The gradient w.r.t. the k-th tensor is a slice of gz along the
        # 'axis' dimension.
        return [gz[[slice(None)] * axis + [slice(idx[k], idx[k + 1])] + \
                [slice(None)] * (n_dims - axis - 1)] \
                for k in range(len(sizes_along_axis))]

    def vec_length(self, node):
        """Guess the length of a Join Variable"""
        assert isinstance(node.owner.op, Join)
        if node.ndim != 1:
            raise TypeError('argument must be symbolic vector')
        inputs = node.owner.inputs
        axis, tensors = inputs[0], inputs[1:]
        # if v is a vector, then axis must be 0
        # the question is whether all the inputs are broadcastable.
        if all(i.broadcastable[0] for i in tensors):
            return len(tensors)
        raise ValueError("could not determine vector length")

@_redefine_asRoutine(Join())
def join(axis, *tensors):
    """
    Convenience function to concatenate `TensorType`s along the given axis.

    :Parameters:
     - `tensors` : list of tensors (or list-like)
       A list of tensors to be concatenated along the given axis.
     - `axis` : int (symbolic or literal)
       On which dimension should the tensors be joined?  The `axis` must be a valid index into
       the shape of the tensors to be concatenated.
       The `axis` parameter may either be an integer or an object that can be converted to a
       scalar using `as_scalar`(`axis`). In the former case, the axis is fixed at construction,
       while in the latter it may vary over time depending on the value of the `axis` variable.

    The shapes of the tensors to be concatenated must be all identical, except in the dimension
    (`axis`) on which they are to be joined.

    """

pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, Join),
              printing.FunctionPrinter('join'))



@constructor
def shape_padleft(t, n_ones=1):
    """Reshape `t` by left-padding the shape with `n_ones` 1s
    
    See also: `shape_padright` and `Dimshuffle`
    """
    _t = as_tensor_variable(t)

    pattern = ['x']*n_ones + [i for i in range(_t.type.ndim)]
    return DimShuffle(_t.broadcastable, pattern)(_t)

@constructor
def shape_padright(t, n_ones=1):
    """Reshape `t` by right-padding the shape with `n_ones` 1s
    
    See also: `shape_padleft` and `Dimshuffle`
    """
    _t = as_tensor_variable(t)

    pattern = [i for i in range(_t.type.ndim)] + ['x']*n_ones
    return DimShuffle(_t.broadcastable, pattern)(_t)

@constructor
def stack(*tensors):
    """Insert the arguments as slices into a tensor of 1 rank greater.
    EXAMPLE
    """
    return join(0, *[shape_padleft(t, 1) for t in tensors])

@constructor
def concatenate(tensor_list, axis=0):
    """Alias for `join`(axis, *tensor_list).
    
    This function is similar to `join`, but uses the signature of numpy's concatenate function.

    This function 
    :Exceptions:
     - `TypeError` : the tensor_list must be a tuple or list

    """
    # Check someone did not make the common mistake to do something like:
    #   c = concatenate(x, y)
    # instead of
    #   c = concatenate((x, y))
    if not isinstance(tensor_list, (tuple, list)):
        raise TypeError("The 'tensors' argument must be either a tuple "
                "or a list, make sure you did not forget () or [] around "
                "arguments of concatenate.", tensor_list)
    return join(axis, *tensor_list)

def get_vector_length(v):
    """Return the run-time length of a symbolic vector.

    :Parameters:
     - `v` : A rank-1 TensorType variable.

    :Exceptions:
     - `TypeError` : `v` hasn't the proper type.
     - `ValueError` : No special case applies, the length is not known.
    
    In general this is not possible, but for a number of special cases the length can be
    determined at compile / graph-construction time.  This function implements these special
    cases.

    """
    v = as_tensor_variable(v)
    if v.ndim != 1:
        raise TypeError('argument must be symbolic vector')
    if isinstance(v, gof.Constant) and v.type.ndim == 1:
        return len(v.data)
    if v.owner and isinstance(v.owner.op, Join):
        try:
            return join.vec_length(v)
        except:
            pass
    if v.owner and v.owner.op == shape:
        return v.owner.inputs[0].type.ndim
    raise ValueError("length not known")

@constructor
def horizontal_stack(*args):
    """
    Horizontally stack two L{TensorType}s.
    Stack two L{TensorType}s along the second axis (column wise). These
    L{TensorType}s must have the same shape along all dimensions but the
    second.
    """
    assert len(args) >= 2
    for arg in args: assert arg.type.ndim == 2
    return concatenate(args, axis=1)

@constructor
def vertical_stack(*args):
    assert len(args) >= 2
    for arg in args: assert arg.type.ndim == 2
    return concatenate(args, axis=0)

if 0: #vertical and horizontal stacking are deprecated.  Better to use stack() and join().
    class VerticalStack(Op):
        """
        Vertically stack two L{TensorType}s.
        Stack two L{TensorType}s along the first axis (row wise). These
        L{TensorType}s must have the same shape along all dimensions but the
        first.

        @attention: Because we use vstack as the implementation, if the
        inputs have 1-dimension, the output will have 2-dimensions.
        """
        def make_node(self, x, y):
            x = as_tensor_variable(x)
            y = as_tensor_variable(y)
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

else:
    pass


class MakeVector(Op):
    """WRITEME"""
    def __init__(self, stype):
        self.stype = stype
    def make_node(self, *inputs):
        inputs = map(as_tensor_variable, inputs)
        assert all(a.type == self.stype for a in inputs)
        return Apply(self, inputs, [TensorType(broadcastable = (False,),
                                           dtype = self.stype.dtype)()])
    def perform(self, node, inputs, (out,)):
        out[0] = numpy.asarray(inputs)
    def grad(self, inputs, (gout,)):
        return [None]*len(inputs)

make_lvector = MakeVector(lscalar)
"""WRITEME"""


class MakeVectorPrinter:

    def process(self, r, pstate):
        if r.owner is None:
            raise TypeError("Can only print make_vector.")
        elif isinstance(r.owner.op, MakeVector):
            return "[%s]" % ", ".join(pstate.pprinter.process(input, pstate.clone(precedence = 1000)) for input in r.owner.inputs)
        else:
            raise TypeError("Can only print make_vector.")

pprint.assign(lambda pstate, r: r.owner and isinstance(r.owner.op, MakeVector), MakeVectorPrinter())


class Reshape(Op):
    """Perform a reshape operation of the input x to the new shape shp.
    The number of dimensions to which to reshape to (ndim) must be known at graph 
    build time."""
    view_map = {0: [0]} #output 0 is potentially aliased to inputs [0]
    def __init__(self, ndim):
        self.ndim = ndim
    def __eq__(self, other):
        return (type(other) is Reshape) and (other.ndim == self.ndim)
    def __hash__(self):
        return hash(Reshape) ^ hash(self.ndim)
    def make_node(self, x, shp):
        x = as_tensor_variable(x)
        shp = as_tensor_variable(shp)
        return gof.Apply(self, [x, shp], [tensor(x.type.dtype, [False]*self.ndim)])
    def perform(self, node, (x, shp), (out,)):
        if (len(shp) != self.ndim):
            raise ValueError('shape argument to Reshape.perform has incorrect length %i'
                    ', should be %i' % (len(shp), self.ndim), shp)
        try:
            out[0] = numpy.reshape(x, shp)
        except:
            raise ValueError('Cannot reshape input of shape %s to shape %s' % (x.shape,shp))
    def grad(self, (x, shp), (g_out,)):
        return [reshape(g_out, shape(x), ndim=x.ndim), None]

def reshape(x, newshape, ndim=None):
    if ndim is None:
        ndim = get_vector_length(newshape)
    op = Reshape(ndim)
    return op(x, newshape)


class Flatten(Op):
    """Flattens a tensor to `outdim` dimensions by preserving the leading outdim-1 shape
    components.
    """
    view_map = {0:[0]}
    def __init__(self, outdim=1):
        self.outdim = int(outdim)
    def __eq__(self, other):
        return type(self) == type(other) and self.outdim == other.outdim
    def __hash__(self):
        return hash(type(self))^hash(self.outdim)
    def make_node(self, x):
        t_x = as_tensor_variable(x)
        if self.outdim < 1 or (x.ndim and self.outdim > x.ndim):
            raise ValueError('invalid output ndimensions(%i) for tensor of rank %i' %(self.outdim, t_x.ndim))
        return gof.Apply(self, [t_x], [tensor(x.type.dtype, (False,)*self.outdim)])
    def perform(self, node, (x,), (out,)):
        outdim = self.outdim
        if outdim == 1:
            out[0] = x.reshape(x.size)
        elif outdim == len(x.shape):
            out[0] = x
        else:
            newshape = x.shape[:outdim-1] + (numpy.prod(x.shape[outdim-1:]),)
            #print 'newshape', newshape, x.shape, x.shape
            out[0] = x.reshape(newshape)
    def grad(self, (x,), (g_out,)):
        return [reshape(g_out, shape(x), x.ndim)]

def flatten(x, outdim=1): 
    return Flatten(outdim)(x)

class TileGrad(Op):
    """Calculates the gradient of the Tile Op"""
    #this is so weird, I can't think of how to make this a general thing.
    def make_node(self, x, reps, g_out):
        return gof.Apply(self, [x, reps, g_out], [x.type()])
    def perform(self, node, (x, reps, g_out), (gx,)):
        xsh = x.shape
        if len(reps)==2 and reps[1] == 1 and len(x.shape) == 1:
            gx[0] = numpy.sum(g_out, axis=0)
        else:
            raise NotImplementedError('x.shape, reps combination not supported',
                    (x.shape, reps))
tilegrad = TileGrad()


class Tile(Op):
    """Tiles its input according to reps. Reps is of same dimension as x
    and contains the number of times to tile x in each dimension"""
    def __init__(self, ndim):
        self.ndim = ndim
    def __eq__(self, other):
        return (type(other) is Tile) and (other.ndim == self.ndim)
    def __hash__(self):
        return hash(Tile) ^ hash(self.ndim)

    def make_node(self, x, reps):
        x = as_tensor_variable(x)
        reps = as_tensor_variable(reps)
        return gof.Apply(self, [x, reps], [tensor(x.type.dtype, [False,] * self.ndim)])
    def perform(self, node, (x, reps), (out,)):
        out[0] = numpy.tile(x, reps)
        if len(out[0].shape) != self.ndim:
            raise ValueError('Tile.perform produced incorrect shape')
    def grad(self, (x, reps), (g_out,)):
        return [tilegrad(x, reps, g_out), None]

def tile(x, reps, ndim=None):
    if not hasattr(tile, 'op'):
        tile.op = {}
    ndim = len(reps) if ndim is None else ndim #not sure if len(shp) is going to work.
    if ndim not in tile.op:
        tile.op[ndim] = Tile(ndim)
    return tile.op[ndim](x, reps)



#########################
# Linalg : Dot
#########################
#
# For BLAS-related ops see blas.py
#
# TODO: Dotinv should go here, Eigs, Svd, etc.

class Dot(Op):
    """Compute matrix-matrix, matrix-vector products and vector inner-products.

    """
    def make_node(self, *inputs):
        inputs = map(as_tensor_variable, inputs)

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
        try:
            z[0] = numpy.asarray(numpy.dot(x, y))
        except ValueError, e:
            # The error raised by numpy has no shape information, we mean to add that
            e.args = e.args + (x.shape, y.shape)
            raise

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
pprint.assign(dot, printing.OperatorPrinter(printing.special['middle_dot'], -1, 'left'))

#########################
# Linalg : TensorDot
#########################
class TensorDotGrad(Op):
    def __init__(self, axes):
        self.axes = axes;

    def __eq__(self, other):
        return type(self) == type(other) and self.axes == other.axes

    def __hash__(self):
        return hash(type(self)) ^ hash(self.axes) ^ 89234

    def make_node(self, x, y, gz):
        assert isinstance(x, Variable)
        assert isinstance(y, Variable)
        assert isinstance(gz, Variable)
        gx = x.type()
        gy = y.type()
        return Apply(self, [x,y,gz], [gx, gy])

    def perform(self, node, (x, y, gz), (gx,gy)):

        sum_over_y = range(y.ndim)
        [sum_over_y.remove(q) for q in self.axes[1]]
        sum_over_x = range(x.ndim)
        [sum_over_x.remove(q) for q in self.axes[0]]

        _gx = numpy.tensordot(gz, y, [range(x.ndim-len(self.axes[0]),gz.ndim), sum_over_y])
        idx = numpy.hstack((sum_over_x, self.axes[0]))
        newshapex = numpy.zeros(x.ndim)
        newshapex[[newpos for newpos in idx]] = [i for i in range(x.ndim)]
        gx[0] = numpy.transpose(_gx, newshapex)
        assert str(gx[0].dtype) == 'float64'

        _gy = numpy.tensordot(x, gz, [sum_over_x, range(x.ndim-len(self.axes[0]))])
        idy = numpy.hstack((self.axes[1], sum_over_y))
        newshapey = numpy.zeros(y.ndim)
        newshapey[[newpos for newpos in idy]] = [i for i in range(y.ndim)]
        gy[0] = numpy.transpose(_gy, newshapey)
        assert str(gy[0].dtype) == 'float64'

tensordot_grad = TensorDotGrad

class TensorDot(Op):
    """Compute tensor-tensor products over the given axes.
    See numpy documentation for details.
    (http://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html)

    """

    def __init__(self, axes):
        self.axes = axes;

    def __eq__(self, other):
        return type(self) == type(other) and self.axes == other.axes

    def __hash__(self):
        return hash(type(self)) ^ hash(self.axes) ^ 89234

    def make_node(self, x, y):

        axesdim = numpy.size(self.axes)/2
        x, y = map(as_tensor_variable, [x, y])

        if axesdim > x.type.ndim or axesdim > y.type.ndim:
            raise TypeError('Cannot sum over more dimensions than input. %i > %i,%i' %
                    axesdim, x.type.ndim, y.type.ndim)
       
        outdim = x.type.ndim + y.type.ndim - 2*axesdim
        output = tensor(dtype=x.dtype, broadcastable=[False]*outdim);
        return Apply(self, inputs=[x,y], outputs=[output,])

    def perform(self, node, (x, y), (z,)):
        try:
            z[0] = numpy.asarray(numpy.tensordot(x, y, self.axes))
            assert str(z[0].dtype) == 'float64'
        except ValueError, e:
            # The error raised by numpy has no shape information, we mean to add that
            e.args = e.args + (x.shape, y.shape, self.axes)
            raise

    def grad(self, (x, y), (gz,)):
        gx, gy = tensordot_grad(self.axes)(x, y, gz)
        return [gx, gy]
    
    def __str__(self):
        return "tensordot"
tensordot = TensorDot

class Outer(Op):
    """ Compute vector-vector outer product
    """
    def make_node(self, *inputs):
        inputs = map(as_tensor_variable, inputs)

        x, y = inputs
        nx = x.type.ndim
        ny = y.type.ndim

        if nx != 1: raise TypeError('non-vector arg0 to outer()', x)
        if ny != 1: raise TypeError('not-vector arg1 to outer()', y)
        
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

#########################
# Gradient
#########################

def grad(cost, wrt, g_cost=None, consider_constant=[]):
    """
    @type cost: L{Variable}
    @type wrt: L{Variable} or list of L{Variable}s.
    @type g_cost: L{Variable} broadcastable to size of I{cost}, or None
    @param g_cost: an expression for the gradient through cost.  The default is
        {{{ones_like(cost)}}}
    @param consider_constant: a list of expressions not to backpropagate through

    @rtype: L{Variable} or list of L{Variable}s (depending upon I{wrt})
    @return: symbolic expression of gradient of I{cost} with respect to I{wrt}.
    If I{wrt} is a list, then return a list containing the gradient of I{cost} wrt
    each element of the list.  If an element of I{wrt} is not differentiable
    with respect to the output, then a L{TensorConstant} with an appropriate
    kind of zero is returned.

    """
    if not isinstance(cost, TensorVariable):
        raise TypeError('In tensor.grad(), cost argument should be a TensorVariable.', cost)

    if g_cost is None:
        g_cost = ones_like(cost)
    inputs = gof.graph.inputs([cost])
    gmap = gradient.grad_sources_inputs([(cost, g_cost)], inputs + consider_constant)

    def zero(p):
        return TensorConstant(
                TensorType(dtype = p.type.dtype, broadcastable = []),
                numpy.asarray(0, dtype=p.type.dtype))

    #try:
        #it = iter(wrt)
    #except:
        #it = None

    #if it: #hasattr(wrt, '__iter__'): # isinstance(wrt, (list, tuple)):
    if isinstance(wrt, (list, tuple)):
        return [gmap.get(p, zero(p)) for p in wrt]
    else:
        return gmap.get(wrt, zero(wrt))

class numeric_grad:
    """WRITEME"""
    type_eps = {'float64': 1e-7,
            'float32': 3e-3}

    def __init__(self, f, pt, eps=None):
        """Return the gradient of f at pt.
        
        This function computes the gradient by a one-sided finite differences of a
        fixed step size (eps).
        
        It is assumed that f(...) will return a scalar.
        It is assumed that all f's inputs are numpy.ndarray objects.

        :param eps: the stepsize for the finite differencing.  None means input
        dtype-dependent. See `type_eps`.
        """

        def prod(inputs):
            rval = 1
            for i in inputs:
                rval *= i
            return rval

        packed_pt = False
        if not isinstance(pt, (list, tuple)):
            pt = [pt]
            packed_pt = True

        apt = [numpy.array(p) for p in pt]

        shapes = [p.shape for p in apt]
        dtypes = [str(p.dtype) for p in apt]

        # TODO: remove this eventually (why was this here in the first place ?)
        # In the case of CSM, the arguments are a mixture of floats and integers...
        #if not dtypes == [dtypes[0]] * len(apt):
            #raise TypeError('All function arguments must have same dtype')

        total_size = __builtin__.sum(prod(sh) for sh in shapes)

        working_dtype = __builtin__.min((self.type_eps[dt], dt) for dt in dtypes)[1]

        #create un-initialized memory
        x = numpy.ndarray((total_size,), dtype=working_dtype)
        gx = numpy.ndarray((total_size,), dtype=working_dtype)

        if eps is None:
            eps = __builtin__.max(self.type_eps[dt] for dt in dtypes)


        #set up aliases so that apt[i] is backed by memory in x
        # and self.gf is backed by memory in gx
        cur_pos = 0
        self.gf = []
        for i,p in enumerate(apt):
            p_size = prod(p.shape)
            # set up alias
            apt[i] = x[cur_pos:cur_pos+p_size].reshape(p.shape)
            self.gf.append(gx[cur_pos:cur_pos+p_size].reshape(p.shape))
            # initialize with p's value
            apt[i][:] = p
            cur_pos += p_size

        f_x = f(*[p.copy() for p in apt])

        # now iterate over the elements of x, and call f on apt.
        x_copy = x.copy()
        for i in xrange(total_size):
            x[:] = x_copy

            x[i] += eps
            f_eps = f(*apt)
            gx[i] = numpy.asarray((f_eps - f_x)/eps)

        if packed_pt:
            self.gf = self.gf[0]

    @staticmethod
    def abs_rel_err(a,b,eps=1.0e-10):
        """Return a small number when a and b are close, relative to how big they are"""
        return abs(a-b) / (abs(a)+abs(b)+eps)

    def max_err(self, g_pt):
        """Return the biggest relative error between g_pt and self.gf"""
        assert len(g_pt) == len(self.gf)
        errs = []
        for a, b in zip(g_pt, self.gf):
            errs.append(numpy.max(numeric_grad.abs_rel_err(a,b)))
        return numpy.max(errs), numpy.argmax(errs)


def verify_grad(op, pt, n_tests=2, rng=None, eps=None, tol=None, mode=None):
    """ WRITEME
    
    Raises an Exception if the difference between the analytic gradient and
    numerical gradient (computed through the Finite Difference Method) exceeds
    the given tolerance.

    :param op: something that behaves like an Op instance with a single output
               (can be a python function combining multiple ops)
    :param pt: the list of numpy.ndarrays to use as inputs to the op
    :param n_tests: number of times to run the test
    :param rng: random number generator from which to draw random samples
    :param eps: stepsize used in the Finite Difference Method (Default None is type-dependent)
    :param tol: relative tolerance used as threshold for gradient comparison
    
    """
    pt = [numpy.array(p) for p in pt]

    _type_tol = dict( # relativ error tolerances for different types
            float32=1e-2,
            float64=1e-4)

    if tol is None:
        tol = __builtin__.max(_type_tol[str(p.dtype)] for p in pt)

    if rng is None:
        rng = numpy.random
        unittest_tools.seed_rng()

    def function(inputs, output):
        if mode is None:
            f = compile.function(inputs, output, accept_inplace=True)
        else:
            f = compile.function(inputs, output, accept_inplace=True, mode=mode)
        return f

    for test_num in xrange(n_tests):

        tensor_pt = [value(p.copy(), name='input %i'%i) for i,p in enumerate(pt)]
        
        #op can be either a function or an actual Op instance
        o_output = op(*tensor_pt) 

        if isinstance(o_output,list) > 1:
            raise NotImplementedError('cant (yet) autotest gradient of op with multiple outputs')
            # we could make loop over outputs making random projections R for each,
            # but this doesn't handle the case where not all the outputs are
            # differentiable... so I leave this as TODO for now -JB.

        o_fn = function(tensor_pt, o_output)
        o_fn_out = o_fn(*[p.copy() for p in pt])

        random_projection = rng.rand(*o_fn_out.shape)
        t_r = as_tensor_variable(random_projection)

        #random projection of o onto t_r
        cost = sum(t_r * o_output)  #This sum() is defined above, it's not the builtin sum.
        cost_fn = function(tensor_pt, cost)

        num_grad = numeric_grad(cost_fn, [p.copy() for p in pt], eps)

        symbolic_grad = grad(cost, tensor_pt,as_tensor_variable(1.0,name='g_cost'))

        grad_fn = function(tensor_pt, symbolic_grad)

        analytic_grad = grad_fn(*[p.copy() for p in pt])

        if not isinstance(analytic_grad, (list, tuple)):
            analytic_grad = [analytic_grad]

        max_err, max_err_pos = num_grad.max_err(analytic_grad)
        if  max_err > tol:
            raise Exception(verify_grad.E_grad, (max_err, tol, max_err_pos))

verify_grad.E_grad = 'gradient error exceeded tolerance'
"""This error is raised when a gradient is calculated, but incorrect."""
