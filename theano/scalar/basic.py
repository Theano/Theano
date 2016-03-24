"""
.. warning::

This directory is for the internal of Theano.

You are strongly advised not to use it, except if you know
what you are doing!

If you want to use a scalar variable in a Theano graph,
you probably want to use theano.tensor.[c,z,f,d,b,w,i,l,]scalar!
"""
from __future__ import absolute_import, print_function, division

from itertools import chain
import math
import warnings
from copy import copy
from textwrap import dedent

import numpy
from six.moves import xrange

import theano
from theano.compat import imap, izip
from theano import gof, printing
from theano.gof import (Op, utils, Variable, Constant, Type, Apply,
                        FunctionGraph)
from functools import partial
from theano.configparser import config

from theano.gradient import DisconnectedType
from theano.gradient import grad_undefined

from theano.printing import pprint
import collections

builtin_complex = complex
builtin_int = int
builtin_float = float


class ComplexError(Exception):
    """
    Raised if complex numbers are used in an unsupported operation.

    """
    pass


class IntegerDivisionError(Exception):
    """
    Raised if someone tries to divide integers with '/' instead of '//'.

    """
    pass


def upcast(dtype, *dtypes):
    # This tries to keep data in floatX or lower precision, unless we
    # explicitely request a higher precision datatype.
    keep_float32 = [(config.cast_policy == 'numpy+floatX' and
                     config.floatX == 'float32')]
    keep_float16 = [(config.cast_policy == 'numpy+floatX' and
                     config.floatX == 'float16')]

    def make_array(dt):
        if dt == 'float64':
            # There is an explicit float64 dtype: we cannot keep float32.
            keep_float32[0] = False
            keep_float16[0] = False
        if dt == 'float32':
            keep_float16[0] = False
        return numpy.zeros((), dtype=dt)
    z = make_array(dtype)
    for dt in dtypes:
        z = z + make_array(dt=dt)
    rval = str(z.dtype)
    if rval == 'float64':
        if keep_float16[0]:
            return 'float16'
        if keep_float32[0]:
            return 'float32'
    elif rval == 'float32':
        if keep_float16[0]:
            return 'float16'
    return rval


def get_scalar_type(dtype):
    """
    Return a Scalar(dtype) object.

    This caches objects to save allocation and run time.

    """
    if dtype not in get_scalar_type.cache:
        get_scalar_type.cache[dtype] = Scalar(dtype=dtype)
    return get_scalar_type.cache[dtype]
get_scalar_type.cache = {}


def as_scalar(x, name=None):
    from ..tensor import TensorType, scalar_from_tensor
    if isinstance(x, gof.Apply):
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a multi-output"
                             " Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, Variable):
        if isinstance(x.type, Scalar):
            return x
        elif isinstance(x.type, TensorType) and x.ndim == 0:
            return scalar_from_tensor(x)
        else:
            raise TypeError("Variable type field must be a Scalar.", x, x.type)
    try:
        return constant(x)
    except TypeError:
        raise TypeError("Cannot convert %s to Scalar" % x, type(x))


def constant(x):
    # pass through numpy scalars, since they are already typed on
    # purpose typically.
    if hasattr(x, 'dtype'):
        assert x.ndim == 0
        return ScalarConstant(get_scalar_type(str(x.dtype)), x)
    if isinstance(x, builtin_float):
        for dtype in ['float32', 'float64']:
            x_ = theano._asarray(x, dtype=dtype)
            if numpy.all(x == x_):
                break
            x_ = None
        assert x_ is not None
        return ScalarConstant(get_scalar_type(str(x_.dtype)), x)
    if isinstance(x, builtin_int):
        for dtype in ['int8', 'int16', 'int32', 'int64']:
            x_ = theano._asarray(x, dtype=dtype)
            if numpy.all(x == x_):
                break
            x_ = None
        assert x_ is not None
        return ScalarConstant(get_scalar_type(str(x_.dtype)), x)
    if isinstance(x, builtin_complex):
        # TODO: We have added the complex type, so this should be tested
        raise NotImplementedError()
    raise TypeError(x)
    # return ScalarConstant(float64, float(x))


class Scalar(Type):

    """
    Internal class, should not be used by clients.

    Primarily used by tensor.elemwise and tensor.reduce.
    Analogous to TensorType, but for zero-dimensional objects.
    Maps directly to C primitives.

    TODO: refactor to be named ScalarType for consistency with TensorType.

    """

    ndim = 0

    def __init__(self, dtype):
        if dtype == 'floatX':
            dtype = config.floatX
        self.dtype = dtype
        self.dtype_specs()  # error checking

    @staticmethod
    def may_share_memory(a, b):
        # This class represent basic c type, represented in python
        # with numpy.scalar. They are read only. So from python, they
        # can never share memory.
        return False

    def filter(self, data, strict=False, allow_downcast=None):
        py_type = self.dtype_specs()[0]
        if strict and not isinstance(data, py_type):
            raise TypeError("%s expected a %s, got %s of type %s" % (
                self, py_type, data, type(data)), data)
        try:
            converted_data = py_type(data)
            if (allow_downcast or
                    (allow_downcast is None and
                        type(data) is float and
                        self.dtype == theano.config.floatX) or
                    data == converted_data):
                return py_type(data)
            else:
                raise TypeError('Value cannot accurately be converted to dtype'
                                ' (%s) and allow_downcast is not True' %
                                self.dtype)
        except Exception as e:
            raise TypeError("Could not convert %s (value=%s) to %s" % (
                type(data), data, self.dtype), e)

    def values_eq_approx(self, a, b, tolerance=1e-4):
        # The addition have risk of overflow especially with [u]int8
        diff = a - b
        if diff == 0:
            return True
        return abs(diff) <= (abs(a) * tolerance) + (abs(b) * tolerance)

    def c_headers(self, c_compiler):
        l = ['<math.h>']
        # These includes are needed by Scalar and TensorType,
        # we declare them here and they will be re-used by TensorType
        l.append('<numpy/arrayobject.h>')
        l.append('<numpy/arrayscalars.h>')
        if config.lib.amdlibm and c_compiler.supports_amdlibm:
            l += ['<amdlibm.h>']
        return l

    def c_libraries(self, c_compiler):
        l = []
        if config.lib.amdlibm and c_compiler.supports_amdlibm:
            l += ['amdlibm']
        return l

    def c_compile_args(self, c_compiler):
        if config.lib.amdlibm and c_compiler.supports_amdlibm:
            return ['-DREPLACE_WITH_AMDLIBM']
        else:
            return []

    def __eq__(self, other):
        return type(self) == type(other) and other.dtype == self.dtype

    def __hash__(self):
        return hash('theano.scalar.Scalar') ^ hash(self.dtype)

    def dtype_specs(self):
        try:
            # To help debug dtype/typenum problem, here is code to get
            # the list of numpy typenum.  This list change between 32
            # and 64 bit platform and probably also also between
            # Windows and Linux.
            # NOTE: equivalent type on a platform can have different typenum.
            #     This is the source of all dtype/typenum problem found up to
            #     now, as Theano always expect the exact typenum that
            #     correspond to our supported dtype.
            """
            for dtype in ['int8', 'uint8', 'short', 'ushort', 'intc', 'uintc',
                          'longlong', 'ulonglong', 'single', 'double',
                          'longdouble', 'csingle', 'cdouble', 'clongdouble',
                          'float32', 'float64', 'int8', 'int16', 'int32',
                          'int64', 'uint8', 'uint16', 'uint32', 'uint64',
                          'complex64', 'complex128', 'float', 'double',
                          'int', 'uint']:
                print(dtype, np.zeros(1, dtype=dtype).dtype.num)
            """
            return {  # dtype: (py_type, c_type, cls_name)
                'float16': (numpy.float16, 'npy_float16', 'Float16'),
                'float32': (numpy.float32, 'npy_float32', 'Float32'),
                'float64': (numpy.float64, 'npy_float64', 'Float64'),
                'complex128': (numpy.complex128, 'theano_complex128',
                               'Complex128'),
                'complex64': (numpy.complex64, 'theano_complex64', 'Complex64'),
                'uint8': (numpy.uint8, 'npy_uint8', 'UInt8'),
                'int8': (numpy.int8, 'npy_int8', 'Int8'),
                'uint16': (numpy.uint16, 'npy_uint16', 'UInt16'),
                'int16': (numpy.int16, 'npy_int16', 'Int16'),
                'uint32': (numpy.uint32, 'npy_uint32', 'UInt32'),
                'int32': (numpy.int32, 'npy_int32', 'Int32'),
                'uint64': (numpy.uint64, 'npy_uint64', 'UInt64'),
                'int64': (numpy.int64, 'npy_int64', 'Int64')
            }[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (
                self.__class__.__name__, self.dtype))

    def upcast(self, *others):
        return upcast(*[x.dtype for x in [self] + list(others)])

    def make_variable(self, name=None):
        return ScalarVariable(self, name=name)

    def __str__(self):
        return str(self.dtype)

    def __repr__(self):
        return "Scalar(%s)" % self.dtype

    def c_literal(self, data):
        if 'complex' in self.dtype:
            raise NotImplementedError("No literal for complex values.")
        return str(data)

    def c_declare(self, name, sub, check_input=True):
        if(check_input):
            pre = """
                typedef %(dtype)s %(name)s_dtype; // Deprecated use dtype_%(name)s instead.
                typedef %(dtype)s dtype_%(name)s;
            """ % dict(name=name, dtype=self.dtype_specs()[1])
        else:
            pre = ""
        return pre + """
        %(dtype)s %(name)s;
        """ % dict(name=name, dtype=self.dtype_specs()[1])

    def c_init(self, name, sub):
        return """
        %(name)s = 0;
        """ % locals()

    def c_extract(self, name, sub, check_input=True):
        specs = self.dtype_specs()
        if(check_input):
            pre = """
            if (!PyObject_TypeCheck(py_%(name)s, &%(pyarr_type)s))
            {
                PyErr_Format(PyExc_ValueError,
                    "Scalar check failed (%(dtype)s)");
                %(fail)s
            }
            """ % dict(sub,
                       name=name,
                       dtype=specs[1],
                       pyarr_type='Py%sArrType_Type' % specs[2])
        else:
            pre = ""
        return pre + """
        PyArray_ScalarAsCtype(py_%(name)s, &%(name)s);
        """ % dict(sub, name=name)

    def c_sync(self, name, sub):
        specs = self.dtype_specs()
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = PyArrayScalar_New(%(cls)s);
        if (!py_%(name)s)
        {
            Py_XINCREF(Py_None);
            py_%(name)s = Py_None;
            PyErr_Format(PyExc_MemoryError,
                "Instantiation of new Python scalar failed (%(dtype)s)");
            %(fail)s
        }
        PyArrayScalar_ASSIGN(py_%(name)s, %(cls)s, %(name)s);
        """ % dict(sub,
                   name=name,
                   dtype=specs[1],
                   cls=specs[2])

    def c_cleanup(self, name, sub):
        return ""

    def c_support_code(self):

        if self.dtype.startswith('complex'):
            cplx_types = ['theano_complex64', 'theano_complex128']
            real_types = ['npy_int8', 'npy_int16', 'npy_int32', 'npy_int64',
                          'npy_float32', 'npy_float64']
            # If the 'int' C type is not exactly the same as an existing
            # 'npy_intX', some C code may not compile, e.g. when assigning
            # the value 0 (cast to 'int' in C) to a theano_complex64.
            if (numpy.dtype('intc').num not in
                    [numpy.dtype(d[4:]).num for d in real_types]):
                # In that case we add the 'int' type to the real types.
                real_types.append('int')

            template = """
            struct theano_complex%(nbits)s : public npy_complex%(nbits)s
            {
                typedef theano_complex%(nbits)s complex_type;
                typedef npy_float%(half_nbits)s scalar_type;

                complex_type operator +(const complex_type &y) const {
                    complex_type ret;
                    ret.real = this->real + y.real;
                    ret.imag = this->imag + y.imag;
                    return ret;
                }

                complex_type operator -() const {
                    complex_type ret;
                    ret.real = -this->real;
                    ret.imag = -this->imag;
                    return ret;
                }
                bool operator ==(const complex_type &y) const {
                    return (this->real == y.real) && (this->imag == y.imag);
                }
                bool operator ==(const scalar_type &y) const {
                    return (this->real == y) && (this->imag == 0);
                }
                complex_type operator -(const complex_type &y) const {
                    complex_type ret;
                    ret.real = this->real - y.real;
                    ret.imag = this->imag - y.imag;
                    return ret;
                }
                complex_type operator *(const complex_type &y) const {
                    complex_type ret;
                    ret.real = this->real * y.real - this->imag * y.imag;
                    ret.imag = this->real * y.imag + this->imag * y.real;
                    return ret;
                }
                complex_type operator /(const complex_type &y) const {
                    complex_type ret;
                    scalar_type y_norm_square = y.real * y.real + y.imag * y.imag;
                    ret.real = (this->real * y.real + this->imag * y.imag) / y_norm_square;
                    ret.imag = (this->imag * y.real - this->real * y.imag) / y_norm_square;
                    return ret;
                }
                template <typename T>
                complex_type& operator =(const T& y);

                theano_complex%(nbits)s() {}

                template <typename T>
                theano_complex%(nbits)s(const T& y) { *this = y; }

                template <typename TR, typename TI>
                theano_complex%(nbits)s(const TR& r, const TI& i) { this->real=r; this->imag=i; }
            };
            """

            def operator_eq_real(mytype, othertype):
                return '''
                template <> %(mytype)s & %(mytype)s::operator=<%(othertype)s>(const %(othertype)s & y)
                { this->real=y; this->imag=0; return *this; }
                ''' % dict(mytype=mytype, othertype=othertype)

            def operator_eq_cplx(mytype, othertype):
                return '''
                template <> %(mytype)s & %(mytype)s::operator=<%(othertype)s>(const %(othertype)s & y)
                { this->real=y.real; this->imag=y.imag; return *this; }
                ''' % dict(mytype=mytype, othertype=othertype)

            operator_eq = (''.join(operator_eq_real(ctype, rtype)
                                   for ctype in cplx_types
                                   for rtype in real_types) +
                           ''.join(operator_eq_cplx(ctype1, ctype2)
                                   for ctype1 in cplx_types
                                   for ctype2 in cplx_types))

            # We are not using C++ generic templating here, because this would
            # generate two different functions for adding a complex64 and a
            # complex128, one returning a complex64, the other a complex128,
            # and the compiler complains it is ambiguous.
            # Instead, we generate code for known and safe types only.

            def operator_plus_real(mytype, othertype):
                return '''
                const %(mytype)s operator+(const %(mytype)s &x, const %(othertype)s &y)
                { return %(mytype)s(x.real+y, x.imag); }

                const %(mytype)s operator+(const %(othertype)s &y, const %(mytype)s &x)
                { return %(mytype)s(x.real+y, x.imag); }
                ''' % dict(mytype=mytype, othertype=othertype)

            operator_plus = ''.join(operator_plus_real(ctype, rtype)
                                    for ctype in cplx_types
                                    for rtype in real_types)

            def operator_minus_real(mytype, othertype):
                return '''
                const %(mytype)s operator-(const %(mytype)s &x, const %(othertype)s &y)
                { return %(mytype)s(x.real-y, x.imag); }

                const %(mytype)s operator-(const %(othertype)s &y, const %(mytype)s &x)
                { return %(mytype)s(y-x.real, -x.imag); }
                ''' % dict(mytype=mytype, othertype=othertype)

            operator_minus = ''.join(operator_minus_real(ctype, rtype)
                                     for ctype in cplx_types
                                     for rtype in real_types)

            def operator_mul_real(mytype, othertype):
                return '''
                const %(mytype)s operator*(const %(mytype)s &x, const %(othertype)s &y)
                { return %(mytype)s(x.real*y, x.imag*y); }

                const %(mytype)s operator*(const %(othertype)s &y, const %(mytype)s &x)
                { return %(mytype)s(x.real*y, x.imag*y); }
                ''' % dict(mytype=mytype, othertype=othertype)

            operator_mul = ''.join(operator_mul_real(ctype, rtype)
                                   for ctype in cplx_types
                                   for rtype in real_types)

            return (template % dict(nbits=64, half_nbits=32) +
                    template % dict(nbits=128, half_nbits=64) +
                    operator_eq +
                    operator_plus +
                    operator_minus +
                    operator_mul)

        else:
            return ""

    def c_init_code(self):
        return ["import_array();"]

    def c_code_cache_version(self):
        return (13, numpy.__version__)

    def get_shape_info(self, obj):
        return obj.itemsize

    def get_size(self, shape_info):
        return shape_info

# Register C code for ViewOp on Scalars.
theano.compile.register_view_op_c_code(
    Scalar,
    """
    %(oname)s = %(iname)s;
    """,
    1)


int8 = get_scalar_type('int8')
int16 = get_scalar_type('int16')
int32 = get_scalar_type('int32')
int64 = get_scalar_type('int64')
uint8 = get_scalar_type('uint8')
uint16 = get_scalar_type('uint16')
uint32 = get_scalar_type('uint32')
uint64 = get_scalar_type('uint64')
float16 = get_scalar_type('float16')
float32 = get_scalar_type('float32')
float64 = get_scalar_type('float64')
complex64 = get_scalar_type('complex64')
complex128 = get_scalar_type('complex128')

int_types = int8, int16, int32, int64
uint_types = uint8, uint16, uint32, uint64
float_types = float16, float32, float64
complex_types = complex64, complex128

discrete_types = int_types + uint_types
continuous_types = float_types + complex_types
all_types = discrete_types + continuous_types


class _scalar_py_operators:
    # So that we can simplify checking code when we have a mixture of Scalar
    # variables and Tensor variables
    ndim = 0

    dtype = property(lambda self: self.type.dtype)
    """The dtype of this scalar."""

    # UNARY
    def __abs__(self):
        return abs_(self)

    def __neg__(self):
        return neg(self)

    # CASTS
    # def __int__(self): return AsInt(self).out
    # def __float__(self): return AsDouble(self).out
    # def __complex__(self): return AsComplex(self).out

    # BITWISE
    def __invert__(self):
        return invert(self)

    def __and__(self, other):
        return and_(self, other)

    def __or__(self, other):
        return or_(self, other)

    def __xor__(self, other):
        return xor(self, other)

    def __rand__(self, other):
        return and_(other, self)

    def __ror__(self, other):
        return or_(other, self)

    def __rxor__(self, other):
        return xor(other, self)

    # COMPARISONS
    def __lt__(self, other):
        return lt(self, other)

    def __le__(self, other):
        return le(self, other)

    def __gt__(self, other):
        return gt(self, other)

    def __ge__(self, other):
        return ge(self, other)

    # ARITHMETIC - NORMAL
    def __add__(self, other):
        return add(self, other)

    def __sub__(self, other):
        return sub(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __truediv__(self, other):
        return div_proxy(self, other)

    def __div__(self, other):
        return div_proxy(self, other)

    def __floordiv__(self, other):
        return int_div(self, other)

    def __mod__(self, other):
        return mod_check(self, other)

    def __pow__(self, other):
        return pow(self, other)

    # ARITHMETIC - RIGHT-OPERAND
    def __radd__(self, other):
        return add(other, self)

    def __rsub__(self, other):
        return sub(other, self)

    def __rmul__(self, other):
        return mul(other, self)

    def __rdiv__(self, other):
        return div_proxy(other, self)

    def __rmod__(self, other):
        return mod(other, self)

    def __rpow__(self, other):
        return pow(other, self)

    def zeros_like(self, dtype=None):
        # The second is needed for Elemwise ops to work right
        if dtype is None:
            dtype = str(self.type.dtype)
        return second(self, ScalarConstant(get_scalar_type(dtype), 0))

    def astype(self, dtype):
        return cast(self, dtype)


class ScalarVariable(_scalar_py_operators, Variable):
    pass


class ScalarConstant(_scalar_py_operators, Constant):
    pass

# Register ScalarConstant as the type of Constant corresponding to Scalar
Scalar.Constant = ScalarConstant


# Easy constructors

def _multi(*fns):
    def f2(f, names):
        if len(names) == 1:
            return f(names)
        else:
            return [f(name) for name in names]
    if len(fns) == 1:
        return partial(f2, fns[0])
    else:
        return [partial(f2, f) for f in fns]

ints = _multi(int64)
floats = _multi(float64)
complexs = _multi(complex128)
complexs64 = _multi(complex64)
complexs128 = _multi(complex128)


# Using a class instead of a function makes it possible to deep-copy it in
# Python 2.4.
# Note that currently only a few functions use this mechanism, because it is
# enough to make the test-suite pass with Python 2.4. However, it may prove
# necessary to use this same mechanism in other places as well in the future.
class upcast_out(object):
    def __new__(self, *types):
        dtype = Scalar.upcast(*types)
        return get_scalar_type(dtype),


class upgrade_to_float(object):
    def __new__(self, *types):
        """
        Upgrade any int types to float32 or float64 to avoid losing precision.

        """
        conv = {int8: float32,
                int16: float32,
                int32: float64,
                int64: float64,
                uint8: float32,
                uint16: float32,
                uint32: float64,
                uint64: float64}
        return get_scalar_type(Scalar.upcast(*[conv.get(type, type)
                               for type in types])),


class same_out(object):
    def __new__(self, type):
        return type,


def upcast_out_no_complex(*types):
    if any([type in complex_types for type in types]):
        raise TypeError('complex type are not supported')
    return get_scalar_type(dtype=Scalar.upcast(*types)),


def same_out_float_only(type):
    if type not in float_types:
        raise TypeError('only float type are supported')
    return type,


class transfer_type(gof.utils.object2):
    def __init__(self, *transfer):
        assert all(type(x) in [int, str] or x is None for x in transfer)
        self.transfer = transfer

    def __str__(self):
        return 'transfer_type{%s}' % self.transfer

    def __call__(self, *types):
        upcast = upcast_out(*types)
        retval = []
        for i in self.transfer:
            if i is None:
                retval += [upcast]
            elif isinstance(i, str):
                retval += [i]
            else:
                retval += [types[i]]
        return retval
        # return [upcast if i is None else types[i] for i in self.transfer]

    def __eq__(self, other):
        return type(self) == type(other) and self.transfer == other.transfer

    def __hash__(self):
        return hash(self.transfer)


class specific_out(gof.utils.object2):
    def __init__(self, *spec):
        self.spec = spec

    def __call__(self, *types):
        return self.spec

    def __eq__(self, other):
        return type(self) == type(other) and self.spec == other.spec

    def __hash__(self):
        return hash(self.spec)


def int_out(*types):
    return int64,


def float_out(*types):
    return float64,


def upgrade_to_float_no_complex(*types):
    """
    Don't accept complex, otherwise call upgrade_to_float().

    """
    for type in types:
        if type in complex_types:
            raise TypeError('complex argument not supported')
    return upgrade_to_float(*types)


def same_out_nocomplex(type):
    if type in complex_types:
        raise TypeError('complex argument not supported')
    return type,


def int_out_nocomplex(*types):
    for type in types:
        if type in complex_types:
            raise TypeError('complex argument not supported')
    return int64,


def float_out_nocomplex(*types):
    for type in types:
        if type in complex_types:
            raise TypeError('complex argument not supported')
    return float64,


class unary_out_lookup(gof.utils.object2):
    """
    Get a output_types_preference object by passing a dictionary:

    unary_out_lookup({int8:int32, float32:complex128})

    The result is an op that maps in8 to int32 and float32 to
    complex128 and other input types lead to a TypeError.

    """
    def __init__(self, type_table):
        self.tbl = type_table

    def __call__(self, *types):
        if len(types) == 1:
            types = types[0]
        try:
            rval = self.tbl[types]
        except Exception:
            raise TypeError(types)
        if isinstance(types, (list, tuple)):
            return rval
        else:
            return [rval]

    def __eq__(self, other):
        return type(self) == type(other) and self.tbl == other.tbl

    def __hash__(self):
        return hash(type(self))  # ignore hash of table


def real_out(type):
    if type == complex64:
        return float32,
    if type == complex128:
        return float64,
    return type,


class ScalarOp(Op):

    nin = -1
    nout = 1

    def __init__(self, output_types_preference=None, name=None):
        self.name = name
        if output_types_preference is not None:
            if not isinstance(output_types_preference, collections.Callable):
                raise TypeError(
                    "Expected a callable for the 'output_types_preference' argument to %s. (got: %s)" %
                    (self.__class__, output_types_preference))
            self.output_types_preference = output_types_preference

    def make_node(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s.make_node (got %i(%s), expected %i)" %
                                (self, len(inputs), str(inputs), self.nin))
        inputs = [as_scalar(input) for input in inputs]
        outputs = [t() for t in self.output_types([input.type
                                                   for input in inputs])]
        if len(outputs) != self.nout:
            raise TypeError("Not the right number of outputs produced for %s(%s). Expected %s, got %s."
                            % (self, ", ".join(str(input) for input in inputs), self.nout, len(outputs)))
        return Apply(self, inputs, outputs)

    def output_types(self, types):
        if hasattr(self, 'output_types_preference'):
            variables = self.output_types_preference(*types)
            if not isinstance(variables, (list, tuple)) or any(not isinstance(x, Type) for x in variables):
                raise TypeError(
                    "output_types_preference should return a list or a tuple of types", self.output_types_preference, variables)
            if len(variables) != self.nout:
                raise TypeError("Not the right number of outputs types produced for %s(%s) by %s. Expected %s, got %s."
                                % (self, ", ".join(str(type) for type in variables),
                                   self.output_types_preference, self.nout, len(variables)))
            return variables
        else:
            raise NotImplementedError(
                "Cannot calculate the output types for %s" % self)

    def perform(self, node, inputs, output_storage):
        if self.nout == 1:
            output_storage[0][0] = self.impl(*inputs)
        else:
            variables = utils.from_return_values(self.impl(*inputs))
            assert len(variables) == len(output_storage)
            for storage, variable in zip(output_storage, variables):
                storage[0] = variable

    def impl(self, *inputs):
        raise utils.MethodNotDefined("impl", type(self),
                                     self.__class__.__name__)

    def grad(self, inputs, output_gradients):
        raise utils.MethodNotDefined("grad", type(self),
                                     self.__class__.__name__)

    def __eq__(self, other):
        test = (type(self) == type(other) and
                getattr(self, 'output_types_preference', None) ==
                getattr(other, 'output_types_preference', None))
        return test

    def __hash__(self):
        return hash(type(self).__name__) ^ hash(
            getattr(self, 'output_types_preference', 0))

    def __str__(self):
        if hasattr(self, 'name') and self.name:
            return self.name
        else:
            param = [(k, v) for k, v in self.__dict__.items()
                     if k not in ["name", "_op_use_c_code",
                                  "output_types_preference"]]
            if param:
                return "%s{%s}" % (self.__class__.__name__,
                                   ", ".join("%s=%s" % (k, v)
                                             for k, v in param))
            else:
                return self.__class__.__name__

    def c_code_cache_version(self):
        return (4,)

    def c_code_contiguous(self, node, name, inp, out, sub):
        """
        This function is called by Elemwise when all inputs and outputs are
        c_contiguous. This allows to use the SIMD version of this op.

        The inputs are the same as c_code except that:

            - inp and out must be the names of the variables associated to the
              ndarrays in the C code
            - node must be the elemwise node (this is needed to know
              the inputs/outputs types)

        """
        raise theano.gof.utils.MethodNotDefined()


class UnaryScalarOp(ScalarOp):
    nin = 1
    amd_float32 = None
    amd_float64 = None

    def c_code_contiguous(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if (not theano.config.lib.amdlibm or
                # We compare the dtype AND the broadcast flag
                # as this function do not broadcast
                node.inputs[0].type != node.outputs[0].type):
            raise theano.gof.utils.MethodNotDefined()

        dtype = node.inputs[0].type.dtype_specs()[1]
        fct_call = self.c_code_contiguous_raw(dtype, 'n', 'x', 'z')
        return """
{
        npy_intp n = PyArray_SIZE(%(z)s);
        %(dtype)s * x = (%(dtype)s*) PyArray_DATA(%(x)s);
        %(dtype)s * z = (%(dtype)s*) PyArray_DATA(%(z)s);
        %(fct_call)s;
}
        """ % locals()

    def c_code_contiguous_raw(self, dtype, n, i, o):
        if not config.lib.amdlibm:
            raise theano.gof.utils.MethodNotDefined()
        if dtype.startswith('npy_'):
            dtype = dtype[4:]
        if dtype == 'float32' and self.amd_float32 is not None:
            dtype = 'float'
            fct = self.amd_float32
        elif dtype == 'float64' and self.amd_float64 is not None:
            dtype = 'double'
            fct = self.amd_float64
        else:
            raise theano.gof.utils.MethodNotDefined()
        return "%(fct)s(%(n)s, %(i)s, %(o)s)" % locals()


class BinaryScalarOp(ScalarOp):
    # One may define in subclasses the following fields:
    #   - `identity`: for an associative operation, identity corresponds to
    #     the neutral element. For instance, it will be 0 for addition, 1 for
    #     multiplication, True for "and", False for "or".
    #   - `commutative`: whether op(a, b) == op(b, a)
    #   - `associative`: whether op(op(a, b), c) == op(a, op(b, c))
    nin = 2


###############
# Comparisons
###############

class LogicalComparison(BinaryScalarOp):
    def output_types(self, *input_dtypes):
        return [int8]

    def grad(self, inputs, output_gradients):
        x, y = inputs
        out = self(x, y)
        assert str(out.type.dtype).find('int') != -1
        return [x.zeros_like().astype(theano.config.floatX),
                y.zeros_like().astype(theano.config.floatX)]


class FixedLogicalComparison(UnaryScalarOp):
    """
    Comparison to a fixed value.

    """
    def output_types(self, *input_dtypes):
        return [int8]

    def grad(self, inputs, output_gradients):
        x, = inputs
        out = self(x)
        assert str(out.type.dtype).find('int') != -1
        return [x.zeros_like().astype(theano.config.floatX)]


class LT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ('less', 2, 1)

    def impl(self, x, y):
        # built-in < don't support complex
        return numpy.less(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s < %(y)s);" % locals()
lt = LT()


class GT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ('greater', 2, 1)

    def impl(self, x, y):
        # built-in > don't support complex
        return numpy.greater(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s > %(y)s);" % locals()
gt = GT()


class LE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ('less_equal', 2, 1)

    def impl(self, x, y):
        # built-in <= don't support complex
        return numpy.less_equal(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s <= %(y)s);" % locals()
le = LE()


class GE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    nfunc_spec = ('greater_equal', 2, 1)

    def impl(self, x, y):
        # built-in >= don't support complex
        return numpy.greater_equal(x, y)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s >= %(y)s);" % locals()
ge = GE()


class EQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    nfunc_spec = ('equal', 2, 1)

    def impl(self, x, y):
        return x == y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return "%(z)s = (%(x)s == %(y)s);" % locals()
eq = EQ()


class NEQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    nfunc_spec = ('not_equal', 2, 1)

    def impl(self, x, y):
        return x != y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s != %(y)s);" % locals()
neq = NEQ()


class IsNan(FixedLogicalComparison):
    nfunc_spec = ('isnan', 1, 1)

    def impl(self, x):
        return numpy.isnan(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        # Windows tries to be different and sometimes return -1, but we want
        # to be consistent with numpy (which returns True), hence the "abs".
        return "%(z)s = abs(isnan(%(x)s));" % locals()

    def c_code_cache_version(self):
        scalarop_version = super(IsNan, self).c_code_cache_version()
        return tuple(scalarop_version) + (2,)
isnan = IsNan()


class IsInf(FixedLogicalComparison):
    nfunc_spec = ('isinf', 1, 1)

    def impl(self, x):
        return numpy.isinf(x)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        # Note that the C isinf returns -1 for -Inf and +1 for +Inf, while
        # numpy simply returns True: we mimic numpy's behavior here, thus
        # the absolute value.
        return "%(z)s = abs(isinf(%(x)s));" % locals()
isinf = IsInf()


class InRange(LogicalComparison):
    nin = 3

    def __init__(self, openlow, openhi):
        self.openlow = openlow
        self.openhi = openhi

    def impl(self, x, low, hi):
        if self.openlow and x <= low:
            return False
        elif not self.openlow and x < low:
            return False
        if self.openhi and x >= hi:
            return False
        elif not self.openhi and x > hi:
            return False
        return True

    def c_code(self, node, name, inputs, outputs, sub):
        (x, low, hi) = inputs
        (z,) = outputs
        if self.openlow:
            cmp1 = '>'
        else:
            cmp1 = '>='

        # backport
        # cmp1 = '>' if self.openlow else '>='

        if self.openhi:
            cmp2 = '<'
        else:
            cmp2 = '<='

        # backport
        # cmp2 = '<' if self.openhi else '<='
        return ("%(z)s = %(x)s %(cmp1)s %(low)s &&"
                " %(x)s %(cmp2)s %(hi)s;" % locals())

    def get_grad(self, elem):
        if elem.type in complex_types:
            msg = ("No gradient implemented for complex numbers in "
                   "class scalar.basic.InRange")
            raise NotImplementedError(msg)
        elif elem.type in discrete_types:
            return elem.zeros_like().astype(theano.config.floatX)
        else:
            return elem.zeros_like()

    def grad(self, inputs, gout):
        (x, low, hi) = inputs
        (gz,) = gout
        grads = []
        for elem in [x, low, hi]:
            grads.append(self.get_grad(elem))
        return grads

inopenrange = InRange(True, True)
inclosedrange = InRange(False, False)


class Switch(ScalarOp):
    nin = 3
    nfunc_spec = ('where', 3, 1)

    def impl(self, cond, ift, iff):
        if cond:
            return ift
        else:
            return iff

            # backport
            # return ift if cond else iff
    def c_code(self, node, name, inputs, outputs, sub):
        (cond, ift, iff) = inputs
        (z,) = outputs
        return "%(z)s = %(cond)s ? %(ift)s : %(iff)s;" % locals()

    def grad(self, inputs, gout):
        (cond, ift, iff) = inputs
        (gz,) = gout
        first_part = switch(cond, gz, 0.)
        second_part = switch(cond, 0., gz)

        out = self(cond, ift, iff)
        if out.type.dtype in discrete_types:
            first_part = 0.
            second_part = 0.

        # cond does affect the elements of the output so it is connected.
        # For the sake of making the gradient convenient we assume that
        # condition + epsilon always triggers the same branch as condition
        condition_grad = cond.zeros_like().astype(theano.config.floatX)

        return (condition_grad, first_part, second_part)

    def output_types(self, types):
        (cond_t, ift_t, iff_t) = types
        return upcast_out(ift_t, iff_t)
switch = Switch()

####################
# BIT-WISE OPERATORS
####################


class UnaryBitOp(UnaryScalarOp):
    def output_types(self, *input_types):
        for i in input_types[0]:
            if i not in (int8, int16, int32, int64):
                raise TypeError('input to a BitOp must have type int8,'
                                ' int16, int32 or int64... not %s' % i)
        return upcast_out(*input_types[0])

    def grad(self, inputs, output_gradients):
        return [inputs[0].zeros_like().astype(theano.config.floatX)]


class BinaryBitOp(BinaryScalarOp):
    def output_types(self, *input_types):
        t0, t1 = input_types[0]
        for i in input_types[0]:
            if i not in (int8, int16, int32, int64):
                raise TypeError('input to a BitOp must have type int8,'
                                ' int16, int32 or int64... not %s' % i)
        return upcast_out(*input_types[0])

    def grad(self, inputs, output_gradients):
        a, b = inputs
        return [a.zeros_like().astype(theano.config.floatX),
                b.zeros_like().astype(theano.config.floatX)]


class OR(BinaryBitOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ('bitwise_or', 2, 1)

    def impl(self, x, y):
        return x | y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return "%(z)s = (%(x)s | %(y)s);" % locals()
or_ = OR()


class XOR(BinaryBitOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ('bitwise_xor', 2, 1)

    def impl(self, x, y):
        return x ^ y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return "%(z)s = (%(x)s ^ %(y)s);" % locals()
xor = XOR()


class AND(BinaryBitOp):
    identity = 1
    commutative = True
    associative = True
    nfunc_spec = ('bitwise_and', 2, 1)

    def impl(self, x, y):
        return x & y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return "%(z)s = (%(x)s & %(y)s);" % locals()
and_ = AND()


class Invert(UnaryBitOp):
    nfunc_spec = ('invert', 1, 1)

    def impl(self, x):
        return ~x

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = (~%(x)s);" % locals()
invert = Invert()


##############
# Arithmetic
##############
class Maximum(BinaryScalarOp):
    commutative = True
    associative = True
    nfunc_spec = ('maximum', 2, 1)

    def impl(self, *inputs):
        # The built-in max function don't support complex type
        return numpy.maximum(*inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if any([i.type in complex_types for i in node.inputs]):
            raise NotImplementedError()
        # Test for both y>x and x>=y to detect NaN
        return ('%(z)s = ((%(y)s)>(%(x)s)? (%(y)s): '
                '((%(x)s)>=(%(y)s)? (%(x)s): nan("")));' % locals())

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            # max is currently defined for complex_types,
            # but the gradient for complex is not.
            raise NotImplementedError()

        output = self(x, y)

        if output.type in discrete_types:
            return [x.zeros_like().astype(theano.config.floatX),
                    y.zeros_like().astype(theano.config.floatX)]

        gx = eq(output, x) * gz
        gy = eq(output, y) * gz
        return (gx, gy)

maximum = Maximum(upcast_out, name='maximum')


class Minimum(BinaryScalarOp):
    commutative = True
    associative = True
    nfunc_spec = ('minimum', 2, 1)

    def impl(self, *inputs):
        # The built-in min function don't support complex type
        return numpy.minimum(*inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if any([i.type in complex_types for i in node.inputs]):
            raise NotImplementedError()
        return ('%(z)s = ((%(y)s)<(%(x)s)? (%(y)s): '
                '((%(x)s)<=(%(y)s)? (%(x)s): nan("")));' % locals())

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            # min is currently defined for complex_types,
            # but the gradient for complex is not.
            raise NotImplementedError()

        output = minimum(x, y)
        if output.type in discrete_types:
            return [x.zeros_like().astype(theano.config.floatX),
                    y.zeros_like().astype(theano.config.floatX)]
        gx = eq(output, x) * gz
        gy = eq(output, y) * gz
        return (gx, gy)
minimum = Minimum(upcast_out, name='minimum')


class Add(ScalarOp):
    identity = 0
    commutative = True
    associative = True
    nfunc_spec = ('add', 2, 1)

    def impl(self, *inputs):
        return sum(inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (z,) = outputs
        if not inputs:
            return z + " = 0;"
        else:
            return z + " = " + " + ".join(inputs) + ";"

    def grad(self, inputs, gout):
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(*inputs).type in discrete_types:
            assert gz is not None
            retval = []
            for ii, inp in enumerate(inputs):
                if hasattr(inp, 'zeros_like'):
                    retval.append(
                        inp.zeros_like().astype(theano.config.floatX))
                else:
                    retval.append(grad_undefined(self, ii, inp))
        else:
            retval = []
            for i in inputs:
                    retval += [gz]
        return retval


add = Add(upcast_out, name='add')


class Mul(ScalarOp):
    identity = 1
    commutative = True
    associative = True
    nfunc_spec = ('multiply', 2, 1)

    def impl(self, *inputs):
        return numpy.product(inputs)

    def c_code(self, node, name, inputs, outputs, sub):
        (z,) = outputs
        if not inputs:
            return z + " = 1;"
        else:
            return z + " = " + " * ".join(inputs) + ";"

    def grad(self, inputs, gout):
        (gz,) = gout
        retval = []

        # The following 3 lines verify that gz is complex when the
        # output is complex. The rest of this function make this supposition.
        output_type = self.output_types([i.type for i in inputs])[0]
        if output_type in complex_types:
            if gz.type not in complex_types:
                raise TypeError(
                    'Mul with output_type ' + str(output_type) +
                    ' expected gz type to be complex, got gz with type ' +
                    str(gz.type))

        if output_type in discrete_types:
            return [ipt.zeros_like().astype(theano.config.floatX)
                    for ipt in inputs]

        for input in inputs:
            if gz.type in complex_types:
                # zr+zi = (xr + xi)(yr + yi)
                # zr+zi = (xr*yr - xi*yi) + (xr yi + xi yr )
                otherprod = mul(*(utils.difference(inputs, [input])))
                yr = real(otherprod)
                yi = imag(otherprod)
                if input.type in complex_types:
                    retval += [complex(yr * real(gz) + yi * imag(gz),
                                       yr * imag(gz) - yi * real(gz))]
                else:
                    retval += [yr * real(gz) + yi * imag(gz)]
            else:
                retval += [mul(*([gz] + utils.difference(inputs,
                                                         [input])))]
        return retval


mul = Mul(upcast_out, name='mul')


class Sub(BinaryScalarOp):
    nfunc_spec = ('subtract', 2, 1)

    def impl(self, x, y):
        return x - y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return "%(z)s = %(x)s - %(y)s;" % locals()

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()

        if (x - y).type in discrete_types:
            return [x.zeros_like().astype(theano.config.floatX),
                    y.zeros_like().astype(theano.config.floatX)]

        first_part = gz
        second_part = -gz

        return first_part, second_part
sub = Sub(upcast_out, name='sub')


def int_or_true_div(x_discrete, y_discrete):
    """
    Return 'int' or 'true' depending on the type of division used for x / y.

    Parameters
    ----------
    x_discrete : bool
        True if `x` is discrete ([unsigned] integer).
    y_discrete : bool
        True if `y` is discrete ([unsigned] integer).

    Returns
    -------
    str
        'int' if `x / y` should be an integer division, or `true` if it
        should be a true division.

    Raises
    ------
    IntegerDivisionError
        If both `x_discrete` and `y_discrete` are True and `config.int_division`
        is set to 'raise'.

    Notes
    -----
    This function is used by both scalar/basic.py and tensor/basic.py.

    """
    if (x_discrete and y_discrete):
        if config.int_division == 'raise':
            raise IntegerDivisionError(
                "With `config.int_division` set to 'raise', dividing two "
                "integer types with '/' is forbidden to avoid confusion "
                "between integer and floating point divisions. Please "
                "use // for integer division, or if you want a float result "
                "either cast one of the arguments to a float or directly call "
                "`x.__truediv__(y)`.")
        elif config.int_division == 'int':
            warnings.warn(
                "Division of two integer types with x / y is deprecated, "
                "please use x // y for an integer division.",
                DeprecationWarning,
                stacklevel=4)
            return int_div
        elif config.int_division == 'floatX':
            return true_div
        else:
            raise NotImplementedError(config.int_division)
    else:
        return true_div


def div_proxy(x, y):
    """
    Proxy for either true_div or int_div, depending on types of x, y.

    """
    f = int_or_true_div(as_scalar(x).type in discrete_types,
                        as_scalar(y).type in discrete_types)
    return f(x, y)


class TrueDiv(BinaryScalarOp):
    nfunc_spec = ('true_divide', 2, 1)

    def output_types(self, types):
        if all(t in discrete_types for t in types):
            return [get_scalar_type(config.floatX)]
        else:
            return super(TrueDiv, self).output_types(types)

    def impl(self, x, y):
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        if all(a.dtype in discrete_types for a in (x, y)):
            return numpy.sctypeDict[config.floatX](float(x) / y)
        else:
            return x / y

    def c_code(self, node, name, inputs, outputs, sub):
        # we generate good c code only when both are complex!
        (x, y) = inputs
        (z,) = outputs
        if sum([node.inputs[0].type in complex_types,
                node.inputs[1].type in complex_types]) == 1:
            raise NotImplementedError('type not supported', type)
        if (node.inputs[0].type in discrete_types and
                node.inputs[1].type in discrete_types):
            return "%(z)s = ((double)%(x)s) / %(y)s;" % locals()
        return "%(z)s = %(x)s / %(y)s;" % locals()

    def grad(self, inputs, gout):

        (x, y) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()

        # If the output of this op is discrete, then it
        # it is locally flat everywhere, so the gradient
        # through it is 0.
        # This is different from it not being connected
        # to the output; x/y is still a function of x
        # and y; it's just a step function.
        if all(a.dtype in discrete_types for a in (x, y)):
            return [x.zeros_like(), y.zeros_like()]

        first_part = gz / y

        if y.type in complex_types:
            raise NotImplementedError()

        second_part = -(gz * x) / (y * y)

        return first_part, second_part

true_div = TrueDiv(upcast_out, name='true_div')


class IntDiv(BinaryScalarOp):
    nfunc_spec = ('floor_divide', 2, 1)
    complex_error = ComplexError(
        "Theano does not support integer division (//) on "
        "complex numbers, since numpy deprecated it.")

    def impl(self, x, y):
        return x // y

    def c_support_code(self):
        # We use a macro as python use % as a special string character,
        # and the output of c_code may be run through another level
        # of string formatting.
        return "#define THEANO_MACRO_MOD(x,y) (x % y)"

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        t = node.inputs[0].type.upcast(*[i.type for i in node.inputs[1:]])
        if t in imap(str, discrete_types):
            x_div_y_pp = '(%(x)s / %(y)s)' % locals()
            x_div_y_mp = '((-%(x)s) / %(y)s)' % locals()
            x_mod_y_mp = 'THEANO_MACRO_MOD((-%(x)s), %(y)s)' % locals()
            x_div_y_pm = '(%(x)s / (-%(y)s))' % locals()
            x_mod_y_pm = 'THEANO_MACRO_MOD(%(x)s, (-%(y)s))' % locals()
            x_div_y_mm = '((-%(x)s) / (-%(y)s))' % locals()
        elif t in imap(str, float_types):
            # We need to call different functions of math.h
            # depending on the type
            if t == 'float32':
                floor = 'floorf'
                fmod = 'fmodf'
            elif t == 'float64':
                floor = 'floor'
                fmod = 'fmod'
            else:
                raise NotImplementedError('type not supported', t)

            x_div_y_pp = '%(floor)s(%(x)s / %(y)s)' % locals()
            x_div_y_mp = '%(floor)s((-%(x)s) / %(y)s)' % locals()
            x_mod_y_mp = '%(fmod)s((-%(x)s), %(y)s)' % locals()
            x_div_y_pm = '%(floor)s(%(x)s / (-%(y)s))' % locals()
            x_mod_y_pm = '%(fmod)s(%(x)s, (-%(y)s))' % locals()
            x_div_y_mm = '%(floor)s((-%(x)s) / (-%(y)s))' % locals()
        elif t in complex_types:
            raise self.complex_error
        else:
            raise NotImplementedError('type not supported', t)

        return dedent("""
            if (%(x)s < 0) {
                if (%(y)s < 0) {
                    %(z)s = %(x_div_y_mm)s;
                } else {
                    %(z)s = - %(x_div_y_mp)s - ((%(x_mod_y_mp)s == 0) ? 0 : 1);
                }
            } else {
                if (%(y)s < 0) {
                    %(z)s = - %(x_div_y_pm)s - ((%(x_mod_y_pm)s == 0) ? 0 : 1);
                } else {
                    %(z)s = %(x_div_y_pp)s;
                }
            }
            """) % locals()

    def c_code_cache_version(self):
        return (2,)

    def grad(self, inputs, g_output):
        return [inp.zeros_like(dtype=theano.config.floatX)
                for inp in inputs]
int_div = IntDiv(upcast_out, name='int_div')


floor_div = int_div


def mod_check(x, y):
    if (as_scalar(x).type in complex_types or
            as_scalar(y).type in complex_types):
        # Currently forbidden.
        raise Mod.complex_error
    else:
        return mod(x, y)


class Mod(BinaryScalarOp):
    nfunc_spec = ('mod', 2, 1)
    complex_error = ComplexError(
        "Theano does not support the mod operator (%) on "
        "complex numbers, since numpy deprecated it.")

    def impl(self, x, y):
        if isinstance(x, numpy.complex) or isinstance(y, numpy.complex):
            raise self.complex_error
        return x % y

    def c_code_cache_version(self):
        return (5,)

    def c_support_code(self):
        # We use a macro as python use % as a special string character,
        # and the output of c_code may be run through another level
        # of string formatting.
        return "#define THEANO_MACRO_MOD(x,y) (x % y)"

    def c_code(self, node, name, inputs, outputs, sub):
        """
        We want the result to have the same sign as Python, not the other
        implementation of mod.

        """
        (x, y) = inputs
        (z,) = outputs
        t = node.inputs[0].type.upcast(*[i.type for i in node.inputs[1:]])
        if (str(t) in imap(str, discrete_types) or
                t in ['uint8', 'int8', 'uint16', 'int16'] or
                t in ['uint32', 'int32', 'uint64', 'int64'] or
                t in discrete_types):
            # The above or's should not be needed anymore. However, for now we
            # keep them out of safety, and verify they are useless with an
            # assert.
            assert str(t) in imap(str, discrete_types)
            x_mod_y = "THEANO_MACRO_MOD(%(x)s, %(y)s)" % locals()
            x_mod_ymm = "THEANO_MACRO_MOD(-%(x)s, -%(y)s)" % locals()
            x_mod_ypm = "THEANO_MACRO_MOD(%(x)s, -%(y)s)" % locals()
            x_mod_ymp = "THEANO_MACRO_MOD(-%(x)s, %(y)s)" % locals()
        elif (str(t) in imap(str, float_types) or
              t in ['float32', 'float64'] or
              t in float_types):
            # The above or's should not be needed anymore. However, for now we
            # keep them out of safety, and verify they are useless with an
            # assert.
            assert str(t) in imap(str, float_types)
            x_mod_y = "fmod(%(x)s,%(y)s)" % locals()
            x_mod_ymm = "fmod(-%(x)s,-%(y)s)" % locals()
            x_mod_ypm = "fmod(%(x)s,-%(y)s)" % locals()
            x_mod_ymp = "fmod(-%(x)s,%(y)s)" % locals()
        elif str(t) in imap(str, complex_types):
            raise self.complex_error
        else:
            raise NotImplementedError('type not supported', t)

        return dedent("""
            if (%(x)s < 0){
               if (%(y)s < 0){
                  %(z)s = -(%(x_mod_ymm)s);
               }else{
                  %(z)s = - %(x_mod_ymp)s + (%(x_mod_ymp)s != 0 ? %(y)s : 0);
               }
            }else if (%(y)s < 0){
               %(z)s = (%(x_mod_ypm)s) + (%(x_mod_ypm)s != 0 ? %(y)s : 0);
            }else{
               %(z)s = %(x_mod_y)s;
            }
            """) % locals()

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        z = self(x, y)
        if z.type.dtype in discrete_types:
            # The gradient does not flow in if the output is discrete
            return [x.zeros_like(dtype=theano.config.floatX),
                    y.zeros_like(dtype=theano.config.floatX)]
        return [gz,
                -(x // y) * gz]

mod = Mod(upcast_out, name='mod')


class Pow(BinaryScalarOp):
    nfunc_spec = ('power', 2, 1)

    def impl(self, x, y):
        return x ** y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if (node.inputs[0].type in complex_types or
                node.inputs[1].type in complex_types):
            raise NotImplementedError('type not supported', type)
        return "%(z)s = pow(%(x)s, %(y)s);" % locals()

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()

        if self(x, y).type in discrete_types:
            return [x.zeros_like().astype(theano.config.floatX),
                    y.zeros_like().astype(theano.config.floatX)]

        first_part = gz * y * x ** (y - 1)

        second_part = gz * log(x) * x ** y
        second_part = switch(eq(x, 0), 0, second_part)

        return (first_part, second_part)

    def c_code_contiguous(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        if not theano.config.lib.amdlibm:
            raise theano.gof.utils.MethodNotDefined()

        # We compare the dtype AND the broadcast flag
        # as this function do not broadcast
        if (node.inputs[0].type == node.outputs[0].type and
                node.inputs[1].type == node.outputs[0].type and
                # amdlibm 3.0 do not have a float64 version of this SIMD function
                node.inputs[0].dtype == 'float32' and
                node.inputs[1].dtype == 'float32'):
            dtype = 'float'
            fct = "amd_vrsa_powf"
            return """
        npy_intp n = PyArray_SIZE(%(z)s);
        %(dtype)s * x = (%(dtype)s*) PyArray_DATA(%(x)s);
        %(dtype)s * y = (%(dtype)s*) PyArray_DATA(%(y)s);
        %(dtype)s * z = (%(dtype)s*) PyArray_DATA(%(z)s);
        %(fct)s(n, x, y, z);
        """ % locals()
        # We compare the dtype and check we broadcast a scalar
        elif (node.inputs[0].type == node.outputs[0].type and
              node.inputs[1].dtype == node.outputs[0].dtype and
              all(node.inputs[1].broadcastable) and
              # amdlibm 3.0 do not have a float64 version of this SIMD function
              node.inputs[0].dtype == 'float32' and
              node.inputs[1].dtype == 'float32'):
            dtype = 'float'
            fct = "amd_vrsa_powxf"
            return """
        npy_intp n = PyArray_SIZE(%(z)s);
        %(dtype)s * x = (%(dtype)s*) PyArray_DATA(%(x)s);
        %(dtype)s * y = (%(dtype)s*) PyArray_DATA(%(y)s);
        %(dtype)s * z = (%(dtype)s*) PyArray_DATA(%(z)s);
        %(fct)s(n, x, *y, z);
        """ % locals()

        raise theano.gof.utils.MethodNotDefined()


pow = Pow(upcast_out, name='pow')


class Clip(ScalarOp):
    nin = 3
    # The numpy.clip don't work correctly when the min is bigger then the max,
    # So we do not use nfunc_spec = ('clip', 3, 1)

    def impl(self, x, min, max):
        if x < min:
            return min
        elif x > max:
            return max
        else:
            return x

    def c_code(self, node, name, inputs, outputs, sub):
        (x, min, max) = inputs
        (z,) = outputs
        return "%(z)s = %(x)s < %(min)s ? %(min)s : %(x)s > %(max)s ? %(max)s : %(x)s;" % locals()

    def grad(self, inputs, gout):
        (x, mn, mx) = inputs
        (gz,) = gout
        assert gz.type not in complex_types
        gx = ((x >= mn) & (x <= mx)) * gz
        gmn = (x < mn) * gz
        gmx = (x > mx) * gz

        out = self(x, mn, mx)

        def handle_int(v):
            if out.type in int_types:
                return v.zeros_like().astype(config.floatX)
            return v

        return list(map(handle_int, [gx, gmn, gmx]))

# Don't allow complex even if numpy do
# As there is no mathematical reason for this function on complex
clip = Clip(upcast_out_no_complex, name='clip')


class Second(BinaryScalarOp):
    def impl(self, x, y):
        return y

    def c_code(self, node, name, inputs, outputs, sub):
        (x, y) = inputs
        (z,) = outputs
        return "%(z)s = %(y)s;" % locals()

    def connection_pattern(self, node):

        # x is never connected because its elements are never used
        # y is connected because its elements are copied over

        return [[False], [True]]

    def grad(self, inputs, gout):

        (x, y) = inputs
        (gz,) = gout
        if y.type in continuous_types:
            # x is disconnected because the elements of x are not used
            return DisconnectedType()(), gz
        else:
            # when y is discrete, we assume the function can be extended
            # to deal with real-valued inputs by rounding them to the
            # nearest integer. f(x+eps) thus equals f(x) so the gradient
            # is zero, not disconnected or undefined
            return DisconnectedType()(), y.zeros_like()

second = Second(transfer_type(1), name='second')


class Identity(UnaryScalarOp):
    def impl(self, input):
        return input

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = %(x)s;" % locals()

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in continuous_types:
            return gz,
        else:
            return x.zeros_like(dtype=theano.config.floatX),
identity = Identity(same_out, name='identity')


# CASTING OPERATIONS
class Cast(UnaryScalarOp):
    def __init__(self, o_type, name=None):
        if not isinstance(o_type, Scalar):
            raise TypeError(o_type)
        super(Cast, self).__init__(specific_out(o_type), name=name)
        self.o_type = o_type
        self.ctor = getattr(numpy, o_type.dtype)

    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.o_type.dtype)

    def make_new_inplace(self, output_types_preference=None, name=None):
        """
        This op.__init__ fct don't have the same parameter as other scalar op.
        This breaks the insert_inplace_optimizer optimization.
        This function is a fix to patch this, by ignoring the
        output_types_preference passed by the optimization, and replacing it
        by the current output type. This should only be triggered when
        both input and output have the same dtype anyway.

        """
        return self.__class__(self.o_type, name)

    def impl(self, input):
        return self.ctor(input)

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%s = (%s)%s;" % (z, node.outputs[0].type.dtype_specs()[1], x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if self.o_type in continuous_types:
            return [gz]
        else:
            return [x.zeros_like().astype(theano.config.floatX)]

    def c_code_cache_version(self):
        s = super(Cast, self).c_code_cache_version()
        if s:
            return (3,) + s
        else:
            return s

convert_to_int8 = Cast(int8, name='convert_to_int8')
convert_to_int16 = Cast(int16, name='convert_to_int16')
convert_to_int32 = Cast(int32, name='convert_to_int32')
convert_to_int64 = Cast(int64, name='convert_to_int64')
convert_to_uint8 = Cast(uint8, name='convert_to_uint8')
convert_to_uint16 = Cast(uint16, name='convert_to_uint16')
convert_to_uint32 = Cast(uint32, name='convert_to_uint32')
convert_to_uint64 = Cast(uint64, name='convert_to_uint64')
convert_to_float16 = Cast(float16, name='convert_to_float16')
convert_to_float32 = Cast(float32, name='convert_to_float32')
convert_to_float64 = Cast(float64, name='convert_to_float64')
convert_to_complex64 = Cast(complex64, name='convert_to_complex64')
convert_to_complex128 = Cast(complex128, name='convert_to_complex128')

_cast_mapping = {
    'int8': convert_to_int8,
    'int16': convert_to_int16,
    'int32': convert_to_int32,
    'int64': convert_to_int64,
    'uint8': convert_to_uint8,
    'uint16': convert_to_uint16,
    'uint32': convert_to_uint32,
    'uint64': convert_to_uint64,
    'float16': convert_to_float16,
    'float32': convert_to_float32,
    'float64': convert_to_float64,
    'complex64': convert_to_complex64,
    'complex128': convert_to_complex128}


def cast(x, dtype):
    """
    Symbolically cast `x` to a Scalar of given `dtype`.

    """
    if dtype == 'floatX':
        dtype = config.floatX

    _x = as_scalar(x)
    if _x.type.dtype == dtype:
        return _x
    if _x.type.dtype.startswith('complex') and not dtype.startswith('complex'):
        raise TypeError('Casting from complex to real is ambiguous: consider'
                        ' real(), imag(), angle() or abs()')
    return _cast_mapping[dtype](_x)


class Abs(UnaryScalarOp):
    nfunc_spec = ('abs', 1, 1)

    def make_node(self, x):
        inputs = [as_scalar(input) for input in [x]]
        if inputs[0].type == complex64:
            outputs = [float32()]
        elif inputs[0].type == complex128:
            outputs = [float64()]
        else:
            outputs = [t() for t in self.output_types(
                [input.type for input in inputs])]
        return Apply(self, inputs, outputs)

    def impl(self, x):
        return numpy.abs(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * x / abs(x),  # formula works for complex and real

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        type = node.inputs[0].type
        if type in int_types:
            return "%(z)s = abs(%(x)s);" % locals()
        if type in float_types:
            return "%(z)s = fabs(%(x)s);" % locals()
        if type in complex_types:
            return "%(z)s = sqrt(%(x)s.real*%(x)s.real + %(x)s.imag*%(x)s.imag);" % locals()
        raise NotImplementedError('type not supported', type)
abs_ = Abs(same_out)


class Sgn(UnaryScalarOp):
    nfunc_spec = ('sign', 1, 1)

    def impl(self, x):
        # casting to output type is handled by filter
        return numpy.sign(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        # casting is done by compiler
        # TODO: use copysign
        (x,) = inputs
        (z,) = outputs
        type = node.inputs[0].type
        if type in float_types:
            return '%(z)s = (%(x)s > 0) ? 1. : ((%(x)s < 0) ? -1. : (isnan(%(x)s) ? NAN : 0.));' % locals()
        if type in int_types:
            return "%(z)s = (%(x)s >= 0) ? (%(x)s == 0) ? 0 : 1 : -1;" % locals()
        raise TypeError()  # complex has no sgn

    def c_code_cache_version(self):
        s = super(Sgn, self).c_code_cache_version()
        if s:
            return (4,) + s
        else:  # if parent is unversioned, we are too
            return s
sgn = Sgn(same_out_nocomplex, name='sgn')


class Ceil(UnaryScalarOp):
    nfunc_spec = ('ceil', 1, 1)

    def impl(self, x):
        return numpy.ceil(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = ceil(%(x)s);" % locals()
ceil = Ceil(same_out_nocomplex, name='ceil')


class Floor(UnaryScalarOp):
    nfunc_spec = ('floor', 1, 1)

    def impl(self, x):
        return numpy.floor(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = floor(%(x)s);" % locals()
floor = Floor(same_out_nocomplex, name='floor')


class Trunc(UnaryScalarOp):
    nfunc_spec = ('trunc', 1, 1)

    def impl(self, x):
        return numpy.trunc(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        return [x.zeros_like().astype(theano.config.floatX)]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = %(x)s >= 0? floor(%(x)s): -floor(-%(x)s);" % locals()
trunc = Trunc(same_out_nocomplex, name='trunc')


class RoundHalfToEven(UnaryScalarOp):
    """
    This function implement the same rounding than numpy: Round half to even.

    c/c++ round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more details.

    """
    nfunc_spec = ('around', 1, 1)

    def impl(self, x):
        return numpy.round(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code___(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        typ = node.outputs[0].type.dtype
        if typ not in ['float32', 'float64']:
            Exception("The output should be float32 or float64")

        return dedent("""
            #ifndef ROUNDING_EPSILON
            #define ROUNDING_EPSILON 0.0000001
            #endif

            if (%(x)s < 0.0){
              // We implement the else part like that: -else( -%(x)s);
              %(typ)s i;
              std::modf( -%(x)s, &i );

              // If %(x)s is exactly halfway between two integers
              if ((-%(x)s -(i +0.5)) < epsilon){
                  // If 'i' is even then return 'i'
                if (std::fmod( i, 2.0 ) < epsilon){
                  %(z)s = - i;
                }else{
                  // Else return the nearest even integer
                  %(z)s = - ceil( i +0.5 );
                }
              }else{
                // round to closest
                %(z)s = - round(%(x)s+5);
              }
            }else{
              %(typ)s i;
              std::modf( %(x)s, &i );

              // If %(x)s is exactly halfway between two integers
              if ((%(x)s -(i +0.5)) < epsilon){
                  // If 'i' is even then return 'i'
                if (std::fmod( i, 2.0 ) < epsilon){
                  %(z)s = i;
                }else{
                  // Else return the nearest even integer
                  %(z)s =  ceil( i +0.5 );
                }
              }else{
                // round to closest
                %(z)s = round(%(x)s+5);
              }
            }

            #undef ROUNDING_EPSILON

            """ % locals())
round_half_to_even = RoundHalfToEven(same_out_float_only)


def round_half_away_from_zero_(a):
    if a > 0:
        return numpy.floor(a + 0.5)
    else:
        return numpy.ceil(a - 0.5)

round_half_away_from_zero_vec64 = numpy.vectorize(
    round_half_away_from_zero_,
    doc='round_half_away_from_zero_vec64')
round_half_away_from_zero_vec32 = numpy.vectorize(
    round_half_away_from_zero_,
    doc='round_half_away_from_zero_vec32',
    otypes=['float32'])


def round_half_away_from_zero_vec(a):
    if getattr(a, 'dtype', None) == numpy.float32:
        return round_half_away_from_zero_vec32(a)
    return round_half_away_from_zero_vec64(a)


class RoundHalfAwayFromZero(UnaryScalarOp):
    """
    Implement the same rounding algo as c round() fct.

    numpy.round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more details.

    """
    def impl(self, x):
        return round_half_away_from_zero_vec(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        rval = x.zeros_like()

        if rval.type.dtype in discrete_types:
            rval = rval.astype(theano.config.floatX)

        return [rval]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.outputs[0].type.dtype in ['float32', 'float64']:
            return "%(z)s = round(%(x)s);" % locals()
        else:
            Exception("The output should be float32 or float64")
round_half_away_from_zero = RoundHalfAwayFromZero(same_out_float_only)


class Neg(UnaryScalarOp):
    # We can use numpy.negative here, because even if it gives unexpected
    # results on Boolean arrays, it will be passed other dtypes as Theano
    # does not have a Boolean type for tensors.
    nfunc_spec = ('negative', 1, 1)

    def impl(self, x):
        # We have to make sure x is not a numpy.bool_, because
        # `-numpy.bool_(True)` is `False` (we want 0), and
        # `-numpy.bool_(False)` is `True` (we want 1).
        # This happens for Composite, as the intermediate results are not
        # casted in the dtype of the intermediate variable in general.
        if isinstance(x, numpy.bool_):
            x = numpy.int8(x)
        return -x

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return -gz,

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = -%(x)s;" % locals()
neg = Neg(same_out, name='neg')

pprint.assign(add, printing.OperatorPrinter('+', -2, 'either'))
pprint.assign(mul, printing.OperatorPrinter('*', -1, 'either'))
pprint.assign(sub, printing.OperatorPrinter('-', -2, 'left'))
pprint.assign(neg, printing.OperatorPrinter('-', 0, 'either'))
pprint.assign(true_div, printing.OperatorPrinter('/', -1, 'left'))
pprint.assign(int_div, printing.OperatorPrinter('//', -1, 'left'))
pprint.assign(pow, printing.OperatorPrinter('**', 1, 'right'))
pprint.assign(mod, printing.OperatorPrinter('%', -1, 'left'))


class Inv(UnaryScalarOp):
    """
    Multiplicative inverse. Also called reciprocal.

    """
    def impl(self, x):
        return numpy.float32(1.0) / x

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return -gz / (x * x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = 1.0 / %(x)s;" % locals()
inv = Inv(upgrade_to_float, name='inv')


class Log(UnaryScalarOp):
    """
    log base e.

    """
    nfunc_spec = ('log', 1, 1)
    amd_float32 = "amd_vrsa_logf"
    amd_float64 = "amd_vrda_log"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.log(x, sig='f')
        return numpy.log(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / x,

    def c_code(self, node, name, inputs, outputs, sub):
        # todo: the version using log2 seems to be very slightly faster
        # on some machines for some reason, check if it's worth switching
        # return "%(z)s = log2(%(x)s) * 0.69314718055994529;" % locals()
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log(%(x)s);" % locals()
log = Log(upgrade_to_float, name='log')


class Log2(UnaryScalarOp):
    """
    log base 2.

    """
    nfunc_spec = ('log2', 1, 1)
    amd_float32 = "amd_vrsa_log2f"
    amd_float64 = "amd_vrda_log2"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log2 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.log2(x, sig='f')
        return numpy.log2(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / (x * math.log(2.0)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log2(%(x)s);" % locals()
log2 = Log2(upgrade_to_float, name='log2')


class Log10(UnaryScalarOp):
    """
    log base 10.

    """
    nfunc_spec = ('log10', 1, 1)
    amd_float32 = "amd_vrsa_log10f"
    amd_float64 = "amd_vrda_log10"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log10 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.log10(x, sig='f')
        return numpy.log10(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / (x * numpy.log(10.0)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log10(%(x)s);" % locals()
log10 = Log10(upgrade_to_float, name='log10')


class Log1p(UnaryScalarOp):
    """
    log(1+x).

    """
    nfunc_spec = ('log1p', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.log1p will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.log1p(x, sig='f')
        return numpy.log1p(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz / (1 + x)]

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log1p(%(x)s);" % locals()
log1p = Log1p(upgrade_to_float, name='log1p')


class Exp(UnaryScalarOp):
    nfunc_spec = ('exp', 1, 1)
    amd_float32 = "amd_vrsa_expf"
    amd_float64 = "amd_vrda_exp"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.exp will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.exp(x, sig='f')
        return numpy.exp(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * exp(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = exp(%(x)s);" % locals()
exp = Exp(upgrade_to_float, name='exp')


class Exp2(UnaryScalarOp):
    nfunc_spec = ('exp2', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.exp2 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.exp2(x, sig='f')
        return numpy.exp2(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * exp2(x) * log(numpy.cast[x.type](2)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = exp2(%(x)s);" % locals()
exp2 = Exp2(upgrade_to_float, name='exp2')


class Expm1(UnaryScalarOp):
    nfunc_spec = ('expm1', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.expm1 will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.expm1(x, sig='f')
        return numpy.expm1(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * exp(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = expm1(%(x)s);" % locals()

    def c_code_cache_version(self):
        return (5,)
expm1 = Expm1(upgrade_to_float, name='expm1')


class Sqr(UnaryScalarOp):
    nfunc_spec = ('square', 1, 1)

    def impl(self, x):
        return x * x

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * x * 2,

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        return "%(z)s = %(x)s * %(x)s;" % locals()
sqr = Sqr(same_out, name='sqr')


class Sqrt(UnaryScalarOp):
    nfunc_spec = ('sqrt', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sqrt will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.sqrt(x, sig='f')
        return numpy.sqrt(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return (gz * 0.5) / sqrt(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = sqrt(%(x)s);" % locals()
sqrt = Sqrt(upgrade_to_float, name='sqrt')


class Deg2Rad(UnaryScalarOp):
    nfunc_spec = ('deg2rad', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.deg2rad will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.deg2rad(x, sig='f')
        return numpy.deg2rad(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * numpy.asarray(numpy.pi / 180, gz.type),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = %(x)s * (M_PI / 180.0);" % locals()
deg2rad = Deg2Rad(upgrade_to_float, name='deg2rad')


class Rad2Deg(UnaryScalarOp):
    nfunc_spec = ('rad2deg', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.rad2deg will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.rad2deg(x, sig='f')
        return numpy.rad2deg(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * numpy.asarray(180. / numpy.pi, gz.type),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = %(x)s * (180.0 / M_PI);" % locals()
rad2deg = Rad2Deg(upgrade_to_float, name='rad2deg')


class Cos(UnaryScalarOp):
    nfunc_spec = ('cos', 1, 1)
    amd_float32 = "amd_vrsa_cosf"
    amd_float64 = "amd_vrda_cos"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.cos will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.cos(x, sig='f')
        return numpy.cos(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return -gz * sin(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = cos(%(x)s);" % locals()
cos = Cos(upgrade_to_float, name='cos')


class ArcCos(UnaryScalarOp):
    nfunc_spec = ('arccos', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arccos will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.arccos(x, sig='f')
        return numpy.arccos(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return - gz / sqrt(numpy.cast[x.type](1) - sqr(x)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = acos(%(x)s);" % locals()
arccos = ArcCos(upgrade_to_float, name='arccos')


class Sin(UnaryScalarOp):
    nfunc_spec = ('sin', 1, 1)
    amd_float32 = "amd_vrsa_sinf"
    amd_float64 = "amd_vrda_sin"

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sin will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.sin(x, sig='f')
        return numpy.sin(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * cos(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = sin(%(x)s);" % locals()
sin = Sin(upgrade_to_float, name='sin')


class ArcSin(UnaryScalarOp):
    nfunc_spec = ('arcsin', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arcsin will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.arcsin(x, sig='f')
        return numpy.arcsin(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / sqrt(numpy.cast[x.type](1) - sqr(x)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = asin(%(x)s);" % locals()
arcsin = ArcSin(upgrade_to_float, name='arcsin')


class Tan(UnaryScalarOp):
    nfunc_spec = ('tan', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.tan will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.tan(x, sig='f')
        return numpy.tan(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / sqr(cos(x)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = tan(%(x)s);" % locals()
tan = Tan(upgrade_to_float, name='tan')


class ArcTan(UnaryScalarOp):
    nfunc_spec = ('arctan', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arctan will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.arctan(x, sig='f')
        return numpy.arctan(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / (numpy.cast[x.type](1) + sqr(x)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = atan(%(x)s);" % locals()
arctan = ArcTan(upgrade_to_float, name='arctan')


class ArcTan2(BinaryScalarOp):
    nfunc_spec = ('arctan2', 1, 1)

    def impl(self, y, x):
        # If x and y are int8 or uint8, numpy.arctan2 will compute the result
        # in half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            y_dtype = str(getattr(x, 'dtype', ''))
            if y_dtype in ('int8', 'uint8'):
                return numpy.arctan2(y, x, sig='f')
        return numpy.arctan2(y, x)

    def grad(self, inputs, gout):
        (y, x) = inputs
        (gz,) = gout
        if gz.type in complex_types:
            raise NotImplementedError()
        else:
            if self(x, y).type in discrete_types:
                if x.type in discrete_types:
                    gx = x.zeros_like(dtype=theano.config.floatX)
                else:
                    gx = x.zeros_like()
                if y.type in discrete_types:
                    gy = y.zeros_like(dtype=theano.config.floatX)
                else:
                    gy = y.zeros_like()
                return [gx, gy]

            # If the output is float, the gradient should flow,
            # even if the inputs are ints
            return [gz * x / (sqr(x) + sqr(y)),
                    gz * neg(y) / (sqr(x) + sqr(y))]

    def c_code(self, node, name, inputs, outputs, sub):
        (y, x) = inputs
        (z,) = outputs
        if (node.inputs[0].type in complex_types or
                node.inputs[1].type in complex_types):
            raise NotImplementedError('type not supported', type)
        return "%(z)s = atan2(%(y)s, %(x)s);" % locals()
arctan2 = ArcTan2(upgrade_to_float, name='arctan2')


class Cosh(UnaryScalarOp):
    """
    cosh(x) = (exp(x) + exp(-x)) / 2.

    """
    nfunc_spec = ('cosh', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.cosh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.cosh(x, sig='f')
        return numpy.cosh(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * sinh(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = cosh(%(x)s);" % locals()
cosh = Cosh(upgrade_to_float, name='cosh')


class ArcCosh(UnaryScalarOp):
    nfunc_spec = ('arccosh', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arccosh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.arccosh(x, sig='f')
        return numpy.arccosh(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / sqrt(sqr(x) - numpy.cast[x.type](1)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = acosh(%(x)s);" % locals()
arccosh = ArcCosh(upgrade_to_float, name='arccosh')


class Sinh(UnaryScalarOp):
    """
    sinh(x) = (exp(x) - exp(-x)) / 2.

    """
    nfunc_spec = ('sinh', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.sinh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.sinh(x, sig='f')
        return numpy.sinh(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * cosh(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = sinh(%(x)s);" % locals()
sinh = Sinh(upgrade_to_float, name='sinh')


class ArcSinh(UnaryScalarOp):
    nfunc_spec = ('arcsinh', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arcsinh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.arcsinh(x, sig='f')
        return numpy.arcsinh(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / sqrt(sqr(x) + numpy.cast[x.type](1)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = asinh(%(x)s);" % locals()
arcsinh = ArcSinh(upgrade_to_float, name='arcsinh')


class Tanh(UnaryScalarOp):
    """
    tanh(x) = sinh(x) / cosh(x)
            = (exp(2*x) - 1) / (exp(2*x) + 1).

    """
    nfunc_spec = ('tanh', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.tanh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.tanh(x, sig='f')
        return numpy.tanh(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * (1 - sqr(tanh(x))),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = tanh(%(x)s);" % locals()
tanh = Tanh(upgrade_to_float, name='tanh')


class ArcTanh(UnaryScalarOp):
    nfunc_spec = ('arctanh', 1, 1)

    def impl(self, x):
        # If x is an int8 or uint8, numpy.arctanh will compute the result in
        # half-precision (float16), where we want float32.
        x_dtype = str(getattr(x, 'dtype', ''))
        if x_dtype in ('int8', 'uint8'):
            return numpy.arctanh(x, sig='f')
        return numpy.arctanh(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz / (numpy.cast[x.type](1) - sqr(x)),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = atanh(%(x)s);" % locals()
arctanh = ArcTanh(upgrade_to_float, name='arctanh')


class Real(UnaryScalarOp):
    """
    Extract the real coordinate of a complex number.

    """
    # numpy.real(float32) return a view on the inputs.
    # nfunc_spec = ('real', 1, 1)

    def impl(self, x):
        return numpy.real(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        return [complex(gz, 0)]

real = Real(real_out, name='real')


class Imag(UnaryScalarOp):
    nfunc_spec = ('imag', 1, 1)

    def impl(self, x):
        return numpy.imag(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            return [complex(0, gz)]
        elif x.type in float_types:
            return [second(x, 0)]
        else:
            return [x.zeros_like(dtype=theano.config.floatX)]

imag = Imag(real_out, name='imag')


class Angle(UnaryScalarOp):
    nfunc_spec = ('angle', 1, 1)

    def impl(self, x):
        return numpy.angle(x)

    def grad(self, inputs, gout):
        # y = x.imag
        # r = sqrt(y**2 + x.real**2)
        # g = y/r
        # if x == 0 and y == 0:
        #     theta = 0
        # elif x >= 0:
        #     theta = numpy.arcsin(g)
        # else:
        #     theta = -numpy.arcsin(g)+numpy.pi

        (c,) = inputs
        (gtheta,) = gout
        x = real(c)
        y = imag(c)
        r = abs(c)

        gr = -gtheta * y / (r ** 2 * sqrt(1 - (y / r) ** 2))
        gx = gr * x / r
        gy = gr * y / r
        if c in complex_types:
            return [cast(complex(gx, gy), x.type.dtype)]
        elif c in float_types:
            return [cast(second(x, 0), x.type.dtype)]
        else:
            return [c.zeros_like(dtype=theano.config.floatX)]

angle = Angle(specific_out(float64), name='angle')


class Complex(BinaryScalarOp):
    @staticmethod
    def output_types_preference(x, y):
        if x in complex_types:
            raise TypeError(x)
        if y in complex_types:
            raise TypeError(y)

        up = Scalar.upcast(x, y)
        if up in ('float64', 'int64', 'uint64', 'int32', 'uint32'):
            return [complex128]
        else:
            return [complex64]

    def impl(self, x, y):
        return numpy.complex(x, y)

    def grad(self, inputs, gout):
        (x, y) = inputs
        (gz,) = gout
        return [cast(real(gz), x.type.dtype),
                cast(imag(gz), y.type.dtype)]
complex = Complex(name='complex')


class Conj(UnaryScalarOp):
    nfunc_spec = ('conj', 1, 1)

    def impl(self, x):
        return numpy.conj(x)
conj = Conj(same_out, name='conj')


class ComplexFromPolar(BinaryScalarOp):
    @staticmethod
    def output_types_preference(x, y):
        return Complex.output_types_preference(x, y)

    def impl(self, r, theta):
        if r < 0:
            raise ValueError('polar radius must be non-negative', r)
        x = r * numpy.cos(theta)
        y = r * numpy.sin(theta)
        if x.dtype == 'float32':
            return numpy.complex64(numpy.complex(x, y))
        else:
            return numpy.complex128(numpy.complex(x, y))

    def grad(self, inputs, gout):
        (r, theta) = inputs
        (gz,) = gout
        gr = gz * complex_from_polar(1, theta)
        gtheta = gz * complex_from_polar(r, -theta)
        return [gr, gtheta]
complex_from_polar = ComplexFromPolar(name='complex_from_polar')


class Composite(ScalarOp):
    """
    Composite is an Op that takes a graph of scalar operations and
    produces c code for the whole graph. Its purpose is to implement loop
    fusion.

    Composite depends on all the Ops in its graph having C code.

    """
    init_param = ('inputs', 'outputs')

    def __str__(self):
        return self.name

    def make_new_inplace(self, output_types_preference=None, name=None):
        """
        This op.__init__ fct don't have the same parameter as other scalar op.
        This break the insert_inplace_optimizer optimization.
        This fct allow fix patch this.

        """
        d = dict([(k, getattr(self, k)) for k in self.init_param])
        out = self.__class__(**d)
        if name:
            out.name = name
        else:
            name = out.name
        super(Composite, out).__init__(output_types_preference, name)
        return out

    def init_c_code(self):
        """
        Return the C code for this Composite Op.

        """
        subd = dict(chain(
            ((e, "%%(i%i)s" % i) for i, e in enumerate(self.fgraph.inputs)),
            ((e, "%%(o%i)s" % i) for i, e in enumerate(self.fgraph.outputs))))

        for var in self.fgraph.variables:
            if var.owner is None:
                if var not in self.fgraph.inputs:
                    # This is an orphan
                    if isinstance(var, Constant):
                        subd[var] = var.type.c_literal(var.data)
                    else:
                        raise ValueError(
                            "All orphans in the fgraph to Composite must"
                            " be Constant instances.")
            elif (any(i.dtype == 'float16' for i in var.owner.inputs) or
                  any(o.dtype == 'float16' for o in var.owner.outputs)):
                # flag for elemwise ops to check.
                self.inner_float16 = True

        _c_code = "{\n"
        self.nodenames = ["%(nodename)s_" + ('subnode%i' % j)
                          for j, n in enumerate(self.fgraph.toposort())]

        i = 0
        for j, node in enumerate(self.fgraph.toposort()):
            for output in node.outputs:
                if output not in subd:
                    i += 1
                    name = "V%%(id)s_tmp%i" % i
                    subd[output] = name
                    _c_code += "%s %s;\n" % (
                        output.type.dtype_specs()[1], name)
            s = node.op.c_code(
                node,
                self.nodenames[j],
                [subd[input] for input in node.inputs],
                [subd[output] for output in node.outputs],
                dict(fail="%(fail)s", id="%%(id)s_%i" % j))
            _c_code += s
            _c_code += "\n"
        _c_code += "}\n"
        self._c_code = _c_code

    def init_py_impls(self):
        """
        Return a list of functions that compute each output of self.

        """
        def compose_impl(r):
            # this is not optimal at all eg in add(*1 -> mul(x, y), *1)
            # it will calculate *1 twice
            # it also doesn't follow fgraph.toposort but that's (presumably)
            # still correct since we only have scalar ops
            if r in self.fgraph.inputs:
                idx = self.fgraph.inputs.index(r)
                return lambda inputs: inputs[idx]
            elif r.owner is None:  # in fgraph.orphans:
                return lambda inputs: r.data
            node = r.owner
            producers = [compose_impl(input) for input in node.inputs]

            def f(inputs):
                return node.op.impl(*[p(inputs) for p in producers])
            return f
        self._impls = [compose_impl(r) for r in self.fgraph.outputs]

    def init_name(self):
        """
        Return a readable string representation of self.fgraph.

        """
        try:
            rval = self.name
        except AttributeError:
            if 0:
                l = []
                for n in self.fgraph.toposort():
                    if hasattr(n.op, "name") and n.op.name is not None:
                        v = n.op.name
                        if v.startswith("Composite"):
                            v = v[len("Composite"):]
                    else:
                        v = n.op.__class__.__name__
                    l.append(v)
                rval = "Composite{" + ",".join(l) + "}"
            else:
                for i, r in enumerate(self.fgraph.inputs):
                    r.name = 'i%i' % i
                for i, r in enumerate(self.fgraph.outputs):
                    r.name = 'o%i' % i
                io = set(self.fgraph.inputs + self.fgraph.outputs)
                for i, r in enumerate(self.fgraph.variables):
                    if r not in io and len(r.clients) > 1:
                        r.name = 't%i' % i
                rval = "Composite{%s}" % ', '.join([pprint(output) for output
                                                    in self.fgraph.outputs])
        self.name = rval

    def init_fgraph(self):
        # The clone done by FunctionGraph is needed as we don't want
        # the fgraph to be set to the variable as we need to pickle
        # them for the cache of c module to work.
        fgraph = FunctionGraph(self.inputs, self.outputs)
        gof.MergeOptimizer().optimize(fgraph)
        for node in fgraph.apply_nodes:
            if not isinstance(node.op, ScalarOp):
                raise ValueError("The fgraph to Composite must be exclusively"
                                 " composed of ScalarOp instances.")
        self.fgraph = fgraph

    def __init__(self, inputs, outputs):
        # We need to clone the graph as sometimes its nodes already
        # contain a reference to an fgraph. As we want the Composite
        # to be pickable, we can't have reference to fgraph.

        # Also, if there is Composite in the inner graph, we want to
        # remove them. In that case, we do a more complicated clone
        # that will flatten Composite. We don't need to do this
        # recusively, as the way the fusion optimizer work, we have
        # only 1 new Composite each time at the output.
        for i in inputs:
            assert i not in outputs  # This isn't supported, use identity
        if len(outputs) > 1 or not any([isinstance(var.owner.op, Composite)
                                        for var in outputs]):
            # No inner Composite
            inputs, outputs = gof.graph.clone(inputs, outputs)
        else:
            # Inner Composite that we need to flatten
            assert len(outputs) == 1
            # 1. Create a new graph from inputs up to the
            # Composite
            res = theano.compile.rebuild_collect_shared(
                inputs=inputs,
                outputs=outputs[0].owner.inputs,
                copy_inputs_over=False)  # Clone also the inputs
            # 2. We continue this partial clone with the graph in
            # the inner Composite
            res2 = theano.compile.rebuild_collect_shared(
                inputs=outputs[0].owner.op.inputs,
                outputs=outputs[0].owner.op.outputs,
                replace=dict(izip(outputs[0].owner.op.inputs, res[1]))
            )
            assert len(res2[1]) == len(outputs)
            assert len(res[0]) == len(inputs)
            assert res[0] != inputs
            inputs, outputs = res[0], res2[1]
            # Next assert comment just for speed
            # assert not any([isinstance(node.op, Composite) for node in
            #                theano.gof.graph.ops(inputs, outputs)])

        self.inputs = copy(inputs)
        self.outputs = copy(outputs)
        self.inputs_type = tuple([input.type for input in inputs])
        self.outputs_type = tuple([output.type for output in outputs])
        self.nin = len(inputs)
        self.nout = len(outputs)
        self.init_fgraph()       # self.fgraph
        self.init_name()      # self.name
        self.init_c_code()    # self._c_code and self.nodenames
        self.init_py_impls()  # self._impls

    def output_types(self, input_types):
        if tuple(input_types) != self.inputs_type:
            raise TypeError("Wrong types for Composite. Expected %s, got %s."
                            % (self.inputs_type, tuple(input_types)))
        return self.outputs_type

    def make_node(self, *inputs):
        if (tuple([i.type for i in self.inputs]) ==
                tuple([i.type for i in inputs])):
            return super(Composite, self).make_node(*inputs)
        else:
            # Make a new op with the right input type.
            assert len(inputs) == self.nin
            res = theano.compile.rebuild_collect_shared(
                self.outputs,
                replace=dict(izip(self.inputs, inputs)),
                rebuild_strict=False)
            # After rebuild_collect_shared, the Variable in inputs
            # are not necessarily in the graph represented by res.
            # res[2][0] is a dict that map from the original variable to the
            # cloned variable.
            cloned_inputs = [res[2][0][i] for i in inputs]
            node = Composite(cloned_inputs, res[1]).make_node(*inputs)
            return node

    def perform(self, node, inputs, output_storage):
        for storage, impl in zip(output_storage, self._impls):
            storage[0] = impl(inputs)

    def impl(self, *inputs):
        output_storage = [[None] for i in xrange(self.nout)]
        self.perform(None, inputs, output_storage)
        ret = utils.to_return_values([storage[0] for storage in
                                      output_storage])
        if self.nout > 1:
            ret = tuple(ret)
        return ret

    def grad(self, inputs, output_grads):
        raise NotImplementedError("grad is not implemented for Composite")

    def c_code(self, node, nodename, inames, onames, sub):
        d = dict(chain(izip(("i%i" % i for i in xrange(len(inames))), inames),
                       izip(("o%i" % i for i in xrange(len(onames))),
                            onames)), **sub)
        d['nodename'] = nodename
        if 'id' not in sub:
            # The use of a dummy id is safe as the code is in a separate block.
            # It won't generate conflicting variable name.
            d['id'] = '_DUMMY_ID_'

        return self._c_code % d

    def c_code_cache_version(self):
        rval = [3]
        for x in self.fgraph.toposort():
            xv = x.op.c_code_cache_version()
            if xv:
                rval.append(xv)
            else:
                return ()
        return tuple(rval)

    def c_support_code(self):
        rval = []
        for subnode in self.fgraph.toposort():
            try:
                rval.append(subnode.op.c_support_code().strip())
            except gof.utils.MethodNotDefined:
                pass
        # remove duplicate code blocks
        return "\n".join(sorted(set(rval)))

    def c_support_code_apply(self, node, name):
        rval = []
        for subnode, subnodename in zip(self.fgraph.toposort(), self.nodenames):
            try:
                subnode_support_code = subnode.op.c_support_code_apply(
                    subnode,
                    subnodename % dict(nodename=name))
                if subnode_support_code:
                    rval.append(subnode_support_code)
            except gof.utils.MethodNotDefined:
                pass
        # there should be no need to remove duplicate code blocks because
        # each block should have been specialized for the given nodename.
        # Any block that isn't specialized should be returned via
        # c_support_code instead of c_support_code_apply.
        return "\n".join(rval)

    def __eq__(self, other):
        if self is other:
            return True
        if (type(self) != type(other) or
                self.nin != other.nin or
                self.nout != other.nout):
            return False
        # see __hash__ for comment on why there is no mention of fgraph
        # or module cache key here.
        return (self._c_code == other._c_code)

    def __hash__(self):
        rval = hash((type(self),
                    self.nin,
                    self.nout,
                    self._c_code))
        # Note that in general, the configparser settings at the time
        # of code generation (__init__) affect the semantics of this Op.
        # This function assumes that all relevant info about the configparser
        # is embodied in _c_code.  So the _c_code, rather than self.fgraph,
        # is the signature of the semantics of this Op.
        # _c_code is preserved through unpickling, so the Op will not change
        # semantics when it is reloaded with different configparser
        # settings.
        return rval

    def __getstate__(self):
        rval = dict(self.__dict__)
        del rval['_impls']
        del rval['fgraph']
        return rval

    def __setstate__(self, d):
        self.__dict__.update(d)
        # We must call init to set fgraph and _impls again, as otherwise
        # self.perform will not work.
        self.init_fgraph()
        self.init_py_impls()
        assert self._c_code
