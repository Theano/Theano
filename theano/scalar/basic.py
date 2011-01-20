import operator
import math
from copy import copy

import numpy, theano

from theano import gof
from theano.gof import Op, utils, Variable, Constant, Type, Apply, Env
from theano.gof.python25 import partial, all, any
from theano.configparser import config

builtin_complex = complex
builtin_int = int
builtin_float = float

def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return str(z.dtype)

def as_scalar(x, name = None):
    if isinstance(x, gof.Apply):
        if len(x.outputs) != 1:
            raise ValueError("It is ambiguous which output of a multi-output Op has to be fetched.", x)
        else:
            x = x.outputs[0]
    if isinstance(x, Variable):
        if not isinstance(x.type, Scalar):
            raise TypeError("Variable type field must be a Scalar.", x, x.type)
        return x
    try:
        return constant(x)
    except TypeError:
        raise TypeError("Cannot convert %s to Scalar" % x, type(x))

def constant(x):
    # pass through numpy scalars, since they are already typed on purpose typically.
    if hasattr(x,'dtype'):
        assert x.ndim==0
        return ScalarConstant(Scalar(str(x.dtype)), x)
    if isinstance(x, builtin_float):
        for dtype in ['float32', 'float64']:
            x_ = theano._asarray(x, dtype=dtype)
            if numpy.all(x == x_):
                break
            x_ = None
        assert x_ is not None
        return ScalarConstant(Scalar(str(x_.dtype)), x)
    if isinstance(x, builtin_int):
        for dtype in ['int8', 'int16', 'int32', 'int64']:
            x_ = theano._asarray(x, dtype=dtype)
            if numpy.all(x == x_):
                break
            x_ = None
        assert x_ is not None
        return ScalarConstant(Scalar(str(x_.dtype)), x)
    if isinstance(x, builtin_complex):
        #TODO: We have added the complex type, so this should be tested
        raise NotImplementedError()
    raise TypeError(x)
    #return ScalarConstant(float64, float(x))


class Scalar(Type):

    def __init__(self, dtype):
        if dtype == 'floatX':
            dtype = config.floatX
        self.dtype = dtype
        self.dtype_specs() # error checking

    def filter(self, data, strict=False, allow_downcast=None):
        py_type = self.dtype_specs()[0]
        if strict and not isinstance(data, py_type):
            raise TypeError("%s expected a %s, got %s of type %s" % (self, py_type, data,
                type(data)),
                    data)
        try:
            converted_data = py_type(data)
            if (allow_downcast or
                    (allow_downcast is None and
                        type(data) is float and
                        self.dtype==theano.config.floatX) or
                    data == converted_data):
                return py_type(data)
            else:
                raise TypeError('Value cannot accurately be converted to dtype (%s) and allow_downcast is not True' % self.dtype)
        except Exception, e:
            raise TypeError("Could not convert %s (value=%s) to %s" % (type(data), data, self.dtype), e)

    def values_eq_approx(self, a, b, tolerance = 1e-4):
        return abs(a - b) / (a+b) < tolerance

    def c_headers(self):
        l=['<math.h>']
        if config.lib.amdlibm:
            l+=['<amdlibm.h>']
        return l

    def c_libraries(self):
        l=[]
        if config.lib.amdlibm:
            l+=['amdlibm']
        return l

    def c_compile_args(self):
        if config.lib.amdlibm:
            return ['-DREPLACE_WITH_AMDLIBM']
        else: return []

    def __eq__(self, other):
        return type(self) == type(other) and other.dtype == self.dtype

    def __hash__(self):
        return hash('theano.scalar.Scalar') ^ hash(self.dtype)

    def dtype_specs(self):
        try:
            return {'float32': (numpy.float32, 'npy_float32', 'PyFloat_Check', 'PyFloat_AsDouble', 'PyFloat_FromDouble'),
                    'float64': (numpy.float64, 'npy_float64', 'PyFloat_Check', 'PyFloat_AsDouble', 'PyFloat_FromDouble'),
                    'complex128': (numpy.complex128, 'theano_complex128', 'PyComplex_Check', 'PyComplex_AsCComplex', 'PyComplex_FromCComplex'),
                    'complex64': (numpy.complex64, 'theano_complex64', None, None, None),
                    'uint8':  (numpy.uint8, 'npy_uint8', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int8':  (numpy.int8, 'npy_int8', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'uint16':  (numpy.uint16, 'npy_uint16', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int16': (numpy.int16, 'npy_int16', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'uint32':  (numpy.uint32, 'npy_uint32', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int32': (numpy.int32, 'npy_int32', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'uint64':  (numpy.uint64, 'npy_uint64', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int64': (numpy.int64, 'npy_int64', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong')
                    }[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (self.__class__.__name__, self.dtype))

    def upcast(self, *others):
        return upcast(*[x.dtype for x in [self]+list(others)])

    def make_variable(self, name = None):
        return ScalarVariable(self, name = name)

    def __str__(self):
        return str(self.dtype)

    def __repr__(self):
        return "Scalar(%s)" % self.dtype

    def c_literal(self, data):
        if 'complex' in self.dtype:
            raise NotImplementedError("No literal for complex values.")
        return str(data)

    def c_declare(self, name, sub):
        return """
        %(dtype)s %(name)s;
        typedef %(dtype)s %(name)s_dtype;
        """ % dict(name = name, dtype = self.dtype_specs()[1])

    def c_init(self, name, sub):
        return """
        %(name)s = 0;
        """ % locals()

    def c_extract(self, name, sub):
        specs = self.dtype_specs()
        #TODO: This is the wrong code, but we don't know what to change it to.
        # For example, a numpy.uint8 is not a PyInt, so PyInt_Check
        # is simply the wrong function to
        # call.
        # Look at PyArrayScalar api for how to cast to/from PyArrayScalar objects.
        # numpy.uint* numpy.float* are all constructors of PyArrayScalar objects.
        #
        return """
        if (!%(check)s(py_%(name)s))
        {
            PyErr_Format(PyExc_ValueError,
                "Scalar check failed");
            %(fail)s
        }
        %(name)s = (%(dtype)s)%(conv)s(py_%(name)s);
        """ % dict(sub,
                   name = name,
                   dtype = specs[1],
                   check = specs[2],
                   conv = specs[3])

    def c_sync(self, name, sub):
        specs = self.dtype_specs()
        return """
        Py_XDECREF(py_%(name)s);
        py_%(name)s = %(conv)s((%(dtype)s)%(name)s);
        if (!py_%(name)s)
            py_%(name)s = Py_None;
        """ % dict(name = name,
                   dtype = specs[1],
                   conv = specs[4])

    def c_cleanup(self, name, sub):
        return ""

    def c_support_code(self):

        if self.dtype.startswith('complex'):
            cplx_types = ['theano_complex64', 'theano_complex128']
            real_types = ['npy_int8', 'npy_int16', 'npy_int32', 'npy_int64', 'npy_float32', 'npy_float64']

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
                ''' % dict(mytype = mytype, othertype = othertype)

            def operator_eq_cplx(mytype, othertype):
                return '''
                template <> %(mytype)s & %(mytype)s::operator=<%(othertype)s>(const %(othertype)s & y)
                { this->real=y.real; this->imag=y.imag; return *this; }
                ''' % dict(mytype = mytype, othertype = othertype)

            operator_eq = ''.join(operator_eq_real(ctype, rtype)
                                for ctype in cplx_types
                                for rtype in real_types) \
                        + ''.join(operator_eq_cplx(ctype1, ctype2)
                                for ctype1 in cplx_types
                                for ctype2 in cplx_types)

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
                ''' % dict(mytype = mytype, othertype = othertype)

            operator_plus = ''.join(operator_plus_real(ctype, rtype)
                                for ctype in cplx_types
                                for rtype in real_types)

            def operator_minus_real(mytype, othertype):
                return '''
                const %(mytype)s operator-(const %(mytype)s &x, const %(othertype)s &y)
                { return %(mytype)s(x.real-y, x.imag); }

                const %(mytype)s operator-(const %(othertype)s &y, const %(mytype)s &x)
                { return %(mytype)s(y-x.real, -x.imag); }
                ''' % dict(mytype = mytype, othertype = othertype)

            operator_minus = ''.join(operator_minus_real(ctype, rtype)
                                for ctype in cplx_types
                                for rtype in real_types)

            def operator_mul_real(mytype, othertype):
                return '''
                const %(mytype)s operator*(const %(mytype)s &x, const %(othertype)s &y)
                { return %(mytype)s(x.real*y, x.imag*y); }

                const %(mytype)s operator*(const %(othertype)s &y, const %(mytype)s &x)
                { return %(mytype)s(x.real*y, x.imag*y); }
                ''' % dict(mytype = mytype, othertype = othertype)

            operator_mul = ''.join(operator_mul_real(ctype, rtype)
                                for ctype in cplx_types
                                for rtype in real_types)

            return template % dict(nbits = 64, half_nbits = 32) \
                    + template % dict(nbits = 128, half_nbits = 64) \
                    + operator_eq \
                    + operator_plus \
                    + operator_minus \
                    + operator_mul

        else:
            return ""

    def c_code_cache_version(self):
        return (9, numpy.__version__) # Make operators work with 64 and 128 arguments at the same time
        return (8, numpy.__version__) # put const around operators and added unary '-' operator
        # no need to put lib.amdlibm here as c_compile_args() are put in the key.
        return (7,)  # make complex c code optional
        return (6,)  # added implemeentations of operators that work with scalar arguments
        return (5,)  #added constructors to theano_complex class
        return (4,)  #explicit T given in specialization of operator= lines.  This makes it compile with open64


int8 = Scalar('int8')
int16 = Scalar('int16')
int32 = Scalar('int32')
int64 = Scalar('int64')
uint8 = Scalar('uint8')
uint16 = Scalar('uint16')
uint32 = Scalar('uint32')
uint64 = Scalar('uint64')
float32 = Scalar('float32')
float64 = Scalar('float64')
complex64 = Scalar('complex64')
complex128 = Scalar('complex128')

int_types = int8, int16, int32, int64
uint_types = uint8, uint16, uint32, uint64
float_types = float32, float64
complex_types = complex64, complex128

continuous_types = float_types + complex_types

class _scalar_py_operators:

    #UNARY
    def __abs__(self): return abs_(self)
    def __neg__(self): return neg(self)

    #CASTS
    #def __int__(self): return AsInt(self).out
    #def __float__(self): return AsDouble(self).out
    #def __complex__(self): return AsComplex(self).out

    #BITWISE
    def __invert__(self): return invert(self)
    def __and__(self,other): return and_(self, other)
    def __or__(self,other): return or_(self, other)
    def __xor__(self,other): return xor(self, other)
    def __rand__(self,other): return and_(other,self)
    def __ror__(self,other): return or_(other, self)
    def __rxor__(self,other): return xor(other, self)

    #COMPARISONS
    def __lt__(self,other): return lt(self, other)
    def __le__(self,other): return le(self, other)
    def __gt__(self,other): return gt(self, other)
    def __ge__(self,other): return ge(self, other)

    #ARITHMETIC - NORMAL
    def __add__(self,other): return add(self,other)
    def __sub__(self,other): return sub(self,other)
    def __mul__(self,other): return mul(self,other)
    def __div__(self,other): return div_proxy(self,other)
    def __mod__(self,other): return mod(self,other)
    def __pow__(self,other): return pow(self,other)

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div_proxy(other,self)
    def __rmod__(self,other): return mod(other,self)
    def __rpow__(self,other): return pow(other,self)

class ScalarVariable(_scalar_py_operators, Variable):
    pass

class ScalarConstant(_scalar_py_operators, Constant):
    pass


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




def upcast_out(*types):
    return Scalar(dtype = Scalar.upcast(*types)),
def upcast_out_no_complex(*types):
    if any([type in complex_types for type in types]):
        raise TypeError('complex type are not supported')
    return Scalar(dtype = Scalar.upcast(*types)),
def same_out(type):
    return type,
def same_out_float_only(type):
    if type not in float_types:
        raise TypeError('only float type are supported')
    return type,

class transfer_type(gof.utils.object2):
    def __init__(self, *transfer):
        assert all(type(x) == int for x in transfer)
        self.transfer = transfer
    def __str__(self):
        return 'transfer_type{%s}'%self.transfer
    def __call__(self, *types):
        upcast = upcast_out(*types)
        retval = []
        for i in self.transfer:
            if i is None:
                retval += [upcast]
            else:
                retval += [types[i]]
        return retval
        #return [upcast if i is None else types[i] for i in self.transfer]
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
def upgrade_to_float(*types):
    """
    Upgrade any int types to float32 or float64 to avoid losing any precision.
    """
    conv = {int8: float32,
            int16: float32,
            int32: float64,
            int64: float64}
    return Scalar(Scalar.upcast(*[conv.get(type, type) for type in types])),
def upgrade_to_float_no_complex(*types):
    """
    don't accept complex, otherwise call upgrade_to_float().
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
    get a output_types_preference object by passing a dictionary:

    unary_out_lookup({int8:int32, float32:complex128})

    The result is an op that maps in8 to int32 and float32 to complex128 and other input types
    lead to a TypeError.
    """
    def __init__(self, type_table):
        self.tbl = type_table
    def __call__(self, *types):
        if len(types) == 1:
            types = types[0]
        try:
            rval = self.tbl[types]
        except:
            raise TypeError(types)
        if isinstance(types, (list, tuple)):
            return rval
        else:
            return [rval]
    def __eq__(self, other):
        return type(self) == type(other) and self.tbl == other.tbl
    def __hash__(self):
        return hash(type(self)) # ignore hash of table

def real_out(type):
    if type == complex64:
        return float32,
    if type == complex128:
        return float64,
    return type,

class ScalarOp(Op):

    nin = -1
    nout = 1

    def __init__(self, output_types_preference = None, name = None):
        self.name = name
        if output_types_preference is not None:
            if not callable(output_types_preference):
                raise TypeError("Expected a callable for the 'output_types_preference' argument to %s. (got: %s)" % (self.__class__, output_types_preference))
            self.output_types_preference = output_types_preference

    def make_node(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s.make_node (got %i(%s), expected %i)" \
                                    % (self, len(inputs), str(inputs), self.nin))
        inputs = [as_scalar(input) for input in inputs]
        outputs = [t() for t in self.output_types([input.type for input in inputs])]
        if len(outputs) != self.nout:
            raise TypeError("Not the right number of outputs produced for %s(%s). Expected %s, got %s."
                            % (self, ", ".join(str(input) for input in inputs), self.nout, len(outputs)))
        return Apply(self, inputs, outputs)

    def output_types(self, types):
        if hasattr(self, 'output_types_preference'):
            variables = self.output_types_preference(*types)
            if not isinstance(variables, (list, tuple)) or any(not isinstance(x, Type) for x in variables):
                raise TypeError("output_types_preference should return a list or a tuple of types", self.output_types_preference, variables)
            if len(variables) != self.nout:
                raise TypeError("Not the right number of outputs produced for %s(%s) by %s. Expected %s, got ?s."
                                % (self, ", ".join(str(input.type) for input in inputs),
                                   self.output_types_preference, self.nout, len(variables)))
            return variables
        else:
            raise NotImplementedError("Cannot calculate the output types for %s" % self)

    def perform(self, node, inputs, output_storage):
        if self.nout == 1:
            output_storage[0][0] = self.impl(*inputs)
        else:
            variables = utils.from_return_values(self.impl(*inputs))
            assert len(variables) == len(output_storage)
            for storage, variable in zip(output_storage, variables):
                storage[0] = variable

    def impl(self, *inputs):
        raise utils.MethodNotDefined("impl", type(self), self.__class__.__name__)

    def grad(self, inputs, output_gradients):
        raise utils.MethodNotDefined("grad", type(self), self.__class__.__name__)

    def __eq__(self, other):
        test =  type(self) == type(other) \
            and getattr(self, 'output_types_preference', None) \
            == getattr(other, 'output_types_preference', None)
        return test

    def __hash__(self):
        return hash(type(self).__name__) ^ hash(getattr(self, 'output_types_preference', 0))

    def __str__(self):
        if hasattr(self, 'name') and self.name:
            return self.name
        else:
            return "%s{%s}" % (self.__class__.__name__, ", ".join("%s=%s" % (k, v) for k, v in self.__dict__.items() if k != "name"))

    def c_code_cache_version(self):
        return (3,)


class UnaryScalarOp(ScalarOp):
    nin = 1

class BinaryScalarOp(ScalarOp):
    nin = 2


###############
# Comparisons
###############

class LogicalComparison(BinaryScalarOp):
    def output_types(self, *input_dtypes):
        return [int8]
    def grad(self, inputs, output_gradients):
        return [None, None]

class LT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        # built-in < don't support complex
        return numpy.less(x, y)
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s < %(y)s);" % locals()
lt = LT()

class GT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        # built-in > don't support complex
        return numpy.greater(x, y)
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s > %(y)s);" % locals()
gt = GT()

class LE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        # built-in <= don't support complex
        return numpy.less_equal(x, y)
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s <= %(y)s);" % locals()
le = LE()

class GE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        # built-in >= don't support complex
        return numpy.greater_equal(x, y)
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s >= %(y)s);" % locals()
ge = GE()

class EQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x == y
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s == %(y)s);" % locals()
eq = EQ()

class NEQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x != y
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError()
        return "%(z)s = (%(x)s != %(y)s);" % locals()
neq = NEQ()

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
    def c_code(self, node, name, (x, low, hi), (z, ), sub):
        if self.openlow:
            cmp1 = '>'
        else:
            cmp1 = '>='

        #backport
        #cmp1 = '>' if self.openlow else '>='

        if self.openhi:
            cmp2 = '<'
        else:
            cmp2 = '<='

        #backport
        #cmp2 = '<' if self.openhi else '<='
        return "%(z)s = %(x)s %(cmp1)s %(low)s && %(x)s %(cmp2)s %(hi)s;" % locals()
    def grad(self, (x, low, hi), (gz, )):
        return None, None, None
inopenrange = InRange(True, True)
inclosedrange = InRange(False, False)

class Switch(ScalarOp):
    nin = 3
    def impl(self, cond, ift, iff):
        if cond:
            return ift
        else:
            return iff

            #backport
            #return ift if cond else iff
    def c_code(self, node, name, (cond, ift, iff), (z, ), sub):
        return "%(z)s = %(cond)s ? %(ift)s : %(iff)s;" % locals()
    def grad(self, (cond, ift, iff), (gz, )):
        if ift.type in continuous_types:
            first_part = switch(cond, gz, 0)
        else:
            first_part = None

        if iff.type in continuous_types:
            second_part = switch(cond, 0, gz)
        else:
            second_part = None

        return (None, first_part, second_part)

    def output_types(self, (cond_t, ift_t, iff_t)):
        return upcast_out(ift_t, iff_t)
switch = Switch()

####################
# BIT-WISE OPERATORS
####################

class UnaryBitOp(UnaryScalarOp):
    def output_types(self, *input_types):
        for i in input_types[0]:
            if i not in (int8, int32, int64):
                raise TypeError('input to a BitOp must have type int8, int32 or int64... not %s' % i)
        return upcast_out(*input_types[0])
    def grad(self, inputs, output_gradients):
        return [None]

class BinaryBitOp(BinaryScalarOp):
    def output_types(self, *input_types):
        t0, t1 = input_types[0]
        for i in input_types[0]:
            if i not in (int8, int32, int64):
                raise TypeError('input to a BitOp must have type int8, int32 or int64... not %s' % i)
        return upcast_out(*input_types[0])
    def grad(self, inputs, output_gradients):
        return [None, None]

class OR(BinaryBitOp):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x | y
or_ = OR()

class XOR(BinaryBitOp):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x ^ y
xor = XOR()

class AND(BinaryBitOp):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x & y
and_ = AND()

class Invert(UnaryBitOp):
    identity = False
    def impl(self, x):
        return ~x
invert = Invert()



##############
# Arithmetic
##############

class Maximum(BinaryScalarOp):
    commutative = True
    associative = True
    def impl(self, *inputs):
        # The built-in max function don't support complex type
        return numpy.maximum(*inputs)
    def c_code(self, node, name, (x,y), (z, ), sub):
        if any([i.type in complex_types for i in node.inputs]):
            raise NotImplementedError()
        return "%(z)s = ((%(y)s)>(%(x)s)? (%(y)s):(%(x)s));" %locals()

    def grad(self, (x, y), (gz, )):
        assert gz.type not in complex_types
        # max is not defined for complex_types
        gx, gy = None, None
        if x.type in float_types:
            gx = eq(maximum(x,y),  x)*gz
        if y.type in float_types:
            gy = eq(maximum(x,y),  y)*gz
        return (gx,gy)

maximum = Maximum(upcast_out, name = 'maximum')

class Minimum(BinaryScalarOp):
    commutative = True
    associative = True
    def impl(self, *inputs):
        # The built-in min function don't support complex type
        return numpy.minimum(*inputs)
    def c_code(self, node, name, (x,y), (z, ), sub):
        if any([i.type in complex_types for i in node.inputs]):
            raise NotImplementedError()
        return "%(z)s = ((%(y)s)<(%(x)s)? (%(y)s):(%(x)s));" %locals()

    def grad(self, (x, y), (gz, )):
        assert gz.type not in complex_types
        # max is not defined for complex_types
        gx, gy = None, None
        if x.type in float_types:
            gx = eq(minimum(x,y),  x)*gz
        if y.type in float_types:
            gy = eq(minimum(x,y),  y)*gz
        return (gx,gy)

minimum = Minimum(upcast_out, name = 'minimum')

class Add(ScalarOp):
    identity = 0
    commutative = True
    associative = True
    def impl(self, *inputs):
        return sum(inputs)
    def c_code(self, node, name, inputs, (z, ), sub):
        if not inputs:
            return z + " = 0;"
        else:
            return z + " = " + " + ".join(inputs) + ";"
    def grad(self, inputs, (gz, )):
        retval = []
        if gz.type in complex_types:
            for i in inputs:
                if i.type in complex_types:
                    retval += [cast(gz, i.type.dtype)]
                elif i.type in float_types:
                    retval += [cast(real(gz), i.type.dtype)]
                else:
                    retval += [None]
        elif gz.type in float_types:
            for i in inputs:
                if i.type in float_types:
                    retval += [cast(gz, i.type.dtype)]
                else:
                    retval += [None]
        else:
            retval += [None] * len(inputs)
        return retval
add = Add(upcast_out, name = 'add')

class Mul(ScalarOp):
    identity = 1
    commutative = True
    associative = True
    def impl(self, *inputs):
        return numpy.product(inputs)
    def c_code(self, node, name, inputs, (z, ), sub):
        if not inputs:
            return z + " = 1;"
        else:
            return z + " = " + " * ".join(inputs) + ";"
    def grad(self, inputs, (gz, )):
        retval = []
        for input in inputs:
            if input.type in continuous_types:
                if gz.type in complex_types:
                    # zr+zi = (xr + xi)(yr + yi)
                    # zr+zi = (xr*yr - xi*yi) + (xr yi + xi yr )
                    otherprod = mul(*(utils.difference(inputs, [input])))
                    yr = real(otherprod)
                    yi = imag(otherprod)
                    if input.type in complex_types:
                        retval += [complex(yr*real(gz)+yi*imag(gz), yr*imag(gz)-yi*real(gz))]
                    else:
                        retval += [cast(yr*real(gz)+yi*imag(gz), input.type.dtype)]
                else:
                    retval += [cast(mul(*([gz] + utils.difference(inputs, [input]))), input.type.dtype)]
            else:
                retval += [None]
        return retval
mul = Mul(upcast_out, name = 'mul')

class Sub(BinaryScalarOp):
    def impl(self, x, y):
        return x - y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s - %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if gz.type in complex_types:
            raise NotImplementedError()

        if x.type in float_types:
            first_part = cast(gz, x.type.dtype)
        else:
            first_part = None

        if y.type in float_types:
            second_part = cast(-gz, y.type.dtype)
        else:
            second_part = None
        return first_part, second_part
sub = Sub(upcast_out, name = 'sub')

def div_proxy(x, y):
    """Proxy for either true_div or int_div, depending on types of x, y.
    """
    if as_scalar(x).type.dtype.startswith('int') and as_scalar(y).type.dtype.startswith('int'):
        return int_div(x, y)
    else:
        return true_div(x, y)

class TrueDiv(BinaryScalarOp):
    def output_types(self, types):
        if all(t.dtype.startswith('int') for t in types):
            return [float64]
        else:
            return super(TrueDiv, self).output_types(types)
    def impl(self, x, y):
        x = numpy.asarray(x)
        y = numpy.asarray(y)
        if str(x.dtype).startswith('int') and str(y.dtype).startswith('int'):
            return float(x) / y
        else:
            return x / y
    def c_code(self, node, name, (x, y), (z, ), sub):
        #we generate good c code only when both are complex!
        if sum([node.inputs[0].type in complex_types, node.inputs[1].type in complex_types])==1:
            raise NotImplementedError('type not supported', type)
        if node.inputs[0].type in int_types and node.inputs[1].type in int_types:
            return "%(z)s = ((double)%(x)s) / %(y)s;" % locals()
        return "%(z)s = %(x)s / %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            first_part = cast(gz / y, x.type.dtype)
        else:
            first_part = None

        if y.type in float_types:
            second_part = cast(-(gz * x) / (y * y), y.type.dtype)
        else:
            second_part = None
        return first_part, second_part
true_div = TrueDiv(upcast_out, name = 'true_div')

class IntDiv(BinaryScalarOp):
    def impl(self, x, y):
        return x // y
    def c_code(self, node, name, (x,y), (z,), sub):
        raise NotImplementedError("For integer arguments the behavior of division in C and in Python [can] differ when the quotient is negative.  C actually does not even specify a correct behaviour in this case, it is up to the chip.")
    def grad(self, inputs, g_output):
        return [None] * len(inputs)
int_div = IntDiv(upcast_out, name = 'int_div')

floor_div = int_div

class Mod(BinaryScalarOp):
    def impl(self, x, y):
        return x % y
    def c_code_cache_version(self):
        return (5,)

    def c_support_code(self):
        #We use a macro as python use % as a special string caractere.
        return "#define THEANO_MACRO_MOD(x,y) (x % y)"

    def c_code(self, node, name, (x, y), (z, ), sub):
        """
        We want the result to have the same sign as python, not the other implementaiton of mod.
        """
        #raise NotImplementedError("Unlike Python, C's modulo returns negative modulo on negative dividend (to implement)")
        t = node.inputs[0].type.upcast(*[ i.type for i in node.inputs[1:]])
        if t in int_types or t in ['uint8','int8','uint16','int16','uint32','int32','uint64','int64']:
            x_mod_y = "THEANO_MACRO_MOD(%(x)s, %(y)s)"%locals()
            x_mod_ymm = "THEANO_MACRO_MOD(-%(x)s, -%(y)s)"%locals()
            x_mod_ypm = "THEANO_MACRO_MOD(%(x)s, -%(y)s)"%locals()
            x_mod_ymp = "THEANO_MACRO_MOD(-%(x)s, %(y)s)"%locals()
        elif t in float_types or t in ['float32','float64']:
            x_mod_y = "fmod(%(x)s,%(y)s)"%locals()
            x_mod_ymm = "fmod(-%(x)s,-%(y)s)"%locals()
            x_mod_ypm = "fmod(%(x)s,-%(y)s)"%locals()
            x_mod_ymp = "fmod(-%(x)s,%(y)s)"%locals()
        else:
            raise NotImplementedError('type not supported', type)

        return """
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
        """%locals()
    def grad(self, (x, y), (gz, )):
        return None, None
mod = Mod(upcast_out, name = 'mod')

class Pow(BinaryScalarOp):
    def impl(self, x, y):
        return x ** y
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in complex_types or node.inputs[1].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = pow(%(x)s, %(y)s);" % locals()
    def grad(self, (x, y), (gz, )):
        if gz.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            first_part = gz * y * x**(y - 1)
        else:
            first_part = None

        if y.type in float_types:
            second_part = gz * log(x) * x**y
        else:
            second_part = None

        return (first_part, second_part)

pow = Pow(upcast_out, name = 'pow')

class Clip(ScalarOp):
    nin = 3
    def impl(self, x, min, max):
        if x < min:
            return min
        elif x > max:
            return max
        else:
            return x
    def c_code(self, node, name, (x, min, max), (z, ), sub):
        return "%(z)s = %(x)s < %(min)s ? %(min)s : %(x)s > %(max)s ? %(max)s : %(x)s;" % locals()
    def grad(self, (x, min, max), (gz, )):
        assert gz.type not in complex_types
        gx = ((x > min) & (x < max)) * gz
        if x.type in float_types:
            return gx, None, None
        else:
            return None, None, None
clip = Clip(upcast_out, name = 'clip')

class First(BinaryScalarOp):
    def impl(self, x, y):
        return x
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if x.type in continuous_types:
            return gz, None
        else:
            return None,None
first = First(transfer_type(0), name = 'first')

class Second(BinaryScalarOp):
    def impl(self, x, y):
        return y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if y.type in continuous_types:
            return None, gz
        else:
            return None

second = Second(transfer_type(1), name = 'second')



class Identity(UnaryScalarOp):
    def impl(self, input):
        return input
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, ), (gz, )):
        if x.type in continuous_types:
            return gz,
        else:
            return None,
identity = Identity(same_out, name = 'identity')

#### CASTING OPERATIONS
class Cast(UnaryScalarOp):
    def __init__(self, o_type, name=None):
        if not isinstance(o_type, Scalar):
            raise TypeError(o_type)
        super(Cast, self).__init__(specific_out(o_type), name=name)
        self.o_type = o_type
        self.ctor = getattr(numpy, o_type.dtype)
    def __str__(self):
        return '%s{%s}' % (self.__class__.__name__, self.o_type.dtype)
    def impl(self, input):
        return self.ctor(input)
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%s = (%s)%s;" % (z, node.outputs[0].type.dtype_specs()[1], x)
    def grad(self, (x, ), (gz, )):
        if x.type in continuous_types:
            return [cast(gz, x.type.dtype)]
        else:
            return None,
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
           'float32': convert_to_float32,
           'float64': convert_to_float64,
           'complex64': convert_to_complex64,
           'complex128': convert_to_complex128}
def cast(x, dtype):
    """Symbolically cast `x` to a Scalar of given `dtype`."""
    if dtype == 'floatX': dtype = config.floatX

    _x = as_scalar(x)
    if _x.type.dtype == dtype:
        return _x
    if _x.type.dtype.startswith('complex') and not dtype.startswith('complex'):
        raise TypeError('Casting from complex to real is ambiguous: consider real(), imag(), angle() or abs()')
    return _cast_mapping[dtype](_x)

class Abs(UnaryScalarOp):
    def make_node(self, x):
        inputs = [as_scalar(input) for input in [x]]
        if inputs[0].type == complex64:
            outputs = [float32()]
        elif inputs[0].type == complex128:
            outputs = [float64()]
        else:
            outputs = [t() for t in self.output_types([input.type for input in inputs])]
        return Apply(self, inputs, outputs)
    def impl(self, x):
        return numpy.abs(x)
    def grad(self, (x, ), (gz, )):
        if x.type in float_types + complex_types:
            return gz * x / abs(x), # formula works for complex and real
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
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
    def impl(self, x):
        #casting to output type is handled by filter
        return numpy.sign(x)
    def grad(self, (x, ), (gz, )):
        return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        #casting is done by compiler
        #TODO: use copysign
        type = node.inputs[0].type
        if type in float_types:
            return "%(z)s = (%(x)s >= 0) ? (%(x)s == 0) ? 0.0 : 1.0 : -1.0;" % locals()
        if type in int_types:
            return "%(z)s = (%(x)s >= 0) ? (%(x)s == 0) ? 0 : 1 : -1;" % locals()
        raise TypeError() #complex has no sgn
    def c_code_cache_version(self):
        s = super(Sgn, self).c_code_cache_version()
        if s:
            return (3,) + s
        else: #if parent is unversioned, we are too
            return s
sgn = Sgn(same_out_nocomplex, name = 'sgn')

class Ceil(UnaryScalarOp):
    def impl(self, x):
        return numpy.ceil(x)
    def grad(self, (x,), (gz,)):
        return None,
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = ceil(%(x)s);" % locals()
ceil = Ceil(same_out_nocomplex, name = 'ceil')

class Floor(UnaryScalarOp):
    def impl(self, x):
        return numpy.floor(x)
    def grad(self, (x,), (gz,)):
        return None,
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = floor(%(x)s);" % locals()
floor = Floor(same_out_nocomplex, name = 'floor')

class RoundHalfToEven(UnaryScalarOp):
    """
    This function implement the same rounding than numpy: Round half to even

    c/c++ round fct IS DIFFERENT!
    See http://en.wikipedia.org/wiki/Rounding for more detail
    """
    def impl(self, x):
        return numpy.round(x)
    def c_code___(self, node, name, (x, ), (z, ), sub):
        typ = node.outputs[0].type.dtype
        if not node.outputs[0].type.dtype in ['float32', 'float64']:
            Exception("The output should be float32 or float64")

        return """
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

        """
round_half_to_even = RoundHalfToEven(same_out_float_only)

def round_half_away_from_zero_(a):
    if a>0:
        return numpy.floor(a + 0.5)
    else:
        return numpy.ceil(a - 0.5)

round_half_away_from_zero_vec64 = numpy.vectorize(round_half_away_from_zero_,
                                                  doc='round_half_away_from_zero_vec64')
round_half_away_from_zero_vec32 = numpy.vectorize(round_half_away_from_zero_,
                                                  doc='round_half_away_from_zero_vec32',
                                                  otypes=['float32'])

def round_half_away_from_zero_vec(a):
    if getattr(a, 'dtype',None) == numpy.float32:
        return round_half_away_from_zero_vec32(a)
    return round_half_away_from_zero_vec64(a)

class RoundHalfAwayFromZero(UnaryScalarOp):
    """
    Implement the same rounding algo as c round() fct.
    numpy.round fct IS DIFFERENT!

    See http://en.wikipedia.org/wiki/Rounding for more detail
    """
    def impl(self, x):

        return round_half_away_from_zero_vec(x)
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.outputs[0].type.dtype in ['float32', 'float64']:
            return "%(z)s = round(%(x)s);" % locals()
        else:
            Exception("The output should be float32 or float64")
round_half_away_from_zero = RoundHalfAwayFromZero(same_out_float_only)

class Neg(UnaryScalarOp):
    def impl(self, x):
        return -x
    def grad(self, (x, ), (gz, )):
        if x.type in continuous_types:
            return -gz,
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = -%(x)s;" % locals()
neg = Neg(same_out, name = 'neg')

class Inv(UnaryScalarOp):
    """ multiplicative inverse. Also called reciprocal"""
    def impl(self, x):
        return 1.0 / x
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return -gz / (x * x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = 1.0 / %(x)s;" % locals()
inv = Inv(upgrade_to_float, name = 'inv')

class Log(UnaryScalarOp):
    """ log base e """
    def impl(self, x):
        return numpy.log(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz / x,
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        #todo: the version using log2 seems to be very slightly faster
        # on some machines for some reason, check if it's worth switching
        #return "%(z)s = log2(%(x)s) * 0.69314718055994529;" % locals()
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log(%(x)s);" % locals()
log = Log(upgrade_to_float, name = 'log')

class Log2(UnaryScalarOp):
    """ log base 2 """
    def impl(self, x):
        return numpy.log2(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz / (x * math.log(2.0)),
        else:
            return None,

    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log2(%(x)s);" % locals()
log2 = Log2(upgrade_to_float, name = 'log2')

class Log10(UnaryScalarOp):
    """ log base 10 """
    def impl(self, x):
        return numpy.log10(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz / (x * numpy.log(10.0)),
        else:
            return None

    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log10(%(x)s);" % locals()
log10 = Log10(upgrade_to_float, name = 'log10')

class Log1p(UnaryScalarOp):
    """ log(1+x) """
    def impl(self, x):
        return numpy.log1p(x)
    def grad(self, (x,), (gz,)):
        if gz.type in complex_types:
            raise NotImplementedError()
        if gz.type in float_types:
            return [gz / (1+x)]
        return [None]
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = log1p(%(x)s);" % locals()
log1p = Log1p(upgrade_to_float, name = 'log1p')

class Exp(UnaryScalarOp):
    def impl(self, x):
        return numpy.exp(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            return gz * exp(x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = exp(%(x)s);" % locals()
exp = Exp(upgrade_to_float, name = 'exp')

class Sqr(UnaryScalarOp):
    def impl(self, x):
        return x*x
    def grad(self, (x, ), (gz, )):
        if gz.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz * x * 2,
        else:
            return None,

    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = %(x)s * %(x)s;" % locals()
sqr = Sqr(same_out, name = 'sqr')

class Sqrt(UnaryScalarOp):
    def impl(self, x):
        return numpy.sqrt(x)
    def grad(self, (x, ), (gz, )):
        if gz.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return (gz * 0.5) / sqrt(x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = sqrt(%(x)s);" % locals()
sqrt = Sqrt(upgrade_to_float, name = 'sqrt')

class Cos(UnaryScalarOp):
    def impl(self, x):
        return numpy.cos(x)
    def grad(self, (x, ), (gz, )):
        if gz.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return -gz * sin(x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = cos(%(x)s);" % locals()
cos = Cos(upgrade_to_float, name = 'cos')

class Sin(UnaryScalarOp):
    def impl(self, x):
        return numpy.sin(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz * cos(x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = sin(%(x)s);" % locals()
sin = Sin(upgrade_to_float, name = 'sin')

class Tan(UnaryScalarOp):
    def impl(self, x):
        return numpy.tan(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz / sqr(cos(x)),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = tan(%(x)s);" % locals()
tan = Tan(upgrade_to_float, name = 'tan')

class Cosh(UnaryScalarOp):
    """
    cosh(x) = (exp(x) + exp(-x)) / 2
    """
    def impl(self, x):
        return numpy.cosh(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz * sinh(x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = cosh(%(x)s);" % locals()
cosh = Cosh(upgrade_to_float, name = 'cosh')

class Sinh(UnaryScalarOp):
    """
    sinh(x) = (exp(x) - exp(-x)) / 2
    """
    def impl(self, x):
        return numpy.sinh(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz * cosh(x),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = sinh(%(x)s);" % locals()
sinh = Sinh(upgrade_to_float, name = 'sinh')

class Tanh(UnaryScalarOp):
    """
    tanh(x) = sinh(x) / cosh(x)
            = (exp(2*x) - 1) / (exp(2*x) + 1)
    """
    def impl(self, x):
        return numpy.tanh(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            raise NotImplementedError()
        if x.type in float_types:
            return gz * (1 - sqr(tanh(x))),
        else:
            return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = tanh(%(x)s);" % locals()
tanh = Tanh(upgrade_to_float, name = 'tanh')

class Real(UnaryScalarOp):
    """Extract the real coordinate of a complex number.  """
    def impl(self, x):
        return numpy.real(x)
    def grad(self, (x, ), (gz, )):
        return [complex(gz, 0)]

real = Real(real_out, name='real')

class Imag(UnaryScalarOp):
    def impl(self, x):
        return numpy.imag(x)
    def grad(self, (x, ), (gz, )):
        if x.type in complex_types:
            return [complex(0, gz)]
        elif x.type in float_types:
            return [second(x,0)]
        else:
            return [None]
imag = Imag(real_out, name='imag')

class Angle(UnaryScalarOp):
    def impl(self, x):
        return numpy.angle(x)
    def grad(self, (c, ), (gtheta, )):
        # y = x.imag
        # r = sqrt(y**2 + x.real**2)
        # g = y/r
        # if x == 0 and y == 0:
        #     theta = 0
        # elif x >= 0:
        #     theta = numpy.arcsin(g)
        # else:
        #     theta = -numpy.arcsin(g)+numpy.pi

        x = real(c)
        y = imag(c)
        r = abs(c)

        gr = -gtheta * y / (r**2 * sqrt(1 - (y/r)**2))
        gx = gr * x/r
        gy = gr * y/r
        if c in complex_types:
            return [cast(complex(gx, gy), x.type.dtype)]
        elif c in float_types:
            return [cast(second(x,0), x.type.dtype)]
        else:
            return [None]

angle = Angle(specific_out(float64), name='angle')

class Complex(BinaryScalarOp):
    @staticmethod
    def output_types_preference(x,y):
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
    def grad(self, (x,y), (gz,)):
        return [cast(real(gz), x.type.dtype),
                cast(imag(gz), y.type.dtype)]
complex = Complex(name='complex')

class Conj(UnaryScalarOp):
    def impl(self, x):
        return numpy.conj(x)
    def grad(self, (x, ), (gz, )):
        return [conj(gz)]
conj = Conj(same_out, name='conj')

class ComplexFromPolar(BinaryScalarOp):
    @staticmethod
    def output_types_preference(x,y):
        return Complex.output_types_preference(x,y)
    def impl(self, r, theta):
        if r < 0:
            raise ValueError('polar radius must be non-negative', r)
        x = r*numpy.cos(theta)
        y = r*numpy.sin(theta)
        if x.dtype == 'float32':
            return numpy.complex64(numpy.complex(x,y))
        else:
            return numpy.complex128(numpy.complex(x,y))
    def grad(self, (r,theta), (gz,)):
        gr = cos(theta) * real(gz) + sin(theta) * imag(gz)
        gtheta = -real(gz) * r * sin(theta) + imag(gz) * r * cos(theta)
        return [cast(gr, r.type.dtype),
                cast(gtheta, theta.type.dtype)]
complex_from_polar = ComplexFromPolar(name='complex_from_polar')


class Composite(ScalarOp):
    """
    Composite is an Op that takes a graph of scalar operations and
    produces c code for the whole graph. Its biggest use would be to
    implement the loop fusion optimizer (which I have yet to do
    someday...)
    """
    def __str__(self):
        if hasattr(self, 'name') and self.name:
            return self.name
        else:
            return "%s{%s}" % (self.__class__.__name__, ", ".join(
                "%s=%s" % (k, v) for k, v in self.__dict__.items()
                if k not in ["env","_c_code", "_cmodule_key", "_impls",
                             "_hashval", "inputs_type"] ))

    def make_new_inplace(self, output_types_preference = None, name = None):
        """
        This op.__init__ fct don't have the same parameter as other scalar op.
        This break the insert_inplace_optimizer optimization.
        This fct allow fix patch this.
        """
        out = self.__class__(self.inputs,self.outputs)
        if name:
            out.name = name
        else:
            name = out.name
        super(Composite,out).__init__(output_types_preference, name)
        return out

    def __init__(self, inputs, outputs):
        self.inputs=copy(inputs)
        self.outputs=copy(outputs)

        env = Env(*gof.graph.clone(inputs, outputs))
        gof.MergeOptimizer().optimize(env)
        inputs, outputs = env.inputs, env.outputs

        for node in env.nodes:
            if not isinstance(node.op, ScalarOp):
                raise ValueError("The env to Composite must be exclusively composed of ScalarOp instances.")

        subd = dict(zip(inputs,
                        ["%%(i%i)s"%i for i in range(len(inputs))]) +
                    zip(outputs,
                        ["%%(o%i)s"%i for i in range(len(outputs))]))

        for orphan in env.variables: #env.orphans:
            if orphan.owner is None and orphan not in env.inputs:
                if isinstance(orphan, Constant):
                    subd[orphan] = orphan.type.c_literal(orphan.data)
                else:
                    raise ValueError("All orphans in the env to Composite must be Constant instances.")

        if not hasattr(self,"name"):
            l=[]
            for n in env.nodes:
                if hasattr(n.op,"name") and n.op.name is not None:
                    v=n.op.name
                else: v=n.op.__class__.__name__
                l.append(v)
            self.name="Composite{"+",".join(l)+"}"

        _c_code = "{\n"
        i = 0
        j = 0
        for node in env.toposort():
            j += 1
            for output in node.outputs:
                if output not in subd:
                    i += 1
                    name = "V%%(id)s_tmp%i" % i
                    subd[output] = name
                    _c_code += "%s %s;\n" % (output.type.dtype_specs()[1], name)

            s =     node.op.c_code(node,
                                      "%(name)s",
                                      [subd[input] for input in node.inputs],
                                      [subd[output] for output in node.outputs],
                                      dict(fail = "%(fail)s",
                                           id = "%%(id)s_%i" % j))
            _c_code += s
            _c_code += "\n"
        _c_code += "}\n"


        def compose_impl(r):
            # this is not optimal at all eg in add(*1 -> mul(x, y), *1)
            # it will calculate *1 twice
            # it also doesn't follow env.toposort but that's (presumably)
            # still correct since we only have scalar ops
            if r in env.inputs:
                idx = env.inputs.index(r)
                return lambda inputs: inputs[idx]
            elif r.owner is None: # in env.orphans:
                return lambda inputs: r.data
            node = r.owner
            producers = [compose_impl(input) for input in node.inputs]
            return lambda inputs: node.op.impl(*[p(inputs) for p in producers])

        _impls = [compose_impl(r) for r in env.outputs]

        self._c_code = _c_code
        self._impls = _impls
        self.nin = len(inputs)
        self.nout = len(outputs)
        self.env = env
        self.inputs_type = tuple([input.type for input in self.env.inputs])
        self.outputs_type = tuple([output.type for output in self.env.outputs])
        self._rehash()

    def output_types(self, input_types):
        if tuple(input_types) != self.inputs_type:
            raise TypeError("Wrong types for Composite. Expected %s, got %s."
                            % (self.inputs_type, tuple(input_types)))
        return self.outputs_type

    def perform(self, node, inputs, output_storage):
        for storage, impl in zip(output_storage, self._impls):
            storage[0] = impl(inputs)

    def impl(self, *inputs):
        output_storage = [[None] for i in xrange(self.nout)]
        self.perform(None, inputs, output_storage)
        return utils.to_return_values([storage[0] for storage in output_storage])

    def grad(self, inputs, output_grads):
        raise NotImplementedError("grad is not implemented for Composite")

    def c_code(self, node, name, inames, onames, sub):
        d = dict(zip(["i%i"%i for i in range(len(inames))],
                     inames) +
                 zip(["o%i"%i for i in range(len(onames))],
                     onames),
                 **sub)
        d['name'] = name
        if not sub.has_key('id'):
            #The use of a dummy id is safe as the code is in a separate block.
            #It won't generate conflicting variable name.
            d['id']='_DUMMY_ID_'

        return self._c_code % d

    def c_code_cache_version(self):
        return (1,)+tuple([x.op.c_code_cache_version() for x in self.env.toposort()])

    def c_support_code(self):
        str = ""
        for node in self.env.toposort():
            try:
                str += node.op.c_support_code()+"\n"
            except gof.utils.MethodNotDefined:
                pass
        return str

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, self.__class__): return False
        if self.nin!=other.nin or self.nout != other.nout: return False
        return self._hashval == other._hashval
        return self._cmodule_key == other._cmodule_key

    def _rehash(self):
#TODO: What no_recycling is used for? What I need to put their?
#        no_recycling = []
        self._cmodule_key = gof.CLinker.cmodule_key_(self.env, [])
        self._hashval = hash(self._cmodule_key)

    def __hash__(self):
        return self._hashval

    def __getstate__(self):
        d = copy(self.__dict__)
        d.pop('env')
        d.pop('_impls')
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
        #we must call init to set env and _impls again.
        #otherwise self.perform won't work.
        self.__init__(self.inputs, self.outputs)
