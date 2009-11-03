import operator
import math
from copy import copy

import numpy

from theano import gof
from theano.gof import Op, utils, Variable, Constant, Type, Apply, Env
from theano.gof.python25 import partial, all, any

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
    if isinstance(x, float):
        for dtype in ['float32', 'float64']:
            x_ = numpy.asarray(x, dtype=dtype)
            if numpy.all(x == x_):
                break
            x_ = None
        assert x_ is not None
        return ScalarConstant(Scalar(str(x_.dtype)), x)
    if isinstance(x, int):
        for dtype in ['int8', 'int16', 'int32', 'int64']:
            x_ = numpy.asarray(x, dtype=dtype)
            if numpy.all(x == x_):
                break
            x_ = None
        assert x_ is not None
        return ScalarConstant(Scalar(str(x_.dtype)), x)
    if isinstance(x, complex):
        raise NotImplementedError()
    raise TypeError(x)
    #return ScalarConstant(float64, float(x))


class Scalar(Type):

    def __init__(self, dtype):
        self.dtype = dtype
        self.dtype_specs() # error checking
    
    def filter(self, data, strict = False):
        py_type = self.dtype_specs()[0]
        if strict and not isinstance(data, py_type):
            raise TypeError("%s expected a %s, got %s of type %s" % (self, py_type, data,
                type(data)), 
                    data)
        try:
            return py_type(data)
        except Exception, e:
            raise TypeError("Could not convert %s (value=%s) to %s" % (type(data), data, self.dtype), e)

    def values_eq_approx(self, a, b, tolerance = 1e-4):
        return abs(a - b) / (a+b) < tolerance

    def c_headers(self):
        l=['<math.h>']
        if utils.config.getboolean('lib.amdlibm'):
            l+=['<amdlibm.h>']
        return l

    def c_libraries(self):
        l=[]
        if utils.config.getboolean('lib.amdlibm'):
            l+=['amdlibm']
        return l

    def c_compile_args(self):
        if utils.config.getboolean('lib.amdlibm'):
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
        return """
        if (!%(check)s(py_%(name)s))
            %(fail)s
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
            template <typename T>
            complex_type& operator =(const T& y);
         };
         """
        operator_eq = """
        template <> %(mytype)s & %(mytype)s::operator=<npy_int8>(const npy_int8 & y)
        { this->real=y; this->imag=0; return *this; }

        template <> %(mytype)s & %(mytype)s::operator=<npy_int16>(const npy_int16 & y)
        { this->real=y; this->imag=0; return *this; }

        template <> %(mytype)s & %(mytype)s::operator=<npy_int32>(const npy_int32 & y)
        { this->real=y; this->imag=0; return *this; }

        template <> %(mytype)s & %(mytype)s::operator=<npy_int64>(const npy_int64 & y)
        { this->real=y; this->imag=0; return *this; }

        template <> %(mytype)s & %(mytype)s::operator=<npy_float32>(const npy_float32 & y)
        { this->real=y; this->imag=0; return *this; }

        template <> %(mytype)s & %(mytype)s::operator=<npy_float64>(const npy_float64 & y)
        { this->real=y; this->imag=0; return *this; }

        template <> %(mytype)s & %(mytype)s::operator=<theano_complex128>(const theano_complex128 & y)
        { this->real=y.real; this->imag=y.imag; return *this; }

        template <> %(mytype)s & %(mytype)s::operator=<theano_complex64>(const theano_complex64 & y)
        { this->real=y.real; this->imag=y.imag; return *this; }

        """
        # todo: use C templating
        return template % dict(nbits = 64, half_nbits = 32) \
                + template % dict(nbits = 128, half_nbits = 64) \
                + operator_eq % dict(mytype='theano_complex128') \
                + operator_eq % dict(mytype='theano_complex64')

    def c_code_cache_version(self):
        #return ()
        return (4,)  #explicit T given in specialization of operator= lines.  This makes it compile with open64
        #2,


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

grad_types = float_types + complex_types # these are the types for which gradients can be defined.

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

class ScalarVariable(Variable, _scalar_py_operators):
    pass

class ScalarConstant(Constant, _scalar_py_operators):
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
def same_out(type):
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
    This upgrade the types to float32 or float64 to don't loose any precision.
    """
    conv = {int8: float32,
            int16: float32,
            int32: float64,
            int64: float64}
    return Scalar(Scalar.upcast(*[conv.get(type, type) for type in types])),


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
        return (2,)


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
        return x < y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = (%(x)s < %(y)s);" % locals()
lt = LT()

class GT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        return x > y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = (%(x)s > %(y)s);" % locals()
gt = GT()

class LE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        return x <= y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = (%(x)s <= %(y)s);" % locals()
le = LE()

class GE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        return x >= y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = (%(x)s >= %(y)s);" % locals()
ge = GE()

class EQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x == y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = (%(x)s == %(y)s);" % locals()
eq = EQ()

class NEQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x != y
    def c_code(self, node, name, (x, y), (z, ), sub):
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
        if ift.type in grad_types:
          first_part = switch(cond, gz, 0)
        else:
          first_part = None

        if iff.type in grad_types:
          second_part = switch(cond, 0, gz)
        else:
          second_part = None

        return (None, first_part, second_part)

        #return (None,
        #        switch(cond, gz, 0) if ift.type in grad_types else None,
        #        switch(cond, 0, gz) if iff.type in grad_types else None)

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
                raise TypeError('input to a BitOp must have type int8, int32 or int 64... not %s' % i)
        return upcast_out(*input_types[0])
    def grad(self, inputs, output_gradients):
        return [None]

class BinaryBitOp(BinaryScalarOp):
    def output_types(self, *input_types):
        t0, t1 = input_types[0]
        for i in input_types[0]:
            if i not in (int8, int32, int64):
                raise TypeError('input to a BitOp must have type int8, int32 or int 64... not %s' % i)
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
      for i in inputs:
        if i.type in grad_types:
          retval += [cast(gz, i.type.dtype)]
        else:
          retval += [None]
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
        if input.type in grad_types:
          retval += [cast(mul(*([gz] + utils.difference(inputs, [input]))), input.type.dtype)]
        else:
          retval += [None]
      
      return retval

        #return [(mul(*([gz] + utils.difference(inputs, [input]))) 
        #    if input.type in grad_types else None)
        #        for input in inputs]
mul = Mul(upcast_out, name = 'mul')

class Sub(BinaryScalarOp):
    def impl(self, x, y):
        return x - y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s - %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if x.type in grad_types:
            first_part = cast(gz, x.type.dtype)
        else:
            first_part = None

        if y.type in grad_types:
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
        if node.inputs[0].type in int_types and node.inputs[1].type in int_types:
            return "%(z)s = ((double)%(x)s) / %(y)s;" % locals()
        return "%(z)s = %(x)s / %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if x.type in grad_types:
          first_part = cast(gz / y, x.type.dtype)
        else:
          first_part = None

        if y.type in grad_types:
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


class Mod(BinaryScalarOp):
    def impl(self, x, y):
        return x % y
    def c_code(self, node, name, (x, y), (z, ), sub):
        """
        We want the result to have the same sign as python, not the other implementaiton of mod.
        """
        #raise NotImplementedError("Unlike Python, C's modulo returns negative modulo on negative dividend (to implement)")
        t = node.inputs[0].type.upcast(*[ i.type for i in node.inputs[1:]])
        if t in int_types:
            x_mod_y = "%(x)s %% %(y)s"%locals()
        elif t in float_types:
            x_mod_y = "fmod(%(x)s,%(y)s)"%locals()
        else:
            raise NotImplementedError('type not supported', type)
        
        return """
    if (%(x)s == 0 || %(y)s == 0) {
        if (%(y)s == 0) %(z)s = %(x_mod_y)s;
        %(z)s = 0;
    }
//was #if @neg@, I suspect @neg@ to be platform dependant.
//should be true under X86, but could be false for other architecture!
#if 1
    else if ((%(x)s > 0) == (%(y)s > 0)) {
        %(z)s = %(x_mod_y)s;
    }
    else { /* handled like Python does */
        %(z)s = %(x_mod_y)s;
        if (%(z)s) %(z)s += %(y)s;
    }
#else
    else
        %(z)s = %(x_mod_y)s;
#endif
"""%locals()
    def grad(self, (x, y), (gz, )):
        return None, None
mod = Mod(upcast_out, name = 'mod')

class Pow(BinaryScalarOp):
    def impl(self, x, y):
        return x ** y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = pow(%(x)s, %(y)s);" % locals()
    def grad(self, (x, y), (gz, )):
        if x.type in grad_types:
            first_part = gz * y * x**(y - 1)
        else:
            first_part = None

        if y.type in grad_types:
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

        #backport
        #return min if x < min else max if x > max else x
    def c_code(self, node, name, (x, min, max), (z, ), sub):
        return "%(z)s = %(x)s < %(min)s ? %(min)s : %(x)s > %(max)s ? %(max)s : %(x)s;" % locals()
    def grad(self, (x, min, max), (gz, )):
        gx = ((x > min) & (x < max)) * gz
        if x.type in grad_types:
          return gx, None, None
        else:
          return None, None, None

        #return gx if x.type in grad_types else None, None, None
clip = Clip(transfer_type(0), name = 'clip')

class First(BinaryScalarOp):
    def impl(self, x, y):
        return x
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if x.type in grad_types:
          return gz, None
        else:
          return None,None
        #backport
        #return gz if x.type in grad_types else None, None
first = First(transfer_type(0), name = 'first')

class Second(BinaryScalarOp):
    def impl(self, x, y):
        return y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        if y.type in grad_types:
          return None, gz
        else:
          return None

        #backport
        #return None, gz if y.type in grad_types else None
second = Second(transfer_type(1), name = 'second')



class Identity(UnaryScalarOp):
    def impl(self, input):
        return input
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, ), (gz, )):
        if x.type in grad_types:
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
        if x.type in grad_types:
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
        if x.type in grad_types:
          return gz * sgn(x),
        else:
          return None,
        #backport
        #return gz * sgn(x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        type = node.inputs[0].type
        if type in int_types:
            return "%(z)s = abs(%(x)s);" % locals()
        if type in float_types:
            return "%(z)s = fabs(%(x)s);" % locals()
        if type in complex_types:
            return "%(z)s = sqrt(%(x)s.real*%(x)s.real + %(x)s.imag*%(x)s.imag);" % locals()
        #complex, other?
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
        return "%(z)s = (%(x)s >= 0) ? (%(x)s == 0) ? 0.0 : 1.0 : -1.0;" % locals()
sgn = Sgn(same_out, name = 'sgn')

class IRound(UnaryScalarOp):
    def impl(self, x):
        return numpy.asarray(numpy.round(x), dtype = 'int64')
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = round(%(x)s);" % locals()
iround = IRound(int_out)

class Neg(UnaryScalarOp):
    def impl(self, x):
        return -x
    def grad(self, (x, ), (gz, )):
        if x.type in grad_types:
          return -gz,
        else:
          return None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = -%(x)s;" % locals()
neg = Neg(same_out, name = 'neg')

class Inv(UnaryScalarOp):
    def impl(self, x):
        return 1.0 / x
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return -gz / (x * x),
      else:
        return None,

      #backport
      #return -gz / (x * x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = 1.0 / %(x)s;" % locals()
inv = Inv(upgrade_to_float, name = 'inv')

class Log(UnaryScalarOp):
    """ log base e """
    def impl(self, x):
        return math.log(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz / x,
      else:
        return None,
      #backport
      #return gz / x if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        #todo: the version using log2 seems to be very slightly faster
        # on some machines for some reason, check if it's worth switching
        #return "%(z)s = log2(%(x)s) * 0.69314718055994529;" % locals()
        return "%(z)s = log(%(x)s);" % locals()
log = Log(upgrade_to_float, name = 'log')

class Log2(UnaryScalarOp):
    """ log base 2 """
    def impl(self, x):
        return numpy.log2(x)
    def grad(self, (x, ), (gz, )):
        if x.type in grad_types:
          return gz / (x * math.log(2.0)),
        else:
          return None,

        #backport
        #return gz / (x * math.log(2.0)) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = log2(%(x)s);" % locals()
log2 = Log2(upgrade_to_float, name = 'log2')

class Log10(UnaryScalarOp):
    """ log base 10 """
    def impl(self, x):
        return numpy.log10(x)
    def grad(self, (x, ), (gz, )):
        if x.type in grad_types:
           return gz / (x * math.log(10.0)),
        else:
           return None

        #backport
        #return gz / (x * math.log(10.0)) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = log10(%(x)s);" % locals()
log10 = Log10(upgrade_to_float, name = 'log10')

class Exp(UnaryScalarOp):
    def impl(self, x):
        return math.exp(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz * exp(x),
      else:
        return None,

     #backport
     #return gz * exp(x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = exp(%(x)s);" % locals()
exp = Exp(upgrade_to_float, name = 'exp')

class Sqr(UnaryScalarOp):
    def impl(self, x):
        return x*x
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz * x * 2,
      else:
        return None,

       #backport
       # return gz * x * 2 if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = %(x)s * %(x)s;" % locals()
sqr = Sqr(same_out, name = 'sqr')

class Sqrt(UnaryScalarOp):
    def impl(self, x):
        return math.sqrt(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return (gz * 0.5) / sqrt(x),
      else:
        return None,
      #backport
      #return (gz * 0.5) / sqrt(x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = sqrt(%(x)s);" % locals()
sqrt = Sqrt(upgrade_to_float, name = 'sqrt')

class Cos(UnaryScalarOp):
    def impl(self, x):
        return math.cos(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return -gz * sin(x), 
      else:
        return None,
      #backport
      #  return -gz * sin(x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = cos(%(x)s);" % locals()
cos = Cos(upgrade_to_float, name = 'cos')

class Sin(UnaryScalarOp):
    def impl(self, x):
        return math.sin(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz * cos(x), 
      else:
        return None,
      #backport
      #  return gz * cos(x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = sin(%(x)s);" % locals()
sin = Sin(upgrade_to_float, name = 'sin')

class Tan(UnaryScalarOp):
    def impl(self, x):
        return math.tan(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz / sqr(cos(x)),
      else:
        return None,
      #backport
      #return gz / sqr(cos(x)) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = tan(%(x)s);" % locals()
tan = Tan(upgrade_to_float, name = 'tan')

class Cosh(UnaryScalarOp):
    """
    cosh(x) = (exp(x) + exp(-x)) / 2
    """
    def impl(self, x):
        return math.cosh(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz * sinh(x),
      else:
        return None,
      #backport
      #return gz * sinh(x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = cosh(%(x)s);" % locals()
cosh = Cosh(upgrade_to_float, name = 'cosh')

class Sinh(UnaryScalarOp):
    """
    sinh(x) = (exp(x) - exp(-x)) / 2
    """
    def impl(self, x):
        return math.sinh(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz * cosh(x),
      else:
        return None,
    #backport
    #return gz * cosh(x) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = sinh(%(x)s);" % locals()
sinh = Sinh(upgrade_to_float, name = 'sinh')

class Tanh(UnaryScalarOp):
    """
    tanh(x) = sinh(x) / cosh(x)
            = (exp(2*x) - 1) / (exp(2*x) + 1)
    """
    def impl(self, x):
        return math.tanh(x)
    def grad(self, (x, ), (gz, )):
      if x.type in grad_types:
        return gz * (1 - sqr(tanh(x))),
      else:
        return None,
    #backport
    #return gz * (1 - sqr(tanh(x))) if x.type in grad_types else None,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = tanh(%(x)s);" % locals()
tanh = Tanh(upgrade_to_float, name = 'tanh')



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
                             "_hashval"] ))

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
            self.name="Composite{"+"".join([n.op.__class__.__name__ if not hasattr(n,"name") else n.name for n in env.nodes])+"}"

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
            _c_code += node.op.c_code(node,
                                      "%(name)s",
                                      [subd[input] for input in node.inputs],
                                      [subd[output] for output in node.outputs],
                                      dict(fail = "%(fail)s",
                                           id = "%%(id)s_%i" % j))
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

