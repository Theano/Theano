import operator
import math
from copy import copy

import numpy

from .. import gof
from ..gof import Op, utils, Result, Constant, Type, Apply, Env
from ..gof.python25 import partial

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
    if isinstance(x, Result):
        if not isinstance(x.type, Scalar):
            raise TypeError("Result type field must be a Scalar.", x, x.type)
        return x
    try:
        return constant(x)
    except TypeError:
        raise TypeError("Cannot convert %s to Scalar" % x, type(x))

def constant(x):
    if isinstance(x, float):
        return ScalarConstant(float64, x)
    if isinstance(x, int):
        return ScalarConstant(int64, x)
    return ScalarConstant(float64, float(x))


class Scalar(Type):

    def __init__(self, dtype):
        self.dtype = dtype
        self.dtype_specs() # error checking
    
    def filter(self, data, strict = False):
        py_type = self.dtype_specs()[0]
        if strict and not isinstance(data, py_type):
            raise TypeError("%s expected a %s" % (self, self.dtype), data)
        try:
            return py_type(data)
        except Exception, e:
            raise TypeError("Could not convert %s (value=%s) to %s" % (type(data), data, self.dtype), e)

    def __eq__(self, other):
        return type(self) == type(other) and other.dtype == self.dtype

    def __hash__(self):
        return hash(self.dtype)

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

    def make_result(self, name = None):
        return ScalarResult(self, name = name)

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
        };
        """
        return template % dict(nbits = 64, half_nbits = 32) + template % dict(nbits = 128, half_nbits = 64)


int8 = Scalar('int8')
int16 = Scalar('int16')
int32 = Scalar('int32')
int64 = Scalar('int64')
float32 = Scalar('float32')
float64 = Scalar('float64')
complex64 = Scalar('complex64')
complex128 = Scalar('complex128')

int_types = int8, int16, int32, int64
float_types = float32, float64
complex_types = complex64, complex128

class _scalar_py_operators:

    #UNARY
    def __abs__(self): return _abs(self)
    def __neg__(self): return neg(self)

    #CASTS
    def __int__(self): return AsInt(self).out
    def __float__(self): return AsInt(self).out
    def __complex__(self): return AsComplex(self).out

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
    def __div__(self,other): return div(self,other)
    def __mod__(self,other): return mod(self,other)
    def __pow__(self,other): return pow(self,other)

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div(other,self)
    def __rmod__(self,other): return mod(other,self)
    def __rpow__(self,other): return pow(other,self)

class ScalarResult(Result, _scalar_py_operators):
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




def upcast_out(*types):
    return Scalar(dtype = Scalar.upcast(*types)),
def same_out(type):
    return type,
class transfer_type:
    def __init__(self, i):
        assert type(i) == int
        self.i = i
    def __call__(self, *types):
        return types[self.i],
class specific_out:
    def __init__(self, *spec):
        self.spec = spec
    def __call__(self, *types):
        return self.spec
def int_out(*types):
    return int64,
def float_out(*types):
    return float64,
def upgrade_to_float(*types):
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
                raise TypeError("Wrong number of inputs for %s.make_node (got %i, expected %i)" \
                                    % (self, len(inputs), self.nin))
        inputs = [as_scalar(input) for input in inputs]
        outputs = [t() for t in self.output_types([input.type for input in inputs])]
        if len(outputs) != self.nout:
            raise TypeError("Not the right number of outputs produced for %s(%s). Expected %s, got %s."
                            % (self, ", ".join(str(input) for input in inputs), self.nout, len(outputs)))
        return Apply(self, inputs, outputs)

    def output_types(self, types):
        if hasattr(self, 'output_types_preference'):
            results = self.output_types_preference(*types)
            if not isinstance(results, (list, tuple)) or any(not isinstance(x, Type) for x in results):
                raise TypeError("output_types_preference should return a list or a tuple of types", self.output_types_preference, results)
            if len(results) != self.nout:
                raise TypeError("Not the right number of outputs produced for %s(%s) by %s. Expected %s, got ?s."
                                % (self, ", ".join(str(input.type) for input in inputs),
                                   self.output_types_preference, self.nout, len(results)))
            return results
        else:
            raise NotImplementedError("Cannot calculate the output types for %s" % self)

    def perform(self, node, inputs, output_storage):
        if self.nout == 1:
            output_storage[0][0] = self.impl(*inputs)
        else:
            results = utils.from_return_values(self.impl(*inputs))
            assert len(results) == len(output_storage)
            for storage, result in zip(output_storage, results):
                storage[0] = result

    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def grad(self, inputs, output_gradients):
        raise AbstractFunctionError()

    def __eq__(self, other):
        return type(self) == type(other) \
            and getattr(self, 'output_types_preference', None) \
            == getattr(other, 'output_types_preference', None)

    def __hash__(self):
        return hash(getattr(self, 'output_types_preference', 0))

    def __str__(self):
        if hasattr(self, 'name') and self.name:
            return self.name
        else:
            return "%s{%s}" % (self.__class__.__name__, ", ".join("%s=%s" % (k, v) for k, v in self.__dict__.items() if k != "name"))


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
lt = LT()

class GT(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        return x > y
gt = GT()

class LE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        return x <= y
le = LE()

class GE(LogicalComparison):
    identity = False
    commutative = False
    associative = False
    def impl(self, x, y):
        return x >= y
ge = GE()

class EQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x == y
eq = EQ()

class NEQ(LogicalComparison):
    identity = False
    commutative = True
    associative = False
    def impl(self, x, y):
        return x != y
neq = NEQ()

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
        return (gz, ) * len(inputs)
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
        return [mul(*([gz] + utils.difference(inputs, [input])))
                for input in inputs]
mul = Mul(upcast_out, name = 'mul')

class Sub(BinaryScalarOp):
    def impl(self, x, y):
        return x - y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s - %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz, -gz
sub = Sub(upcast_out, name = 'sub')

class Div(BinaryScalarOp):
    def impl(self, x, y):
        return x / y
    def c_code(self, node, name, (x, y), (z, ), sub):
        if node.inputs[0].type in int_types and node.inputs[1].type in int_types:
            raise NotImplementedError("For integer arguments the behavior of division in C and in Python differ when the quotient is negative (to implement).")
        return "%(z)s = %(x)s / %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz / y, -(gz * x) / (y * y)
div = Div(upcast_out, name = 'div')

class Mod(BinaryScalarOp):
    def impl(self, x, y):
        return x % y
    def c_code(self, node, name, (x, y), (z, ), sub):
        raise NotImplementedError("Unlike Python, C's modulo returns negative modulo on negative dividend (to implement)")
    def grad(self, (x, y), (gz, )):
        return None, None
mod = Mod(upcast_out, name = 'mod')

class Pow(BinaryScalarOp):
    def impl(self, x, y):
        return x ** y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = pow(%(x)s, %(y)s);" % locals()
    def grad(self, (x, y), (gz, )):
        return gz * y * x**(y - 1), gz * log(x) * x**y
pow = Pow(upcast_out, name = 'pow')

class First(BinaryScalarOp):
    def impl(self, x, y):
        return x
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz, None
first = First(transfer_type(0), name = 'first')

class Second(BinaryScalarOp):
    def impl(self, x, y):
        return y
    def c_code(self, node, name, (x, y), (z, ), sub):
        return "%(z)s = %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return None, gz
second = Second(transfer_type(1), name = 'second')



class Identity(UnaryScalarOp):
    def impl(self, x):
        return x
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, ), (gz, )):
        return gz,
identity = Identity(same_out, name = 'identity')

class Neg(UnaryScalarOp):
    def impl(self, x):
        return -x
    def grad(self, (x, ), (gz, )):
        return -gz,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = -%(x)s;" % locals()
neg = Neg(same_out, name = 'neg')

class Abs(UnaryScalarOp):
    #TODO: for complex input, output is some flavour of float
    def impl(self, x):
        return numpy.abs(x)
    def grad(self, (x, ), (gz, )):
        return gz * sgn(x),
    def c_code(self, node, name, (x, ), (z, ), sub):
        type = node.inputs[0].type
        if type in int_types:
            return "%(z)s = abs(%(x)s);" % locals()
        if type in float_types:
            return "%(z)s = fabs(%(x)s);" % locals()
        #complex, other?
        raise NotImplementedError('type not supported', type)
_abs = Abs(same_out)

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

class Inv(UnaryScalarOp):
    def impl(self, x):
        return 1.0 / x
    def grad(self, (x, ), (gz, )):
        return -gz / (x * x),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = 1.0 / %(x)s;" % locals()
inv = Inv(upgrade_to_float, name = 'inv')

class Log(UnaryScalarOp):
    def impl(self, x):
        return math.log(x)
    def grad(self, (x, ), (gz, )):
        return gz / x,
    def c_code(self, node, name, (x, ), (z, ), sub):
        #todo: the version using log2 seems to be very slightly faster
        # on some machines for some reason, check if it's worth switching
        #return "%(z)s = log2(%(x)s) * 0.69314718055994529;" % locals()
        return "%(z)s = log(%(x)s);" % locals()
log = Log(upgrade_to_float, name = 'log')

class Log2(UnaryScalarOp):
    def impl(self, x):
        return numpy.log2(x)
    def grad(self, (x, ), (gz, )):
        return gz / (x * math.log(2.0)),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = log2(%(x)s);" % locals()
log2 = Log2(upgrade_to_float, name = 'log2')

class Exp(UnaryScalarOp):
    def impl(self, x):
        return math.exp(x)
    def grad(self, (x, ), (gz, )):
        return gz * exp(x),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = exp(%(x)s);" % locals()
exp = Exp(upgrade_to_float, name = 'exp')

class Sqr(UnaryScalarOp):
    def impl(self, x):
        return x*x
    def grad(self, (x, ), (gz, )):
        return gz * x * 2,
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = %(x)s * %(x)s;" % locals()
sqr = Sqr(same_out, name = 'sqr')

class Sqrt(UnaryScalarOp):
    def impl(self, x):
        return math.sqrt(x)
    def grad(self, (x, ), (gz, )):
        return (gz * 0.5) / sqrt(x),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = sqrt(%(x)s);" % locals()
sqrt = Sqrt(upgrade_to_float, name = 'sqrt')

class Cos(UnaryScalarOp):
    def impl(self, x):
        return math.cos(x)
    def grad(self, (x, ), (gz, )):
        return -gz * sin(x),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = cos(%(x)s);" % locals()
cos = Cos(upgrade_to_float, name = 'cos')

class Sin(UnaryScalarOp):
    def impl(self, x):
        return math.sin(x)
    def grad(self, (x, ), (gz, )):
        return gz * cos(x),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = sin(%(x)s);" % locals()
sin = Sin(upgrade_to_float, name = 'sin')

class Tan(UnaryScalarOp):
    def impl(self, x):
        return math.tan(x)
    def grad(self, (x, ), (gz, )):
        return gz / sqr(cos(x)),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = tan(%(x)s);" % locals()
tan = Tan(upgrade_to_float, name = 'tan')

class Cosh(UnaryScalarOp):
    """
    sinh(x) = (exp(x) + exp(-x)) / 2
    """
    def impl(self, x):
        return math.cosh(x)
    def grad(self, (x, ), (gz, )):
        return gz * sinh(x),
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
        return gz * cosh(x),
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
        return gz * (1 - sqr(tanh(x))),
    def c_code(self, node, name, (x, ), (z, ), sub):
        return "%(z)s = tanh(%(x)s);" % locals()
tanh = Tanh(upgrade_to_float, name = 'tanh')



class Composite(ScalarOp):

    def __init__(self, inputs, outputs):
        env = Env(*gof.graph.clone(inputs, outputs))
        inputs, outputs = env.inputs, env.outputs

        for node in env.nodes:
            if not isinstance(node.op, ScalarOp):
                raise ValueError("The env to Composite must be exclusively composed of ScalarOp instances.")

        subd = dict(zip(inputs,
                        ["%%(i%i)s"%i for i in range(len(inputs))]) +
                    zip(outputs,
                        ["%%(o%i)s"%i for i in range(len(outputs))]))

        for orphan in env.results: #env.orphans:
            if orphan.owner is None and orphan not in env.inputs:
                if isinstance(orphan, Constant):
                    subd[orphan] = orphan.type.c_literal(orphan.data)
                else:
                    raise ValueError("All orphans in the env to Composite must be Constant instances.")

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

    def output_types(self, input_types):
        if tuple(input_types) != tuple([input.type for input in self.env.inputs]):
            raise TypeError("Wrong types for Composite. Expected %s, got %s."
                            % (tuple([input.type for input in self.env.inputs]), tuple(input_types)))
        return [output.type for output in self.env.outputs]

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
        return self._c_code % d

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)
