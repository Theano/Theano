
import numpy
import math

from copy import copy
import inspect

import gof
from gof import Result, GuardedOp, Env, utils


def as_scalar(x, name = None):
    if isinstance(x, float):
        s = Scalar('float64', name = name)
        s.data = x
        return s
    if isinstance(x, int):
        s = Scalar('int32', name = name)
        s.data = x
        return s
    if isinstance(x, Scalar):
        return x


class Scalar(Result):

    def __init__(self, dtype, name = None):
        Result.__init__(self, role = None, name = name)
        self.dtype = dtype
        self.dtype_specs()

    def __get_constant(self):
        if not hasattr(self, '_constant'):
            return False
        return self._constant

    def __set_constant(self, value):
        if value:
            self.indestructible = True
        self._constant = value

    constant = property(__get_constant, __set_constant)
        
    def filter(self, data):
        py_type = self.dtype_specs()[0]
        return py_type(data)

    def same_properties(self, other):
        return other.dtype == self.dtype

    def dtype_specs(self):
        try:
            return {'float32': (float, 'npy_float32', 'PyFloat_Check', 'PyFloat_AsDouble', 'PyFloat_FromDouble'),
                    'float64': (float, 'npy_float64', 'PyFloat_Check', 'PyFloat_AsDouble', 'PyFloat_FromDouble'),
                    'int8': (int, 'npy_int8', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int16': (int, 'npy_int16', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int32': (int, 'npy_int32', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int64': (int, 'npy_int64', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'complex128': (complex, 'theano_complex128', 'PyComplex_Check', 'PyComplex_AsCComplex', 'PyComplex_FromCComplex'),
                    'complex64': (complex, 'theano_complex64', None, None, None)}[self.dtype]
        except KeyError:
            raise TypeError("Unsupported dtype for %s: %s" % (self.__class__.__name__, self.dtype))

    def c_literal(self):
        if 'complex' in self.dtype:
            raise NotImplementedError("No literal for complex values.")
        return str(self.data)

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

    def c_support_code(cls):
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

    def __copy__(self):
        """
        Return a copy of this instance (with its own attributes)
        """
        cpy = self.__class__(self.dtype, self.name)
        cpy.data = self.data
        return cpy

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

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div(other,self)
    def __rpow__(self,other): return pow(other,self)



class ScalarMixedOp(GuardedOp):
    """Olivier: document this stuff! -JB"""

    nin = -1
    nout = 1
    
    def __init__(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)" \
                                    % (self.__class__.__name__, len(inputs), self.nin))
        
        inputs = [as_scalar(input) for input in inputs]
        i_dtypes = [getattr(input, 'dtype', None) for input in inputs]
        o_dtypes = self.propagate_dtypes(*i_dtypes)

        self.inputs = inputs
        self.outputs = [Scalar(dtype) for dtype in o_dtypes]

    def propagate_dtypes(self, *inputs):
        raise AbstractFunctionError()
    
    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def grad(self, inputs, output_gradients):
        raise AbstractFunctionError()
    
    def perform(self):
        self.outputs[0].data = self.impl(*[input.data for input in self.inputs])


def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return str(z.dtype)


class PureScalarOp(ScalarMixedOp):

    cast_method = lambda self, *args: upcast(*args)
    
    def propagate_dtypes(self, *i_dtypes):
        for dtype in i_dtypes:
            if dtype is None:
                raise TypeError("Expected a Scalar.")
        return [self.cast_method(*i_dtypes)] * self.nout


class UnaryScalarOp(PureScalarOp):
    nin = 1

class BinaryScalarOp(PureScalarOp):
    nin = 2



class Add(BinaryScalarOp):
    identity = 0
    def impl(self, x, y):
        return x + y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s + %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz, gz

class Sub(BinaryScalarOp):
    def impl(self, x, y):
        return x - y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s - %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz, -gz

class Mul(BinaryScalarOp):
    def impl(self, x, y):
        return x * y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s * %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz * y, gz * x

class Div(BinaryScalarOp):
    def impl(self, x, y):
        return x / y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s / %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz / y, -(gz * x) / (y * y)

class Pow(BinaryScalarOp):
    def impl(self, x, y):
        return x ** y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = pow(%(x)s, %(y)s);" % locals()
    def grad(self, (x, y), (gz, )):
        return gz * y * x**(y - 1), gz * log(x) * x**y

class First(BinaryScalarOp):
    def impl(self, x, y):
        return x
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz, None

class Second(BinaryScalarOp):
    def impl(self, x, y):
        return y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return None, gz


class Identity(UnaryScalarOp):
    def impl(self, x):
        return x
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()
    def grad(self, (x, ), (gz, )):
        return gz,

class Neg(UnaryScalarOp):
    def impl(self, x):
        return -x
    def grad(self, (x, ), (gz, )):
        return -gz,
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = -%(x)s;" % locals()

class Abs(UnaryScalarOp):
    def impl(self, x):
        return numpy.abs(x)
    def grad(self, (x, ), (gz, )):
        return gz * sgn(x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = abs(%(x)s);" % locals()

class Sgn(UnaryScalarOp):
    def impl(self, x):
        return numpy.abs(x) / x
    def grad(self, (x, ), (gz, )):
        return None,
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = %(x)s/abs(%(x)s);" % locals() # TODO: C use copysign

class Inv(UnaryScalarOp):
    def impl(self, x):
        return 1 / x
    def grad(self, (x, ), (gz, )):
        return -gz / (x * x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = 1 / %(x)s;" % locals()

class Log(UnaryScalarOp):
    def impl(self, x):
        return math.log(x)
    def grad(self, (x, ), (gz, )):
        return gz / x,
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = log(%(x)s);" % locals()

class Log2(UnaryScalarOp):
    def impl(self, x):
        return numpy.log2(x)
    def grad(self, (x, ), (gz, )):
        return gz / (x * math.log(2.0)),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = log2(%(x)s);" % locals()

class Exp(UnaryScalarOp):
    def impl(self, x):
        return math.exp(x)
    def grad(self, (x, ), (gz, )):
        return gz * exp(x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = exp(%(x)s);" % locals()

class Sqr(UnaryScalarOp):
    def impl(self, x):
        return x*x
    def grad(self, (x, ), (gz, )):
        return gz * x * 2,
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = %(x)s * %(x)s;" % locals()

class Sqrt(UnaryScalarOp):
    def impl(self, x):
        return math.sqrt(x)
    def grad(self, (x, ), (gz, )):
        return (gz * 0.5) / sqrt(x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = sqrt(%(x)s);" % locals()



#NOTE WELL!!!
# The following adds functions to this module automatically.
# For every scalar op class, a lower-case symbol is added which is a constructor
# for that class.

from gof import modes
modes.make_constructors(globals())



def composite(inputs, outputs):
    """
    Usage: composite(inputs, outputs)

    Produces an Op class which represents the computations
    between the provided inputs and outputs as a single
    operation.
    
    The operations between inputs and outputs (as given by
    Env(inputs, outputs).ops()) must all be instances of
    PureScalarOp.

    Examples:
      x, y = Scalar(), Scalar()
      SquareDiff = composite([x, y], [(x - y)**2])
      TimesTen = composite([x], [x * 10.0])
      Neighbors = composite([x], [x - 1, x + 1])
    """
    
    env = Env(inputs, outputs).clone()
    gof.opt.ConstantFinder().apply(env)
    
    inputs, outputs = env.inputs, env.outputs

    for op in env.ops():
        if not isinstance(op, PureScalarOp):
            raise ValueError("The input env to composite must be exclusively composed of PureScalarOp instances.")

    subd = dict(zip(inputs,
                    ["%%(i%i)s"%i for i in range(len(inputs))]) +
                zip(outputs,
                    ["%%(o%i)s"%i for i in range(len(outputs))]))
    
    for orphan in env.orphans():
        if orphan.constant:
            subd[orphan] = orphan.c_literal()
        else:
            raise ValueError("All orphans in the input env to composite must be constant.")

    _c_code = "{\n"
    i = 0
    j = 0
    for op in env.toposort():
        j += 1
        for output in op.outputs:
            if output not in subd:
                i += 1
                name = "V%%(id)s_tmp%i" % i
                subd[output] = name
                # the c code is not robust to any other dtypes than those of the specified inputs
                # a solution would be to require Composite.c_code to fill in the dtypes using
                # a proper upcast
                _c_code += "%s %s;\n" % (output.dtype_specs()[1], name)
        _c_code += op.c_code([subd[input] for input in op.inputs],
                             [subd[output] for output in op.outputs],
                             dict(fail = "%(fail)s",
                                  id = "%%(id)s_%i" % j))
        _c_code += "\n"
    _c_code += "}\n"

    
    def compose_impl(r):
        # this is not optimal at all eg in add(*1 -> mul(x, y), *1)
        # it will calculate *1 twice
        # it also doesn't follow env.toposort but that's (presumably)
        # still correct since we only have pure scalar ops
        if r in env.inputs:
            idx = env.inputs.index(r)
            return lambda inputs: inputs[idx]
        elif r in env.orphans():
            return lambda inputs: r.data
        op = r.owner
        producers = [compose_impl(input) for input in op.inputs]
        return lambda inputs: op.impl(*[p(inputs) for p in producers])

    _impls = [compose_impl(r) for r in env.outputs]
    
    class Composite(PureScalarOp):

        nin = len(inputs)
        nout = len(outputs)

        # todo: propagate_dtypes?
        
        def perform(self):
            inputs = [input.data for input in self.inputs]
            for output, impl in zip(self.outputs, _impls):
                output.data = impl(inputs)

        def grad(self, inputs, output_grads):
            raise NotImplementedError("grad is not implemented for Composite")

        def c_code(self, inames, onames, sub):
            d = dict(zip(["i%i"%i for i in range(len(inames))],
                         inames) +
                     zip(["o%i"%i for i in range(len(onames))],
                         onames),
                     **sub)
            return _c_code % d

    return Composite

