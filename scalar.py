
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

def constant(x):
    res = as_scalar(x)
    res.constant = True
    return res


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

    def desc(self):
        return (self.dtype, self.data)
    
    def filter(self, data):
        py_type = self.dtype_specs()[0]
        return py_type(data)

    def same_properties(self, other):
        return other.dtype == self.dtype

    def dtype_specs(self):
        try:
            return {'float32': (numpy.float32, 'npy_float32', 'PyFloat_Check', 'PyFloat_AsDouble', 'PyFloat_FromDouble'),
                    'float64': (numpy.float64, 'npy_float64', 'PyFloat_Check', 'PyFloat_AsDouble', 'PyFloat_FromDouble'),
                    'complex128': (numpy.complex128, 'theano_complex128', 'PyComplex_Check', 'PyComplex_AsCComplex', 'PyComplex_FromCComplex'),
                    'complex64': (numpy.complex64, 'theano_complex64', None, None, None),
                    'int8':  (numpy.int8, 'npy_int8', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int16': (numpy.int16, 'npy_int16', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int32': (numpy.int32, 'npy_int32', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong'),
                    'int64': (numpy.int64, 'npy_int64', 'PyInt_Check', 'PyInt_AsLong', 'PyInt_FromLong')
                    }[self.dtype]
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
        """Return a copy of this instance (with its own attributes)"""
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



def upcast(dtype, *dtypes):
    z = numpy.zeros((), dtype = dtype)
    for dtype in dtypes:
        z = z + numpy.zeros((), dtype = dtype)
    return str(z.dtype)

class ScalarOp(GuardedOp):

    nin = -1
    nout = 1
        
    def __init__(self, *inputs):
        if self.nin >= 0:
            if len(inputs) != self.nin:
                raise TypeError("Wrong number of inputs for %s (got %i, expected %i)" \
                                    % (self.__class__.__name__, len(inputs), self.nin))
        else:
            self.nin = len(inputs)
        
        inputs = [as_scalar(input) for input in inputs]
        i_dtypes = [getattr(input, 'dtype', None) for input in inputs]
        o_dtypes = self.output_dtypes(*i_dtypes)

        self.inputs = inputs
        self.outputs = [Scalar(dtype) for dtype in o_dtypes]

    def output_dtypes(self, *dtypes):
        if self.nout != 1:
            raise NotImplementedError()
        return upcast(*dtypes),

    def impl(self, *inputs):
        raise AbstractFunctionError()
    
    def grad(self, inputs, output_gradients):
        raise AbstractFunctionError()
    
    def perform(self):
        if self.nout == 1:
            self.outputs[0].data = self.impl(*[input.data for input in self.inputs])
        else:
            results = utils.from_return_values(self.impl(*[input.data for input in self.inputs]))
            for output, result in zip(self.outputs, results):
                output.data = result

class UnaryScalarOp(ScalarOp):
    nin = 1

class BinaryScalarOp(ScalarOp):
    nin = 2

class FloatUnaryScalarOp(UnaryScalarOp):
    def output_dtypes(self, input_dtype):
        if 'int' in input_dtype: return 'float64',
        if 'float' in input_dtype: return input_dtype,
        raise NotImplementedError()



class Add(ScalarOp):
    identity = 0
    def impl(self, *inputs):
        return sum(inputs)
    def c_code(self, inputs, (z, ), sub):
        if not inputs:
            return z + " = 0;"
        else:
            return z + " = " + " + ".join(inputs) + ";"
    def grad(self, inputs, (gz, )):
        return (gz, ) * len(inputs)

class Mul(ScalarOp):
    identity = 1
    def impl(self, *inputs):
        return numpy.product(inputs)
    def c_code(self, inputs, (z, ), sub):
        if not inputs:
            return z + " = 1;"
        else:
            return z + " = " + " * ".join(inputs) + ";"
    def grad(self, inputs, (gz, )):
        return [mul(*([gz] + utils.difference(inputs, [input])))
                for input in inputs]


class Sub(BinaryScalarOp):
    def impl(self, x, y):
        return x - y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s - %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return gz, -gz

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
    #TODO: for complex input, output is some flavour of float
    def impl(self, x):
        return numpy.abs(x)
    def grad(self, (x, ), (gz, )):
        return gz * sgn(x),
    def c_code(self, (x, ), (z, ), sub):
        dtype = str(self.inputs[0].dtype)
        if 'int' in dtype:
            return "%(z)s = abs(%(x)s);" % locals()
        if 'float' in dtype:
            return "%(z)s = fabs(%(x)s);" % locals()
        #complex, other?
        raise NotImplementedError('dtype not supported', dtype)

class Sgn(UnaryScalarOp):
    def impl(self, x):
        #casting to output type is handled by filter
        return 1.0 if x >= 0 else -1.0
    def grad(self, (x, ), (gz, )):
        return None,
    def c_code(self, (x, ), (z, ), sub):
        #casting is done by compiler
        #TODO: use copysign
        return "%(z)s = (%(x)s >= 0) ? 1.0 : -1.0;" % locals()

class Inv(FloatUnaryScalarOp):
    def impl(self, x):
        return 1.0 / x
    def grad(self, (x, ), (gz, )):
        return -gz / (x * x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = 1.0 / %(x)s;" % locals()

class Log(FloatUnaryScalarOp):
    def impl(self, x):
        return math.log(x)
    def grad(self, (x, ), (gz, )):
        return gz / x,
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = log(%(x)s);" % locals()

class Log2(FloatUnaryScalarOp):
    def impl(self, x):
        return numpy.log2(x)
    def grad(self, (x, ), (gz, )):
        return gz / (x * math.log(2.0)),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = log2(%(x)s);" % locals()

class Exp(FloatUnaryScalarOp):
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

class Sqrt(FloatUnaryScalarOp):
    def impl(self, x):
        return math.sqrt(x)
    def grad(self, (x, ), (gz, )):
        return (gz * 0.5) / sqrt(x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = sqrt(%(x)s);" % locals()

class Cos(FloatUnaryScalarOp):
    def impl(self, x):
        return math.cos(x)
    def grad(self, (x, ), (gz, )):
        return gz * sin(x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = cos(%(x)s);" % locals()

class Sin(FloatUnaryScalarOp):
    def impl(self, x):
        return math.sin(x)
    def grad(self, (x, ), (gz, )):
        return -gz * cos(x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = sin(%(x)s);" % locals()

class Tan(FloatUnaryScalarOp):
    def impl(self, x):
        return math.tan(x)
    def grad(self, (x, ), (gz, )):
        raise NotImplementedError()
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = tan(%(x)s);" % locals()

class Cosh(FloatUnaryScalarOp):
    def impl(self, x):
        return math.cosh(x)
    def grad(self, (x, ), (gz, )):
        raise NotImplementedError()
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = cosh(%(x)s);" % locals()

class Sinh(FloatUnaryScalarOp):
    def impl(self, x):
        return math.sinh(x)
    def grad(self, (x, ), (gz, )):
        raise NotImplementedError()
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = sin(%(x)s);" % locals()

class Tanh(FloatUnaryScalarOp):
    def impl(self, x):
        return math.tanh(x)
    def grad(self, (x, ), (gz, )):
        return gz * (1 - tanh(x))**2
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = tanh(%(x)s);" % locals()


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
    ScalarOp.

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
        if not isinstance(op, ScalarOp):
            raise ValueError("The input env to composite must be exclusively composed of ScalarOp instances.")

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
        # still correct since we only have scalar ops
        if r in env.inputs:
            idx = env.inputs.index(r)
            return lambda inputs: inputs[idx]
        elif r in env.orphans():
            return lambda inputs: r.data
        op = r.owner
        producers = [compose_impl(input) for input in op.inputs]
        return lambda inputs: op.impl(*[p(inputs) for p in producers])

    _impls = [compose_impl(r) for r in env.outputs]
    
    class Composite(ScalarOp):

        nin = len(inputs)
        nout = len(outputs)

        def output_dtypes(self, *input_dtypes):
            assert input_dtypes == tuple([input.dtype for input in inputs])
            return [output.dtype for dtype in outputs]

        def perform(self):
            inputs = [input.data for input in self.inputs]
            for output, impl in zip(self.outputs, _impls):
                output.data = impl(inputs)

        def impl(self, *inputs):
            for r, input in zip(self.inputs, inputs):
                r.data = input
            self.perform()
            return utils.to_return_values([output.data for output in self.outputs])

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

