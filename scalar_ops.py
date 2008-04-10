
from scalar import *
import math


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
        return gz, neg(gz)

class Mul(BinaryScalarOp):
    def impl(self, x, y):
        return x * y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s * %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return mul(y, gz), mul(x, gz)

class Div(BinaryScalarOp):
    def impl(self, x, y):
        return x / y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s / %(y)s;" % locals()
    def grad(self, (x, y), (gz, )):
        return div(gz, y), neg(div(mul(x, gz), mul(y, y)))

class Pow(BinaryScalarOp):
    def impl(self, x, y):
        return x ** y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = pow(%(x)s, %(y)s);" % locals()
    def grad(self, (x, y), (gz, )):
        return mul(gz, mul(y, pow(x, sub(y, as_scalar(1))))), mul(gz, mul(log(x), pow(x, y)))

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
    def grad(self, (x, y), (gz, )):
        return gz,

class Neg(UnaryScalarOp):
    def impl(self, x):
        return -x
    def grad(self, (x, ), (gz, )):
        return neg(gz),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = -%(x)s;" % locals()

class Abs(UnaryScalarOp):
    def impl(self, x):
        return numpy.abs(x)
    def grad(self, (x, ), (gz, )):
        return mul(gz, sgn(x)),
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
        return div(neg(gz), mul(x, x)),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = 1 / %(x)s;" % locals()

class Log(UnaryScalarOp):
    def impl(self, x):
        return math.log(x)
    def grad(self, (x, ), (gz, )):
        return div(gz, x),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = log(%(x)s);" % locals()

class Log2(UnaryScalarOp):
    def impl(self, x):
        return numpy.log2(x)
    def grad(self, (x, ), (gz, )):
        return div(gz, mul(x, as_scalar(math.log(2.0)))),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = log2(%(x)s);" % locals()

class Exp(UnaryScalarOp):
    def impl(self, x):
        return math.exp(x)
    def grad(self, (x, ), (gz, )):
        return mul(gz, exp(x)),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = exp(%(x)s);" % locals()

class Sqr(UnaryScalarOp):
    def impl(self, x):
        return x*x
    def grad(self, (x, ), (gz, )):
        return mul(gz, mul(x, as_scalar(2))),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = %(x)s * %(x)s;" % locals()

class Sqrt(UnaryScalarOp):
    def impl(self, x):
        return math.sqrt(x)
    def grad(self, (x, ), (gz, )):
        return div(mul(gz, as_scalar(0.5)), sqrt(x)),
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = sqrt(%(x)s);" % locals()


# class Sigmoid(UnaryComposite):
#     def expand_impl(self, x):
#         return 1.0 / (1.0 + exp(-x))


from gof import modes
modes.make_constructors(globals())

