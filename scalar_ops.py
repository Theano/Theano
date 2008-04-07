
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
        return gz, -gz

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
        return div(gz, y), -div(mul(x, gz), y*y)

class Pow(BinaryScalarOp):
    def impl(self, x, y):
        return x ** y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = pow(%(x)s, %(y)s);" % locals()

class First(BinaryScalarOp):
    def impl(self, x, y):
        return x
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s;" % locals()

class Second(BinaryScalarOp):
    def impl(self, x, y):
        return y
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(y)s;" % locals()

class SquareDiff(BinaryScalarOp):
    def impl(self, x, y):
        diff = (x - y)
        return diff * diff
    def c_code(self, (x, y), (z, ), sub):
        return "%(z)s = %(x)s - %(y)s; %(z)s *= %(z)s;" % locals()


class Neg(UnaryScalarOp):
    def impl(self, x):
        return -x
    def grad(self, (x, ), (gz, )):
        return -gz
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = -%(x)s;" % locals()

class Inv(UnaryScalarOp):
    def impl(self, x):
        return 1 / x
    def grad(self, (x, ), (gz, )):
        return -gz / (x*x)
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = 1 / %(x)s;" % locals()

class Log(UnaryScalarOp):
    def impl(self, x):
        return math.log(x)
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = log(%(x)s);" % locals()

class Exp(UnaryScalarOp):
    def impl(self, x):
        return math.exp(x)
    def c_code(self, (x, ), (z, ), sub):
        return "%(z)s = exp(%(x)s);" % locals()


# class Sigmoid(UnaryComposite):
#     def expand_impl(self, x):
#         return 1.0 / (1.0 + exp(-x))


from gof import modes
modes.make_constructors(globals())

