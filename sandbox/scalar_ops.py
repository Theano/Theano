
from scalar import *
import math


class Add(BinaryScalarOp):
    def impl(self, x, y):
        return x + y
    def c_impl(self, (x, y), z):
        return "%(z)s = %(x)s + %(y)s;"
    def grad(self, (x, y), gz):
        return gz, gz

class Sub(BinaryScalarOp):
    def impl(self, x, y):
        return x - y
    def c_impl(self, (x, y), z):
        return "%(z)s = %(x)s - %(y)s;"
    def grad(self, (x, y), gz):
        return gz, -gz

class Mul(BinaryScalarOp):
    def impl(self, x, y):
        return x * y
    def c_impl(self, (x, y), z):
        return "%(z)s = %(x)s * %(y)s;"
    def grad(self, (x, y), gz):
        return mul(y, gz), mul(x, gz)

class Div(BinaryScalarOp):
    def impl(self, x, y):
        return x / y
    def c_impl(self, (x, y), z):
        return "%(z)s = %(x)s / %(y)s;"
    def grad(self, (x, y), gz):
        return div(gz, y), -div(mul(x, gz), y*y)

class Pow(BinaryScalarOp):
    def impl(self, x, y):
        return x ** y
    def c_impl(self, (x, y), z):
        return "%(z)s = pow(%(x)s, %(y)s);"


class Neg(UnaryScalarOp):
    def impl(self, x):
        return -x
    def grad(self, x, gz):
        return -gz
    def c_impl(self, x, z):
        return "%(z)s = -%(x)s;"

class Inv(UnaryScalarOp):
    def impl(self, x):
        return 1 / x
    def grad(self, x, gz):
        return -gz / (x*x)
    def c_impl(self, x, z):
        return "%(z)s = 1 / %(x)s;"

class Log(UnaryScalarOp):
    def impl(self, x):
        return math.log(x)
    def c_impl(self, x, z):
        return "%(z)s = log(%(x)s);"

class Exp(UnaryScalarOp):
    def impl(self, x):
        return math.exp(x)
    def c_impl(self, x, z):
        return "%(z)s = exp(%(x)s);"


# class Sigmoid(UnaryComposite):
#     def expand_impl(self, x):
#         return 1.0 / (1.0 + exp(-x))


from gof import modes
modes.make_constructors(globals())

