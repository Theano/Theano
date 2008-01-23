
from utils import OmegaError

class OmegaTypeError(OmegaError, TypeError):
    pass


############################
# Dispatcher
############################

class Dispatcher(list):

    all_dispatchers = {}

    def __init__(self, name, description):
        self.name = name
        self.description = description
        self.all_dispatchers[name] = self

    def __call__(self, *inputs, **opts):
        for candidate in self:
            try:
                return candidate(*inputs, **opts)
            except OmegaTypeError:
                continue
        if opts:
            s = " with options %s" % opts
        else:
            s = ""
        raise OmegaTypeError("No candidate found for %s(%s) %s" \
                             % (self.name,
                                ", ".join([input.__class__.__name__ for input in inputs]),
                                s))

    def add_handler(self, x):
        self.insert(0, x)

    def fallback_handler(self, x):
        self.append(x)



# Dispatchers for all python operators
Add = Dispatcher("Add", "x + y")
Subtract = Dispatcher("Subtract", "x - y")
Multiply = Dispatcher("Multiply", "x * y")
Divide = Dispatcher("Divide", "x / y")
FloorDivide = Dispatcher("FloorDivide", "x // y")
Modulo = Dispatcher("Modulo", "x % y")
Power = Dispatcher("Power", "x ** y")
Negate = Dispatcher("Negate", "-x")
Abs = Dispatcher("Abs", "abs(x)")
LeftShift = Dispatcher("LeftShift", "x << y")
RightShift = Dispatcher("RightShift", "x >> y")
Equals = Dispatcher("Equals", "x == y")
NotEquals = Dispatcher("NotEquals", "x != y")
Less = Dispatcher("Less", "x < y")
LessOrEqual = Dispatcher("LessOrEqual", "x <= y")
Greater = Dispatcher("Greater", "x > y")
GreaterOrEqual = Dispatcher("GreaterOrEqual", "x >= y")
Contains = Dispatcher("Contains", "x in y")
BinaryOr = Dispatcher("BinaryOr", "x | y")
BinaryAnd = Dispatcher("BinaryAnd", "x & y")
BinaryXor = Dispatcher("BinaryXor", "x ^ y")
BinaryInverse = Dispatcher("BinaryInverse", "~x")

# Dispatchers for special operations
Transpose = Dispatcher("Transpose", "x.T")

# Others
Log = Dispatcher("Log", 'log(x)')
Exp = Dispatcher("Exp", 'exp(x)')
Sin = Dispatcher("Sin", 'sin(x)')
Cos = Dispatcher("Cos", 'cos(x)')
Tan = Dispatcher("Tan", 'tan(x)')



############################
# PythonOperatorSupport
############################

class PythonOperatorSupport(object):
    """Support for built-in Python operators."""

    # Common arithmetic operations
    def __add__(self, x):
        return Add(self, x)

    def __radd__(self, x):
        return Add(x, self)

    def __sub__(self, x):
        return Subtract(self, x)

    def __rsub__(self, x):
        return Subtract(x, self)

    def __mul__(self, x):
        return Multiply(self, x)

    def __rmul__(self, x):
        return Multiply(x, self)

    def __div__(self, x):
        return Divide(self, x)

    def __rdiv__(self, x):
        return Divide(x, self)

    def __floordiv__(self, x):
        return FloorDivide(self, x)

    def __rfloordiv__(self, x):
        return FloorDivide(x, self)

    def __mod__(self, x):
        return Modulo(self, x)

    def __rmod__(self, x):
        return Modulo(x, self)

    def __pow__(self, x):
        return Power(self, x)

    def __rpow__(self, x):
        return Power(x, self)

    def __neg__(self):
        return Negate(self)

    def __abs__(self):
        return Abs(self)


    # Less common arithmetic operations
    def __lshift__(self, x):
        return LeftShift(self, x)

    def __rlshift__(self, x):
        return LeftShift(x, self)

    def __rshift__(self, x):
        return RightShift(self, x)

    def __rrshift__(self, x):
        return RightShift(x, self)


    # Comparison operations
#     def __eq__(self, x):
#         return Equals(self, x)

#     def __ne__(self, x):
#         return NotEquals(self, x)

    def __lt__(self, x):
        return Less(self, x)

    def __le__(self, x):
        return LessOrEqual(self, x)

    def __gt__(self, x):
        return Greater(self, x)

    def __ge__(self, x):
        return GreaterOrEqual(self, x)

    def __contains__(self, x):
        return Contains(self, x)


    # Binary operations
    def __or__(self, x):
        return BinaryOr(self, x)

    def __ror__(self, x):
        return BinaryOr(x, self)

    def __and__(self, x):
        return BinaryAnd(self, x)

    def __rand__(self, x):
        return BinaryAnd(x, self)

    def __xor__(self, x):
        return BinaryXor(self, x)

    def __rxor__(self, x):
        return BinaryXor(x, self)

    def __invert__(self):
        return BinaryInverse(self)


    # Other operations
    T = property(lambda self: Transpose(self))

    norm = property(lambda self: Norm(self))


    # Always nonzero
    def __nonzero__(self):
        return True


__all__ = globals().keys()



