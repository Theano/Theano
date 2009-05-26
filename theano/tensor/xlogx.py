
import theano
import numpy

from elemwise import Elemwise
from theano import scalar

class XlogX(scalar.UnaryScalarOp):
    """
    Compute X * log(X), with special case 0 log(0) = 0.
    """
    @staticmethod
    def st_impl(x):
        if x == 0.0:
            return 0.0
        return x * numpy.log(x)
    def impl(self, x):
        return XlogX.st_impl(x)
    def grad(self, (x,), (gz,)):
        return [gz * (1 + scalar.log(x))]
    def c_code(self, node, name, (x,), (z,), sub):
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                %(x)s == 0.0
                ? 0.0
                : %(x)s * log(%(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
scalar_xlogx  = XlogX(scalar.upgrade_to_float, name='scalar_xlogx')
xlogx = Elemwise(scalar_xlogx, name='xlogx')


class XlogY0(scalar.BinaryScalarOp):
    """
    Compute X * log(Y), with special case 0 log(0) = 0.
    """
    @staticmethod
    def st_impl(x, y):
        if x == 0.0:
            return 0.0
        return x * numpy.log(y)
    def impl(self, x, y):
        return XlogY0.st_impl(x, y)
    def grad(self, (x, y), (gz,)):
        return [gz * scalar.log(y), gz * x / y]
    def c_code(self, node, name, (x, y), (z,), sub):
        if node.inputs[0].type in [scalar.float32, scalar.float64]:
            return """%(z)s =
                %(x)s == 0.0
                ? 0.0
                : %(x)s * log(%(y)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
scalar_xlogy0  = XlogY0(scalar.upgrade_to_float, name='scalar_xlogy0')
xlogy0 = Elemwise(scalar_xlogy0, name='xlogy0')
