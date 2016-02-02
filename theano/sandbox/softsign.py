from __future__ import absolute_import, print_function, division

import theano
import theano.tensor


class ScalarSoftsign(theano.scalar.UnaryScalarOp):
    # TODO : need description for class
    @staticmethod
    def static_impl(x):
        return x / (1.0 + abs(x))

    def impl(self, x):
        return ScalarSoftsign.static_impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if 'float' in x.type.dtype:
            d = (1.0 + abs(x))
            return [gz / (d * d)]
        else:
            return NotImplemented

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in [theano.scalar.float32,
                                   theano.scalar.float64]:
            return "%(z)s = %(x)s / (1.0+fabs(%(x)s));" % locals()
        raise NotImplementedError('only floating point x is implemented')

scalar_softsign = ScalarSoftsign(theano.scalar.upgrade_to_float,
                                 name='scalar_softsign')
softsign = theano.tensor.Elemwise(scalar_softsign, name='softsign')
