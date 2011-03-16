#definition theano.scalar op that have their python implementation taked from scipy
#as scipy is not always available, we put threat them separatly
import numpy

from theano.scalar.basic import UnaryScalarOp,exp,upgrade_to_float,float_types
from theano.scalar.basic import upgrade_to_float_no_complex,complex_types,upcast

imported_scipy_special = False
try:
    import scipy.special
    imported_scipy_special = True
except ImportError:
    pass

class Erf(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erf(x)
        else:
            super(Erf,self).impl(x)
    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),dtype=upcast(x.type.dtype,gz.type.dtype))
            return gz * cst * exp(-x*x),
        else:
            return None,
    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erf(%(x)s);" % locals()
erf = Erf(upgrade_to_float, name= 'erf')

class Erfc(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfc(x)
        else:
            super(Erfc,self).impl(x)
    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        elif x.type in float_types:
            cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),dtype=upcast(x.type.dtype,gz.type.dtype))
            return - gz * cst * exp(-x*x),
        else:
            return None,
    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfc(%(x)s);" % locals()

# scipy.special.erfc don't support complex. Why?
erfc = Erfc(upgrade_to_float_no_complex, name = 'erfc')
