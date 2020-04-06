from __future__ import absolute_import, print_function, division
# Definitions of theano.scalar ops that have their python implementation taken
# from SciPy. As SciPy is not always available, we treat them separately.

import numpy as np
import os

import theano
from theano.gradient import grad_not_implemented
from theano.scalar.basic import (UnaryScalarOp, BinaryScalarOp,
                                 exp, upgrade_to_float,
                                 upgrade_to_float64,
                                 float_types)
from theano.scalar.basic import (upgrade_to_float_no_complex,
                                 complex_types, discrete_types,
                                 upcast)

imported_scipy_special = False
try:
    import scipy.special
    import scipy.stats
    imported_scipy_special = True
# Importing scipy.special may raise ValueError.
# See http://projects.scipy.org/scipy/ticket/1739
except (ImportError, ValueError):
    pass


class Erf(UnaryScalarOp):
    nfunc_spec = ('scipy.special.erf', 1, 1)

    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erf(x)
        else:
            super(Erf, self).impl(x)

    def L_op(self, inputs, outputs, grads):
        x, = inputs
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(2. / np.sqrt(np.pi),
                         dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * cst * exp(-x * x),

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return "%(z)s = erf((%(cast)s)%(x)s);" % locals()
erf = Erf(upgrade_to_float, name='erf')


class Erfc(UnaryScalarOp):
    nfunc_spec = ('scipy.special.erfc', 1, 1)

    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfc(x)
        else:
            super(Erfc, self).impl(x)

    def L_op(self, inputs, outputs, grads):
        x, = inputs
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(2. / np.sqrt(np.pi),
                         dtype=upcast(x.type.dtype, gz.type.dtype))
        return - gz * cst * exp(-x * x),

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        cast = node.outputs[0].type.dtype_specs()[1]
        return "%(z)s = erfc((%(cast)s)%(x)s);" % locals()

# scipy.special.erfc don't support complex. Why?
erfc = Erfc(upgrade_to_float_no_complex, name='erfc')


class Erfcx(UnaryScalarOp):
    """
    Implements the scaled complementary error function exp(x**2)*erfc(x) in a
    numerically stable way for large x. This is useful for calculating things
    like log(erfc(x)) = log(erfcx(x)) - x ** 2 without causing underflow.
    Should only be used if x is known to be large and positive, as using
    erfcx(x) for large negative x may instead introduce overflow problems.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU an optimization will replace it with a gpu version.

    """
    nfunc_spec = ('scipy.special.erfcx', 1, 1)

    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfcx(x)
        else:
            super(Erfcx, self).impl(x)

    def L_op(self, inputs, outputs, grads):
        x, = inputs
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(2. / np.sqrt(np.pi),
                         dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * (-cst + (2. * x) * erfcx(x)),

erfcx = Erfcx(upgrade_to_float_no_complex, name='erfcx')


class Erfinv(UnaryScalarOp):
    """
    Implements the inverse error function.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU, an optimization will replace it with a GPU version.

    (TODO) Find a C implementation of erfinv for CPU.
    """
    nfunc_spec = ('scipy.special.erfinv', 1, 1)

    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfinv(x)
        else:
            super(Erfinv, self).impl(x)

    def L_op(self, inputs, outputs, grads):
        x, = inputs
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(np.sqrt(np.pi) / 2.,
                         dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * cst * exp(erfinv(x) ** 2),

    # TODO: erfinv() is not provided by the C standard library
    # def c_code(self, node, name, inp, out, sub):
    #    x, = inp
    #    z, = out
    #    if node.inputs[0].type in complex_types:
    #        raise NotImplementedError('type not supported', type)
    #    return "%(z)s = erfinv(%(x)s);" % locals()

erfinv = Erfinv(upgrade_to_float_no_complex, name='erfinv')


class Erfcinv(UnaryScalarOp):
    nfunc_spec = ('scipy.special.erfcinv', 1, 1)

    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfcinv(x)
        else:
            super(Erfcinv, self).impl(x)

    def L_op(self, inputs, outputs, grads):
        x, = inputs
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = np.asarray(np.sqrt(np.pi) / 2.,
                         dtype=upcast(x.type.dtype, gz.type.dtype))
        return - gz * cst * exp(erfcinv(x) ** 2),

    # TODO: erfcinv() is not provided by the C standard library
    # def c_code(self, node, name, inp, out, sub):
    #    x, = inp
    #    z, = out
    #    if node.inputs[0].type in complex_types:
    #        raise NotImplementedError('type not supported', type)
    #    return "%(z)s = erfcinv(%(x)s);" % locals()

erfcinv = Erfcinv(upgrade_to_float_no_complex, name='erfcinv')


class Gamma(UnaryScalarOp):
    nfunc_spec = ('scipy.special.gamma', 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.gamma(x)

    def impl(self, x):
        if imported_scipy_special:
            return Gamma.st_impl(x)
        else:
            super(Gamma, self).impl(x)

    def L_op(self, inputs, outputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * gamma(x) * psi(x),

    def c_code(self, node, name, inputs, outputs, sub):
        (x,) = inputs
        (z,) = outputs
        if node.inputs[0].type in float_types:
            return """%(z)s = tgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
gamma = Gamma(upgrade_to_float, name='gamma')


class GammaLn(UnaryScalarOp):
    """
    Log gamma function.

    """
    nfunc_spec = ('scipy.special.gammaln', 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.gammaln(x)

    def impl(self, x):
        if imported_scipy_special:
            return GammaLn.st_impl(x)
        else:
            super(GammaLn, self).impl(x)

    def L_op(self, inputs, outputs, grads):
        x, = inputs
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * psi(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        # no c code for complex
        # [u]int* will be casted to float64 before computation
        if node.inputs[0].type in complex_types:
            raise NotImplementedError(
                'gammaln complex c code is not implemented')
        # For some reason, on the GPU, uint64 inputs don't get casted
        # automatically to float64. This make the compilation crash
        dtype = ""
        cast = node.outputs[0].type.dtype_specs()[1]
        return """%(z)s = lgamma((%(cast)s)%(x)s);""" % locals()
gammaln = GammaLn(upgrade_to_float, name='gammaln')


class Psi(UnaryScalarOp):
    """
    Derivative of log gamma function.

    """
    nfunc_spec = ('scipy.special.psi', 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.psi(x)

    def impl(self, x):
        if imported_scipy_special:
            return Psi.st_impl(x)
        else:
            super(Psi, self).impl(x)

    def L_op(self, inputs, outputs, grads):
        x, = inputs
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if outputs[0].type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * tri_gamma(x)]

    def c_support_code(self):
        return (
            """
            // For GPU support
            #ifdef WITHIN_KERNEL
            #define DEVICE WITHIN_KERNEL
            #else
            #define DEVICE
            #endif

            #ifndef ga_double
            #define ga_double double
            #endif

            #ifndef _PSIFUNCDEFINED
            #define _PSIFUNCDEFINED
            DEVICE double _psi(ga_double x) {

            /*taken from
            Bernardo, J. M. (1976). Algorithm AS 103:
            Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
            http://www.uv.es/~bernardo/1976AppStatist.pdf */

            ga_double y, R, psi_ = 0;
            ga_double S  = 1.0e-5;
            ga_double C = 8.5;
            ga_double S3 = 8.333333333e-2;
            ga_double S4 = 8.333333333e-3;
            ga_double S5 = 3.968253968e-3;
            ga_double D1 = -0.5772156649;

            y = x;

            if (y <= 0.0)
               return psi_;

            if (y <= S)
                return D1 - 1.0/y;

            while (y < C) {
                psi_ = psi_ - 1.0 / y;
                y = y + 1;
            }

            R = 1.0 / y;
            psi_ = psi_ + log(y) - .5 * R ;
            R= R*R;
            psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

            return psi_;
            }
            #endif
            """)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _psi(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
psi = Psi(upgrade_to_float, name='psi')


class TriGamma(UnaryScalarOp):
    """
    Second derivative of log gamma function.

    """

    @staticmethod
    def st_impl(x):
        return scipy.special.polygamma(1, x)

    def impl(self, x):
        if imported_scipy_special:
            return TriGamma.st_impl(x)
        else:
            super(TriGamma, self).impl(x)

    def grad(self, inputs, outputs_gradients):
        raise NotImplementedError()

    def c_support_code(self):
        # The implementation has been copied from
        # http://people.sc.fsu.edu/~jburkardt/cpp_src/asa121/asa121.html
        return (
            """
            // For GPU support
            #ifdef WITHIN_KERNEL
            #define DEVICE WITHIN_KERNEL
            #else
            #define DEVICE
            #endif

            #ifndef ga_double
            #define ga_double double
            #endif

            #ifndef _TRIGAMMAFUNCDEFINED
            #define _TRIGAMMAFUNCDEFINED

            DEVICE double _tri_gamma(ga_double x) {

                double a = 0.0001;
                double b = 5.0;
                double b2 =  0.1666666667;
                double b4 = -0.03333333333;
                double b6 =  0.02380952381;
                double b8 = -0.03333333333;
                double value;
                double y;
                double z;

                if (x <= 0) {
                    return 0.0;
                }

                if ( x <= a ) {
                    value = 1.0 / x / x;
                    return value;
                }

                value = 0.0;
                z = x;

                while ( z < b ) {
                    value += 1.0 / z / z;
                    z += 1.0;
                }

                y = 1.0 / z / z;

                value +=  0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8 )))) / z;

                return value;
            }
            #endif
            """)

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _tri_gamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')


tri_gamma = TriGamma(upgrade_to_float, name='tri_gamma')


class Chi2SF(BinaryScalarOp):
    """
    Compute (1 - chi2_cdf(x))
        ie. chi2 pvalue (chi2 'survival function')
    """
    nfunc_spec = ('scipy.stats.chi2.sf', 2, 1)

    @staticmethod
    def st_impl(x, k):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, x, k):
        if imported_scipy_special:
            return Chi2SF.st_impl(x, k)
        else:
            super(Chi2SF, self).impl(x, k)

    def c_support_code(self):
        with open(os.path.join(
                os.path.dirname(__file__),
                'c_code',
                'gamma.c')) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        x, k = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) 1 - GammaP(%(k)s/2., %(x)s/2.);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


chi2sf = Chi2SF(upgrade_to_float64, name='chi2sf')


class GammaInc(BinaryScalarOp):
    """
    Compute the regularized lower gamma function (P).
    """
    nfunc_spec = ('scipy.special.gammainc', 2, 1)

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammainc(k, x)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaInc.st_impl(k, x)
        else:
            super(GammaInc, self).impl(k, x)

    def c_support_code(self):
        with open(os.path.join(
                os.path.dirname(__file__),
                'c_code',
                'gamma.c')) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) GammaP(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammainc = GammaInc(upgrade_to_float, name='gammainc')


class GammaIncC(BinaryScalarOp):
    """
    Compute the regularized upper gamma function (Q).
    """
    nfunc_spec = ('scipy.special.gammaincc', 2, 1)

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammaincc(x, k)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaIncC.st_impl(k, x)
        else:
            super(GammaIncC, self).impl(k, x)

    def c_support_code(self):
        with open(os.path.join(
                os.path.dirname(__file__),
                'c_code',
                'gamma.c')) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) GammaQ(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammaincc = GammaIncC(upgrade_to_float, name='gammaincc')


class GammaU(BinaryScalarOp):
    """
    compute the upper incomplete gamma function.
    """
    # Note there is no basic SciPy version so no nfunc_spec.

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammaincc(k, x) * scipy.special.gamma(k)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaU.st_impl(k, x)
        else:
            super(GammaU, self).impl(k, x)

    def c_support_code(self):
        with open(os.path.join(
                os.path.dirname(__file__),
                'c_code',
                'gamma.c')) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) upperGamma(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammau = GammaU(upgrade_to_float, name='gammau')


class GammaL(BinaryScalarOp):
    """
    Compute the lower incomplete gamma function.
    """
    # Note there is no basic SciPy version so no nfunc_spec.

    @staticmethod
    def st_impl(k, x):
        return scipy.special.gammainc(k, x) * scipy.special.gamma(k)

    def impl(self, k, x):
        if imported_scipy_special:
            return GammaL.st_impl(k, x)
        else:
            super(GammaL, self).impl(k, x)

    def c_support_code(self):
        with open(os.path.join(
                os.path.dirname(__file__),
                'c_code',
                'gamma.c')) as f:
            raw = f.read()
            return raw

    def c_code(self, node, name, inp, out, sub):
        k, x = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = 'npy_' + node.outputs[0].dtype
            return """%(z)s =
                (%(dtype)s) lowerGamma(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))


gammal = GammaL(upgrade_to_float, name='gammal')


class Jv(BinaryScalarOp):
    """
    Bessel function of the first kind of order v (real).
    """
    nfunc_spec = ('scipy.special.jv', 2, 1)

    @staticmethod
    def st_impl(v, x):
        return scipy.special.jv(v, x)

    def impl(self, v, x):
        if imported_scipy_special:
            return self.st_impl(v, x)
        else:
            super(Jv, self).impl(v, x)

    def grad(self, inputs, grads):
        v, x = inputs
        gz, = grads
        return [grad_not_implemented(self, 0, v),
                gz * (jv(v - 1, x) - jv(v + 1, x)) / 2.]

jv = Jv(upgrade_to_float, name='jv')


class J1(UnaryScalarOp):
    """
    Bessel function of the first kind of order 1.
    """
    nfunc_spec = ('scipy.special.j1', 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.j1(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(J1, self).impl(x)

    def grad(self, inputs, grads):
        x, = inputs
        gz, = grads
        return [gz * (j0(x) - jv(2, x)) / 2.]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                j1(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

j1 = J1(upgrade_to_float, name='j1')


class J0(UnaryScalarOp):
    """
    Bessel function of the first kind of order 0.
    """
    nfunc_spec = ('scipy.special.j0', 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.j0(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(J0, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * -1 * j1(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                j0(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

j0 = J0(upgrade_to_float, name='j0')


class Iv(BinaryScalarOp):
    """
    Modified Bessel function of the first kind of order v (real).
    """
    nfunc_spec = ('scipy.special.iv', 2, 1)

    @staticmethod
    def st_impl(v, x):
        return scipy.special.iv(v, x)

    def impl(self, v, x):
        if imported_scipy_special:
            return self.st_impl(v, x)
        else:
            super(Iv, self).impl(v, x)

    def grad(self, inputs, grads):
        v, x = inputs
        gz, = grads
        return [grad_not_implemented(self, 0, v),
                gz * (iv(v - 1, x) + iv(v + 1, x)) / 2.]

iv = Iv(upgrade_to_float, name='iv')


class I1(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 1.
    """
    nfunc_spec = ('scipy.special.i1', 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.i1(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I1, self).impl(x)

    def grad(self, inputs, grads):
        x, = inputs
        gz, = grads
        return [gz * (i0(x) + iv(2, x)) / 2.]

i1 = I1(upgrade_to_float, name='i1')


class I0(UnaryScalarOp):
    """
    Modified Bessel function of the first kind of order 0.
    """
    nfunc_spec = ('scipy.special.i0', 1, 1)

    @staticmethod
    def st_impl(x):
        return scipy.special.i0(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(I0, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        return [gz * i1(x)]

i0 = I0(upgrade_to_float, name='i0')
