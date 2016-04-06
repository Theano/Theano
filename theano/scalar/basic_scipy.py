# definition theano.scalar op that have their python implementation taked from scipy
# as scipy is not always available, we treat them separatly
import numpy

import theano
from theano.scalar.basic import (UnaryScalarOp, BinaryScalarOp,
                                 exp, upgrade_to_float,
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
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erf(x)
        else:
            super(Erf, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * cst * exp(-x * x),

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erf(%(x)s);" % locals()
erf = Erf(upgrade_to_float, name='erf')


class Erfc(UnaryScalarOp):
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfc(x)
        else:
            super(Erfc, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return - gz * cst * exp(-x * x),

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfc(%(x)s);" % locals()

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
    running on GPU, sandbox.cuda.opt.local_gpu_elemwise_[0,1] replaces this op
    with sandbox.cuda.elemwise.ErfcxGPU.

    """
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfcx(x)
        else:
            super(Erfcx, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                            dtype=upcast(x.type.dtype, gz.type.dtype))
        return gz * (-cst + (2. * x) * erfcx(x)),

erfcx = Erfcx(upgrade_to_float_no_complex, name='erfcx')


class Erfinv(UnaryScalarOp):
    """
    Implements the inverse error function.

    Notes
    -----
    This op can still be executed on GPU, despite not having c_code. When
    running on GPU, sandbox.cuda.opt.local_gpu_elemwise_[0,1] replaces this op
    with sandbox.cuda.elemwise.ErfinvGPU.

    (TODO) Find a C implementation of erfinv for CPU.
    """
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfinv(x)
        else:
            super(Erfinv, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
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
    def impl(self, x):
        if imported_scipy_special:
            return scipy.special.erfcinv(x)
        else:
            super(Erfcinv, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
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
    @staticmethod
    def st_impl(x):
        return scipy.special.gamma(x)

    def impl(self, x):
        if imported_scipy_special:
            return Gamma.st_impl(x)
        else:
            super(Gamma, self).impl(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
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
    @staticmethod
    def st_impl(x):
        return scipy.special.gammaln(x)

    def impl(self, x):
        if imported_scipy_special:
            return GammaLn.st_impl(x)
        else:
            super(GammaLn, self).impl(x)

    def grad(self, inp, grads):
        x, = inp
        gz, = grads
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return [gz * psi(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                lgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')
gammaln = GammaLn(upgrade_to_float, name='gammaln')

class Polygamma(BinaryScalarOp):
    """
    Polygamma function, derivative of Psi (digamma function)
    """
    @staticmethod
    def st_impl(k, x):
        if k < 0:
            raise ValueError('polygamma order must be non-negative', k)
        return scipy.special.polygamma(k, x)

    def impl(self, k, x):
        if imported_scipy_special:
            return Polygamma.st_impl(k, x)
        else:
            super(Polygamma, self).impl(k, x)

    def grad(self, inputs, gout):
        (k,x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]

        return gz * polygamma(k+1, x),

    def c_support_code(self):
        return (
"""
#ifndef _POLYGAMMAFUNCDEFINED
#define _POLYGAMMAFUNCDEFINED

#include "nmath.h"
#ifdef MATHLIB_STANDALONE
#include <errno.h>
#endif
#define n_max (100)

void dpsifn(double x, int n, int kode, int m, double *ans, int *nz, int *ierr)
{
    const static double bvalues[] = {/* Bernoulli Numbers */
 1.00000000000000000e+00,
-5.00000000000000000e-01,
 1.66666666666666667e-01,
-3.33333333333333333e-02,
 2.38095238095238095e-02,
-3.33333333333333333e-02,
 7.57575757575757576e-02,
-2.53113553113553114e-01,
 1.16666666666666667e+00,
-7.09215686274509804e+00,
 5.49711779448621554e+01,
-5.29124242424242424e+02,
 6.19212318840579710e+03,
-8.65802531135531136e+04,
 1.42551716666666667e+06,
-2.72982310678160920e+07,
 6.01580873900642368e+08,
-1.51163157670921569e+10,
 4.29614643061166667e+11,
-1.37116552050883328e+13,
 4.88332318973593167e+14,
-1.92965793419400681e+16
    };

    int i, j, k, mm, mx, nn, np, nx, fn;
    double arg, den, elim, eps, fln, fx, rln, rxsq,
r1m4, r1m5, s, slope, t, ta, tk, tol, tols, tss, tst,
tt, t1, t2, wdtol, xdmln, xdmy, xinc, xln = 0.0 /* -Wall */,
xm, xmin, xq, yint;
    double trm[23], trmr[n_max + 1];

    *ierr = 0;
    if (n < 0 || kode < 1 || kode > 2 || m < 1) {
*ierr = 1;
return;
    }
    if (x <= 0.) {
/* useAbramowitz & Stegun 6.4.7 "Reflection Formula"
 *psi(k, x) = (-1)^k psi(k, 1-x)-  pi^{n+1} (d/dx)^n cot(x)
 */
if (x == round(x)) {
    /* non-positive integer : +Inf or NaN depends on n */
    for(j=0; j < m; j++) /* k = j + n : */
ans[j] = ((j+n) % 2) ? ML_POSINF : ML_NAN;
    return;
}
/* This could cancel badly */
dpsifn(1. - x, n, /*kode = */ 1, m, ans, nz, ierr);
/* ans[j] == (-1)^(k+1) / gamma(k+1) * psi(k, 1 - x)
 *     for j = 0:(m-1) ,k = n + j
 */

/* Cheat for now: only work for m = 1, n in {0,1,2,3} : */
if(m > 1 || n > 3) {/* doesn't happen for digamma() .. pentagamma() */
    /* not yet implemented */
    *ierr = 4; return;
}
x *= M_PI; /* pi * x */
if (n == 0)
    tt = cos(x)/sin(x);
else if (n == 1)
    tt = -1/JR_pow_di(sin(x), 2);
else if (n == 2)
    tt = 2*cos(x)/JR_pow_di(sin(x), 3);
else if (n == 3)
    tt = -2*(2*JR_pow_di(cos(x), 2) + 1.)/JR_pow_di(sin(x), 4);
else /* can not happen! */
    tt = ML_NAN;
/* end cheat */

s = (n % 2) ? -1. : 1.;/* s = (-1)^n */
/* t := pi^(n+1) * d_n(x) / gamma(n+1), where
 *   d_n(x) := (d/dx)^n cot(x)*/
t1 = t2 = s = 1.;
for(k=0, j=k-n; j < m; k++, j++, s = -s) {
    /* k == n+j , s = (-1)^k */
    t1 *= M_PI;/* t1 == pi^(k+1) */
    if(k >= 2)
t2 *= k;/* t2 == k! == gamma(k+1) */
    if(j >= 0) /* by cheat above,  tt === d_k(x) */
ans[j] = s*(ans[j] + t1/t2 * tt);
}
if (n == 0 && kode == 2) /* unused from R, but "wrong": xln === 0 :*/
    ans[0] += xln;
return;
    } /* x <= 0 */

    /* else :  x > 0 */
    *nz = 0;
    xln = log(x);
    if(kode == 1 && m == 1) {/* the R case  ---  for very large x: */
double lrg = 1/(2. * DBL_EPSILON);
if(n == 0 && x * xln > lrg) {
    ans[0] = -xln;
    return;
}
else if(n >= 1 && x > n * lrg) {
    ans[0] = exp(-n * xln)/n; /* == x^-n / n  ==  1/(n * x^n) */
    return;
}
    }
    mm = m;
    nx = imin2(-jags_i1mach(15), jags_i1mach(16));/* = 1021 */
    r1m5 = jags_d1mach(5);
    r1m4 = jags_d1mach(4) * 0.5;
    wdtol = fmax2(r1m4, 0.5e-18); /* 1.11e-16 */

    /* elim = approximate exponential over and underflow limit */
    elim = 2.302 * (nx * r1m5 - 3.0);/* = 700.6174... */
    for(;;) {
nn = n + mm - 1;
fn = nn;
t = (fn + 1) * xln;

/* overflow and underflow test for small and large x */

if (fabs(t) > elim) {
    if (t <= 0.0) {
*nz = 0;
*ierr = 2;
return;
    }
}
else {
    if (x < wdtol) {
ans[0] = JR_pow_di(x, -n-1);
if (mm != 1) {
    for(k = 1; k < mm ; k++)
ans[k] = ans[k-1] / x;
}
if (n == 0 && kode == 2)
    ans[0] += xln;
return;
    }

    /* compute xmin and the number of terms of the series,  fln+1 */

    rln = r1m5 * jags_i1mach(14);
    rln = fmin2(rln, 18.06);
    fln = fmax2(rln, 3.0) - 3.0;
    yint = 3.50 + 0.40 * fln;
    slope = 0.21 + fln * (0.0006038 * fln + 0.008677);
    xm = yint + slope * fn;
    mx = (int)xm + 1;
    xmin = mx;
    if (n != 0) {
xm = -2.302 * rln - fmin2(0.0, xln);
arg = xm / n;
arg = fmin2(0.0, arg);
eps = exp(arg);
xm = 1.0 - eps;
if (fabs(arg) < 1.0e-3)
    xm = -arg;
fln = x * xm / eps;
xm = xmin - x;
if (xm > 7.0 && fln < 15.0)
    break;
    }
    xdmy = x;
    xdmln = xln;
    xinc = 0.0;
    if (x < xmin) {
nx = (int)x;
xinc = xmin - nx;
xdmy = x + xinc;
xdmln = log(xdmy);
    }

    /* generate w(n+mm-1, x) by the asymptotic expansion */

    t = fn * xdmln;
    t1 = xdmln + xdmln;
    t2 = t + xdmln;
    tk = fmax2(fabs(t), fmax2(fabs(t1), fabs(t2)));
    if (tk <= elim) /* for all but large x */
goto L10;
}
nz++; /* underflow */
mm--;
ans[mm] = 0.;
if (mm == 0)
    return;
    } /* end{for()} */
    nn = (int)fln + 1;
    np = n + 1;
    t1 = (n + 1) * xln;
    t = exp(-t1);
    s = t;
    den = x;
    for(i=1; i <= nn; i++) {
den += 1.;
trm[i] = pow(den, (double)-np);
s += trm[i];
    }
    ans[0] = s;
    if (n == 0 && kode == 2)
ans[0] = s + xln;

    if (mm != 1) { /* generate higher derivatives, j > n */

tol = wdtol / 5.0;
for(j = 1; j < mm; j++) {
    t /= x;
    s = t;
    tols = t * tol;
    den = x;
    for(i=1; i <= nn; i++) {
den += 1.;
trm[i] /= den;
s += trm[i];
if (trm[i] < tols)
    break;
    }
    ans[j] = s;
}
    }
    return;

  L10:
    tss = exp(-t);
    tt = 0.5 / xdmy;
    t1 = tt;
    tst = wdtol * tt;
    if (nn != 0)
t1 = tt + 1.0 / fn;
    rxsq = 1.0 / (xdmy * xdmy);
    ta = 0.5 * rxsq;
    t = (fn + 1) * ta;
    s = t * bvalues[2];
    if (fabs(s) >= tst) {
tk = 2.0;
for(k = 4; k <= 22; k++) {
    t = t * ((tk + fn + 1)/(tk + 1.0))*((tk + fn)/(tk + 2.0)) * rxsq;
    trm[k] = t * bvalues[k-1];
    if (fabs(trm[k]) < tst)
break;
    s += trm[k];
    tk += 2.;
}
    }
    s = (s + t1) * tss;
    if (xinc != 0.0) {

/* backward recur from xdmy to x */

nx = (int)xinc;
np = nn + 1;
if (nx > n_max) {
    *nz = 0;
    *ierr = 3;
    return;
}
else {
    if (nn==0)
goto L20;
    xm = xinc - 1.0;
    fx = x + xm;

    /* this loop should not be changed. fx is accurate when x is small */
    for(i = 1; i <= nx; i++) {
trmr[i] = pow(fx, (double)-np);
s += trmr[i];
xm -= 1.;
fx = x + xm;
    }
}
    }
    ans[mm-1] = s;
    if (fn == 0)
goto L30;

    /* generate lower derivatives,  j < n+mm-1 */

    for(j = 2; j <= mm; j++) {
fn--;
tss *= xdmy;
t1 = tt;
if (fn!=0)
    t1 = tt + 1.0 / fn;
t = (fn + 1) * ta;
s = t * bvalues[2];
if (fabs(s) >= tst) {
    tk = 4 + fn;
    for(k=4; k <= 22; k++) {
trm[k] = trm[k] * (fn + 1) / tk;
if (fabs(trm[k]) < tst)
    break;
s += trm[k];
tk += 2.;
    }
}
s = (s + t1) * tss;
if (xinc != 0.0) {
    if (fn == 0)
goto L20;
    xm = xinc - 1.0;
    fx = x + xm;
    for(i=1 ; i<=nx ; i++) {
trmr[i] = trmr[i] * fx;
s += trmr[i];
xm -= 1.;
fx = x + xm;
    }
}
ans[mm - j] = s;
if (fn == 0)
    goto L30;
    }
    return;

  L20:
    for(i = 1; i <= nx; i++)
s += 1. / (x + (nx - i)); /* avoid disastrous cancellation, PR#13714 */

  L30:
    if (kode != 2) /* always */
ans[0] = s - xdmln;
    else if (xdmy != x) {
xq = xdmy / x;
ans[0] = s - log(xq);
    }
    return;
}

double _polygamma(double x, int n){
    /*taken from
    Bernardo, J. M. (1976). Algorithm AS 103:
    Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
    http://www.uv.es/~bernardo/1976AppStatist.pdf */
    double ans;
    int nz, ierr, k, n;
    
    dpsifn(x, n, 1, 1, &ans, &nz, &ierr);

    /* Now, ans ==  A := (-1)^(n+1) / gamma(n+1) * psi(n, x) */
    ans = -ans; /* = (-1)^(0+1) * gamma(0+1) * A */
    for(k = 1; k <= n; k++) ans *= (-k);
    return ans;}
    #endif
        """ )

    def c_code(self, node, name, inp, out, sub):
        x,n, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =_polygamma(%(x)s, %(n)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

polygamma = Polygamma(upgrade_to_float, name='polygamma')

class Psi(UnaryScalarOp):
    """
    Derivative of log gamma function.

    """
    @staticmethod
    def st_impl(x):
        return scipy.special.psi(x)

    def impl(self, x):
        if imported_scipy_special:
            return Psi.st_impl(x)
        else:
            super(Psi, self).impl(x)

    def grad(self, inputs, gout):
        (x,) = inputs
        (gz,) = gout
        if x.type in complex_types:
            raise NotImplementedError()
        if self(x).type in discrete_types:
            if x.type in discrete_types:
                return [x.zeros_like(dtype=theano.config.floatX)]
            else:
                return [x.zeros_like()]
        return gz * polygamma(1, x),


    def c_support_code(self):
        return (
            """
            // For GPU support
            #ifdef __CUDACC__
            #define DEVICE __device__
            #else
            #define DEVICE
            #endif

            #ifndef _PSIFUNCDEFINED
            #define _PSIFUNCDEFINED
            DEVICE double _psi(double x){

            /*taken from
            Bernardo, J. M. (1976). Algorithm AS 103:
            Psi (Digamma) Function. Applied Statistics. 25 (3), 315-317.
            http://www.uv.es/~bernardo/1976AppStatist.pdf */

            double y, R, psi_ = 0;
            double S  = 1.0e-5;
            double C = 8.5;
            double S3 = 8.333333333e-2;
            double S4 = 8.333333333e-3;
            double S5 = 3.968253968e-3;
            double D1 = -0.5772156649;

            y = x;

            if (y <= 0.0)
               return psi_;

            if (y <= S )
                return D1 - 1.0/y;

            while (y < C){
                psi_ = psi_ - 1.0 / y;
                y = y + 1;}

            R = 1.0 / y;
            psi_ = psi_ + log(y) - .5 * R ;
            R= R*R;
            psi_ = psi_ - R * (S3 - R * (S4 - R * S5));

            return psi_;}
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


class Chi2SF(BinaryScalarOp):
    """
    Compute (1 - chi2_cdf(x)) ie. chi2 pvalue (chi2 'survival function').

    C code is provided in the Theano_lgpl repository.
    This make it faster.

    https://github.com/Theano/Theano_lgpl.git

    """

    @staticmethod
    def st_impl(x, k):
        return scipy.stats.chi2.sf(x, k)

    def impl(self, x, k):
        if imported_scipy_special:
            return Chi2SF.st_impl(x, k)
        else:
            super(Chi2SF, self).impl(x, k)
chi2sf = Chi2SF(upgrade_to_float, name='chi2sf')


class J1(UnaryScalarOp):
    """
    Bessel function of the 1'th kind
    """

    @staticmethod
    def st_impl(x):
        return scipy.special.j1(x)

    def impl(self, x):
        if imported_scipy_special:
            return self.st_impl(x)
        else:
            super(J1, self).impl(x)

    def grad(self, inp, grads):
        raise NotImplementedError()

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
    Bessel function of the 0'th kind
    """

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
