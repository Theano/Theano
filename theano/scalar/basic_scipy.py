#definition theano.scalar op that have their python implementation taked from scipy
#as scipy is not always available, we treat them separatly
import numpy

from theano.scalar.basic import (UnaryScalarOp, BinaryScalarOp,
                                 exp, upgrade_to_float,
                                 float_types)
from theano.scalar.basic import (upgrade_to_float_no_complex,
                                 complex_types,
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
        elif x.type in float_types:
            cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return gz * cst * exp(-x * x),
        else:
            return None,

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
        elif x.type in float_types:
            cst = numpy.asarray(2. / numpy.sqrt(numpy.pi),
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return - gz * cst * exp(-x * x),
        else:
            return None,

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in complex_types:
            raise NotImplementedError('type not supported', type)
        return "%(z)s = erfc(%(x)s);" % locals()

# scipy.special.erfc don't support complex. Why?
erfc = Erfc(upgrade_to_float_no_complex, name='erfc')


class Erfinv(UnaryScalarOp):
    """
    Implements the inverse error function.

    Note: This op can still be executed on GPU, despite not having c_code.  When
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
        elif x.type in float_types:
            cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return gz * cst * exp(erfinv(x) ** 2),
        else:
            return None,

    # TODO: erfinv() is not provided by the C standard library
    #def c_code(self, node, name, inp, out, sub):
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
        elif x.type in float_types:
            cst = numpy.asarray(numpy.sqrt(numpy.pi) / 2.,
                                dtype=upcast(x.type.dtype, gz.type.dtype))
            return - gz * cst * exp(erfcinv(x) ** 2),
        else:
            return None,

    # TODO: erfcinv() is not provided by the C standard library
    #def c_code(self, node, name, inp, out, sub):
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

    def grad(self, (x, ), (gz, )):
        return gz * gamma(x) * psi(x),

    def c_code(self, node, name, (x, ), (z, ), sub):
        if node.inputs[0].type in float_types:
            return """%(z)s = tgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))
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
        return [gz * psi(x)]

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                lgamma(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))
gammaln = GammaLn(upgrade_to_float, name='gammaln')


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

    def grad(self, inputs, outputs_gradients):
        raise NotImplementedError()
        return [None]

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
        """ )

    def c_code(self, node, name, inp, out, sub):
        x, = inp
        z, = out
        if node.inputs[0].type in float_types:
            return """%(z)s =
                _psi(%(x)s);""" % locals()
        raise NotImplementedError('only floating point is implemented')

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))
psi = Psi(upgrade_to_float, name='psi')

class Chi2SF(BinaryScalarOp):
    """
    Compute (1 - chi2_cdf(x))
        ie. chi2 pvalue (chi2 'survival function')
    """
    
    @staticmethod
    def st_impl(x, k):
        return scipy.stats.chi2.sf(x, k)
    def impl(self, x, k):
        if imported_scipy_special:
            return Chi2SF.st_impl(x, k)
        else:
            super(Chi2SF, self).impl(x, k)
    def c_support_code(self):
        return(
               """
                   //For GPU support
                   #ifdef __CUDACC__
                   #define DEVICE __device__
                   #else
                   #define DEVICE
                   #endif
                   
                   #ifndef _CHI2FUNCDEFINED
                   #define _CHI2FUNCDEFINED
                   
                   /*----------------------------------------------------------------------
                   File    : gamma.c
                   Contents: computation of the (incomplete/regularized) gamma function
                   Author  : Christian Borgelt
                   History : 2002.07.04 file created
                   2003.05.19 incomplete Gamma function added
                   2008.03.14 more incomplete Gamma functions added
                   2008.03.15 table of factorials and logarithms added
                   2008.03.17 gamma distribution functions added
                   ----------------------------------------------------------------------*/
                   #include <stdio.h>
                   #include <stdlib.h>
                   
                   
                   #include <assert.h>
                   #include <float.h>
                   #include <math.h>
                   /*----------------------------------------------------------------------
                   Preprocessor Definitions
                   ----------------------------------------------------------------------*/
                   #define LN_BASE      2.71828182845904523536028747135  /* e */
                   #define SQRT_PI      1.77245385090551602729816748334  /* \sqrt(\pi) */
                   #define LN_PI        1.14472988584940017414342735135  /* \ln(\pi) */
                   #define LN_SQRT_2PI  0.918938533204672741780329736406
                   /* \ln(\sqrt(2\pi)) */
                   #define EPSILON      2.2204460492503131e-16
                   #define EPS_QTL      1.4901161193847656e-08
                   #define MAXFACT      170
                   #define MAXITER      1024
                   #define TINY         (EPSILON *EPSILON *EPSILON)
                   
                   /*----------------------------------------------------------------------
                   Table of Factorials/Gamma Values
                   ----------------------------------------------------------------------*/
                   static double _facts[MAXFACT+1] = { 0 };
                   static double _logfs[MAXFACT+1];
                   static double _halfs[MAXFACT+1];
                   static double _loghs[MAXFACT+1];
                   
                   /*----------------------------------------------------------------------
                   Functions
                   ----------------------------------------------------------------------*/
                   static void _init (void)
                   {                               /* --- init. factorial tables */
                   int    i;                     /* loop variable */
                   double x = 1;                 /* factorial */
                   
                   _facts[0] = _facts[1] = 1;    /* store factorials for 0 and 1 */
                   _logfs[0] = _logfs[1] = 0;    /* and their logarithms */
                   for (i = 1; ++i <= MAXFACT; ) {
                   _facts[i] = x *= i;         /* initialize the factorial table */
                   _logfs[i] = log(x);         /* and the table of their logarithms */
                   }
                   _halfs[0] = x = SQRT_PI;      /* store Gamma(0.5) */
                   _loghs[0] = 0.5*LN_PI;        /* and its logarithm */
                   for (i = 0; ++i < MAXFACT; ) {
                   _halfs[i] = x *= i-0.5;     /* initialize the table for */
                   _loghs[i] = log(x);         /* the Gamma function of half numbers */
                   }                             /* and the table of their logarithms */
                   }  /* _init() */
                   
                   /*--------------------------------------------------------------------*/
                   #if 0
                   
                   double logGamma (double n)
                   {                               /* --- compute ln(Gamma(n))         */
                   double s;                     /*           = ln((n-1)!), n \in IN */
                   
                   assert(n > 0);                /* check the function argument */
                   if (_facts[0] <= 0) _init();  /* initialize the tables */
                   if (n < MAXFACT +1 +4 *EPSILON) {
                   if (fabs(  n -floor(  n)) < 4 *EPSILON)
                   return _logfs[(int)floor(n)-1];
                   if (fabs(2*n -floor(2*n)) < 4 *EPSILON)
                   return _loghs[(int)floor(n)];
                   }                             /* try to get the value from a table */
                   s =  1.000000000190015        /* otherwise compute it */
                   + 76.18009172947146      /(n+1)
                   - 86.50532032941677      /(n+2)
                   + 24.01409824083091      /(n+3)
                   -  1.231739572450155     /(n+4)
                   +  0.1208650972866179e-2 /(n+5)
                   -  0.5395239384953e-5    /(n+6);
                   return (n+0.5) *log((n+5.5)/LN_BASE) +(LN_SQRT_2PI +log(s/n) -5.0);
                   }  /* logGamma() */
                   
                   #else /*--------------------------------------------------------------*/
                   
                   double logGamma (double n)
                   {                               /* --- compute ln(Gamma(n))         */
                   double s;                     /*           = ln((n-1)!), n \in IN */
                   
                   assert(n > 0);                /* check the function argument */
                   if (_facts[0] <= 0) _init();  /* initialize the tables */
                   if (n < MAXFACT +1 +4 *EPSILON) {
                   if (fabs(  n -floor(  n)) < 4 *EPSILON)
                   return _logfs[(int)floor(n)-1];
                   if (fabs(2*n -floor(2*n)) < 4 *EPSILON)
                   return _loghs[(int)floor(n)];
                   }                             /* try to get the value from a table */
                   s =    0.99999999999980993227684700473478  /* otherwise compute it */
                   +  676.520368121885098567009190444019 /(n+1)
                   - 1259.13921672240287047156078755283  /(n+2)
                   +  771.3234287776530788486528258894   /(n+3)
                   -  176.61502916214059906584551354     /(n+4)
                   +   12.507343278686904814458936853    /(n+5)
                   -    0.13857109526572011689554707     /(n+6)
                   +    9.984369578019570859563e-6       /(n+7)
                   +    1.50563273514931155834e-7        /(n+8);
                   return (n+0.5) *log((n+7.5)/LN_BASE) +(LN_SQRT_2PI +log(s/n) -7.0);
                   }  /* logGamma() */
                   
                   #endif
                   /*----------------------------------------------------------------------
                   Use Lanczos' approximation
                   \Gamma(n+1) = (n+\gamma+0.5)^(n+0.5)
                   * e^{-(n+\gamma+0.5)}
                   * \sqrt{2\pi}
                   * (c_0 +c_1/(n+1) +c_2/(n+2) +...+c_n/(n+k) +\epsilon)
                   and exploit the recursion \Gamma(n+1) = n *\Gamma(n) once,
                   i.e., compute \Gamma(n) as \Gamma(n+1) /n.
                   
                   For the choices \gamma = 5, k = 6, and c_0 to c_6 as defined
                   in the first version, it is |\epsilon| < 2e-10 for all n > 0.
                   
                   Source: W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery
                   Numerical Recipes in C - The Art of Scientific Computing
                   Cambridge University Press, Cambridge, United Kingdom 1992
                   pp. 213-214
                   
                   For the choices gamma = 7, k = 8, and c_0 to c_8 as defined
                   in the second version, the value is slightly more accurate.
                   ----------------------------------------------------------------------*/
                   
                   double Gamma (double n)
                   {                               /* --- compute Gamma(n) = (n-1)! */
                   assert(n > 0);                /* check the function argument */
                   if (_facts[0] <= 0) _init();  /* initialize the tables */
                   if (n < MAXFACT +1 +4 *EPSILON) {
                   if (fabs(  n -floor(  n)) < 4 *EPSILON)
                   return _facts[(int)floor(n)-1];
                   if (fabs(2*n -floor(2*n)) < 4 *EPSILON)
                   return _halfs[(int)floor(n)];
                   }                             /* try to get the value from a table */
                   return exp(logGamma(n));      /* compute through natural logarithm */
                   }  /* Gamma() */
                   
                   /*--------------------------------------------------------------------*/
                   
                   static double _series (double n, double x)
                   {                               /* --- series approximation */
                   int    i;                     /* loop variable */
                   double t, sum;                /* buffers */
                   
                   sum = t = 1/n;                /* compute initial values */
                   for (i = MAXITER; --i >= 0; ) {
                   sum += t *= x/++n;          /* add one term of the series */
                   if (fabs(t) < fabs(sum) *EPSILON) break;
                   }                             /* if term is small enough, abort */
                   return sum;                   /* return the computed factor */
                   }  /* _series() */
                   
                   /*----------------------------------------------------------------------
                   series approximation:
                   P(a,x) =    \gamma(a,x)/\Gamma(a)
                   \gamma(a,x) = e^-x x^a \sum_{n=0}^\infty (\Gamma(a)/\Gamma(a+1+n)) x^n
                   
                   Source: W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery
                   Numerical Recipes in C - The Art of Scientific Computing
                   Cambridge University Press, Cambridge, United Kingdom 1992
                   formula: pp. 216-219
                   
                   The factor exp(n *log(x) -x) is added in the functions below.
                   ----------------------------------------------------------------------*/
                   
                   static double _cfrac (double n, double x)
                   {                               /* --- continued fraction approx. */
                   int    i;                     /* loop variable */
                   double a, b, c, d, e, f;      /* buffers */
                   
                   b = x+1-n; c = 1/TINY; f = d = 1/b;
                   for (i = 1; i < MAXITER; i++) {
                   a = i*(n-i);                /* use Lentz's algorithm to compute */
                   d = a *d +(b += 2);         /* consecutive approximations */
                   if (fabs(d) < TINY) d = TINY;
                   c = b +a/c;
                   if (fabs(c) < TINY) c = TINY;
                   d = 1/d; f *= e = d *c;
                   if (fabs(e-1) < EPSILON) break;
                   }                             /* if factor is small enough, abort */
                   return f;                     /* return the computed factor */
                   }  /* _cfrac() */
                   
                   /*----------------------------------------------------------------------
                   continued fraction approximation:
                   P(a,x) = 1 -\Gamma(a,x)/\Gamma(a)
                   \Gamma(a,x) = e^-x x^a (1/(x+1-a- 1(1-a)/(x+3-a- 2*(2-a)/(x+5-a- ...))))
                   
                   Source: W.H. Press, S.A. Teukolsky, W.T. Vetterling, and B.P. Flannery
                   Numerical Recipes in C - The Art of Scientific Computing
                   Cambridge University Press, Cambridge, United Kingdom 1992
                   formula:           pp. 216-219
                   Lentz's algorithm: p.  171
                   
                   The factor exp(n *log(x) -x) is added in the functions below.
                   ----------------------------------------------------------------------*/
                   
                   double lowerGamma (double n, double x)
                   {                               /* --- lower incomplete Gamma fn. */
                   assert((n > 0) && (x > 0));   /* check the function arguments */
                   return _series(n, x) *exp(n *log(x) -x);
                   }  /* lowerGamma() */
                   
                   /*--------------------------------------------------------------------*/
                   
                   double upperGamma (double n, double x)
                   {                               /* --- upper incomplete Gamma fn. */
                   assert((n > 0) && (x > 0));   /* check the function arguments */
                   return _cfrac(n, x) *exp(n *log(x) -x);
                   }  /* upperGamma() */
                   
                   /*--------------------------------------------------------------------*/
                   
                   
                   double GammaP (double n, double x)
                   {                               /* --- regularized Gamma function P */
                   assert((n > 0) && (x >= 0));  /* check the function arguments */
                   if (x <=  0) return 0;        /* treat x = 0 as a special case */
                   if (x < n+1) return _series(n, x) *exp(n *log(x) -x -logGamma(n));
                   return 1 -_cfrac(n, x) *exp(n *log(x) -x -logGamma(n));
                   }  /* GammaP() */
                   
                   
                   //ebuchman: this function is equivalent to scipy.stats.chi2.sf
                   //it's the pvalue (survival function) of a chi2 distribution
                   DEVICE double Chi2SF (double k, double x)
                   {
                   return 1 - GammaP(k/2., x/2.);
                   }
                   """)
    
    
    def c_code(self, node, name, inp, out, sub):
        
        x, k = inp
        z, = out
        if node.inputs[0].type in float_types:
            dtype = z.dtype
            return """%(z)s =
                (%(dtype)s)Chi2SF(%(k)s, %(x)s);""" % locals()
        raise NotImplementedError('only floatingpoint is implemented')
    
    def __eq__(self, other):
        return type(self) == type(other)
    def __hash__(self):
        return hash(type(self))

chi2sf = Chi2SF(upgrade_to_float, name='chi2sf')
