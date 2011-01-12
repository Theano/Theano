"""Ops and optimizations for using BLAS function calls to evaluate linear algebra expressions"""

import sys, traceback, logging, copy, os
import numpy
import numpy.distutils
from theano.configparser import config, AddConfigVar, StrParam
from theano.gof import (utils, Op, view_roots, PatternSub, DestroyHandler,
        SeqOptimizer, local_optimizer, Optimizer, LocalOptimizer, OpKeyOptimizer,
        InconsistencyError, toolbox, SequenceDB, EquilibriumOptimizer)
from theano.printing import pprint, FunctionPrinter, debugprint
from theano.compile.mode import optdb
from theano.gof.python25 import all, any
import theano.scalar
import basic as T


from theano.gof.apply_shape import Apply

#NB: this clobbers the builtin 'compile' symbol
from theano import compile  #to register the optimizer built by this file

from theano.tensor.blas_headers import cblas_header_text, blas_header_text

_logger = logging.getLogger('theano.tensor.blas')
_logger.setLevel(logging.WARN)
def debug(*msg): _logger.debug(' '.join(str(m) for m in msg))
def info(*msg): _logger.info(' '.join(str(m) for m in msg))
def warn(*msg): _logger.warn(' '.join(str(m) for m in msg))
def warning(*msg): _logger.warning(' '.join(str(m) for m in msg))
def error(*msg): _logger.error(' '.join(str(m) for m in msg))

try:
    import scipy.linalg.blas
    _have_fblas = True
    _blas_gemv_fns = {
            numpy.dtype('float32'):scipy.linalg.blas.fblas.sgemv,
            numpy.dtype('float64'):scipy.linalg.blas.fblas.dgemv,
            numpy.dtype('complex64'):scipy.linalg.blas.fblas.cgemv,
            numpy.dtype('complex128'):scipy.linalg.blas.fblas.zgemv,
            }
except ImportError, e:
    _have_fblas = False
    warning('Failed to import scipy.linalg.blas.fblas. Falling back on slower implementations (%s)' % str(e))

class Gemv(Op):
    """
    expression is beta * y + alpha * A x

    A is matrix
    x, y are vectors
    alpha, beta are scalars

    """
    def __init__(self, inplace):
        self.inplace=inplace
        if inplace:
            self.destroy_map={0:[0]}
    def __eq__(self, other):
        return type(self)==type(other) and self.inplace == other.inplace
    def __str__(self):
        if self.inplace:
            return 'Gemv{inplace}'
        else:
            return 'Gemv{no_inplace}'
    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)
    def make_node(self, y, alpha, A, x, beta):
        y = T.as_tensor_variable(y)
        x = T.as_tensor_variable(x)
        A = T.as_tensor_variable(A)
        alpha = T.as_tensor_variable(alpha)
        beta = T.as_tensor_variable(beta)
        if y.dtype != A.dtype or y.dtype != x.dtype:
            raise TypeError('Gemv requires matching dtypes', (y.dtype, A.dtype, x.dtype))
        if A.ndim != 2: raise TypeError('gemv requires matrix for A', A.type)
        if x.ndim != 1: raise TypeError('gemv requires vector for x', x.type)
        if y.ndim != 1: raise TypeError('gemv requires vector for y', y.type)
        if y.broadcastable[0] != A.broadcastable[0]:
            raise TypeError('broadcastable mismatch between y and A', (y.type, A.type))
        # The following is not grounds for error
        # because as long as sizes are 1 at time of perform() there is no problem
        #if x.broadcastable[0] != A.broadcastable[1]:
            #raise TypeError('broadcastable mismatch between x and A', (x.type, A.type))
        return Apply(self, [y, alpha, A, x, beta], [y.type()])
    def perform(self, node, inputs, out_storage):
        y, alpha, A, x, beta = inputs
        if _have_fblas:
            gemv = _blas_gemv_fns[y.dtype]

            #Here I suppose that A is in c order. If we don't make it explicitly
            #  as fortran order, scipy 0.7.2 seam to create a copy in fortran
            #  order instead of just reshaping it and using the trans flag.
            #If A is already in fortran order, make it in c order and using the
            #  trans flag don't seam to cause slowdown.
            #out_storage[0][0] = gemv(alpha, A, x, beta, y, overwrite_y=self.inplace)
            out_storage[0][0] = gemv(alpha, A.T, x, beta, y, overwrite_y=self.inplace, trans=True)
        else:
            out_storage[0][0] = numpy.asarray(
                    beta * y + alpha * numpy.dot(A, x)
                    , dtype=y.dtype)

gemv_no_inplace = Gemv(inplace=False)
gemv_inplace = Gemv(inplace=True)

def default_blas_ldflags():
    try:
        #if numpy was linked with library that are not installed, we can't reuse them.
        if all(not os.path.exists(dir) for dir in numpy.distutils.__config__.blas_opt_info['library_dirs']):
            return "-lblas"
        return ' '.join(
                        #TODO: the Gemm op below should separate the -L and -l arguments into the two callbacks that CLinker uses for that stuff.
                        # for now, we just pass the whole ldflags as the -l options part.
                        ['-L%s'%l for l in numpy.distutils.__config__.blas_opt_info['library_dirs']] +
                        ['-l%s'%l for l in numpy.distutils.__config__.blas_opt_info['libraries']])
#                       ['-I%s'%l for l in numpy.distutils.__config__.blas_opt_info['include_dirs']])
    except KeyError:
        return "-lblas"

AddConfigVar('blas.ldflags',
        "lib[s] to include for [Fortran] level-3 blas implementation",
        StrParam(default_blas_ldflags()))

@utils.memoize
def ldflags(libs=True, flags=False, libs_dir=False, include_dir=False):
    """Return a list of libraries against which an Op's object file should be
    linked to benefit from a BLAS implementation.

    Default: ['blas'], but configuration variable config.blas.ldflags overrides this.
    """
    rval = []
    if libs_dir:
        found_dyn=False
        dirs = [x[2:] for x in config.blas.ldflags.split() if x.startswith('-L')]
        l = ldflags()
        for d in dirs:
            for f in os.listdir(d):
                if f.endswith('.so') or f.endswith('.dylib') or f.endswith('.dll'):
                    if any([f.find(ll)>=0 for ll in l]):
                        found_dyn=True
        if not found_dyn and dirs:
            warning("We did not found a dynamic library into the library_dir of the library we use for blas. If you use ATLAS, make sure to compile it with dynamics library.")

    for t in config.blas.ldflags.split():
        try:
            t0, t1, t2 = t[0:3]
            assert t0 == '-'
        except:
            raise ValueError('invalid token in config.blas.ldflags', t)
        if libs_dir and t1 == 'L':
            rval.append(t[2:])
        elif include_dir and t1 == 'I':
            raise ValueError('Include dirs are not used for blas. We disable this as this can hide other headers and this is not wanted.', t)
            rval.append(t[2:])
        elif libs and t1=='l': # example -lmkl
            rval.append(t[2:])
        elif flags and t1 not in ['L','I','l']: # example -openmp
            rval.append(t)
        elif flags and t1 == 'L':
            #to find it when we load the compiled op if the env of the used is not well configured.
            rval.append('-Wl,-rpath,'+t[2:])
    return rval

class GemmRelated(Op):
    """Base class for Gemm and Dot22

    This class provides a kind of templated gemm Op.
    """
    def __eq__(self, other):
        return (type(self) == type(other))
    def __hash__(self):
        return hash(type(self))
    def __str__(self):
        return self.__class__.__name__
    def c_support_code(self):
        #return cblas_header_text()
        mod_str = """
        #ifndef MOD
        #define MOD %
        #endif
        static double time_time() // a time function like time.time()
        {
            struct timeval tv;
            gettimeofday(&tv, 0);
            return (double) tv.tv_sec + (double) tv.tv_usec / 1000000.0;
        }
        """
        return blas_header_text() + mod_str
    def c_headers(self):
        # std.cout doesn't require the '%' symbol to print stuff...
        # so it works much better with python's string-substitution stuff.
        return ['<iostream>', '<time.h>', '<sys/time.h>']

    def c_libraries(self):
        return ldflags()

    # code_cache_version is built by subclasses from
    #  build_gemm_version

    def c_compile_args(self):
        return ldflags(libs=False, flags=True)

    def c_lib_dirs(self):
        return ldflags(libs=False, libs_dir=True)

    def c_header_dirs(self):
        return ldflags(libs=False, include_dir=True)

    declare_NS = """
        int unit = 0;

        int type_num = %(_x)s->descr->type_num;
        int type_size = %(_x)s->descr->elsize; // in bytes

        npy_intp* Nx = %(_x)s->dimensions;
        npy_intp* Ny = %(_y)s->dimensions;
        npy_intp* Nz = 0; //%(_zout)s->dimensions;

        npy_intp* Sx = %(_x)s->strides;
        npy_intp* Sy = %(_y)s->strides;
        npy_intp* Sz = 0; //%(_zout)s->strides;

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;
        """

    #setup_z_Nz_Sz = None

    check_xyz_rank2 = """
        if (%(_x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(_y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(_zout)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(z) != 2"); %(fail)s;}
        """
    check_xyz_double_or_float = """
        if ((%(_x)s->descr->type_num != PyArray_DOUBLE)
            && (%(_x)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((%(_y)s->descr->type_num != PyArray_DOUBLE)
            && (%(_y)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((%(_zout)s->descr->type_num != PyArray_DOUBLE)
            && (%(_zout)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((%(_x)s->descr->type_num != %(_y)s->descr->type_num)
            ||(%(_x)s->descr->type_num != %(_zout)s->descr->type_num))
        { PyErr_SetString(PyExc_NotImplementedError, "type(z), type(y), type(z) are not all the same"); %(fail)s; }
        """

    #it is not necessary that a or b have the same type as x,y,z
    check_ab_double_or_float = """
        if ((%(_a)s->descr->type_num != PyArray_DOUBLE)
            && (%(_a)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        if ((%(_b)s->descr->type_num != PyArray_DOUBLE)
            && (%(_b)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(b) is not double or float"); %(fail)s;}
        """

    check_dims_strides = """
        if (Nx[0] != Nz[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld rows but z has %%ld rows",
                (long int)Nx[0], (long int)Nz[0]);
            %(fail)s;
        }
        if (Nx[1] != Ny[0])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: x has %%ld cols but y has %%ld rows",
                (long int)Nx[1], (long int)Ny[0]);
            %(fail)s;
        }
        if (Ny[1] != Nz[1])
        {
            PyErr_Format(PyExc_ValueError,
                "Shape mismatch: y has %%ld cols but z has %%ld cols",
                (long int)Ny[1], (long int)Nz[1]);
            %(fail)s;
        }
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
           || (Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
           || (Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size))
        {
            PyErr_SetString(PyExc_NotImplementedError, "stride is not multiple of element size"); %(fail)s;
        }
        """

    encode_strides_in_unit = """
        /*
        encode the stride structure of _x,_y,_zout into a single integer
        */
        unit |= ((Sx[1] == type_size) ? 0x0 : (Sx[0] == type_size) ? 0x1 : 0x2) << 8;
        unit |= ((Sy[1] == type_size) ? 0x0 : (Sy[0] == type_size) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size) ? 0x0 : (Sz[0] == type_size) ? 0x1 : 0x2) << 0;
        """

    compute_strides = """
        /* create appropriate strides for malformed matrices that are row or column
         * vectors
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : Nx[1];
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : Nx[0];
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : Ny[1];
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : Ny[0];
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : Nz[1];
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : Nz[0];
        """

    begin_switch_typenum = """
        switch (type_num)
        {
        """

    case_float = """
            case PyArray_FLOAT:
            {
        """

    #case_float_ab_constants = None

    case_float_gemm = """
                float* x = (float*)PyArray_DATA(%(_x)s);
                float* y = (float*)PyArray_DATA(%(_y)s);
                float* z = (float*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                //double t0 = time_time();
                switch(unit)
                {
                    case 0x000: sgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: sgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: sgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: sgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: sgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: sgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: sgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: sgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
                //fprintf(stderr, "Calling sgemm %%i %%i %%i %%i took %%f\\n", unit, Nz1, Nz0, Nx1, time_time() - t0);
        """

    case_double = """
            }
            break;
            case PyArray_DOUBLE:
            {
        """

    #case_double_ab_constants = None

    case_double_gemm = """
                double* x = (double*)PyArray_DATA(%(_x)s);
                double* y = (double*)PyArray_DATA(%(_y)s);
                double* z = (double*)PyArray_DATA(%(_zout)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
                //double t0 = time_time();
                switch(unit)
                {
                    case 0x000: dgemm_(&N, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_0, &b, z, &sz_0); break;
                    case 0x100: dgemm_(&N, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_0, x, &sx_1, &b, z, &sz_0); break;
                    case 0x010: dgemm_(&T, &N, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_0, &b, z, &sz_0); break;
                    case 0x110: dgemm_(&T, &T, &Nz1, &Nz0, &Nx1, &a, y, &sy_1, x, &sx_1, &b, z, &sz_0); break;
                    case 0x001: dgemm_(&T, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_0, &b, z, &sz_1); break;
                    case 0x101: dgemm_(&N, &T, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_0, &b, z, &sz_1); break;
                    case 0x011: dgemm_(&T, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_0, y, &sy_1, &b, z, &sz_1); break;
                    case 0x111: dgemm_(&N, &N, &Nz0, &Nz1, &Nx1, &a, x, &sx_1, y, &sy_1, &b, z, &sz_1); break;
                    default: PyErr_SetString(PyExc_ValueError, "some matrix has no unit stride"); %(fail)s;
                };
                //fprintf(stderr, "Calling dgemm %%i %%i %%i %%i took %%f\\n", unit, Nz1, Nz0, Nx1, time_time()- t0);
        """

    end_switch_typenum = """
            }
            break;
        }
        """

    def build_gemm_call(self):

        return reduce(str.__add__, (
            self.declare_NS,
            self.setup_z_Nz_Sz,
            self.check_xyz_rank2,
            self.check_xyz_double_or_float,
            self.check_ab_double_or_float,
            self.check_dims_strides,
            self.encode_strides_in_unit,
            self.compute_strides,
            self.begin_switch_typenum,
            self.case_float,
            self.case_float_ab_constants,
            self.case_float_gemm,
            self.case_double,
            self.case_double_ab_constants,
            self.case_double_gemm,
            self.end_switch_typenum), '')

    def build_gemm_version(self):
        return (4,)

class Gemm(GemmRelated):
    """In-place version of matrix-matrix multiplication (with accumulation):

    When a and b are scalars and x, y, and z are matrices, then

        gemm(z,a,x,y,b)

    is similar to

        b*z + a*dot(x,y)

    The difference between the two is that the top form is destructive on z,
    whereas the bottom form is not.  Gemm works in-place on the storage
    associated with z, and the L{Variable} returned by Gemm has a storage that
    will be aliased to the storage of the z argument. Because of this in-place
    computation, an L{Apply} of this op will destroy the L{Variable} z on
    which it operates.  (See L{DestructiveOps} for an explanation of what
    destroying means in the context of theano graphs. See L{BlasLapackSupport} for
    more optimized linear algebra operations.)

    """
    E_rank = 'gemm only works for rank 2'
    E_scalar = 'gemm requires scalar argument'
    E_z_uniq = 'argument z aliased to x or y'
    def __init__(self, inplace):
        self.__setstate__({'inplace':inplace})

    def __eq__(self, other):
        return (type(self) == type(other)\
                and self.inplace == other.inplace)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def __str__(self):
        if self.inplace: inplace_str = 'inplace'
        else: inplace_str = 'no_inplace'
        return '%s{%s}' % (self.__class__.__name__, inplace_str)

    def __setstate__(self, dct):
        inplace = dct.get('inplace', True)
        if inplace:
            self.destroy_map = {0: [0]}
            self.setup_z_Nz_Sz = self.setup_z_Nz_Sz_inplace
        else:
            self.setup_z_Nz_Sz = self.setup_z_Nz_Sz_outplace
        self.inplace = inplace
    def __getstate__(self):
        return dict(inplace=self.inplace)


    def make_node(self, *inputs):
        inputs = map(T.as_tensor_variable, inputs)
        if len(inputs) != 5:
            raise TypeError("Wrong number of inputs for %s (expected 5, got %s)" % (self, len(inputs)))
        z, a, x, y, b = inputs
        zr, xr, yr = [set(view_roots(i)) for i in z,x,y]
        if zr.intersection(xr):
            raise InconsistencyError(Gemm.E_z_uniq, (z, x))
        if zr.intersection(yr):
            raise InconsistencyError(Gemm.E_z_uniq, (z, y))
        bz, ba, bx, by, bb = [r.type.broadcastable for r in inputs]
        if bz != (False,False): raise ValueError(Gemm.E_rank, bz)
        if bx != (False,False): raise ValueError(Gemm.E_rank, bx)
        if by != (False,False): raise ValueError(Gemm.E_rank, by)
        if len(ba): raise ValueError(Gemm.E_scalar, ba)
        if len(bb): raise ValueError(Gemm.E_scalar, bb)
        output = z.type()
        return Apply(self, inputs, [output])
    def perform(self, node, (z, a, x, y, b), (zout, )):
        assert a.shape == ()
        assert b.shape == ()
        if not self.inplace:
            z = z.copy() # the original z will not be changed
        if z.shape == ():
            z.itemset(z*a + b*numpy.dot(x,y))
            zout[0] = z
        else:
            if b == 0.0:
                if a == 1.0:
                    z[:] = numpy.dot(x,y)
                elif a == -1.0:
                    z[:] = -numpy.dot(x,y)
                else:
                    z[:] = a * numpy.dot(x,y)
            elif b == 1.0:
                if a == 1.0:
                    z += numpy.dot(x,y)
                elif a == -1.0:
                    z -= numpy.dot(x,y)
                else:
                    z += a * numpy.dot(x,y)
            else:
                z *= b
                z += a * numpy.dot(x,y)
            zout[0] = z

    setup_z_Nz_Sz_inplace = """
        if (%(_zout)s != %(_z)s)
        {
            if (%(_zout)s)
            {
                Py_DECREF(%(_zout)s);
            }
            %(_zout)s = %(_z)s;
            Py_INCREF(%(_zout)s);
        }
        Nz = %(_z)s->dimensions;
        Sz = %(_z)s->strides;
        """

    setup_z_Nz_Sz_outplace = """
        if ((NULL == %(_zout)s)
            || (%(_zout)s->dimensions[0] != %(_z)s->dimensions[0])
            || (%(_zout)s->dimensions[1] != %(_z)s->dimensions[1]))
        {
            if (%(_zout)s) Py_XDECREF(%(_zout)s);
            npy_intp dims[2];
            dims[0] = %(_z)s->dimensions[0];
            dims[1] = %(_z)s->dimensions[1];
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, type_num_%(_z)s);
            //fprintf(stderr, "Gemm Allocating %%i %%i\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc gemm_no_inplace output");
                %(fail)s
            }
        }
        Nz = %(_zout)s->dimensions;
        Sz = %(_zout)s->strides;
        if (1) // COPY z -> zout
        {
            if (%(_zout)s->descr->type_num == PyArray_FLOAT)
            {
                float * zoutdata = (float*)%(_zout)s->data;
                const float * zdata = (float*)%(_z)s->data;
                int zi = %(_z)s->strides[0]/sizeof(float);
                int zj = %(_z)s->strides[1]/sizeof(float);
                for (int i = 0; i < Nz[0]; ++i)
                {
                    for (int j = 0; j < Nz[1]; ++j)
                    {
                        zoutdata[i*Nz[1]+j] = zdata[zi*i+zj*j];
                    }
                }
            }
            else if (%(_zout)s->descr->type_num == PyArray_DOUBLE)
            {
                double * zoutdata = (double*) %(_zout)s->data;
                const double * zdata = (double*)%(_z)s->data;
                int zi = %(_z)s->strides[0]/sizeof(double);
                int zj = %(_z)s->strides[1]/sizeof(double);
                for (int i = 0; i < Nz[0]; ++i)
                {
                    for (int j = 0; j < Nz[1]; ++j)
                    {
                        zoutdata[i*Nz[1]+j] = zdata[zi*i+zj*j];
                    }
                }
            }
            else
            {
                PyErr_SetString(PyExc_AssertionError, "neither float nor double dtype");
                %(fail)s
            }
        }
        """

    case_float_ab_constants = """
        #define REAL float
        float a = (%(_a)s->descr->type_num == PyArray_FLOAT)
        ? (REAL)(((float*)%(_a)s->data)[0])
        : (REAL)(((double*)%(_a)s->data)[0]);
        float b = (%(_b)s->descr->type_num == PyArray_FLOAT) ?
        (REAL)(((float*)%(_b)s->data)[0])
        : (REAL)(((double*)%(_b)s->data)[0]);
        #undef REAL
        """
    case_double_ab_constants = """
        #define REAL double
        double a = (%(_a)s->descr->type_num == PyArray_FLOAT)
        ? (REAL)(((float*)%(_a)s->data)[0])
        : (REAL)(((double*)%(_a)s->data)[0]);
        double b = (%(_b)s->descr->type_num == PyArray_FLOAT) ?
        (REAL)(((float*)%(_b)s->data)[0])
        : (REAL)(((double*)%(_b)s->data)[0]);
        #undef REAL
        """

    def c_code(self, node, name, (_z, _a, _x, _y, _b), (_zout, ), sub): #DEBUG
        if node.inputs[0].type.dtype.startswith('complex'):
            raise utils.MethodNotDefined('%s.c_code' \
                    % self.__class__.__name__)
        if not config.blas.ldflags:
            return super(Gemm, self).c_code(node, name, (_z, _a, _x, _y, _b), (_zout, ), sub)
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code

    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (3,) + gv
        else:
            return gv

gemm_inplace = Gemm(inplace=True)
gemm_no_inplace = Gemm(inplace=False)
pprint.assign(gemm_inplace, FunctionPrinter('gemm_inplace'))
pprint.assign(gemm_no_inplace, FunctionPrinter('gemm_no_inplace'))

def res_is_a(node, op, maxclients=None):
    if maxclients is not None:
        retval = (len(node.clients) <= maxclients)
    else:
        retval = True

    return node.owner \
              and node.owner.op == op \
              and retval


def _as_scalar(res):
    """Return None or a TensorVariable whose type is in T.float_scalar_types"""
    if numpy.all(res.type.broadcastable):
        while res.owner and isinstance(res.owner.op, T.DimShuffle):
            res = res.owner.inputs[0]
        if res.type.broadcastable: # may still have some number of True's
            rval = res.dimshuffle()
        else:
            rval = res

        if rval.type.dtype[:3] in ('int', 'uin'):
            rval = cast(rval, theano.config.floatX) #may lose precision !?

        #if isinstance(rval, T.Constant):
            #rval = rval.data.flatten()[0]

        return rval

def _is_real_matrix(res):
    return res.type.dtype in ('float32', 'float64') \
            and res.type.ndim == 2 \
            and res.type.broadcastable[0] == False \
            and res.type.broadcastable[1] == False #cope with tuple vs. list
def _is_real_vector(res):
    return res.type.dtype in ('float32', 'float64') \
            and res.type.ndim == 1 \
            and res.type.broadcastable[0] == False

def _beta_L_plus_alpha_M(beta, L, alpha, M, recurse_flip = True):
    #print 'BETA L + ALPHA M', beta, L, alpha, M, recurse_flip
    #EXPRESSION: (beta * L) + (alpha * M)

    # we've already checked the client counts, now just make the type check.
    ####if res_is_a(M, _dot22, 1):
    if M.owner and M.owner.op == _dot22:
        if M.broadcastable == L.broadcastable:
            Ml, Mr = M.owner.inputs
            rval = [gemm_no_inplace(L, alpha, Ml, Mr, beta)]
            #print 'GEMM 0', rval, beta, L, alpha, M
            return rval

    if M.owner and M.owner.op == T.dot\
            and L.broadcastable==(False,) \
            and M.broadcastable==(False,):
        Ml, Mr = M.owner.inputs
        rval = None
        if Ml.ndim == 1:
            if Mr.ndim == 1:
                #TODO: insert a BLAS ddot Op
                pass
            if Mr.ndim == 2:
                #print "RETURNING GEMV (case 2)"
                if Mr.dtype == Ml.dtype:
                    rval = [gemv_no_inplace(L, alpha, Mr.T, Ml, beta)]
                    assert L.type == rval[0].type, (L.type, rval[0].type)
                else:
                    # TODO
                    pass
        if Ml.ndim == 2:
            if Mr.ndim == 1:
                #print "RETURNING GEMV (case 3)"
                if Mr.dtype == Ml.dtype:
                    rval = [gemv_no_inplace(L, alpha, Ml, Mr, beta)]
                    assert L.type == rval[0].type, (L.type, rval[0].type)
                else:
                    # TODO
                    pass
            if Mr.ndim == 2:
                # should have already got this case with a _dot22
                pass
        return rval

    # this is False'd out because of inadequate testing.
    # TODO see ticket #237
    if False and res_is_a(M, gemm_no_inplace, 1):
        #EXPRESSION: (beta * L) + (alpha * (gemm_no_inplace(G, a, u, v, b)))
        #EXPRESSION: (beta * L) + alpha * (b * G) + alpha * a * dot(u, v)
        G, a, u, v, b = M.owner.inputs
        #print 'GEMM', G, L

        if res_is_a(G, _dot22, 1):
            #EXPRESSION: (beta * L) + (alpha * (gemm_no_inplace(dot(x,y), a, u, v, b)))
            x, y = G.owner.inputs

            #EXPRESSION: (beta * L) + (alpha * ((b*dot(x,y) + (a * dot(u, v)))))
            #EXPRESSION: (beta * L) + (alpha*b*dot(x,y)) + (alpha * a * dot(u, v))
            rval = [gemm_no_inplace(gemm_no_inplace(L, alpha * b, x, y, beta), alpha * a, u, v, 1.0)]
            return rval
        if (G is L):
            #EXPRESSION: (beta * L) + (alpha*b*L) + (alpha * a * dot(u, v))
            rval = [gemm_no_inplace(L, alpha*a, u, v, alpha * b + beta)]
            return rval
        if (1.0 != alpha):
            #at the very least, move the alpha inside the gemm_no_inplace
            rval = [beta * L + gemm_no_inplace(G, alpha * a, u, v, alpha * b)]
            return rval

    if recurse_flip:
        return _beta_L_plus_alpha_M(alpha, M, beta, L, recurse_flip = False)
    else:
        return False


def _gemm_canonicalize(r, scale, rval, maxclients):
    # Tries to interpret node as a sum of scalars * (vectors or matrices)
    def scaled(thing):
        if scale == 1:
            return thing
        if scale == -1:
            return -thing
        else:
            return scale*thing
    try:
        r.type.broadcastable
    except:
        return None

    if ((r.type.ndim not in (1, 2)) or
            r.type.dtype not in ('float32', 'float64', 'complex64', 'complex128')):
        rval.append(scaled(r))
        return rval

    if maxclients and len(getattr(r,'clients',[])) > maxclients:
        rval.append((scale, r))
        return rval

    if r.owner and r.owner.op == T.sub:
        _gemm_canonicalize(r.owner.inputs[0], scale, rval, 1)
        _gemm_canonicalize(r.owner.inputs[1], -scale, rval, 1)

    elif r.owner and r.owner.op == T.add:
        for i in r.owner.inputs:
            _gemm_canonicalize(i, scale, rval, 1)

    elif r.owner and r.owner.op == T.neg:
        _gemm_canonicalize(r.owner.inputs[0], -scale, rval, 1)

    elif r.owner and r.owner.op == T.mul:
        scalars = []
        vectors = []
        matrices = []
        for i in r.owner.inputs:
            if numpy.all(i.type.broadcastable):
                while i.owner and isinstance(i.owner.op, T.DimShuffle):
                    i = i.owner.inputs[0]
                if i.type.broadcastable:
                    scalars.append(i.dimshuffle())
                else:
                    scalars.append(i)
            elif _is_real_vector(i):
                vectors.append(i)
            elif _is_real_matrix(i):
                matrices.append(i)
            else:
                # just put the original arguments as in the base case
                rval.append((scale,r))
                return rval
        if len(matrices)==1:
            assert len(vectors)==0
            m = matrices[0]
            if len(scalars) == 0:
                _gemm_canonicalize(m, scale, rval, 1)
            elif len(scalars) == 1:
                _gemm_canonicalize(m, scaled(scalars[0]), rval, 1)
            else:
                _gemm_canonicalize(m, T.mul(scaled(scalars[0]), *scalars[1:]), rval, 1)
        elif len(vectors)==1:
            assert len(matrices)==0
            v = vectors[0]
            if len(scalars) == 0:
                _gemm_canonicalize(v, scale, rval, 1)
            elif len(scalars) == 1:
                _gemm_canonicalize(v, scaled(scalars[0]), rval, 1)
            else:
                _gemm_canonicalize(v, T.mul(scaled(scalars[0]), *scalars[1:]), rval, 1)
        else: #lets not open this up
            rval.append((scale,r))
    else:
        rval.append((scale,r))
    return rval

def _factor_canonicalized(lst):
    # remove duplicates from canonicalized list

    # we only delete out of the right end of the list,
    # once i has touched a list element, it is permantent
    lst = list(lst)
    #print 'FACTOR', lst
    #for (a,b) in lst:
        #theano.printing.debugprint(a)
        #theano.printing.debugprint(b)
    i = 0
    while i < len(lst)-1:
        try:
            s_i,M_i = lst[i]
        except:
            i += 1
            continue

        j = i+1
        while j < len(lst):
            try:
                s_j,M_j = lst[j]
            except:
                j += 1
                continue

            if M_i is M_j:
                s_i = s_i + s_j
                lst[i] = (s_i, M_i)
                del lst[j]
            else:
                j += 1
        i+=1
    return lst

def _gemm_from_factored_list(lst):
    """Returns None, or a list to replace node.outputs
    """
    # Try every pair in the sM_list, trying to turn it into a gemm operation
    for i in xrange(len(lst) - 1):
        try:
            s_i,M_i = lst[i]
        except:
            continue

        for j in xrange(i+1, len(lst)):

            try:
                s_j, M_j = lst[j]
            except:
                continue

            #print 'TRYING', (s_i, M_i, s_j, M_j)

            gemm_of_sM_list = _beta_L_plus_alpha_M(s_i, M_i, s_j, M_j)
            #print 'GOT IT', gemm_of_sM_list
            if gemm_of_sM_list:
                def item_to_var(t):
                    try: s,M = t
                    except: return t
                    if s == 1: return M
                    if s == -1: return -M
                    return s*M

                assert len(gemm_of_sM_list) == 1
                add_inputs = [item_to_var(input)
                        for k, input in enumerate(lst) if k not in (i,j)]
                add_inputs.extend(gemm_of_sM_list)
                if len(add_inputs) > 1:
                    rval = [T.add(*add_inputs)]
                else:
                    rval = add_inputs
                #print "RETURNING GEMM THIGN", rval
                return rval

def _gemm_from_node2(node):
    """
    :todo: In many expressions, there are many ways to turn it into a gemm.  For example
    dot(a,b) + c + d.  This function should return all of them, so that if one version of gemm
    causes a cycle in the graph, then another application of gemm can be tried.

    """
    lst = []
    _gemm_canonicalize(node.outputs[0], 1.0, lst, 0)
    #print "GEMM CANON", lst
    if len(lst) > 1:
        lst = _factor_canonicalized(lst)
        rval = _gemm_from_factored_list(lst)
        #print "RVAL", rval
        if rval:
            assert rval[0].type == node.outputs[0].type, (rval[0].type, node.outputs[0].type)
        return rval

class GemmOptimizer(Optimizer):
    """Graph optimizer for inserting Gemm operations"""
    def __init__(self):
        Optimizer.__init__(self)

    def add_requirements(self, env):
        env.extend(toolbox.ReplaceValidate())
        env.extend(DestroyHandler())

    def apply(self, env):
        did_something = True
        while did_something:
            nodelist = list(env.toposort())
            did_something = False
            nodelist.reverse()
            for node in nodelist:
                try:
                    new_outputs = _gemm_from_node2(node)
                except InconsistencyError, e:
                    continue
                if new_outputs:
                    assert len(new_outputs) == len(node.outputs)
                    try:
                        env.replace_all_validate(
                                zip(node.outputs, new_outputs),
                                reason = 'GemmOptimizer')
                        did_something = True
                        break
                    except InconsistencyError, e:
                        #TODO: retry other applications of gemm (see comment in _gemm_from_node
                        pass

class Dot22(GemmRelated):
    """Compute a matrix-matrix product.
    This is a specialization of the more general Dot()
    """
    def make_node(self, x, y):
        if x.type.ndim != 2 or x.type.dtype not in ('float32', 'float64'):
            raise TypeError(x)
        if y.type.ndim != 2 or y.type.dtype not in ('float32', 'float64'):
            raise TypeError(y)
        if y.type.dtype != x.type.dtype:
            raise TypeError('dtype mismatch to Dot22')
        bz = (x.type.broadcastable[0], y.type.broadcastable[1])
        outputs = [T.tensor(x.type.dtype, bz)]
        return Apply(self, [x,y], outputs)

    def perform(self, node, (x, y), (z, )):
        try:
            z[0] = numpy.asarray(numpy.dot(x, y))
        except ValueError, e:
            # The error raised by numpy has no shape information, we mean to add that
            e.args = e.args + (x.shape, y.shape)
            raise
    def __str__(self):
        return "_dot22"

    setup_z_Nz_Sz = """
        if ((NULL == %(_zout)s)
            || (%(_zout)s->dimensions[0] != %(_x)s->dimensions[0])
            || (%(_zout)s->dimensions[1] != %(_y)s->dimensions[1]))
        {
            if (NULL != %(_zout)s) Py_XDECREF(%(_zout)s);
            npy_intp dims[2];
            dims[0] = %(_x)s->dimensions[0];
            dims[1] = %(_y)s->dimensions[1];
            %(_zout)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, type_num_%(_x)s);
            //fprintf(stderr, "Dot Allocating %%i %%i\\n", dims[0], dims[1]);
            if(!%(_zout)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc dot22 output");
                %(fail)s
            }
        }
        Nz = %(_zout)s->dimensions;
        Sz = %(_zout)s->strides;

        """
    check_ab_double_or_float = ""
    case_float_ab_constants = """
                float a = 1.0;
                float b = 0.0;
        """
    case_double_ab_constants = """
                double a = 1.0;
                double b = 0.0;
        """
    def c_code(self, node, name, (_x, _y), (_zout, ), sub): #DEBUG
        if node.inputs[0].type.dtype.startswith('complex'):
            raise utils.MethodNotDefined('%s.c_code' \
                    % self.__class__.__name__)
        if len(self.c_libraries())<=0:
            return super(Dot22, self).c_code(node, name, (_x, _y), (_zout, ), sub)
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code
    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (1,) + gv
        else:
            return gv

_dot22 = Dot22()

@local_optimizer([T.dot])
def local_dot_to_dot22(node):
    if node.op != T.dot:
        return

    x,y = node.inputs
    if y.type.dtype != x.type.dtype:
        # TODO: upcast one so the types match
        info('Not optimizing dot with inputs', x, y, x.type, y.type)
        return
    if y.type.dtype.startswith('float'):
        if _is_real_matrix(x) and _is_real_matrix(y):
            return [_dot22(*node.inputs)]
        if 0:
            if _is_real_matrix(x) and _is_real_vector(y):
                return [_dot22(x, y.dimshuffle(0,'x')).dimshuffle(0)]
            if _is_real_vector(x) and _is_real_matrix(y):
                return [_dot22(x.dimshuffle('x',0), y).dimshuffle(1)]
            if _is_real_vector(x) and _is_real_vector(x):
                return [_dot22(x.dimshuffle('x',0), y.dimshuffle(0,'x')).dimshuffle()]

    info('Not optimizing dot with inputs', x, y, x.type, y.type)

@local_optimizer([gemm_no_inplace])
def local_inplace_gemm(node):
    if node.op == gemm_no_inplace:
        return [gemm_inplace(*node.inputs)]
@local_optimizer([gemv_no_inplace])
def local_inplace_gemv(node):
    if node.op == gemv_no_inplace:
        return [gemv_inplace(*node.inputs)]

#################################
#
# Set up the BlasOpt optimizer
#
#################################

blas_optdb = SequenceDB()

# run after numerical stability optimizations (1.5)
optdb.register('BlasOpt', blas_optdb, 1.7, 'fast_run')
# run before specialize (2.0) because specialize is basically a free-for-all that makes the
# graph crazy.

blas_optdb.register('local_dot_to_dot22',
        EquilibriumOptimizer([local_dot_to_dot22], max_use_ratio=5),
        0, 'fast_run')
blas_optdb.register('local_dot_to_gemm', GemmOptimizer(), 10, 'fast_run')

# After destroyhandler is in but before we try to make elemwise things inplace
# Try to make gemm inplace
# Also, need to make the gemm optimisation(step 70) happen before the fusion of elemwise(step 71)
optdb.register('InplaceBlasOpt',
        EquilibriumOptimizer([local_inplace_gemm, local_inplace_gemv], failure_callback=EquilibriumOptimizer.warn_inplace,
            max_use_ratio=5),
        70.0, 'fast_run', 'inplace')

class Dot22Scalar(GemmRelated):
    """Compute a matrix-matrix product.
    This is a specialization of the more general Dot()
    Used to call optimized gemm implementation.
    Also used to generate a gemm later.
    compute scalar*dot(x,y)
    """
    def make_node(self, x, y, scalar):
        if not _is_real_matrix(x):
            raise TypeError(x)
        if not _is_real_matrix(x):
            raise TypeError(y)
        if not _as_scalar(scalar):
            raise TypeError(scalar)
        if y.type.dtype != x.type.dtype and y.type.dtype != scalar.type.dtype:
            raise TypeError('dtype mismatch to Dot22Scalar')
        bz = [False, False]
        outputs = [T.tensor(x.type.dtype, bz)]
        return Apply(self, [x,y,scalar], outputs)

    def perform(self, node, (x, y, scalar), (z, )):
        try:
            z[0] = scalar * numpy.asarray(numpy.dot(x, y))
        except ValueError, e:
            # The error raised by numpy has no shape information, we mean to add that
            e.args = e.args + (x.shape, y.shape)
            raise
    def __str__(self):
        return "_dot22scalar"

    setup_z_Nz_Sz = Dot22.setup_z_Nz_Sz

    check_ab_double_or_float = """
        if ((%(_a)s->descr->type_num != PyArray_DOUBLE)
            && (%(_a)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        """
    case_float_ab_constants = """
        #define REAL float
        float a = (%(_a)s->descr->type_num == PyArray_FLOAT)
        ? (REAL)(((float*)%(_a)s->data)[0])
        : (REAL)(((double*)%(_a)s->data)[0]);
        #undef REAL
        float b = 0.0;
        """
    case_double_ab_constants = """
        #define REAL double
        double a = (%(_a)s->descr->type_num == PyArray_FLOAT)
        ? (REAL)(((float*)%(_a)s->data)[0])
        : (REAL)(((double*)%(_a)s->data)[0]);
        #undef REAL
        double b = 0.0;
        """
    def c_code(self, node, name, (_x, _y, _a), (_zout, ), sub): #DEBUG
        if len(self.c_libraries())<=0:
            return super(Dot22Scalar, self).c_code(node, name, (_x, _y), (_zout, ), sub)
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code
    def c_code_cache_version(self):
        gv = self.build_gemm_version()
        if gv:
            return (2,) + gv
        else:
            return gv

_dot22scalar = Dot22Scalar()

@local_optimizer([T.mul])
def local_dot22_to_dot22scalar(node):
    """
    :note: we upcast the scalar if after the multiplication with the dot this give the same type.
    .. note:
        We execute this optimizer after the gemm optimizer. This allow to give more priority to gemm that give more speed up then this optimizer, but allow the gemm optimizer to ignore this op.


    TODO: support when we can reorder the mul to generate a dot22scalar or fix the canonizer to merge them(1 mul with multiple inputs)
    """
    if node.op != T.mul:
        return False
    i_dot22 = [x.owner and x.owner.op==_dot22 for x in node.inputs]
    if not any(i_dot22): return False # no dot22
    if i_dot22.count(True)>1: return False #TODO fix
    #we take the first _dot22 found. TODO check others!
    dot22_idx = i_dot22.index(True)
    d = node.inputs[dot22_idx]
    i_scalar = [_as_scalar(x) for x in node.inputs]
    if not any(i_scalar) and not any([x.owner and x.owner.op ==T.mul for x in node.inputs]):
        #no scalar in input and no multiplication
        #if their was a multiplication we couls reorder the graph by the associativity of the graph.
        return False

    if not any(i_scalar):
        #maybe we can reorder the graph as this mul have a mul in input.
        #The canonizer should have merged those mul together.
        #We support only 1 additional level of mul.
        i_mul = [x.owner and x.owner.op ==T.mul for x in node.inputs]
        mul_idx = i_mul.index(True)#we take the first mul!
        m = node.inputs[mul_idx]

        if len(m.owner.inputs)==2 and any([_as_scalar(x) for x in m.owner.inputs]):
            scalar_idx = 0
            for i,x in enumerate(m.owner.inputs):
                if _as_scalar(x):
                    scalar_idx=i
                    break
            dot=_dot22scalar(d.owner.inputs[0], d.owner.inputs[1],m.owner.inputs[scalar_idx])

            return [T.mul(m.owner.inputs[1-i],dot)]
        elif m.owner and m.owner.op == T.mul:
            info('Not optimizing dot22 with inputs', d, m, d.type, m.type, 'we need to check in a recursive way in the mul if we can reorder the graph. The canonizer should have done this.')
        else:
            return False

    scalar_idx = -1
    for i,x in enumerate(node.inputs):
        if i_scalar[i] and theano.scalar.upcast(x.type.dtype,d.type.dtype) == d.type.dtype:
            scalar_idx = i
            break
    if scalar_idx<0:
        info('Not optimizing dot22 with inputs', node.inputs, [x.type for x in node.inputs], 'as the type of the scalar can\'t be upcasted to the matrix type')
        return False
    assert scalar_idx<len(node.inputs)
    s = node.inputs[scalar_idx]
    o = copy.copy(node.inputs)
    o.remove(d)
    o.remove(s)
    if len(o)==0:
        return [_dot22scalar(d.owner.inputs[0], d.owner.inputs[1], s)]
    else:
        return [T.mul(_dot22scalar(d.owner.inputs[0], d.owner.inputs[1], s), *o)]

#must happen after gemm as the gemm optimizer don't understant dot22scalar and gemm give more speed up then dot22scalar
blas_optdb.register('local_dot22_to_dot22scalar',
        EquilibriumOptimizer([local_dot22_to_dot22scalar ], max_use_ratio=5),
        11, 'fast_run')


from opt import register_specialize, register_canonicalize
#@register_specialize
@local_optimizer([])
def local_print_as_we_go_along(node):
    if node.op in (T.sub, T.add):
        debugprint(node)
