"""Ops and optimizations for using BLAS function calls to evaluate linear algebra expressions"""

import os, sys, traceback
import numpy

from ..gof import (utils, Op, Apply, view_roots, PatternSub, DestroyHandler, 
        SeqOptimizer, warn, local_optimizer, LocalOptimizer, OpKeyOptimizer, 
        InconsistencyError)
from ..printing import pprint, FunctionPrinter
from .opt import register_specialize, out2in, insert_inplace_optimizer

import basic as T

#NB: this clobbers the builtin 'compile' symbol
from .. import compile  #to register the optimizer built by this file 

from .blas_headers import cblas_header_text, blas_header_text

@utils.memoize
def ldflags():
    """Return a list of libraries against which an Op's object file should be
    linked to benefit from a BLAS implementation.
    
    Default: ['blas'], but environment variable THEANO_BLAS_LDFLAGS overrides this.
    """
    if os.getenv('THEANO_BLAS_LDFLAGS'):
        tokens = os.getenv('THEANO_BLAS_LDFLAGS').split()
        for t in tokens:
            try:
                t0, t1, t2 = t[0:3]
                assert t0 == '-'
            except:
                raise ValueError('invalid token in THEANO_BLAS_LDFLAGS', t)
            if t1 == 'L':
                raise ValueError('library dir not allowed in THEANO_BLAS_LDFLAGS', t)
        rval = [token[2:] for token in tokens]
    else:
        rval = ['blas']
    #print "blas linking against", rval
    return rval

class GemmRelated(Op):
    """Base class for Gemm and Dot22
    
    This class provides a kind of templated gemm Op.
    """
    def c_support_code(self):
        #return cblas_header_text()
        mod_str = """
        #ifndef MOD
        #define MOD %
        #endif
        """
        return blas_header_text() + mod_str
    def c_headers(self):
        # std.cout doesn't require the '%' symbol to print stuff... 
        # so it works much better with python's string-substitution stuff.
        return ['<iostream>'] 
    
    def c_libraries(self):
        return ldflags()

    declare_NS = """
        int unit = 0;

        int type_num = %(_x)s->descr->type_num;
        int type_size = %(_x)s->descr->elsize; // in bytes

        npy_intp* Nx = %(_x)s->dimensions;
        npy_intp* Ny = %(_y)s->dimensions;
        npy_intp* Nz = 0; //%(_z)s->dimensions;

        npy_intp* Sx = %(_x)s->strides;
        npy_intp* Sy = %(_y)s->strides;
        npy_intp* Sz = 0; //%(_z)s->strides;

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;
        """

    #setup_z_Nz_Sz = None

    check_xyz_rank2 = """
        if (%(_x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(_y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(_z)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(z) != 2"); %(fail)s;}
        """
    check_xyz_double_or_float = """
        if ((%(_x)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_x)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(x) is not double or float"); %(fail)s;}

        if ((%(_y)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_y)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(y) is not double or float"); %(fail)s;}

        if ((%(_z)s->descr->type_num != PyArray_DOUBLE) 
            && (%(_z)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(z) is not double or float"); %(fail)s;}

        if ((%(_x)s->descr->type_num != %(_y)s->descr->type_num)
            ||(%(_x)s->descr->type_num != %(_z)s->descr->type_num))
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
        if ((Nx[0] != Nz[0]) || (Nx[1] != Ny[0]) || (Ny[1] != Nz[1]))
        {
            PyErr_SetString(PyExc_ValueError, "Input dimensions do not agree");
            %(fail)s;
        }
        if ((Sx[0] < 1) || (Sx[1] < 1) || (Sx[0] MOD type_size) || (Sx[1] MOD type_size)
           || (Sy[0] < 1) || (Sy[1] < 1) || (Sy[0] MOD type_size) || (Sy[1] MOD type_size)
           || (Sz[0] < 1) || (Sz[1] < 1) || (Sz[0] MOD type_size) || (Sz[1] MOD type_size))
        {
            PyErr_SetString(PyExc_ValueError, "stride is not multiple of element size"); %(fail)s;
        }
        """

    encode_strides_in_unit = """
        /*
        encode the stride structure of _x,_y,_z into a single integer
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
                float* z = (float*)PyArray_DATA(%(_z)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
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
                double* z = (double*)PyArray_DATA(%(_z)s);
                char N = 'N';
                char T = 'T';
                int Nz0 = Nz[0], Nz1 = Nz[1], Nx1 = Nx[1];
                //std::cerr << (unit/256) MOD 16 << (unit / 16) MOD 16 << unit MOD 16<< '\\n';
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


class Gemm(GemmRelated):
    """In-place version of matrix-matrix multiplication (with accumulation):

    When a and b are scalars and x, y, and z are matrices, then

        gemm(z,a,x,y,b) 

    is similar to 

        b*z + a*dot(x,y) 

    The difference between the two is that the top form is destructive on z,
    whereas the bottom form is not.  Gemm works in-place on the storage
    associated with z, and the L{Result} returned by Gemm has a storage that
    will be aliased to the storage of the z argument. Because of this in-place
    computation, an L{Apply} of this op will destroy the L{Result} z on
    which it operates.  (See L{DestructiveOps} for an explanation of what
    destroying means in the context of theano graphs. See L{BlasLapackSupport} for
    more optimized linear algebra operations.)

    """
    E_rank = 'gemm only works for rank 2'
    E_scalar = 'gemm requires scalar argument'
    E_z_uniq = 'argument z aliased to x or y'
    destroy_map = {0: [0]}
    def make_node(self, *inputs):
        inputs = map(T.as_tensor, inputs)
        if len(inputs) != 5:
            raise TypeError("Wrong number of inputs for %s (expected 5, got %s)" % (self, len(inputs)))
        z, a, x, y, b = inputs
        zr, xr, yr = [set(view_roots(i)) for i in z,x,y]
        if zr.intersection(xr):
            raise ValueError(Gemm.E_z_uniq, (z, x))
        if zr.intersection(yr):
            raise ValueError(Gemm.E_z_uniq, (z, y))
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

    setup_z_Nz_Sz = """
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
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code
gemm = Gemm()

pprint.assign(gemm, FunctionPrinter('gemm'))

def res_is_a(node, op, maxclients=None):
    return node.owner \
            and node.owner.op == op \
            and (len(node.clients) <= maxclients if maxclients is not None else True)

class GemmLocalOptimizer(LocalOptimizer):
    """This is a massive beast for recognizing all the ways that a subtraction could be
    replaced by a GEMM

    It depends on `local_transposed_dot` to canonicalize the graph a bit by swapping
    dot(a,b).T -> dot(b.T, a.T)
    """

    def __init__(self):
        super(LocalOptimizer, self).__init__()

    def op_key(self):
        return [T.add, T.sub]

    def add_requirements(self, env):
        super(GemmLocalOptimizer,self).add_requirements(env)
        env.extend(DestroyHandler())

    def transform(self, node):
        _as_scalar, _is_real_matrix, _as_isolated_scalar_times_matrix, beta_L_plus_alpha_M\
                = (GemmLocalOptimizer._as_scalar, 
                        GemmLocalOptimizer._is_real_matrix, 
                        GemmLocalOptimizer._as_isolated_scalar_times_matrix, 
                        GemmLocalOptimizer.beta_L_plus_alpha_M)
        if node.op == T.sub:
            L, R = node.inputs
            if not _is_real_matrix(L):
                return False
            if not _is_real_matrix(R):
                return False

            tmp = _as_isolated_scalar_times_matrix(L)
            try:
                sL, mL = tmp
            except:
                sL, mL = 1.0, L

            tmp = _as_isolated_scalar_times_matrix(R)
            try:
                sR, mR = tmp
            except:
                sR, mR = 1.0, R
            rval = beta_L_plus_alpha_M(sL, mL, -sR, mR)
            return rval
        if node.op == T.add:
            sM_list = []
            other_inputs = []
            for input in node.inputs:
                tmp = _as_isolated_scalar_times_matrix(input)
                if tmp:
                    sM_list.append(tmp)
                elif _is_real_matrix(input):
                    sM_list.append((1.0, input))
                else:
                    other_inputs.append(input)

            if len(sM_list) == 2:
                (sL, mL), (sR, mR) = sM_list
                gemm_of_sM_list = beta_L_plus_alpha_M(sL, mL, sR, mR)
                if gemm_of_sM_list: 
                    #we turned the two candidates into a gemm
                    # now we have to add the other_inputs and return the replacement graph
                    if other_inputs:
                        return [T.add(*(other_inputs + gemm_of_sM_list))]
                    else:
                        return gemm_of_sM_list
            else:
                for i in xrange(len(sM_list) - 1):
                    for j in xrange(i+1, len(sM_list)):
                        sL, mL = sM_list[i]
                        sR, mR = sM_list[j]
                        gemm_of_sM_list = beta_L_plus_alpha_M(sL, mL, sR, mR)
                        if gemm_of_sM_list:
                            assert len(gemm_of_sM_list) == 1
                            inputs_without_ij = \
                                    [input for k, input in enumerate(node.inputs) if k not in (i,j)]
                            return [T.add( *(inputs_without_ij + gemm_of_sM_list + other_inputs))]
        return False

    @staticmethod
    def failure_callback(exc, nav, repl_pairs):
        """WRITEME"""
        if not isinstance(exc, InconsistencyError):
            traceback.print_exc()
        else:
            #print 'GEMM caused cycle, forget it.'
            pass

    @staticmethod
    def _as_scalar(res):
        """Return None or a TensorResult whose type is in T.float_scalar_types"""
        if res.owner and isinstance(res.owner.op, T.DimShuffle):
            return GemmLocalOptimizer._as_scalar(res.owner.inputs[0])
        elif res.type in T.float_scalar_types:
            return res
        elif isinstance(res, T.Constant) and res.data.size == 1:
            return res.data.flatten()[0]
        else:
            return None

    @staticmethod
    def _is_real_matrix(res):
        return res.type in T.float_matrix_types \
                and res.broadcastable[0] == False \
                and res.broadcastable[1] == False #cope with tuple vs. list

    @staticmethod
    def _as_isolated_scalar_times_matrix(res):
        _as_scalar, _is_real_matrix, _as_isolated_scalar_times_matrix, beta_L_plus_alpha_M\
                = (GemmLocalOptimizer._as_scalar, 
                        GemmLocalOptimizer._is_real_matrix, 
                        GemmLocalOptimizer._as_isolated_scalar_times_matrix, 
                        GemmLocalOptimizer.beta_L_plus_alpha_M)
        if res_is_a(res, T.mul, 1):
            if len(res.owner.inputs) == 2:
                L, R = res.owner.inputs
                sL = _as_scalar(L)
                sR = _as_scalar(R)
                if (sL is not None) and _is_real_matrix(R):
                    return (sL, R)
                if (sR is not None) and _is_real_matrix(L):
                    return (sR, L)
            else:
                scalars = []
                matrices = []
                for input in res.owner.inputs:
                    scalar_input = _as_scalar(input)
                    if scalar_input is not None:
                        scalars.append(scalar_input)
                    elif _is_real_matrix(input):
                        matrices.append(input)
                    else:
                        return None
                if len(matrices) == 1:
                    rval = (T.mul(*scalars), matrices[0])
                    return rval

    @staticmethod
    def beta_L_plus_alpha_M(beta, L, alpha, M, recurse_flip = True):
        #print 'BETA L + ALPHA M', beta, L, alpha, M, recurse_flip
        #EXPRESSION: (beta * L) + (alpha * M)

        if res_is_a(M, _dot22, 1):
            Ml, Mr = M.owner.inputs
            rval = [gemm(L, alpha, Ml, Mr, beta)]
            #print 'GEMM 0', rval, beta, L, alpha, M
            return rval

        # this is False'd out because of inadequate testing.  
        # TODO see ticket #237
        if False and res_is_a(M, gemm, 1):
            #EXPRESSION: (beta * L) + (alpha * (gemm(G, a, u, v, b)))
            #EXPRESSION: (beta * L) + alpha * (b * G) + alpha * a * dot(u, v)
            G, a, u, v, b = M.owner.inputs
            #print 'GEMM', G, L

            if res_is_a(G, _dot22, 1):
                #EXPRESSION: (beta * L) + (alpha * (gemm(dot(x,y), a, u, v, b)))
                x, y = G.owner.inputs

                #EXPRESSION: (beta * L) + (alpha * ((b*dot(x,y) + (a * dot(u, v)))))
                #EXPRESSION: (beta * L) + (alpha*b*dot(x,y)) + (alpha * a * dot(u, v))
                rval = [gemm(gemm(L, alpha * b, x, y, beta), alpha * a, u, v, 1.0)]
                print 'GEMM 1', rval
                return rval
            if (G is L):
                #EXPRESSION: (beta * L) + (alpha*b*L) + (alpha * a * dot(u, v))
                rval = [gemm(L, alpha*a, u, v, alpha * b + beta)]
                print 'GEMM 2', rval
                return rval
            if (1.0 != alpha):
                #at the very least, move the alpha inside the gemm
                rval = [beta * L + gemm(G, alpha * a, u, v, alpha * b)]
                print 'GEMM 3', rval
                return rval

        if recurse_flip:
            return GemmLocalOptimizer.beta_L_plus_alpha_M(alpha, M, beta, L, recurse_flip = False)
        else:
            return False

#I think that three passes should suffice to catch all the GEMMs.
# TODO: This could be an equilibriumOptmizer, but I don't know how to combine an OpKeyOptimizer and
# an EquilibriumOptimizer.
compile.optdb.register('inplace_gemm_0', OpKeyOptimizer(GemmLocalOptimizer(), 
    failure_callback=GemmLocalOptimizer.failure_callback), 70.00, 'fast_run', 'inplace')
compile.optdb.register('inplace_gemm_1', OpKeyOptimizer(GemmLocalOptimizer(), 
    failure_callback=GemmLocalOptimizer.failure_callback), 70.01, 'fast_run', 'inplace')
compile.optdb.register('inplace_gemm_2', OpKeyOptimizer(GemmLocalOptimizer(), 
    failure_callback=GemmLocalOptimizer.failure_callback), 70.02, 'fast_run', 'inplace')

class Dot22(GemmRelated):
    """Compute a matrix-matrix product.
    This is a specialization of the more general Dot()
    """
    def make_node(self, x, y):
        assert GemmLocalOptimizer._is_real_matrix(x)
        assert y.type == x.type               #makes sure y is a matrix
        bz = [False, False]
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
        if ((NULL == %(_z)s)
            || (%(_z)s->dimensions[0] != %(_x)s->dimensions[0])
            || (%(_z)s->dimensions[1] != %(_y)s->dimensions[1]))
        {
            if (NULL != %(_z)s) Py_XDECREF(%(_z)s);
            npy_intp dims[2];
            dims[0] = %(_x)s->dimensions[0];
            dims[1] = %(_y)s->dimensions[1];
            %(_z)s = (PyArrayObject*)PyArray_SimpleNew(2, dims, type_num_%(_x)s);
            if(!%(_z)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc dot22 output");
                %(fail)s
            }
        }
        Nz = %(_z)s->dimensions;
        Sz = %(_z)s->strides;

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
    def c_code(self, node, name, (_x, _y), (_z, ), sub): #DEBUG
        full_code = self.build_gemm_call() % dict(locals(), **sub)
        return full_code
_dot22 = Dot22()

@local_optimizer([T.dot])
def local_dot_to_dot22(node):
    if node.op == T.dot:
        x,y = node.inputs
        if GemmLocalOptimizer._is_real_matrix(x) and y.type == x.type:
            return [_dot22(*node.inputs)]
    else:
        return False
register_specialize(local_dot_to_dot22)

