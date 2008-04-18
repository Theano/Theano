"""A L{Result} to store L{numpy.ndarray} with basic accompanying L{Op}s"""
import sys # for sys.maxint
import inspect

import numpy

from gof import Result, Op, utils, Destroyer, Viewer, AbstractFunctionError
import gof.result
import gof.op

from base_tensor import BaseTensor, BaseTensorOp
import blas # for gemm, dot

import elemwise as s2t
import scalar as scal


class Tensor(BaseTensor):
    """
    This subclass of L{BaseTensor} provides operator overloading using
    implementations of L{Tensor} operations contained in this file.
    
    Operators:
     - most numeric operators are overloaded (to return L{Op}s that
     perform the corresponding calculation)
    """

    #UNARY
    def __abs__(self): return Abs(self).out
    def __neg__(self): return Neg(self).out

    #CASTS
    def __int__(self): return AsInt(self).out
    def __float__(self): return AsInt(self).out
    def __complex__(self): return AsComplex(self).out

    #COMPARISONS
    def __lt__(self,other): return lt(self, other)
    def __le__(self,other): return le(self, other)
    def __gt__(self,other): return gt(self, other)
    def __ge__(self,other): return ge(self, other)

    #ARITHMETIC - NORMAL
    def __add__(self,other): return add(self,other)
    def __sub__(self,other): return sub(self,other)
    def __mul__(self,other): return mul(self,other)
    def __div__(self,other): return div(self,other)
    def __pow__(self,other): return pow(self,other)

    #ARITHMETIC - INPLACE
    def __iadd__(self,other): return add_inplace(self,other)
    def __isub__(self,other): return sub_inplace(self,other)
    def __imul__(self,other): return mul_inplace(self,other)
    def __idiv__(self,other): return div_inplace(self,other)
    def __ipow__(self,other): return pow_inplace(self,other)

    #ARITHMETIC - RIGHT-OPERAND
    def __radd__(self,other): return add(other,self)
    def __rsub__(self,other): return sub(other,self)
    def __rmul__(self,other): return mul(other,self)
    def __rdiv__(self,other): return div(other,self)
    def __rpow__(self,other): return pow(other,self)

    #TRANSPOSE
    T = property(lambda self: transpose(self))

    #SLICING
    def __getitem__(self, item): return subtensor(self, item)
    def __getslice__(self, *args): return subtensor(self, slice(*args))

    #COPYING
    def copy(self): return tensor_copy(self)
    
s2t.Tensor = Tensor


# alternate Tensor constructor
def astensor(data, broadcastable=None, name=None):
    """Return a L{Tensor} containing given data"""
    if isinstance(data, BaseTensor):
        if broadcastable is not None and list(data.broadcastable) != list(broadcastable):
            raise TypeError("The data to wrap as a Tensor has the wrong broadcastable pattern. Expected %s, got %s." % (broadcastable, data.broadcastable))
        if name is not None and name != data.name:
            raise ValueError("Cannot rename an existing Tensor.")
        return data
    elif isinstance(data, Result):
        raise TypeError("Cannot make a Tensor out of a non-Tensor result:", data)
        
    if data is None and broadcastable is None:
        raise TypeError("Cannot make a Tensor out of None.")
    
    data = numpy.asarray(data)
    if broadcastable is None:
        broadcastable = [s==1 for s in data.shape]
    elif broadcastable in [0, 1]:
        broadcastable = [broadcastable] *  len(data.shape)
    rval = Tensor(data.dtype, broadcastable, name = name)
    rval.data = data # will raise if broadcastable was mis-specified
    return rval
s2t.astensor = astensor


############################
# Supporting Ops
############################

# this has a different name, because _as_tensor is the function which ops use
# to upcast their arguments... this internal-use function is a good place to put debugging stuff, better than the global astensor.
_as_tensor = astensor

class _Op(BaseTensorOp):
    """A convenient base for the ops in this file"""

    out_tensor_class = Tensor

    @classmethod
    def input_wrapper(cls, obj):
        return _as_tensor(obj)
    
    def c_var_names(self):
        (self, inames, onames), _1, _2, _3 = inspect.getargspec(self.c_impl)
        inames = utils.from_return_values(inames)
        onames = utils.from_return_values(onames)
        return [inames, onames]
    
    def c_code(self, input_names, output_names, sub):
        sub = dict(sub)
        icvn, ocvn = self.c_var_names()
        for real, tosub in zip(input_names + output_names, icvn + ocvn):
            sub[tosub] = real
        return self.c_impl(self.inputs, self.outputs) % sub

    def c_impl(self, inputs, outputs):
        raise AbstractFunctionError("No c_impl for %s" % self.__class__.__name__)

class _Unary:
    nin = 1

class _Binary:
    nin = 2


##########################
# Unary Operations
##########################

def broadcast(scalar_opclass, name, inplace_versions = True):
    C = s2t.make_broadcast(scalar_opclass, name = name)
    c = gof.op.constructor(s2t.wrap_broadcast(C))
    if inplace_versions:
        CInplace = s2t.make_broadcast(scalar_opclass, {0:0}, name = name+"Inplace")
        c_inplace = gof.op.constructor(s2t.wrap_broadcast(CInplace))
        return C, c, CInplace, c_inplace
    else:
        return C, c
    
class Argmax(Op):
    """Calculate the max and argmax over a given axis"""
    nin=2 # tensor, axis
    nout=2 # max val, max idx
    E_axis = 'invalid axis'
    debug = 0
    def __init__(self, x, axis=None):
        x = _as_tensor(x)
        if axis is None:
            axis = len(x.broadcastable) -1
        axis = _as_tensor(axis)
        self.inputs = [x, axis]
        broadcastable = [0] * (len(x.broadcastable) - 1)
        self.outputs = [Tensor(x.dtype, broadcastable), 
                Tensor(axis.dtype, broadcastable)]
    def perform(self): 
        axis = self.inputs[1].data
        x = self.inputs[0].data
        self.outputs[0].data = numpy.max(x, axis)
        self.outputs[1].data = numpy.argmax(x,axis)
argmax = gof.op.constructor(Argmax)

def max(x, axis=None):
    """Return maximum elements obtained by iterating over given axis

    Default axis is the last one.
    """
    # In python (using Argmax.perform()) this leads to an wasteful
    # implementation that goes through the data twice instead of once
    # but when Argmax.c_impl() is in place, it should be fine.
    return argmax(x,axis)[0]

Abs, _abs, AbsInplace, abs_inplace = broadcast(scal.Abs, 'Abs')
Exp, exp, ExpInplace, exp_inplace = broadcast(scal.Exp, 'Exp')
Neg, neg, NegInplace, neg_inplace = broadcast(scal.Neg, 'Neg')
Log, log, LogInplace, log_inplace = broadcast(scal.Log, 'Log')
Log2, log2, Log2Inplace, log2_inplace = broadcast(scal.Log2, 'Log2')
Sgn, sgn, SgnInplace, sgn_inplace = broadcast(scal.Sgn, 'Sgn')
Sqr, sqr, SqrInplace, sqr_inplace = broadcast(scal.Sqr, 'Sqr')
Sqrt, sqrt, SqrtInplace, sqrt_inplace = broadcast(scal.Sqrt, 'Sqrt')
Cos, cos, CosInplace, cos_inplace = broadcast(scal.Cos, 'Cos')
Sin, sin, SinInplace, sin_inplace = broadcast(scal.Sin, 'Sin')
Tan, tan, TanInplace, tan_inplace = broadcast(scal.Tan, 'Tan')
Cosh, cosh, CoshInplace, cosh_inplace = broadcast(scal.Cosh, 'Cosh')
Sinh, sinh, SinhInplace, sinh_inplace = broadcast(scal.Sinh, 'Sinh')
Tanh, tanh, TanhInplace, tanh_inplace = broadcast(scal.Tanh, 'Tanh')

Sum = s2t.Sum
sum = gof.op.constructor(Sum)

Fill, fill, FillInplace, fill_inplace = broadcast(scal.Second, 'Fill')

def ones_like(model):
    return fill(model, 1.0)
def zeros_like(model):
    return fill(model, 0.0)

TensorCopy, tensor_copy = broadcast(scal.Identity, 'TensorCopy', False)


##########################
# View Operations
##########################

class TransposeInplace(s2t.DimShuffle):

    def __init__(self, input):
        s2t.DimShuffle.__init__(self, input, range(len(input.broadcastable)-1, -1, -1), True)
    
    def perform(self):
        self.outputs[0].data = self.inputs[0].data.T
    
    def grad(self, (x,), (gz,)):
        return transpose(gz),
    
    def c_code(self, (x, ), (z, ), sub):
        return """
        PyArrayObject* transposed = (PyArrayObject*)PyArray_Transpose(%(x)s, NULL);
        if (%(z)s) {
            Py_XDECREF(%(z)s);
        }
        %(z)s = transposed;
        """ % locals()

transpose_inplace = gof.op.constructor(TransposeInplace)
def transpose(x, **kwargs):
    return transpose_inplace(tensor_copy(x), **kwargs)

class Subtensor(Op, Viewer):
    nin = 2
    nout = 1
    e_invalid = 'invalid index'
    debug = 0
    def __init__(self, *args,**kwargs):
        def as_tuple_result(obj):
            if isinstance(obj, Result):
                return obj
            r = gof.result.PythonResult(None)
            if isinstance(obj, tuple):
                r.data = obj
            else:
                r.data = (obj,)
            return r
        def pad(tplR, N):
            l = list(tplR.data)
            for i in range(len(l), N):
                l.append(slice(0,sys.maxint,1))
            tplR.data = tuple(l)

        if Subtensor.debug:
            print 'Subtensor.__init__', args, kwargs
        #Olivier says not to call this
        #Op.__init__(self,  *args,**kwargs) 
        #Viewer.__init__(self, *args,**kwargs)
        t, coord = args
        t = _as_tensor(t)
        coord = as_tuple_result(coord)
        if len(coord.data) > len(t.broadcastable):
            raise ValueError(Subtensor.e_invalid)
        # add the implicit extra unbounded slices 
        # e.g. n[0] on a 3d tensor pads to n[0,:,:]
        pad(coord, len(t.broadcastable))
        broadcastable = [0 for c in coord.data if isinstance(c, slice)]
        if Subtensor.debug:
            print 'brdcstble', broadcastable
            print 't', t.data
            print 'coord', coord.data
        self.inputs = [t, coord]
        self.outputs = [Tensor(t.dtype, broadcastable)]
    def view_map(self): 
        return {self.out: [self.inputs[0]]}
    def perform(self):
        x = self.inputs[0].data
        c = self.inputs[1].data
        if Subtensor.debug:
            print 'perform: x', x
            print 'perform: c', c
        if len(c) == 1:
            self.outputs[0].data = x.__getitem__(c[0])
        else:
            self.outputs[0].data = x.__getitem__(c)
    def grad(self, (x,), (gz,)):
        # - option: allocate a potentially large matrix of zeros, and fill in
        # the appropriate elements from gz
        # - option: return a sparse matrix
        # - option: return gz, but think about how to include a special addition
        # function that works on a corresponding view of the original data
        raise NotImplementedError() 
subtensor = gof.op.constructor(Subtensor)


##########################
# Arithmetics
##########################

Add, add, AddInplace, add_inplace = broadcast(scal.Add, 'Add')
Sub, sub, SubInplace, sub_inplace = broadcast(scal.Sub, 'Sub')
Mul, mul, MulInplace, mul_inplace = broadcast(scal.Mul, 'Mul')
Div, div, DivInplace, div_inplace = broadcast(scal.Div, 'Div')
Pow, pow, PowInplace, pow_inplace = broadcast(scal.Pow, 'Pow')


#########################
# Linalg : Dot
#########################

class Dot(_Op):
    nin=2
    nout=1
    @staticmethod 
    def broadcastable_rule(bx,by):
        if len(bx) == 0:     # x is a scalar
            rval = by
        else:
            if len(by) >= 2: #y is a matrix or tensor
                rval = bx[:-1] + by[:-2] + by[-1:]
            elif len(by)==1: #y is vector
                rval = bx[:-1]
            else:            #y is a scalar
                rval = bx
        return rval
    def propagate_broadcastable(self, bx, by):
        return [self.broadcastable_rule(bx,by)]
    def impl(self, x, y):
        return numpy.dot(x, y)
    def grad(self, (x, y), (gz,)):
        return dot(gz, y.T), dot(x.T, gz)
    if 0:
        def c_support_code(self):
            return blas.cblas_header_text()
        def c_libs(self):
            return blas.ldflags()
        def c_impl(self, (_x, _y), (_z, )):
            return blas.gemm_code('', '1.0', '0.0')
dot = gof.op.constructor(Dot)

class Gemm(_Op):
    nin=5
    nout=1
    E_rank = 'gemm only works for rank 2'
    E_scalar = 'gemm requires scalar argument'
    E_z_uniq = 'argument z aliased to x or y'
    debug = False
    def __init__(self, *args, **kwargs):
        _Op.__init__(self, *args, **kwargs)
        z, a, x, y, b = self.inputs
        zr, xr, yr = [set(gof.view_roots(i)) for i in z,x,y]
        if zr.intersection(xr):
            raise ValueError(Gemm.E_z_uniq, (z, x))
        if zr.intersection(yr):
            raise ValueError(Gemm.E_z_uniq, (z, y))
    def destroy_map(self):
        return {self.out:[self.inputs[0]]}
    def propagate_broadcastable(self, bz, ba, bx, by, bb):
        if len(bz) != 2: raise ValueError(Gemm.E_rank, len(bz))
        if len(bx) != 2: raise ValueError(Gemm.E_rank, len(bx))
        if len(by) != 2: raise ValueError(Gemm.E_rank, len(by))
        if len(ba): raise ValueError(Gemm.E_scalar, ba)
        if len(bb): raise ValueError(Gemm.E_scalar, bb)

        return [bz]
    def impl(self, z, a, x, y, b):
        assert a.shape == ()
        assert b.shape == ()
        if z.shape == ():
            z.itemset(z*a + b*numpy.dot(x,y))
            return z
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
            return z
    def grad(self, (z, a, x, y, b), (gz,)):
        raise NotImplementedError()

    def c_support_code(self):
        #return blas.cblas_header_text()
        mod_str = """
        #ifndef MOD
        #define MOD %
        #endif
        """
        return blas.blas_proto() + mod_str
    def c_headers(self):
        return ['<iostream>']
    def c_libraries(self):
        return blas.ldflags()
    def c_validate_update(self, *args):
        return ""
    def c_validate_update_cleanup(self, *args):
        return ""
    def c_code(self, (_z, _a, _x, _y, _b), (_zout, ), sub):
        return """
        int unit = 0;

        int type_num = %(_x)s->descr->type_num;
        int type_size = %(_x)s->descr->elsize; // in bytes

        npy_intp* Nx = %(_x)s->dimensions;
        npy_intp* Ny = %(_y)s->dimensions;
        npy_intp* Nz = %(_z)s->dimensions;

        npy_intp* Sx = %(_x)s->strides;
        npy_intp* Sy = %(_y)s->strides;
        npy_intp* Sz = %(_z)s->strides;

        //strides for x, y, z in dimensions 0, 1
        int sx_0, sx_1, sy_0, sy_1, sz_0, sz_1;

        if (%(_zout)s != %(_z)s)
        {
            if (%(_zout)s)
            {
                Py_DECREF(%(_zout)s);
            }
            %(_zout)s = %(_z)s;
            Py_INCREF(%(_zout)s);
        }

        if (%(_x)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(x) != 2"); %(fail)s;}
        if (%(_y)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(y) != 2"); %(fail)s;}
        if (%(_z)s->nd != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(z) != 2"); %(fail)s;}

        if ((%(_a)s->descr->type_num != PyArray_DOUBLE)
            && (%(_a)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(a) is not double or float"); %(fail)s;}

        if ((%(_b)s->descr->type_num != PyArray_DOUBLE)
            && (%(_b)s->descr->type_num != PyArray_FLOAT))
        {PyErr_SetString(PyExc_NotImplementedError, "type(b) is not double or float"); %(fail)s;}

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

        /*
        encode the stride structure of _x,_y,_z into a single integer
        */
        unit |= ((Sx[1] == type_size) ? 0x0 : (Sx[0] == type_size) ? 0x1 : 0x2) << 8;
        unit |= ((Sy[1] == type_size) ? 0x0 : (Sy[0] == type_size) ? 0x1 : 0x2) << 4;
        unit |= ((Sz[1] == type_size) ? 0x0 : (Sz[0] == type_size) ? 0x1 : 0x2) << 0;

        /* create appropriate strides for malformed matrices that are row or column
         * vectors
         */
        sx_0 = (Nx[0] > 1) ? Sx[0]/type_size : Nx[1];
        sx_1 = (Nx[1] > 1) ? Sx[1]/type_size : Nx[0];
        sy_0 = (Ny[0] > 1) ? Sy[0]/type_size : Ny[1];
        sy_1 = (Ny[1] > 1) ? Sy[1]/type_size : Ny[0];
        sz_0 = (Nz[0] > 1) ? Sz[0]/type_size : Nz[1];
        sz_1 = (Nz[1] > 1) ? Sz[1]/type_size : Nz[0];

        switch (type_num)
        {
            case PyArray_FLOAT:
            {
                #define REAL float
                float a = (%(_a)s->descr->type_num == PyArray_FLOAT) 
                ? (REAL)(((float*)%(_a)s->data)[0])
                : (REAL)(((double*)%(_a)s->data)[0]);
                float b = (%(_b)s->descr->type_num == PyArray_FLOAT) ?
                (REAL)(((float*)%(_b)s->data)[0])
                : (REAL)(((double*)%(_b)s->data)[0]);

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
                #undef REAL
            }
            break;
            case PyArray_DOUBLE:
            {
                #define REAL double

                double a = (%(_a)s->descr->type_num == PyArray_FLOAT) 
                ? (REAL)(((float*)%(_a)s->data)[0])
                : (REAL)(((double*)%(_a)s->data)[0]);
                double b = (%(_b)s->descr->type_num == PyArray_FLOAT) ?
                (REAL)(((float*)%(_b)s->data)[0])
                : (REAL)(((double*)%(_b)s->data)[0]);
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
                #undef REAL
            }
            break;
        }

        """ % dict(locals(), **sub)
gemm = gof.op.constructor(Gemm)

