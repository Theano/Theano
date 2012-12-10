"""
Implementation of MRG31k3p random number generator for Theano

Generator code in SSJ package (L'Ecuyer & Simard)
http://www.iro.umontreal.ca/~simardr/ssj/indexe.html

"""
import sys, warnings
import numpy

from theano import Op, Apply, shared, config, Variable
from theano.tensor import (raw_random, TensorType, as_tensor_variable,
        get_vector_length, cast, opt, scal)
from theano.tensor import zeros_like, sqrt, log, sin, cos, join, prod
from theano.compile import optdb
from theano.gof import local_optimizer
from theano.gof.python25 import all, any

import multinomial

from theano.sandbox.cuda import cuda_available, cuda_enabled, GpuOp
if cuda_available:
    from theano.sandbox.cuda import (CudaNdarrayType,
                                     float32_shared_constructor)


def matVecModM(A, s, m):
    # return (A * s) % m
    x = numpy.zeros_like(s)
    for i in xrange(len(x)):
        for j in xrange(len(s)):
            r = numpy.int32((numpy.int64(A[i][j]) * s[j] + x[i]) % m)
            if r >= 0:
                x[i] = r
            else:
                x[i] = r + m
    return x

def multMatVect(v, A, m1, B, m2):
    #multiply the first half of v by A with a modulo of m1
    #and the second half by B with a modulo of m2
    err_orig = numpy.seterr(over='ignore')
    try:
        r = numpy.zeros_like(v)
        r[:3] = matVecModM(A, v[:3], m1)
        r[3:] = matVecModM(B, v[3:], m2)
    finally:
        numpy.seterr(**err_orig)
    return r


#MRG31k3p
#generator constants :
M1 = numpy.int32(2147483647)    #2^31 - 1
M2 = numpy.int32(2147462579)    #2^31 - 21069
MASK12 = numpy.int32(511)       #2^9 - 1
MASK13 = numpy.int32(16777215)  #2^24 - 1
MASK2 = numpy.int32(65535)      #2^16 - 1
MULT2 = numpy.int32(21069)
NORM = 4.656612873077392578125e-10; #1./2^31

A1p0 = numpy.asarray([[0, 4194304, 129], [1, 0, 0], [0, 1, 0]])
A2p0 = numpy.asarray([[32768, 0, 32769], [1, 0, 0], [0, 1, 0]])

A1p72 = numpy.asarray([[1516919229, 758510237, 499121365],
       [1884998244, 1516919229, 335398200],
       [601897748, 1884998244, 358115744]])
A2p72 = numpy.asarray([[1228857673, 1496414766, 954677935],
   [1133297478, 1407477216, 1496414766],
   [2002613992, 1639496704, 1407477216]])

A1p134 = numpy.asarray(
  [[1702500920, 1849582496, 1656874625],
   [828554832, 1702500920, 1512419905],
   [1143731069, 828554832, 102237247]])
A2p134 = numpy.asarray(
  [[796789021, 1464208080, 607337906],
   [1241679051, 1431130166, 1464208080],
   [1401213391, 1178684362, 1431130166]])
np_int32_vals = [numpy.int32(i) for i in (0, 7, 9, 15, 16, 22, 24)]

def ff_2p134(rstate):
    return multMatVect(rstate, A1p134, M1, A2p134, M2)

def ff_2p72(rstate):
    return multMatVect(rstate, A1p72, M1, A2p72, M2)


def mrg_next_value(rstate, new_rstate):
    x11, x12, x13, x21, x22, x23 = rstate
    assert type(x11) == numpy.int32

    #i0, i7, i9, i15, i16, i22, i24 = [numpy.int32(i) for i in (0, 7, 9, 15, 16, 22, 24)]
    i0, i7, i9, i15, i16, i22, i24 = np_int32_vals
    #first component
    y1 = (((x12 & MASK12) << i22) + (x12 >> i9)
        + ((x13 & MASK13) << i7) + (x13 >> i24))

    assert type(y1) == numpy.int32
    if (y1 < 0 or y1 >= M1):     #must also check overflow
        y1 -= M1;
    y1 += x13;
    if (y1 < 0 or y1 >= M1):
        y1 -= M1;

    x13 = x12;
    x12 = x11;
    x11 = y1;

    #second component
    y1 = ((x21 & MASK2) << i15) + (MULT2 * (x21 >> i16));
    assert type(y1) == numpy.int32
    if (y1 < 0 or y1 >= M2):
        y1 -= M2;
    y2 = ((x23 & MASK2) << i15) + (MULT2 * (x23 >> i16));
    assert type(y2) == numpy.int32
    if (y2 < 0 or y2 >= M2):
        y2 -= M2;
    y2 += x23;
    if (y2 < 0 or y2 >= M2):
        y2 -= M2;
    y2 += y1;
    if (y2 < 0 or y2 >= M2):
        y2 -= M2;

    x23 = x22;
    x22 = x21;
    x21 = y2;

    # Must never return either 0 or M1+1
    new_rstate[...] = [x11, x12, x13, x21, x22, x23]
    assert new_rstate.dtype == numpy.int32
    if (x11 <= x21):
        return (x11 - x21 + M1) * NORM
    else:
        return (x11 - x21) * NORM

class mrg_uniform_base(Op):
    def __init__(self, output_type, inplace=False):
        Op.__init__(self)
        self.output_type = output_type
        self.inplace=inplace
        if inplace:
            self.destroy_map = {0:[0]}
        self.warned_numpy_version = False

    def __eq__(self, other):
        return type(self) == type(other) \
                and self.output_type == other.output_type \
                and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ hash(self.output_type) ^ hash(self.inplace)
    def __str__(self):
        if self.inplace:
            s = "inplace"
        else: s = "no_inplace"
        return self.__class__.__name__+"{%s,%s}"%(self.output_type,s)

    def make_node(self, rstate, size):
        # error checking slightly redundant here, since
        # this op should not be called directly.
        #
        # call through MRG_RandomStreams instead.
        return Apply(self,
                [rstate, size],
                [rstate.type(), self.output_type()])

    def grad(self,inputs,ograd):
        return [None for i in inputs]

    def R_op(self, inputs, eval_points):
        return [None for i in eval_points]


class mrg_uniform(mrg_uniform_base):
    #CPU VERSION

    @classmethod
    def new(cls, rstate, ndim, dtype, size):
        v_size = as_tensor_variable(size)
        if ndim is None:
            ndim = get_vector_length(v_size)
        op = cls(TensorType(dtype, (False,)*ndim))
        return op(rstate, cast(v_size, 'int32'))

    def perform(self, node, inp, out):
        rstate, size = inp
        o_rstate, o_sample = out
        numpy_version=numpy.__version__.split('.')
        if not self.warned_numpy_version and int(numpy_version[0])<=1 and int(numpy_version[1])<3:
            print "Warning: you must use numpy version 1.3.0 or higher with the python version of this op. Otherwise numpy leak memory. and numpy"
            self.warned_numpy_version = True

        n_elements = 1

        rstate = numpy.asarray(rstate) # bring state from GPU if necessary
        if not self.inplace:
            rstate = rstate.copy()

        for s in size:
            n_elements *= s

        n_streams,_ = rstate.shape

        rval = numpy.zeros(n_elements, dtype=self.output_type.dtype)

        err_orig = numpy.seterr(over='ignore')
        try:
            for i in xrange(n_elements):
                sample = mrg_next_value(rstate[i%n_streams], rstate[i%n_streams])
                rval[i] = sample
        finally:
            numpy.seterr(**err_orig)

        o_rstate[0] = node.outputs[0].type.filter(rstate) # send to GPU if necessary
        o_sample[0] = node.outputs[1].type.filter(rval.reshape(size))# send to GPU if necessary

    def c_code(self, node, name, inp, out, sub):
        rstate, size = inp
        o_rstate, o_sample = out
        if self.inplace:
            o_rstate_requirement = 'NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED'
        else:
            o_rstate_requirement = 'NPY_ARRAY_ENSURECOPY|NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED'
        ndim = self.output_type.ndim
        o_type_num = numpy.asarray(0, dtype=self.output_type.dtype).dtype.num
        fail = sub['fail']
        if self.output_type.dtype == 'float32':
            otype = 'float'
            NORM = '4.6566126e-10f' #numpy.float32(1.0/(2**31+65))
            # this was determined by finding the biggest number such that
            # numpy.float32(number * M1) < 1.0
        else:
            otype = 'double'
            NORM = '4.656612873077392578125e-10'
        return """
        //////// <code generated by mrg_uniform>

        npy_intp odims[%(ndim)s];
        int n_elements = 1;
        int n_streams = 0;
        int must_alloc_sample = ((NULL == %(o_sample)s)
                                 || (PyArray_NDIM(%(o_sample)s) != %(ndim)s)
                                 || !(PyArray_ISCONTIGUOUS(%(o_sample)s)));
        %(otype)s * sample_data;
        npy_int32 * state_data;

        const npy_int32 i0 = 0;
        const npy_int32 i7 = 7;
        const npy_int32 i9 = 9;
        const npy_int32 i15 = 15;
        const npy_int32 i16 = 16;
        const npy_int32 i22 = 22;
        const npy_int32 i24 = 24;

        const npy_int32 M1 = 2147483647;      //2^31 - 1
        const npy_int32 M2 = 2147462579;      //2^31 - 21069
        const npy_int32 MASK12 = 511;       //2^9 - 1
        const npy_int32 MASK13 = 16777215;  //2^24 - 1
        const npy_int32 MASK2 = 65535;      //2^16 - 1
        const npy_int32 MULT2 = 21069;

        if (PyArray_NDIM(%(size)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "size must be vector");
            %(fail)s
        }
        if (PyArray_DIMS(%(size)s)[0] != %(ndim)s)
        {
            PyErr_Format(PyExc_ValueError, "size must have length %%i (not %%i)",
                %(ndim)s, int(PyArray_DIMS(%(size)s)[0]));
            %(fail)s
        }
        if (PyArray_DESCR(%(size)s)->type_num != NPY_INT32)
        {
            PyErr_SetString(PyExc_ValueError, "size must be int32");
            %(fail)s
        }
        for (int i = 0; i < %(ndim)s; ++i)
        {
            odims[i] = ((npy_int32*)(%(size)s->data + %(size)s->strides[0] * i))[0];
            n_elements *= odims[i];
            must_alloc_sample = must_alloc_sample || (PyArray_DIMS(%(o_sample)s)[i] != odims[i]);
            //fprintf(stderr, "size %%i %%i\\n", i, (int)odims[i]);
        }
        if (must_alloc_sample)
        {
            Py_XDECREF(%(o_sample)s);
            %(o_sample)s = (PyArrayObject*)PyArray_SimpleNew(%(ndim)s, odims, %(o_type_num)s);
            if(!%(o_sample)s) {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc mrg_uniform output");
                %(fail)s
            }
        }
        Py_XDECREF(%(o_rstate)s);
        %(o_rstate)s = (PyArrayObject*)PyArray_FromAny(py_%(rstate)s, NULL, 0, 0, %(o_rstate_requirement)s,NULL);

        if (PyArray_NDIM(%(o_rstate)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "rstate must be matrix");
            %(fail)s
        }
        if (PyArray_DIMS(%(o_rstate)s)[1] != 6)
        {
            PyErr_Format(PyExc_ValueError, "rstate must have 6 columns");
            %(fail)s
        }
        if (PyArray_DESCR(%(o_rstate)s)->type_num != NPY_INT32)
        {
            PyErr_SetString(PyExc_ValueError, "rstate must be int32");
            %(fail)s
        }
        n_streams = PyArray_DIMS(%(o_rstate)s)[0];

        sample_data = (%(otype)s *) %(o_sample)s->data;
        state_data = (npy_int32 *) %(o_rstate)s->data;
        for (int i = 0; i < n_elements; ++i)
        {
            npy_int32 * state_data_i = state_data + (i%%n_streams)*6;
            npy_int32 y1, y2, x11, x12, x13, x21, x22, x23;

            x11 = state_data_i[0];
            x12 = state_data_i[1];
            x13 = state_data_i[2];
            x21 = state_data_i[3];
            x22 = state_data_i[4];
            x23 = state_data_i[5];

            y1 = ((x12 & MASK12) << i22) + (x12 >> i9) + ((x13 & MASK13) << i7) + (x13 >> i24);
            if ((y1 < 0 || y1 >= M1))     //must also check overflow
                y1 -= M1;
            y1 += x13;
            if ((y1 < 0 or y1 >= M1))
                y1 -= M1;
            x13 = x12;
            x12 = x11;
            x11 = y1;

            y1 = ((x21 & MASK2) << i15) + (MULT2 * (x21 >> i16));
            if (y1 < 0 || y1 >= M2)
                y1 -= M2;
            y2 = ((x23 & MASK2) << i15) + (MULT2 * (x23 >> i16));
            if (y2 < 0 || y2 >= M2)
                y2 -= M2;
            y2 += x23;
            if (y2 < 0 || y2 >= M2)
                y2 -= M2;
            y2 += y1;
            if (y2 < 0 or y2 >= M2)
                y2 -= M2;

            x23 = x22;
            x22 = x21;
            x21 = y2;

            if (x11 <= x21) {
                assert((x11 - x21 + M1) <= M1);
                sample_data[i] = (x11 - x21 + M1) * %(NORM)s;
            }
            else
            {
                assert(x11 - x21 <= M1);
                sample_data[i] = (x11 - x21) * %(NORM)s;
            }

            state_data_i[0]= x11;
            state_data_i[1]= x12;
            state_data_i[2]= x13;
            state_data_i[3]= x21;
            state_data_i[4]= x22;
            state_data_i[5]= x23;
        }
        //////// </ code generated by mrg_uniform>
        """ % locals()

    def c_code_cache_version(self):
        return (2,)


class GPU_mrg_uniform(mrg_uniform_base, GpuOp):
    #GPU VERSION

    @classmethod
    def new(cls, rstate, ndim, dtype, size):
        v_size = as_tensor_variable(size)
        if ndim is None:
            ndim = get_vector_length(v_size)
        op = cls(CudaNdarrayType((False,)*ndim))
        return op(rstate, cast(v_size, 'int32'))

    def c_support_code_apply(self, node, nodename):
        if self.output_type.dtype == 'float32':
            otype = 'float'
            NORM = '4.6566126e-10f' #numpy.float32(1.0/(2**31+65))
            # this was determined by finding the biggest number such that
            # numpy.float32(number * M1) < 1.0
        else:
            otype = 'double'
            NORM = '4.656612873077392578125e-10'
        return """
        static int %(nodename)s_printed_warning = 0;

        static __global__ void %(nodename)s_mrg_uniform(
                %(otype)s*sample_data,
                npy_int32*state_data,
                const int Nsamples,
                const int Nstreams_used)
        {
            const npy_int32 i0 = 0;
            const npy_int32 i7 = 7;
            const npy_int32 i9 = 9;
            const npy_int32 i15 = 15;
            const npy_int32 i16 = 16;
            const npy_int32 i22 = 22;
            const npy_int32 i24 = 24;

            const npy_int32 M1 = 2147483647;      //2^31 - 1
            const npy_int32 M2 = 2147462579;      //2^31 - 21069
            const npy_int32 MASK12 = 511;       //2^9 - 1
            const npy_int32 MASK13 = 16777215;  //2^24 - 1
            const npy_int32 MASK2 = 65535;      //2^16 - 1
            const npy_int32 MULT2 = 21069;

            const unsigned int numThreads = blockDim.x * gridDim.x;
            const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            npy_int32 y1, y2, x11, x12, x13, x21, x22, x23;

            if (idx < Nstreams_used)
            {
            x11 = state_data[idx*6+0];
            x12 = state_data[idx*6+1];
            x13 = state_data[idx*6+2];
            x21 = state_data[idx*6+3];
            x22 = state_data[idx*6+4];
            x23 = state_data[idx*6+5];

            for (int i = idx; i < Nsamples; i += Nstreams_used)
            {
                y1 = ((x12 & MASK12) << i22) + (x12 >> i9) + ((x13 & MASK13) << i7) + (x13 >> i24);
                y1 -= (y1 < 0 || y1 >= M1) ? M1 : 0;
                y1 += x13;
                y1 -= (y1 < 0 || y1 >= M1) ? M1 : 0;
                x13 = x12;
                x12 = x11;
                x11 = y1;

                y1 = ((x21 & MASK2) << i15) + (MULT2 * (x21 >> i16));
                y1 -= (y1 < 0 || y1 >= M2) ? M2 : 0;
                y2 = ((x23 & MASK2) << i15) + (MULT2 * (x23 >> i16));
                y2 -= (y2 < 0 || y2 >= M2) ? M2 : 0;
                y2 += x23;
                y2 -= (y2 < 0 || y2 >= M2) ? M2 : 0;
                y2 += y1;
                y2 -= (y2 < 0 or y2 >= M2) ? M2 : 0;

                x23 = x22;
                x22 = x21;
                x21 = y2;

                if (x11 <= x21) {
                    sample_data[i] = (x11 - x21 + M1) * %(NORM)s;
                }
                else
                {
                    sample_data[i] = (x11 - x21) * %(NORM)s;
                }
            }

            state_data[idx*6+0]= x11;
            state_data[idx*6+1]= x12;
            state_data[idx*6+2]= x13;
            state_data[idx*6+3]= x21;
            state_data[idx*6+4]= x22;
            state_data[idx*6+5]= x23;
            }
        }

        """ %locals()

    def c_code(self, node, nodename, inp, out, sub):
        rstate, size = inp
        o_rstate, o_sample = out
        inplace = int(self.inplace)
        ndim = self.output_type.ndim
        o_type_num = numpy.asarray(0, dtype=self.output_type.dtype).dtype.num
        fail = sub['fail']

        if self.output_type.dtype == 'float32':
            otype = 'float'
        else:
            otype = 'double'

        SYNC="CNDA_THREAD_SYNC";
        return """
        //////// <code generated by mrg_uniform>

        int odims[%(ndim)s];
        int n_elements = 1;
        int n_streams, n_streams_used_in_this_call;
        int must_alloc_sample = ((NULL == %(o_sample)s)
                || !CudaNdarray_Check(py_%(o_sample)s)
                || !CudaNdarray_is_c_contiguous(%(o_sample)s)
                || (PyArray_NDIM(%(o_sample)s) != %(ndim)s));

        if (PyArray_NDIM(%(size)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "size must be vector");
            %(fail)s
        }
        if (PyArray_DIMS(%(size)s)[0] != %(ndim)s)
        {
            PyErr_Format(PyExc_ValueError, "size must have length %%i (not %%i)",
                %(ndim)s, PyArray_DIMS(%(size)s)[0]);
            %(fail)s
        }
        if (PyArray_DESCR(%(size)s)->type_num != NPY_INT32)
        {
            PyErr_SetString(PyExc_ValueError, "size must be int32");
            %(fail)s
        }
        for (int i = 0; i < %(ndim)s; ++i)
        {
            odims[i] = ((npy_int32*)(%(size)s->data + %(size)s->strides[0] * i))[0];
            n_elements *= odims[i];
            must_alloc_sample = (must_alloc_sample
                    || CudaNdarray_HOST_DIMS(%(o_sample)s)[i] != odims[i]);
        }
        if (must_alloc_sample)
        {
            Py_XDECREF(%(o_sample)s);
            %(o_sample)s = (CudaNdarray*)CudaNdarray_NewDims(%(ndim)s, odims);
            if(!%(o_sample)s)
            {
                %(fail)s;
            }
        }
        if (!CudaNdarray_Check(py_%(rstate)s))
        {
            PyErr_Format(PyExc_ValueError, "rstate must be cudandarray");
            %(fail)s;
        }

        Py_XDECREF(%(o_rstate)s);
        if (%(inplace)s)
        {
            Py_INCREF(%(rstate)s);
            %(o_rstate)s = %(rstate)s;
        }
        else
        {
            %(o_rstate)s = (CudaNdarray*)CudaNdarray_Copy(%(rstate)s);
        }

        if (PyArray_NDIM(%(o_rstate)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "rstate must be vector");
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(%(o_rstate)s)[0] %% 6)
        {
            PyErr_Format(PyExc_ValueError, "rstate len must be multiple of 6");
            %(fail)s;
        }
        n_streams = CudaNdarray_HOST_DIMS(%(o_rstate)s)[0]/6;
        n_streams_used_in_this_call = std::min(n_streams, n_elements);

        {
            unsigned int threads_per_block = std::min((unsigned int)n_streams_used_in_this_call, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
            unsigned int n_blocks = std::min(ceil_intdiv((unsigned int)n_streams_used_in_this_call, threads_per_block), (unsigned int)NUM_VECTOR_OP_BLOCKS);

            if (threads_per_block * n_blocks < n_streams)
            {
                if (! %(nodename)s_printed_warning)
                  fprintf(stderr, "WARNING: unused streams above %%i (Tune GPU_mrg get_n_streams)\\n", threads_per_block * n_blocks );
                %(nodename)s_printed_warning = 1;
            }
            %(nodename)s_mrg_uniform<<<n_blocks,threads_per_block>>>(
                CudaNdarray_DEV_DATA(%(o_sample)s),
                (npy_int32*)CudaNdarray_DEV_DATA(%(o_rstate)s),
                n_elements, n_streams_used_in_this_call);
        }

        %(SYNC)s;

        {
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError, "Cuda error: %%s: %%s.\\n", "mrg_uniform", cudaGetErrorString(err));
                %(fail)s;
            }
        }

        //////// </ code generated by mrg_uniform>
        """ %locals()
    def c_code_cache_version(self):
        return (6,)


def guess_n_streams(size, warn=True):
    """
    Return a guess at a good number of streams.

    :param warn: If True, warn when a guess cannot be made (in which case
    we return 60 * 256).
    """
    # TODO: a smart way of choosing the number of streams, see #612.
    # Note that this code was moved out of `MRG_RandomStreams` so that it can
    # be easily accessed from tests, where we want to disable the warning.
    if (isinstance(size, (tuple, list)) and
        all([isinstance(i, int) for i in size])):
        # We can make a guess.
        r = 1
        for s in size:
            r *= s
        if r > 6:
            r = r/6 # chosen as fastest for rbm_benchmark

        # The purpose of sampling from many streams is to be able to use
        # the GPU to its full capacity.  It just wastes RAM and stream-initialization time to
        # allocate more streams than necessary for the GPU.
        # XXX: This number is chosen to be good for 280 and 480 architectures,
        #      Better would be to use pycuda to query the number of
        #      processors on the GPU device,
        #      rather than guessing 60.
        return min(r, 60 * 256)
    else:
        if warn:
            warnings.warn((
                    "MRG_RandomStreams Can't determine #streams from "
                    "size (%s), guessing 60*256") % str(size),
                    stacklevel=3)
        return 60 * 256


class MRG_RandomStreams(object):
    """Module component with similar interface to numpy.random (numpy.random.RandomState)"""

    def updates(self):
        return list(self.state_updates)

    def __init__(self, seed=12345, use_cuda=None):
        """
        :type seed: int or list of 6 int.

        :param seed: a default seed to initialize the random state.
            If a single int is given, it will be replicated 6 times.
            The first 3 values of the seed must all be less than M1 = 2147483647,
            and not all 0; and the last 3 values must all be less than
            M2 = 2147462579, and not all 0.

        """
        # A list of pairs of the form (input_r, output_r), representing the
        # update rules of all the random states generated by this RandomStreams.
        self.state_updates = []

        super(MRG_RandomStreams, self).__init__()
        if isinstance(seed, int):
            if seed == 0:
                raise ValueError('seed should not be 0', seed)
            elif seed >= M2:
                raise ValueError('seed should be less than %i' % M2, seed)
            self.rstate = numpy.asarray([seed]*6, dtype='int32')
        elif len(seed)==6:
            if seed[0] == 0 and seed[1] == 0 and seed[2] == 0:
                raise ValueError('The first 3 values of seed should not be all 0', seed)
            if seed[3] == 0 and seed[4] == 0 and seed[5] == 0:
                raise ValueError('The last 3 values of seed should not be all 0', seed)
            if seed[0] >= M1 or seed[1] >= M1 or seed[2] >= M1:
                raise ValueError('The first 3 values of seed should be less than %i' % M1, seed)
            if seed[3] >= M2 or seed[4] >= M2 or seed[5] >= M2:
                raise ValueError('The last 3 values of seed should be less than %i' % M2, seed)
            self.rstate = numpy.asarray(seed, dtype='int32')
        else:
            raise TypeError("seed should be 1 integer or 6 integers")
        if use_cuda is None:
            self.use_cuda = cuda_enabled
        else:
            self.use_cuda = use_cuda

    def inc_rstate(self):
        """Update self.rstate to be skipped 2^134 steps forward to the next stream start"""
        self.rstate = ff_2p134(self.rstate)
        assert self.rstate.dtype == numpy.int32

    def get_substream_rstates(self, n_streams, inc_rstate=True):
        """Initialize a matrix in which each row is a MRG stream state,
        and they are spaced by 2**72 samples.
        """
        assert n_streams < 2**72
        assert n_streams > 0
        rval = numpy.zeros((n_streams,6), dtype='int32')
        rval[0] = self.rstate
        for i in xrange(1, n_streams):
            rval[i] = ff_2p72(rval[i-1])
        if inc_rstate:
            self.inc_rstate()
        return rval

    def n_streams(self, size):
        return guess_n_streams(size, warn=True)

    def pretty_return(self, node_rstate, new_rstate, sample):
        sample.rstate = node_rstate
        sample.update = (node_rstate, new_rstate)
        self.state_updates.append((node_rstate, new_rstate))
        node_rstate.default_update = new_rstate
        return sample

    def uniform(self, size, low=0.0, high=1.0, ndim=None, dtype=None,
                nstreams=None):
        """
        Sample a tensor of given size whose element from a uniform
        distribution between low and high.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing
        information.

        :param low: Lower bound of the interval on which values are sampled.
        If the ``dtype`` arg is provided, ``low`` will be cast into dtype.

        :param high: Higher bound of the interval on which values are sampled.
        If the ``dtype`` arg is provided, ``high`` will be cast into dtype.

        :param size: Can be a list of integer or Theano variable
                (ex: the shape of other Theano Variable)

        :param dtype: The output data type. If dtype is not specified, it will
        be inferred from the dtype of low and high, but will be at least as
        precise as floatX.
        """
        low = as_tensor_variable(low)
        high = as_tensor_variable(high)
        if dtype is None:
            dtype = scal.upcast(config.floatX, low.dtype, high.dtype)

        low = cast(low, dtype=dtype)
        high = cast(high, dtype=dtype)

        if isinstance(size, tuple):
            msg = "size must be a tuple of int or a Theano variable"
            assert all([isinstance(i,int) or isinstance(i,Variable)
                for i in size]), msg
            if any([isinstance(i, int) and i <= 0 for i in size]):
                raise ValueError(
                    "The specified size contains a dimension with value <= 0",
                    size)

        else:
            msg = "size must be a tuple of int or a Theano variable"
            assert isinstance(size, Variable) and size.ndim==1, msg

        if nstreams is None:
            nstreams = self.n_streams(size)

        if self.use_cuda and dtype=='float32':
            rstates = self.get_substream_rstates(nstreams)
            rstates = rstates.flatten()
            # HACK - we use fact that int32 and float32 have same size to
            # sneak ints into the CudaNdarray type.
            # these *SHOULD NEVER BE USED AS FLOATS*
            tmp_float_buf = numpy.frombuffer(rstates.data, dtype='float32')
            assert tmp_float_buf.shape == rstates.shape
            assert tmp_float_buf.data[:24] == rstates.data[:24]
            # transfer to device
            node_rstate = float32_shared_constructor(tmp_float_buf)
            assert isinstance(node_rstate.type, CudaNdarrayType)

            # we can't use the normal mrg_uniform constructor + later
            # optimization
            # because of the tmp_float_buf hack above.  There is
            # currently no Theano node that will do a frombuffer
            # reinterpretation.
            u = self.pretty_return(node_rstate,
                    *GPU_mrg_uniform.new(node_rstate, ndim, dtype, size))
        else:
            node_rstate = shared(self.get_substream_rstates(nstreams))
            u = self.pretty_return(node_rstate,
                    *mrg_uniform.new(node_rstate, ndim, dtype, size))
        r = u * (high-low) + low

        if u.type.broadcastable != r.type.broadcastable:
            raise NotImplementedError( 'Increase the size to match the broadcasting pattern of `low` and `high` arguments')

        assert r.dtype == dtype
        return  r

    def binomial(self, size=None, n=1, p=0.5, ndim=None, dtype='int64',
                 nstreams=None):
        if n == 1:
            if dtype == 'float32' and self.use_cuda:
                x = self.uniform(size=size, dtype=dtype, nstreams=nstreams)
            else:
                x = self.uniform(size=size, nstreams=nstreams)
            return cast(x < p, dtype)
        else:
            raise NotImplementedError("MRG_RandomStreams.binomial with n > 1")

    def multinomial(self, size=None, n=1, pvals=None, ndim=None, dtype='int64',
                    nstreams=None):
        """
        Sample `n` (currently `n` needs to be 1) times from a multinomial
        distribution defined by probabilities pvals.

        Example : pvals = [[.98, .01, .01], [.01, .98, .01]] will
        probably result in [[1,0,0],[0,1,0]].

        .. note::
            -`size` and `ndim` are only there keep the same signature as other
            uniform, binomial, normal, etc.
            todo : adapt multinomial to take that into account

            -Does not do any value checking on pvals, i.e. there is no
             check that the elements are non-negative, less than 1, or
             sum to 1. passing pvals = [[-2., 2.]] will result in
             sampling [[0, 0]]
        """
        if pvals is None:
            raise TypeError("You have to specify pvals")
        pvals = as_tensor_variable(pvals)
        if size is not None:
            if any([isinstance(i, int) and i <= 0 for i in size]):
                raise ValueError(
                    "The specified size contains a dimension with value <= 0",
                    size)

        if n == 1 and pvals.ndim == 2:
            if size is not None:
                raise ValueError("Provided a size argument to "
                        "MRG_RandomStreams.multinomial, which does not use "
                        "the size argument.")
            if ndim is not None:
                raise ValueError("Provided an ndim argument to "
                        "MRG_RandomStreams.multinomial, which does not use "
                        "the ndim argument.")
            ndim, size, bcast = raw_random._infer_ndim_bcast(
                    ndim, size, pvals[:,0])
            assert ndim==1
            bcast = bcast+(pvals.type.broadcastable[-1],)
            unis = self.uniform(size=size, ndim=1, nstreams=nstreams)
            op = multinomial.MultinomialFromUniform(dtype)
            return op(pvals, unis)
        else:
            raise NotImplementedError(("MRG_RandomStreams.multinomial only"
                " implemented with n == 1 and pvals.ndim = 2"))

    def normal(self, size=None, avg=0.0, std=1.0, ndim=None,
               dtype=None, nstreams=None):
        """
        :param size: Can be a list of integers or Theano variables (ex: the
        shape of another Theano Variable)

        :param dtype: The output data type. If dtype is not specified, it will
        be inferred from the dtype of low and high, but will be at least as
        precise as floatX.

        :param nstreams: Number of streams.
        """
        # We need an even number of ]0,1[ samples. Then we split them
        # in two halves. First half becomes our U1's for Box-Muller,
        # second half our U2's. See Wikipedia page:
        # http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
        avg = as_tensor_variable(avg)
        std = as_tensor_variable(std)

        if dtype is None:
            dtype = scal.upcast(config.floatX, avg.dtype, std.dtype)

        avg = cast(avg, dtype)
        std = cast(std, dtype)

        evened = False
        constant = False
        if isinstance(size, tuple) and all([isinstance(i,int) for i in size]):
            constant = True
            n_samples = numpy.prod(size)

            if n_samples % 2 == 1:
                n_samples += 1
                evened = True
        else:
            #if even, don't change, if odd, +1
            n_samples = prod(size)+(prod(size)%2)
        flattened = self.uniform(size=(n_samples,), dtype=dtype,
                                 nstreams=nstreams)

        if constant:
            U1 = flattened[:n_samples // 2]
            U2 = flattened[n_samples // 2:]
        else:
            U1 = flattened[:prod(flattened.shape) // 2]
            U2 = flattened[prod(flattened.shape) // 2:]

        #normal_samples = zeros_like(flattened)
        sqrt_ln_U1 = sqrt(-2.0 * log(U1))
        # TypeError: 'TensorVariable' object does not support item assignment
        # so this doesn't work...
        #normal_samples[:n_samples/2] = sqrt_ln_U1 * cos(2.0*numpy.pi*U2)
        #normal_samples[n_samples/2:] = sqrt_ln_U1 * sin(2.0*numpy.pi*U2)

        # so trying this instead
        first_half = sqrt_ln_U1 * cos(numpy.array(2.0 * numpy.pi, dtype=dtype) * U2)
        second_half = sqrt_ln_U1 * sin(numpy.array(2.0 * numpy.pi, dtype=dtype)*U2)
        normal_samples = join(0, first_half, second_half)

        final_samples = None
        if evened:
            final_samples = normal_samples[:-1]
        elif constant:
            final_samples = normal_samples
        else:
            final_samples = normal_samples[:prod(size)]

        if size:
            final_samples = final_samples.reshape(size)

        final_samples = avg + std * final_samples

        assert final_samples.dtype == dtype
        return final_samples

@local_optimizer([None])
def mrg_random_make_inplace(node):
    op = node.op
    if isinstance(op, mrg_uniform) and not op.inplace:
        # op might be gpu version
        new_op = op.__class__(op.output_type, inplace=True)
        return new_op.make_node(*node.inputs).outputs
    return False
optdb.register('random_make_inplace_mrg', opt.in2out(mrg_random_make_inplace, ignore_newtrees=True), 99, 'fast_run', 'inplace')
