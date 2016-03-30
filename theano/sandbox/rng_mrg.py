"""
Implementation of MRG31k3p random number generator for Theano.

Generator code in SSJ package (L'Ecuyer & Simard).
http://www.iro.umontreal.ca/~simardr/ssj/indexe.html

"""
from __future__ import absolute_import, print_function, division
import warnings

import numpy
from six import integer_types
from six.moves import xrange

from theano import Op, Apply, shared, config, Variable
from theano import gradient, function
from theano import tensor
from theano.tensor import (TensorType, as_tensor_variable, get_vector_length,
                           cast, opt, scal)
from theano.tensor import sqrt, log, sin, cos, join, prod
from theano.compile import optdb
from theano.gof import local_optimizer
from . import multinomial

import theano.sandbox.cuda
from theano.sandbox.cuda import GpuOp
from theano.sandbox.gpuarray.basic_ops import GpuKernelBase, Kernel
from theano.sandbox.gpuarray.type import GpuArrayType
from theano.sandbox.gpuarray.fp16_help import write_w
from theano.sandbox.gpuarray.opt import (register_opt as register_gpua,
                                         host_from_gpu as host_from_gpua)
if theano.sandbox.cuda.cuda_available:
    from theano.sandbox.cuda import (CudaNdarrayType,
                                     float32_shared_constructor)


def matVecModM(A, s, m):
    # TODO : need description for method, parameter and return
    assert A.dtype == 'int64'
    return numpy.int32(numpy.sum((A * s) % m, 1) % m)


def multMatVect(v, A, m1, B, m2):
    # TODO : need description for parameter and return
    """
    Multiply the first half of v by A with a modulo of m1 and the second half
    by B with a modulo of m2.

    Notes
    -----
    The parameters of dot_modulo are passed implicitly because passing them
    explicitly takes more time than running the function's C-code.

    """
    if multMatVect.dot_modulo is None:
        A_sym = tensor.lmatrix('A')
        s_sym = tensor.ivector('s')
        m_sym = tensor.iscalar('m')
        A2_sym = tensor.lmatrix('A2')
        s2_sym = tensor.ivector('s2')
        m2_sym = tensor.iscalar('m2')
        o = DotModulo()(A_sym, s_sym, m_sym, A2_sym, s2_sym, m2_sym)
        multMatVect.dot_modulo = function(
            [A_sym, s_sym, m_sym, A2_sym, s2_sym, m2_sym], o, profile=False)

    # This way of calling the Theano fct is done to bypass Theano overhead.
    f = multMatVect.dot_modulo
    f.input_storage[0].storage[0] = A
    f.input_storage[1].storage[0] = v[:3]
    f.input_storage[2].storage[0] = m1
    f.input_storage[3].storage[0] = B
    f.input_storage[4].storage[0] = v[3:]
    f.input_storage[5].storage[0] = m2
    f.fn()
    r = f.output_storage[0].storage[0]

    return r
multMatVect.dot_modulo = None


class DotModulo(Op):
    """
    Efficient and numerically stable implementation of a dot product followed
    by a modulo operation. This performs the same function as matVecModM.

    We do this 2 times on 2 triple inputs and concatenating the output.

    """
    __props__ = ()

    def make_node(self, A, s, m, A2, s2, m2):
        return Apply(self, [A, s, m, A2, s2, m2], [s.type()])

    def perform(self, node, inputs, outputs):
        (A, s, m, A2, s2, m2) = inputs
        (out,) = outputs
        o1 = matVecModM(A, s, m)
        o2 = matVecModM(A2, s2, m2)
        out[0] = numpy.concatenate((o1, o2))

    def c_code_cache_version(self):
        return (6,)

    def c_code(self, node, name, inputs, outputs, sub):
        (_A, _s, _m, _A2, _s2, _m2) = inputs
        (_z,) = outputs
        return """
        int osize = -1;
        if (PyArray_NDIM(%(_A)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(A) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(_s)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(v) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(_m)s) != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(m) != 0"); %(fail)s;}
        if (PyArray_NDIM(%(_A2)s) != 2) {PyErr_SetString(PyExc_NotImplementedError, "rank(A2) != 2"); %(fail)s;}
        if (PyArray_NDIM(%(_s2)s) != 1) {PyErr_SetString(PyExc_NotImplementedError, "rank(v2) != 1"); %(fail)s;}
        if (PyArray_NDIM(%(_m2)s) != 0) {PyErr_SetString(PyExc_NotImplementedError, "rank(m2) != 0"); %(fail)s;}

        if( PyArray_DIMS(%(_A)s)[1] != PyArray_DIMS(%(_s)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "A and s shapes don't agree."); %(fail)s;}
        if( PyArray_DIMS(%(_A2)s)[1] != PyArray_DIMS(%(_s2)s)[0])
        {PyErr_SetString(PyExc_NotImplementedError, "A2 and s2 shapes don't agree."); %(fail)s;}

        osize = PyArray_DIMS(%(_A)s)[0] + PyArray_DIMS(%(_A2)s)[0];
        if (!%(_z)s
            || (PyArray_DIMS(%(_z)s)[0] != osize))
        {
            {Py_XDECREF(%(_z)s);}
            npy_intp dims[] = {0,};
            dims[0] = osize;
            %(_z)s = (PyArrayObject*) PyArray_SimpleNew(1, dims, PyArray_TYPE(%(_s)s));
        }

        if(!%(_z)s){%(fail)s;}

        {   //makes it compile even though labels jump over variable definitions.

            // A has size MxN, s has N, output M
            npy_intp M = PyArray_DIMS(%(_A)s)[0];
            npy_intp N = PyArray_DIMS(%(_A)s)[1];

            const dtype_%(_A)s* __restrict__ DA = (dtype_%(_A)s*)PyArray_DATA(%(_A)s);
            dtype_%(_s)s* __restrict__ Ds = (dtype_%(_s)s*)PyArray_DATA(%(_s)s);
            dtype_%(_z)s* __restrict__ Dz = (dtype_%(_z)s*)PyArray_DATA(%(_z)s);
            const dtype_%(_m)s m = ((dtype_%(_m)s*)PyArray_DATA(%(_m)s))[0];

            npy_intp SA = PyArray_STRIDES(%(_A)s)[1] / PyArray_DESCR(%(_A)s)->elsize;
            npy_intp Ss = PyArray_STRIDES(%(_s)s)[0] / PyArray_DESCR(%(_s)s)->elsize;
            npy_intp Sz = PyArray_STRIDES(%(_z)s)[0] / PyArray_DESCR(%(_z)s)->elsize;

            for (npy_int32 i = 0; i < M; ++i)
            {
                const dtype_%(_A)s* __restrict__ Ak = (dtype_%(_A)s*)(PyArray_BYTES(%(_A)s) + PyArray_STRIDES(%(_A)s)[0] * i);

                npy_int64 r = 0;

                for (npy_int32 j = 0; j < N; ++j)
                {
                    r += (npy_int64)(Ds[j * Ss] * (npy_int64)(Ak[j * SA])) %% m;
                }

                Dz[i * Sz] = r %% m;
            }
        }

        //redo it with the second triple of inputs
        {
            // A has size MxN, s has N, output M
            npy_intp M = PyArray_DIMS(%(_A2)s)[0];
            npy_intp N = PyArray_DIMS(%(_A2)s)[1];

            const dtype_%(_A2)s* __restrict__ DA = (dtype_%(_A2)s*)PyArray_DATA(%(_A2)s);
            dtype_%(_s2)s* __restrict__ Ds = (dtype_%(_s2)s*)PyArray_DATA(%(_s2)s);
            const dtype_%(_m2)s m = ((dtype_%(_m2)s*)PyArray_DATA(%(_m2)s))[0];

            npy_intp SA = PyArray_STRIDES(%(_A2)s)[1] / PyArray_DESCR(%(_A2)s)->elsize;
            npy_intp Ss = PyArray_STRIDES(%(_s2)s)[0] / PyArray_DESCR(%(_s2)s)->elsize;
            npy_intp Sz = PyArray_STRIDES(%(_z)s)[0] / PyArray_DESCR(%(_z)s)->elsize;

            dtype_%(_z)s* __restrict__ Dz = (dtype_%(_z)s*)PyArray_DATA(%(_z)s) + PyArray_DIMS(%(_A)s)[0] * Sz;

            for (npy_int32 i = 0; i < M; ++i)
            {
                const dtype_%(_A2)s* __restrict__ Ak = (dtype_%(_A2)s*)(PyArray_BYTES(%(_A2)s) + PyArray_STRIDES(%(_A2)s)[0] * i);

                npy_int64 r = 0;

                for (npy_int32 j = 0; j < N; ++j)
                {
                    r += (npy_int64)(Ds[j * Ss] * (npy_int64)(Ak[j * SA])) %% m;
                }

                Dz[i * Sz] = r %% m;
            }

        }

        """ % dict(locals(), **sub)


# MRG31k3p
# generator constants :
M1 = numpy.asarray(numpy.int32(2147483647))    # 2^31 - 1
M2 = numpy.asarray(numpy.int32(2147462579))    # 2^31 - 21069
MASK12 = numpy.int32(511)                      # 2^9 - 1
MASK13 = numpy.int32(16777215)                 # 2^24 - 1
MASK2 = numpy.int32(65535)                     # 2^16 - 1
MULT2 = numpy.int32(21069)
NORM = 4.656612873077392578125e-10  # 1./2^31

# A1p0 = numpy.asarray([[0, 4194304, 129], [1, 0, 0], [0, 1, 0]],
#                      dtype='int64')
# A2p0 = numpy.asarray([[32768, 0, 32769], [1, 0, 0], [0, 1, 0]],
#                      dtype='int64')

A1p72 = numpy.asarray([[1516919229, 758510237, 499121365],
                       [1884998244, 1516919229, 335398200],
                       [601897748, 1884998244, 358115744]],
                      dtype='int64')
A2p72 = numpy.asarray([[1228857673, 1496414766, 954677935],
                       [1133297478, 1407477216, 1496414766],
                       [2002613992, 1639496704, 1407477216]],
                      dtype='int64')

A1p134 = numpy.asarray(
    [[1702500920, 1849582496, 1656874625],
     [828554832, 1702500920, 1512419905],
     [1143731069, 828554832, 102237247]],
    dtype='int64')
A2p134 = numpy.asarray(
    [[796789021, 1464208080, 607337906],
     [1241679051, 1431130166, 1464208080],
     [1401213391, 1178684362, 1431130166]],
    dtype='int64')
np_int32_vals = [numpy.int32(i) for i in (0, 7, 9, 15, 16, 22, 24)]


def ff_2p134(rstate):
    # TODO : need description for method, parameter and return
    return multMatVect(rstate, A1p134, M1, A2p134, M2)


def ff_2p72(rstate):
    # TODO : need description for method, parameter and return
    return multMatVect(rstate, A1p72, M1, A2p72, M2)


def mrg_next_value(rstate, new_rstate):
    # TODO : need description for method, parameter and return
    x11, x12, x13, x21, x22, x23 = rstate
    assert type(x11) == numpy.int32

    i0, i7, i9, i15, i16, i22, i24 = np_int32_vals
    # first component
    y1 = (((x12 & MASK12) << i22) + (x12 >> i9) +
          ((x13 & MASK13) << i7) + (x13 >> i24))

    assert type(y1) == numpy.int32
    if (y1 < 0 or y1 >= M1):  # must also check overflow
        y1 -= M1
    y1 += x13
    if (y1 < 0 or y1 >= M1):
        y1 -= M1

    x13 = x12
    x12 = x11
    x11 = y1

    # second component
    y1 = ((x21 & MASK2) << i15) + (MULT2 * (x21 >> i16))
    assert type(y1) == numpy.int32
    if (y1 < 0 or y1 >= M2):
        y1 -= M2
    y2 = ((x23 & MASK2) << i15) + (MULT2 * (x23 >> i16))
    assert type(y2) == numpy.int32
    if (y2 < 0 or y2 >= M2):
        y2 -= M2
    y2 += x23
    if (y2 < 0 or y2 >= M2):
        y2 -= M2
    y2 += y1
    if (y2 < 0 or y2 >= M2):
        y2 -= M2

    x23 = x22
    x22 = x21
    x21 = y2

    # Must never return either 0 or M1+1
    new_rstate[...] = [x11, x12, x13, x21, x22, x23]
    assert new_rstate.dtype == numpy.int32
    if (x11 <= x21):
        return (x11 - x21 + M1) * NORM
    else:
        return (x11 - x21) * NORM


class mrg_uniform_base(Op):
    # TODO : need description for class, parameter
    __props__ = ("output_type", "inplace")

    def __init__(self, output_type, inplace=False):
        Op.__init__(self)
        self.output_type = output_type
        self.inplace = inplace
        if inplace:
            self.destroy_map = {0: [0]}
        self.warned_numpy_version = False

    def __str__(self):
        if self.inplace:
            s = "inplace"
        else:
            s = "no_inplace"
        return self.__class__.__name__ + "{%s,%s}" % (self.output_type, s)

    def make_node(self, rstate, size):
        # error checking slightly redundant here, since
        # this op should not be called directly.
        #
        # call through MRG_RandomStreams instead.
        return Apply(self,
                     [rstate, size],
                     [rstate.type(), self.output_type()])

    def grad(self, inputs, ograd):
        return [gradient.grad_undefined(self, k, inp,
                                        'No gradient defined through '
                                        'random sampling op')
                for k, inp in enumerate(inputs)]

    def R_op(self, inputs, eval_points):
        return [None for i in eval_points]


class mrg_uniform(mrg_uniform_base):
    # CPU VERSION

    @classmethod
    def new(cls, rstate, ndim, dtype, size):
        v_size = as_tensor_variable(size)
        if ndim is None:
            ndim = get_vector_length(v_size)
        op = cls(TensorType(dtype, (False,) * ndim))
        return op(rstate, v_size)

    def perform(self, node, inp, out):
        rstate, size = inp
        o_rstate, o_sample = out
        n_elements = 1
        for s in size:
            n_elements *= s
        if n_elements > M1:
            # The limit is on the C and GPU code. This perform don't
            # have this limit.  But to have all of them behave the
            # same (and have DebugMode don't use too much memory for
            # some rng_mrg tests) I also add this limit here.
            raise ValueError("rng_mrg does not support more then (2**31 -1) samples")

        rstate = numpy.asarray(rstate)  # bring state from GPU if necessary
        if not self.inplace:
            rstate = rstate.copy()

        n_streams, _ = rstate.shape

        rval = numpy.zeros(n_elements, dtype=self.output_type.dtype)

        err_orig = numpy.seterr(over='ignore')
        try:
            for i in xrange(n_elements):
                sample = mrg_next_value(rstate[i % n_streams],
                                        rstate[i % n_streams])
                rval[i] = sample
        finally:
            numpy.seterr(**err_orig)

        # send to GPU if necessary
        o_rstate[0] = node.outputs[0].type.filter(rstate)
        o_sample[0] = node.outputs[1].type.filter(rval.reshape(size))

    def c_code(self, node, name, inp, out, sub):
        rstate, size = inp
        # If we try to use the C code here with something else than a
        # TensorType, something is wrong (likely one of the GPU ops
        # not defining C code correctly).
        assert isinstance(node.inputs[0].type, TensorType)
        o_rstate, o_sample = out
        if self.inplace:
            o_rstate_requirement = (
                'NPY_ARRAY_C_CONTIGUOUS|NPY_ARRAY_ALIGNED')
        else:
            o_rstate_requirement = (
                'NPY_ARRAY_ENSURECOPY|NPY_ARRAY_C_CONTIGUOUS|'
                'NPY_ARRAY_ALIGNED')
        ndim = self.output_type.ndim
        o_type_num = numpy.asarray(0, dtype=self.output_type.dtype).dtype.num
        fail = sub['fail']
        if self.output_type.dtype == 'float32':
            otype = 'float'
            NORM = '4.6566126e-10f'  # numpy.float32(1.0/(2**31+65))
            # this was determined by finding the biggest number such that
            # numpy.float32(number * M1) < 1.0
        else:
            otype = 'double'
            NORM = '4.656612873077392578125e-10'
        return """
        //////// <code generated by mrg_uniform>
        // The +1 is to avoid odims[0] which fails on windows
        npy_int64 odims[%(ndim)s+1];
        npy_int64 n_elements = 1;
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

        for (int i = 0; i < %(ndim)s; ++i)
        {
            odims[i] = *(dtype_%(size)s *)PyArray_GETPTR1(%(size)s, i);
            n_elements *= odims[i];
            must_alloc_sample = must_alloc_sample || (PyArray_DIMS(%(o_sample)s)[i] != odims[i]);
            //fprintf(stderr, "size %%i %%i\\n", i, (int)odims[i]);
            //printf("%%li", n_elements);
        }
        //fprintf(stderr, "n_elements %%lld\\n", (long long)n_elements);
        if (n_elements > M1)
        {
            PyErr_SetString(
                PyExc_ValueError,
                "rng_mrg cpu-implementation does not support more than (2**31 -1) samples");
            %(fail)s
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
        %(o_rstate)s = (PyArrayObject*)PyArray_FromAny(
            (PyObject*)%(rstate)s,
            NULL, 0, 0, %(o_rstate_requirement)s,NULL);

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

        sample_data = (%(otype)s *) PyArray_DATA(%(o_sample)s);
        state_data = (npy_int32 *) PyArray_DATA(%(o_rstate)s);
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
        return (7, )


class GPU_mrg_uniform(mrg_uniform_base, GpuOp):
    # GPU VERSION

    @classmethod
    def new(cls, rstate, ndim, dtype, size):
        v_size = as_tensor_variable(size)
        if ndim is None:
            ndim = get_vector_length(v_size)
        op = cls(CudaNdarrayType((False,) * ndim))
        return op(rstate, v_size)

    def c_support_code_apply(self, node, nodename):
        if self.output_type.dtype == 'float32':
            otype = 'float'
            NORM = '4.6566126e-10f'  # numpy.float32(1.0/(2**31+65))
            # this was determined by finding the biggest number such that
            # numpy.float32(number * M1) < 1.0
        else:
            otype = 'double'
            NORM = '4.656612873077392578125e-10'
        return """
        // FB: I disable the printing of the warning, as we
        //receive too much email about this and this don't help
        //people. I'm not even sure if the "fix" to give the info about
        //the shape statically give a speed up. So I consider this
        //warning as useless until proved it can speed the user code.
        static int %(nodename)s_printed_warning = 1;

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
                y2 -= (y2 < 0 || y2 >= M2) ? M2 : 0;

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

        """ % locals()

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

        SYNC = "CNDA_THREAD_SYNC"
        return """
        //////// <code generated by mrg_uniform>
        npy_int64 M1 = 2147483647;      //2^31 - 1
        // The +1 is to avoid odims[0] which fails on windows
        npy_int64 odims[%(ndim)s+1];
        npy_int64 n_elements = 1;
        int n_streams, n_streams_used_in_this_call;
        int must_alloc_sample = ((NULL == %(o_sample)s)
                || !CudaNdarray_Check((PyObject*)%(o_sample)s)
                || !CudaNdarray_is_c_contiguous(%(o_sample)s)
                || (CudaNdarray_NDIM(%(o_sample)s) != %(ndim)s));

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

        for (int i = 0; i < %(ndim)s; ++i)
        {
            odims[i] = *(dtype_%(size)s *)PyArray_GETPTR1(%(size)s, i);
            n_elements *= odims[i];
            must_alloc_sample = (must_alloc_sample
                    || CudaNdarray_HOST_DIMS(%(o_sample)s)[i] != odims[i]);
        }

        if (n_elements > M1)
        {
            PyErr_SetString(
                PyExc_ValueError,
                "rng_mrg gpu implementation does not support more than (2**31 -1) samples");
            %(fail)s
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
        if (!CudaNdarray_Check((PyObject*)%(rstate)s))
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
            if (!%(o_rstate)s) {
                PyErr_SetString(PyExc_RuntimeError, "GPU_mrg_uniform: "
                                "could not copy rstate");
                %(fail)s
            }
        }

        if (CudaNdarray_NDIM(%(o_rstate)s) != 1)
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
        n_streams_used_in_this_call = std::min(n_streams, (int)n_elements);

        {
            unsigned int threads_per_block = std::min((unsigned int)n_streams_used_in_this_call, (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK);
            unsigned int n_blocks = std::min(ceil_intdiv((unsigned int)n_streams_used_in_this_call, threads_per_block), (unsigned int)NUM_VECTOR_OP_BLOCKS);

            if (n_streams > (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK * (unsigned int)NUM_VECTOR_OP_BLOCKS)
            {
                PyErr_Format(PyExc_ValueError, "On GPU, n_streams should be at most %%u",
                    (unsigned int)NUM_VECTOR_OP_THREADS_PER_BLOCK * (unsigned int)NUM_VECTOR_OP_BLOCKS);
                %(fail)s;
            }

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
        """ % locals()

    def c_code_cache_version(self):
        return (12,)


class GPUA_mrg_uniform(GpuKernelBase, mrg_uniform_base):
    # GpuArray version
    _f16_ok = True

    def get_params(self, node):
        return node.inputs[0].type.context

    @classmethod
    def new(cls, rstate, ndim, dtype, size):
        v_size = as_tensor_variable(size)
        if ndim is None:
            ndim = get_vector_length(v_size)
        op = cls(GpuArrayType(dtype, (False,) * ndim))
        return op(rstate, v_size)

    def c_headers(self):
        return super(GPUA_mrg_uniform, self).c_headers() + ['numpy_compat.h']

    def gpu_kernels(self, node, name):
        write = write_w(self.output_type.dtype)
        if self.output_type.dtype == 'float16':
            otype = 'ga_half'
            # limit the values of the state that we use.
            mask = '& 0x7fff'
            NORM = '3.0518e-05f'  # numpy.float16(1.0/(2**15+8))
            # this was determined by finding the biggest number such that
            # numpy.float16(number * (M1 & 0x7fff)) < 1.0
        elif self.output_type.dtype == 'float32':
            otype = 'float'
            mask = ''
            NORM = '4.6566126e-10f'  # numpy.float32(1.0/(2**31+65))
            # this was determined by finding the biggest number such that
            # numpy.float32(number * M1) < 1.0
        elif self.output_type.dtype == 'float64':
            otype = 'double'
            mask = ''
            NORM = '4.656612873077392578125e-10'
        else:
            raise ValueError('Unsupported data type for output',
                             self.output_type.dtype)
        code = """
        KERNEL void mrg_uniform(
                GLOBAL_MEM %(otype)s *sample_data,
                GLOBAL_MEM ga_int *state_data,
                const ga_uint Nsamples,
                const ga_uint Nstreams_used)
        {
            /*
             * The cluda backend makes sure that ga_int corresponds to
             * a 32 bit signed type on the target device.  It is not a
             * variable width type.
             */
            const ga_int i7 = 7;
            const ga_int i9 = 9;
            const ga_int i15 = 15;
            const ga_int i16 = 16;
            const ga_int i22 = 22;
            const ga_int i24 = 24;

            const ga_int M1 = 2147483647;      //2^31 - 1
            const ga_int M2 = 2147462579;      //2^31 - 21069
            const ga_int MASK12 = 511;       //2^9 - 1
            const ga_int MASK13 = 16777215;  //2^24 - 1
            const ga_int MASK2 = 65535;      //2^16 - 1
            const ga_int MULT2 = 21069;

            const ga_uint idx = GID_0 * LDIM_0 + LID_0;
            ga_int y1, y2, x11, x12, x13, x21, x22, x23;

            if (idx < Nstreams_used)
            {
            x11 = state_data[idx*6+0];
            x12 = state_data[idx*6+1];
            x13 = state_data[idx*6+2];
            x21 = state_data[idx*6+3];
            x22 = state_data[idx*6+4];
            x23 = state_data[idx*6+5];

            for (ga_uint i = idx; i < Nsamples; i += Nstreams_used)
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
                y2 -= (y2 < 0 || y2 >= M2) ? M2 : 0;

                x23 = x22;
                x22 = x21;
                x21 = y2;

                if (x11 <= x21) {
                    sample_data[i] = %(write)s(((x11 - x21 + M1) %(mask)s) * %(NORM)s);
                }
                else
                {
                    sample_data[i] = %(write)s(((x11 - x21) %(mask)s) * %(NORM)s);
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

        """ % locals()

        # we shouldn't get to this line if it's about to fail
        from pygpu import gpuarray

        return [Kernel(code=code, name="mrg_uniform",
                       params=[gpuarray.GpuArray, gpuarray.GpuArray,
                               'uint32', 'uint32'],
                       flags=Kernel.get_flags(self.output_type.dtype, 'int32'))
                ]

    def c_code(self, node, nodename, inp, out, sub):
        rstate, size = inp
        o_rstate, o_sample = out
        inplace = int(self.inplace)
        ndim = self.output_type.ndim
        o_type_num = numpy.asarray(0, dtype=self.output_type.dtype).dtype.num
        fail = sub['fail']
        ctx = sub['params']
        kname = self.gpu_kernels(node, nodename)[0].objvar
        otypecode = str(self.output_type.typecode)

        return """
        npy_int64 M1 = 2147483647;      //2^31 - 1
        // The +1 is to avoid odims[0] which fails on windows
        size_t odims[%(ndim)s+1];
        size_t n_elements = 1;
        unsigned int n_streams;
        int must_alloc_sample = ((NULL == %(o_sample)s)
                || !pygpu_GpuArray_Check((PyObject*)%(o_sample)s)
                || !(%(o_sample)s->ga.flags & GA_C_CONTIGUOUS)
                || (PyGpuArray_NDIM(%(o_sample)s) != %(ndim)s));

        if (PyArray_NDIM(%(size)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "size must be vector");
            %(fail)s
        }
        if (PyArray_DIMS(%(size)s)[0] != %(ndim)s)
        {
            PyErr_Format(PyExc_ValueError, "size must have length %%i (not %%li)",
                %(ndim)s, PyArray_DIMS(%(size)s)[0]);
            %(fail)s
        }

        for (int i = 0; i < %(ndim)s; ++i)
        {
            odims[i] = *(dtype_%(size)s *)PyArray_GETPTR1(%(size)s, i);
            n_elements *= odims[i];
            must_alloc_sample = (must_alloc_sample
                    || PyGpuArray_DIMS(%(o_sample)s)[i] != odims[i]);
        }

        if (n_elements > M1)
        {
            PyErr_SetString(
                PyExc_ValueError,
                "rng_mrg gpu implementation does not support more than (2**31 -1) samples");
            %(fail)s
        }
        if (must_alloc_sample)
        {
            Py_XDECREF(%(o_sample)s);
            %(o_sample)s = pygpu_empty(%(ndim)s, odims, %(otypecode)s, GA_C_ORDER,
                                       %(ctx)s, Py_None);
            if(!%(o_sample)s)
            {
                %(fail)s;
            }
        }
        if (!pygpu_GpuArray_Check((PyObject*)%(rstate)s))
        {
            PyErr_Format(PyExc_ValueError, "rstate must be gpuarray");
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
            %(o_rstate)s = pygpu_copy(%(rstate)s, GA_ANY_ORDER);
            if (!%(o_rstate)s) {
                %(fail)s
            }
        }

        if (PyGpuArray_NDIM(%(o_rstate)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "rstate must be a matrix");
            %(fail)s
        }
        if (PyGpuArray_DIMS(%(o_rstate)s)[1] != 6)
        {
            PyErr_Format(PyExc_ValueError, "rstate must have 6 columns");
            %(fail)s
        }
        if (%(o_rstate)s->ga.typecode != GA_INT) {
            PyErr_Format(PyExc_ValueError, "rstate must be int32");
            %(fail)s
        }
        if (!GpuArray_CHKFLAGS(&%(o_rstate)s->ga, GA_C_CONTIGUOUS)) {
            PyErr_Format(PyExc_ValueError, "rstate must be C contiguous");
            %(fail)s
        }
        n_streams = PyGpuArray_DIMS(%(o_rstate)s)[0];
        if (n_streams > n_elements)
          n_streams = n_elements;

        {
          void *args[4];
          size_t ls = 0, gs = 0;
          args[0] = %(o_sample)s->ga.data;
          args[1] = %(o_rstate)s->ga.data;
          args[2] = &n_elements;
          args[3] = &n_streams;
          int err = GpuKernel_sched(&%(kname)s, n_elements, &ls, &gs);
          if (err != GA_NO_ERROR) {
              PyErr_Format(PyExc_RuntimeError, "GpuKernel_sched: %%s\\n",
                           GpuKernel_error(&%(kname)s, err));
              %(fail)s
          }
          err = GpuKernel_call(&%(kname)s, 1, &ls, &gs, 0, args);
          if (err != GA_NO_ERROR) {
              PyErr_Format(PyExc_RuntimeError, "GpuKernel_call: %%s\\n",
                           GpuKernel_error(&%(kname)s, err));
              %(fail)s
          }
        }
        """ % locals()

    def c_code_cache_version(self):
        return (11,)


def guess_n_streams(size, warn=False):
    # TODO : need description for parameter 'size'
    """
    Return a guess at a good number of streams.

    Parameters
    ----------
    warn : bool, optional
        If True, warn when a guess cannot be made (in which case we
        return 60 * 256).

    """
    # TODO: a smart way of choosing the number of streams, see #612.
    # Note that this code was moved out of `MRG_RandomStreams` so that it can
    # be easily accessed from tests, where we want to disable the warning.
    if (isinstance(size, (tuple, list)) and
            all([isinstance(i, integer_types) for i in size])):
        # We can make a guess.
        r = 1
        for s in size:
            r *= s
        if r > 6:
            r = r // 6  # chosen as fastest for rbm_benchmark

        # The purpose of sampling from many streams is to be able to use
        # the GPU to its full capacity. It just wastes RAM and
        # stream-initialization time to allocate more streams than necessary
        # for the GPU.
        # XXX: This number is chosen to be good for 280 and 480 architectures,
        #      Better would be to use pycuda to query the number of
        #      processors on the GPU device,
        #      rather than guessing 60.
        return min(r, 60 * 256)
    else:
        if warn:
            warnings.warn(
                ("MRG_RandomStreams Can't determine #streams "
                 "from size (%s), guessing 60*256") % str(size),
                stacklevel=3)
        return 60 * 256


class MRG_RandomStreams(object):
    # TODO : need description for parameter 'use_cuda'
    """
    Module component with similar interface to numpy.random
    (numpy.random.RandomState).

    Parameters
    ----------
    seed : int or list of 6 int
        A default seed to initialize the random state.
        If a single int is given, it will be replicated 6 times.
        The first 3 values of the seed must all be less than M1 = 2147483647,
        and not all 0; and the last 3 values must all be less than
        M2 = 2147462579, and not all 0.

    """

    def updates(self):
        # TODO : need description for method and return
        return list(self.state_updates)

    def __init__(self, seed=12345, use_cuda=None):
        # A list of pairs of the form (input_r, output_r), representing the
        # update rules of all the random states generated
        # by this RandomStreams.
        self.state_updates = []

        super(MRG_RandomStreams, self).__init__()

        # Needed to reset the streams.
        self.default_instance_seed = seed

        self.set_rstate(seed)

        if use_cuda is None:
            self.use_cuda = theano.sandbox.cuda.cuda_enabled
        else:
            self.use_cuda = use_cuda

    def set_rstate(self, seed):
        # TODO : need description for method, parameter
        if isinstance(seed, integer_types):
            if seed == 0:
                raise ValueError('seed should not be 0', seed)
            elif seed >= M2:
                raise ValueError('seed should be less than %i' % M2, seed)
            self.rstate = numpy.asarray([seed] * 6, dtype='int32')
        elif len(seed) == 6:
            if seed[0] == 0 and seed[1] == 0 and seed[2] == 0:
                raise ValueError(
                    'The first 3 values of seed should not be all 0', seed)
            if seed[3] == 0 and seed[4] == 0 and seed[5] == 0:
                raise ValueError(
                    'The last 3 values of seed should not be all 0', seed)
            if seed[0] >= M1 or seed[1] >= M1 or seed[2] >= M1:
                raise ValueError(
                    'The first 3 values of seed should be less than %i' % M1,
                    seed)
            if seed[3] >= M2 or seed[4] >= M2 or seed[5] >= M2:
                raise ValueError(
                    'The last 3 values of seed should be less than %i' % M2,
                    seed)
            self.rstate = numpy.asarray(seed, dtype='int32')
        else:
            raise TypeError("seed should be 1 integer or 6 integers")

    def seed(self, seed=None):
        """
        Re-initialize each random stream.

        Parameters
        ----------
        seed : None or integer in range 0 to 2**30
            Each random stream will be assigned a unique state that depends
            deterministically on this value.

        Returns
        -------
        None

        """
        if seed is None:
            seed = self.default_instance_seed
        self.set_rstate(seed)

        for old_r, new_r, size, nstreams in self.state_updates:
            if nstreams is None:
                nstreams = self.n_streams(size)
            rstates = self.get_substream_rstates(nstreams,
                                                 new_r.owner.outputs[1].dtype)
            assert (old_r.get_value(borrow=True,
                                    return_internal_type=True).shape ==
                    rstates.shape)
            assert rstates.dtype == old_r.dtype
            old_r.set_value(rstates, borrow=True)

    def inc_rstate(self):
        """
        Update self.rstate to be skipped 2^134 steps forward to the next stream
        start.

        """
        # self.rstate = ff_2p134(self.rstate)
        self.rstate = multMatVect(self.rstate, A1p134, M1, A2p134, M2)
        assert self.rstate.dtype == numpy.int32

    def get_substream_rstates(self, n_streams, dtype, inc_rstate=True):
        # TODO : need description for parameter and return
        """
        Initialize a matrix in which each row is a MRG stream state,
        and they are spaced by 2**72 samples.

        """
        assert isinstance(dtype, str)
        assert n_streams < 2**72
        assert n_streams > 0
        rval = numpy.zeros((n_streams, 6), dtype='int32')
        rval[0] = self.rstate

        # If multMatVect.dot_modulo isn't compiled, compile it.
        if multMatVect.dot_modulo is None:
            multMatVect(rval[0], A1p72, M1, A2p72, M2)

        # This way of calling the Theano fct is done to bypass Theano overhead.
        f = multMatVect.dot_modulo
        f.input_storage[0].storage[0] = A1p72
        f.input_storage[2].storage[0] = M1
        f.input_storage[3].storage[0] = A2p72
        f.input_storage[5].storage[0] = M2
        for i in xrange(1, n_streams):
            # Inline the following call to bypass Python overhead
            # rval[i] = ff_2p72(rval[i - 1])
            v = rval[i - 1]
            f.input_storage[1].storage[0] = v[:3]
            f.input_storage[4].storage[0] = v[3:]
            f.fn()
            rval[i] = f.output_storage[0].storage[0]

        if inc_rstate:
            self.inc_rstate()
        if self.use_cuda and dtype == 'float32':
            rval = rval.flatten()
            # HACK - we use fact that int32 and float32 have same size to
            # sneak ints into the CudaNdarray type.
            # these *SHOULD NEVER BE USED AS FLOATS*
            tmp_float_buf = numpy.frombuffer(rval.data, dtype='float32')
            assert tmp_float_buf.shape == rval.shape
            assert (tmp_float_buf.view('int32') == rval).all()
            rval = tmp_float_buf

        return rval

    def n_streams(self, size):
        # TODO : need description for method, parameter and return
        return guess_n_streams(size)

    def pretty_return(self, node_rstate, new_rstate, sample, size, nstreams):
        # TODO : need description for method, parameter and return
        sample.rstate = node_rstate
        sample.update = (node_rstate, new_rstate)
        self.state_updates.append((node_rstate, new_rstate, size, nstreams))
        node_rstate.default_update = new_rstate
        return sample

    def uniform(self, size, low=0.0, high=1.0, ndim=None, dtype=None,
                nstreams=None):
        # TODO : need description for parameter 'size', 'ndim', 'nstreams'
        """
        Sample a tensor of given size whose element from a uniform
        distribution between low and high.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        Parameters
        ----------
        low
            Lower bound of the interval on which values are sampled.
            If the ``dtype`` arg is provided, ``low`` will be cast into
            dtype. This bound is excluded.
        high
            Higher bound of the interval on which values are sampled.
            If the ``dtype`` arg is provided, ``high`` will be cast into
            dtype. This bound is excluded.
        size
          Can be a list of integer or Theano variable (ex: the shape
          of other Theano Variable).
        dtype
            The output data type. If dtype is not specified, it will be
            inferred from the dtype of low and high, but will be at
            least as precise as floatX.

        """
        low = as_tensor_variable(low)
        high = as_tensor_variable(high)
        if dtype is None:
            dtype = scal.upcast(config.floatX, low.dtype, high.dtype)

        low = cast(low, dtype=dtype)
        high = cast(high, dtype=dtype)

        if isinstance(size, tuple):
            msg = "size must be a tuple of int or a Theano variable"
            assert all([isinstance(i, (numpy.integer, integer_types, Variable))
                        for i in size]), msg
            if any([isinstance(i, (numpy.integer, integer_types)) and i <= 0
                    for i in size]):
                raise ValueError(
                    "The specified size contains a dimension with value <= 0",
                    size)

        else:
            if not (isinstance(size, Variable) and size.ndim == 1):
                raise TypeError("size must be a tuple of int or a Theano "
                                "Variable with 1 dimension, got " + str(size) +
                                " of type " + str(type(size)))
        orig_nstreams = nstreams
        if nstreams is None:
            nstreams = self.n_streams(size)
        rstates = self.get_substream_rstates(nstreams, dtype)

        if self.use_cuda and dtype == 'float32':
            node_rstate = float32_shared_constructor(rstates)
            assert isinstance(node_rstate.type, CudaNdarrayType)

            # we can't use the normal mrg_uniform constructor + later
            # optimization
            # because of the tmp_float_buf hack above.  There is
            # currently no Theano node that will do a frombuffer
            # reinterpretation.
            u = self.pretty_return(node_rstate,
                                   *GPU_mrg_uniform.new(node_rstate,
                                                        ndim, dtype, size),
                                   size=size, nstreams=orig_nstreams)
        else:
            node_rstate = shared(rstates)
            u = self.pretty_return(node_rstate,
                                   *mrg_uniform.new(node_rstate,
                                                    ndim, dtype, size),
                                   size=size, nstreams=orig_nstreams)
        # Add a reference to distinguish from other shared variables
        node_rstate.tag.is_rng = True
        r = u * (high - low) + low

        if u.type.broadcastable != r.type.broadcastable:
            raise NotImplementedError(
                'Increase the size to match the broadcasting pattern of '
                '`low` and `high` arguments')

        assert r.dtype == dtype
        return r

    def binomial(self, size=None, n=1, p=0.5, ndim=None, dtype='int64',
                 nstreams=None):
        # TODO : need description for method, parameter and return
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
        # TODO : need description for parameter and return
        """
        Sample `n` (`n` needs to be >= 1, default 1) times from a multinomial
        distribution defined by probabilities pvals.

        Example : pvals = [[.98, .01, .01], [.01, .49, .50]] and n=1 will
        probably result in [[1,0,0],[0,0,1]]. When setting n=2, this
        will probably result in [[2,0,0],[0,1,1]].

        Notes
        -----
        -`size` and `ndim` are only there keep the same signature as other
        uniform, binomial, normal, etc.
        TODO : adapt multinomial to take that into account

        -Does not do any value checking on pvals, i.e. there is no
        check that the elements are non-negative, less than 1, or
        sum to 1. passing pvals = [[-2., 2.]] will result in
        sampling [[0, 0]]

        """
        if pvals is None:
            raise TypeError("You have to specify pvals")
        pvals = as_tensor_variable(pvals)
        if size is not None:
            if any([isinstance(i, integer_types) and i <= 0 for i in size]):
                raise ValueError(
                    "The specified size contains a dimension with value <= 0",
                    size)

        if size is not None:
            raise ValueError(
                "Provided a size argument to MRG_RandomStreams.multinomial, "
                "which does not use the size argument.")
        if ndim is not None:
            raise ValueError(
                "Provided an ndim argument to MRG_RandomStreams.multinomial, "
                "which does not use the ndim argument.")
        if pvals.ndim == 2:
            size = pvals[:, 0].shape * n
            unis = self.uniform(size=size, ndim=1, nstreams=nstreams)
            op = multinomial.MultinomialFromUniform(dtype)
            n_samples = as_tensor_variable(n)
            return op(pvals, unis, n_samples)
        else:
            raise NotImplementedError(("MRG_RandomStreams.multinomial only"
                                       " implemented for pvals.ndim = 2"))

    def multinomial_wo_replacement(self, size=None, n=1, pvals=None,
                                   ndim=None, dtype='int64', nstreams=None):
        # TODO : need description for parameter
        """
        Sample `n` times *WITHOUT replacement* from a multinomial distribution
        defined by probabilities pvals, and returns the indices of the sampled
        elements.
        `n` needs to be in [1, m], where m is the number of elements to select
        from, i.e. m == pvals.shape[1]. By default n = 1.

        Example : pvals = [[.98, .01, .01], [.01, .49, .50]] and n=1 will
        probably result in [[0],[2]]. When setting n=2, this
        will probably result in [[0,1],[2,1]].

        Notes
        -----
        -`size` and `ndim` are only there keep the same signature as other
        uniform, binomial, normal, etc.
        TODO : adapt multinomial to take that into account

        -Does not do any value checking on pvals, i.e. there is no
        check that the elements are non-negative, less than 1, or
        sum to 1. passing pvals = [[-2., 2.]] will result in
        sampling [[0, 0]]

        """
        if pvals is None:
            raise TypeError("You have to specify pvals")
        pvals = as_tensor_variable(pvals)

        if size is not None:
            raise ValueError("Provided a size argument to "
                             "MRG_RandomStreams.multinomial_wo_replacement, "
                             "which does not use the size argument.")
        if ndim is not None:
            raise ValueError("Provided an ndim argument to "
                             "MRG_RandomStreams.multinomial_wo_replacement, "
                             "which does not use the ndim argument.")
        if pvals.ndim == 2:
            # size = [pvals.shape[0], as_tensor_variable(n)]
            size = pvals[:, 0].shape * n
            unis = self.uniform(size=size, ndim=1, nstreams=nstreams)
            op = multinomial.MultinomialWOReplacementFromUniform(dtype)
            n_samples = as_tensor_variable(n)
            return op(pvals, unis, n_samples)
        else:
            raise NotImplementedError(
                "MRG_RandomStreams.multinomial_wo_replacement only implemented"
                " for pvals.ndim = 2")

    def normal(self, size, avg=0.0, std=1.0, ndim=None,
               dtype=None, nstreams=None):
        # TODO : need description for method
        """
        Parameters
        ----------
        size
            Can be a list of integers or Theano variables (ex: the shape
            of another Theano Variable).
        dtype
            The output data type. If dtype is not specified, it will be
            inferred from the dtype of low and high, but will be at
            least as precise as floatX.
        nstreams
            Number of streams.

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
        if (isinstance(size, tuple) and
                all([isinstance(i, (numpy.integer, integer_types)) for i in size])):
            constant = True
            # Force dtype because it defaults to float when size is empty
            n_samples = numpy.prod(size, dtype='int64')

            if n_samples % 2 == 1:
                n_samples += 1
                evened = True
        else:
            # if even, don't change, if odd, +1
            n_samples = prod(size) + (prod(size) % 2)
        flattened = self.uniform(size=(n_samples,), dtype=dtype,
                                 nstreams=nstreams)

        if constant:
            U1 = flattened[:n_samples // 2]
            U2 = flattened[n_samples // 2:]
        else:
            U1 = flattened[:prod(flattened.shape) // 2]
            U2 = flattened[prod(flattened.shape) // 2:]

        # normal_samples = zeros_like(flattened)
        sqrt_ln_U1 = sqrt(-2.0 * log(U1))
        # TypeError: 'TensorVariable' object does not support item assignment
        # so this doesn't work...
        # normal_samples[:n_samples/2] = sqrt_ln_U1 * cos(2.0*numpy.pi*U2)
        # normal_samples[n_samples/2:] = sqrt_ln_U1 * sin(2.0*numpy.pi*U2)

        # so trying this instead
        first_half = sqrt_ln_U1 * cos(
            numpy.array(2.0 * numpy.pi, dtype=dtype) * U2)
        second_half = sqrt_ln_U1 * sin(
            numpy.array(2.0 * numpy.pi, dtype=dtype) * U2)
        normal_samples = join(0, first_half, second_half)

        final_samples = None
        if evened:
            final_samples = normal_samples[:-1]
        elif constant:
            final_samples = normal_samples
        else:
            final_samples = normal_samples[:prod(size)]

        if not size:
            # Force the dtype to be int64, otherwise reshape complains
            size = tensor.constant(size, dtype='int64')
        final_samples = final_samples.reshape(size)

        final_samples = avg + std * final_samples

        assert final_samples.dtype == dtype
        return final_samples


@register_gpua('fast_compile')
@local_optimizer([mrg_uniform])
def local_gpua_mrg(node):
    # TODO : need description for function
    if (type(node.op) == mrg_uniform and
            isinstance(node.inputs[0].type, GpuArrayType)):
        outs = GPUA_mrg_uniform.new(node.inputs[0],
                                    node.op.output_type.ndim,
                                    node.op.output_type.dtype,
                                    node.inputs[1])
        return [outs[0], host_from_gpua(outs[1])]


MRG_RNGs = (mrg_uniform, GPU_mrg_uniform, GPUA_mrg_uniform)


@local_optimizer(MRG_RNGs)
def mrg_random_make_inplace(node):

    op = node.op
    if isinstance(op, MRG_RNGs) and not op.inplace:
        # op might be gpu version
        new_op = op.__class__(op.output_type, inplace=True)
        return new_op.make_node(*node.inputs).outputs
    return False
optdb.register('random_make_inplace_mrg',
               opt.in2out(mrg_random_make_inplace, ignore_newtrees=True),
               99, 'fast_run', 'inplace')
