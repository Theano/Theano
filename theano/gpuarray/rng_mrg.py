"""
GPU implementation of MRG31k3p random number generator for Theano.

Generator code in SSJ package (L'Ecuyer & Simard).
http://www.iro.umontreal.ca/~simardr/ssj/indexe.html

"""
from __future__ import absolute_import, print_function, division

from theano import Apply, tensor
from theano.gof import local_optimizer
from theano.sandbox.rng_mrg import mrg_uniform_base, mrg_uniform
from theano.tensor import as_tensor_variable, get_vector_length
from theano.scalar import int32 as int_t

from .basic_ops import (GpuKernelBase, Kernel, infer_context_name,
                        GpuFromHost, host_from_gpu, as_gpuarray_variable)
from .type import GpuArrayType, gpu_context_type
from .fp16_help import write_w
from .opt import register_opt, register_opt2


class GPUA_mrg_uniform(GpuKernelBase, mrg_uniform_base):
    # GpuArray version
    _f16_ok = True
    params_type = mrg_uniform_base.params_type.extended(otypecode=int_t, context=gpu_context_type)

    otypecode = property(lambda self: self.output_type.typecode)

    def make_node(self, rstate, size):
        # error checking slightly redundant here, since
        # this op should not be called directly.
        #
        # call through MRG_RandomStreams instead.
        broad = []
        for i in range(self.output_type.ndim):
                broad.append(tensor.extract_constant(size[i]) == 1)
        output_type = self.output_type.clone(broadcastable=broad)()
        rstate = as_gpuarray_variable(rstate, infer_context_name(rstate))
        return Apply(self,
                     [rstate, size],
                     [rstate.type(), output_type])

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)

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
            offset = '+ 1'
            NORM = '3.0458e-05f'  # numpy.float16(1.0/(2**15+33))
            # this was determined by finding the biggest number such that
            # numpy.float16(number * ((M1 & 0x7fff) + 1)) < 1.0
        elif self.output_type.dtype == 'float32':
            otype = 'float'
            mask = ''
            offset = ''
            NORM = '4.6566126e-10f'  # numpy.float32(1.0/(2**31+65))
            # this was determined by finding the biggest number such that
            # numpy.float32(number * M1) < 1.0
        elif self.output_type.dtype == 'float64':
            otype = 'double'
            mask = ''
            offset = ''
            NORM = '4.656612873077392578125e-10'
        else:
            raise ValueError('Unsupported data type for output',
                             self.output_type.dtype)
        code = """#include "cluda.h"

        KERNEL void mrg_uniform(
                GLOBAL_MEM %(otype)s *sample_data,
                ga_size sample_offset,
                GLOBAL_MEM ga_int *state_data,
                ga_size state_offset,
                const ga_uint Nsamples,
                const ga_uint Nstreams_used)
        {
            sample_data = (GLOBAL_MEM %(otype)s *)(((GLOBAL_MEM char *)sample_data) + sample_offset);
            state_data = (GLOBAL_MEM ga_int *)(((GLOBAL_MEM char *)state_data) + state_offset);
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
                    sample_data[i] = %(write)s((((x11 - x21 + M1) %(mask)s) %(offset)s) * %(NORM)s);
                }
                else
                {
                    sample_data[i] = %(write)s((((x11 - x21) %(mask)s) %(offset)s) * %(NORM)s);
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
                       params=[gpuarray.GpuArray, gpuarray.SIZE,
                               gpuarray.GpuArray, gpuarray.SIZE,
                               'uint32', 'uint32'],
                       flags=Kernel.get_flags(self.output_type.dtype, 'int32'))
                ]

    def c_code(self, node, nodename, inp, out, sub):
        return """
        npy_int64 M1 = 2147483647;      //2^31 - 1
        size_t n_elements = 1;
        unsigned int n_streams;
        int must_alloc_sample = ((NULL == %(o_sample)s)
                || !pygpu_GpuArray_Check((PyObject*)%(o_sample)s)
                || !(%(o_sample)s->ga.flags & GA_C_CONTIGUOUS)
                || (PyGpuArray_NDIM(%(o_sample)s) != %(params)s->ndim));

        size_t* odims = (size_t*)malloc(%(params)s->ndim * sizeof(size_t));
        if (odims == NULL) {
            PyErr_NoMemory();
            %(just_fail)s
        }

        if (PyArray_NDIM(%(size)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "size must be vector");
            %(fail)s
        }
        if (PyArray_DIMS(%(size)s)[0] != %(params)s->ndim)
        {
            PyErr_Format(PyExc_ValueError, "size must have length %%i (not %%li)",
                %(params)s->ndim, PyArray_DIMS(%(size)s)[0]);
            %(fail)s
        }

        for (int i = 0; i < %(params)s->ndim; ++i)
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
            %(o_sample)s = pygpu_empty(%(params)s->ndim, odims, %(params)s->otypecode, GA_C_ORDER,
                                       %(params)s->context, Py_None);
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
        if (%(params)s->inplace)
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

        if (n_elements > 0){
          size_t ls = 0, gs = 0;
          int err = GpuKernel_sched(&%(kname)s, n_streams, &ls, &gs);
          if (err != GA_NO_ERROR) {
              PyErr_Format(PyExc_RuntimeError, "GpuKernel_sched: %%s\\n",
                           GpuKernel_error(&%(kname)s, err));
              %(fail)s
          }
          // Make sure we run as many blocks as we need to cover the whole n_streams
          gs = (n_streams + ls - 1)/ls;
          err = mrg_uniform_call(1, &ls, &gs, 0, %(o_sample)s->ga.data, %(o_sample)s->ga.offset, %(o_rstate)s->ga.data, %(o_rstate)s->ga.offset, n_elements, n_streams);
          if (err != GA_NO_ERROR) {
              PyErr_Format(PyExc_RuntimeError, "mrg_uniform_call: %%s\\n",
                           GpuKernel_error(&%(kname)s, err));
              %(fail)s
          }
        }

        free(odims);
        """ % dict(rstate=inp[0], size=inp[1],
                   o_rstate=out[0], o_sample=out[1],
                   kname=self.gpu_kernels(node, nodename)[0].objvar,
                   params=sub['params'],
                   just_fail=sub['fail'],
                   fail="""
                   {
                     free(odims);
                     %(fail)s
                   }
                   """ % dict(fail=sub['fail']))

    def c_code_cache_version(self):
        return (17,)


@register_opt2([mrg_uniform], 'fast_compile')
def local_gpua_mrg_graph(op, context_name, inputs, outputs):
    if (type(op) == mrg_uniform and
            isinstance(inputs[0].type, GpuArrayType) and
            (inputs[0].owner is None or
             not isinstance(inputs[0].owner.op,
                            GpuFromHost))):
        outs = GPUA_mrg_uniform.new(inputs[0],
                                    op.output_type.ndim,
                                    op.output_type.dtype,
                                    inputs[1])
        return [outs[0], host_from_gpu(outs[1])]


@register_opt('fast_compile')
@local_optimizer([mrg_uniform])
def local_gpua_mrg(node):
    context_name = infer_context_name(*node.inputs)
    return local_gpua_mrg_graph(node.op, context_name, node.inputs, node.outputs)
