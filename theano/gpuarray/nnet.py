from __future__ import absolute_import, print_function, division
import numpy as np

from theano import Op, Apply
from six import StringIO

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel, gpuarray_helper_inc_dir,
                        infer_context_name)
from .type import GpuArrayType
from .fp16_help import work_dtype, load_w, write_w


class GpuCrossentropySoftmaxArgmax1HotWithBias(GpuKernelBase, Op):
    """
    Implement CrossentropySoftmaxArgmax1HotWithBias on the gpu.

    """
    nin = 3
    nout = 3
    __props__ = ()
    _f16_ok = True

    def make_node(self, x, b, y_idx):
        ctx_name = infer_context_name(x, b, y_idx)
        x = as_gpuarray_variable(x, ctx_name)
        b = as_gpuarray_variable(b, ctx_name)
        y_idx = as_gpuarray_variable(y_idx, ctx_name)
        nll = GpuArrayType(x.type.dtype,
                           y_idx.type.broadcastable,
                           context_name=ctx_name)()
        sm = x.type()
        am = y_idx.type()
        return Apply(self, [x, b, y_idx], [nll, sm, am])

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>', 'gpuarray_helper.h']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def gpu_kernels(self, node, nodename):
        dtype_x = node.inputs[0].dtype
        dtype_b = node.inputs[1].dtype
        dtype_y_idx = node.inputs[2].dtype
        work_x = work_dtype(dtype_x)
        work_b = work_dtype(dtype_b)
        load_x = load_w(dtype_x)
        load_b = load_w(dtype_b)
        write_x = write_w(dtype_x)
        write_b = write_w(dtype_b)
        flags = Kernel.get_flags(dtype_x, dtype_b, dtype_y_idx)
        type_x = gpuarray.dtype_to_ctype(dtype_x)
        type_b = gpuarray.dtype_to_ctype(dtype_b)
        work_x = gpuarray.dtype_to_ctype(work_x)
        type_y_idx = gpuarray.dtype_to_ctype(dtype_y_idx)
        kname = "k_xent_sm_1hot_bias"
        k_var = "k_xent_sm_1hot_bias_" + nodename
        if node.inputs[0].type.context.kind != b'cuda':
            f = ''
        else:
            f = '' if dtype_x == 'float64' else 'f'
        params = [
            gpuarray.SIZE, gpuarray.SIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE
        ]
        sio = StringIO()
        print("""#include "cluda.h"

        KERNEL void %(kname)s(const ga_size M, const ga_size N,
            GLOBAL_MEM const %(type_x)s* x_data, const ga_size offset_x, const ga_ssize xs0, const ga_ssize xs1,
            GLOBAL_MEM const %(type_b)s* b, const ga_size offset_b, const ga_ssize bs0,
            GLOBAL_MEM const %(type_y_idx)s* y_idx_data, const ga_size offset_y_idx, const ga_ssize y_idxs0,
            GLOBAL_MEM %(type_x)s* nll_data, const ga_size offset_nll, const ga_ssize nlls0,
            GLOBAL_MEM %(type_x)s* sm_data, const ga_size offset_sm, const ga_ssize sms0, const ga_ssize sms1,
            GLOBAL_MEM %(type_y_idx)s* am_data, const ga_size offset_am, const ga_ssize ams0 GA_DECL_SHARED_PARAM(%(work_x)s, per_thread_values))
        {
          x_data = (GLOBAL_MEM const %(type_x)s *)(((GLOBAL_MEM char *)x_data)+offset_x);
          b = (GLOBAL_MEM const %(type_b)s *)(((GLOBAL_MEM char *)b)+offset_b);
          y_idx_data = (GLOBAL_MEM const %(type_y_idx)s *)(((GLOBAL_MEM char *)y_idx_data)+offset_y_idx);
          nll_data = (GLOBAL_MEM %(type_x)s *)(((GLOBAL_MEM char *)nll_data)+offset_nll);
          sm_data = (GLOBAL_MEM %(type_x)s *)(((GLOBAL_MEM char *)sm_data)+offset_sm);
          am_data = (GLOBAL_MEM %(type_y_idx)s *)(((GLOBAL_MEM char *)am_data)+offset_am);
          for (ga_int row = GID_0; row < M; row += GDIM_0){
            GLOBAL_MEM const %(type_x)s* x = x_data + xs0 * row;
            GLOBAL_MEM %(type_x)s* sm = sm_data + sms0 * row;
            GA_DECL_SHARED_BODY(%(work_x)s, per_thread_values);
            LOCAL_MEM %(work_x)s row_max, sum, sum_inv;
            LOCAL_MEM ga_int row_max_threadIdx;
            %(work_x)s per_thread_row_max, per_thread_sum;
            ga_int per_thread_row_max_j;
            // COMPUTE ROW MAX AND ARGMAX
            // compute separate per-thread maximums and argmaxes
            per_thread_row_max = NAN;
            per_thread_row_max_j = 0;
            for (ga_int j = LID_0; j < N; j += LDIM_0)
            {
              %(work_x)s row_ij = %(load_x)s(x[j * xs1]) + %(load_b)s(b[j * bs0]);
              per_thread_row_max_j = (row_ij > per_thread_row_max) ? j : per_thread_row_max_j;
              per_thread_row_max = fmax%(f)s(row_ij, per_thread_row_max);
            }
            per_thread_values[LID_0] = per_thread_row_max;
            local_barrier();
            if (LID_0 == 0) {
              row_max = NAN;
              row_max_threadIdx = 0;
              for (ga_int j = 0; j < LDIM_0; j++)
              {
                %(work_x)s per_thread_max = per_thread_values[j];
                row_max_threadIdx = (per_thread_max > row_max) ? j : row_max_threadIdx;
                row_max = fmax%(f)s(per_thread_max, row_max);
              }
            }
            local_barrier();
            // The thread with the highest max writes out which of its
            // values was the winner.
            if (LID_0 == row_max_threadIdx) am_data[row * ams0] = per_thread_row_max_j;
            // COMPUTE SOFTMAX
            per_thread_sum = 0.0;
            for (ga_int j = LID_0; j < N; j += LDIM_0)
            {
              %(work_x)s row_ij = %(load_x)s(x[j * xs1]) + %(load_b)s(b[j * bs0]);
              %(work_x)s sm_ij = exp%(f)s(row_ij - row_max);
              per_thread_sum += sm_ij;
              sm[j * sms1] = %(write_x)s(sm_ij);
            }
            per_thread_values[LID_0] = per_thread_sum;
            local_barrier();
            if (LID_0 == 0) {
              sum = 0.0;
              for (ga_int j = 0; j < LDIM_0; j++) {
                sum += per_thread_values[j];
              }
              sum_inv = 1.0 / sum;
            }
            local_barrier();
            for (ga_int j = LID_0; j < N; j += LDIM_0) {
              sm[j * sms1] = %(write_x)s(%(load_x)s(sm[j * sms1]) * sum_inv);
            }
            if (LID_0 == 0) {
              const %(type_y_idx)s y_idx = (ga_int)y_idx_data[row * y_idxs0];
              if ((y_idx >= N || y_idx < 0)) {
                // raise some suspicion.
                nll_data[row * nlls0] = %(write_x)s(0.0);
              } else {
                nll_data[row * nlls0] = %(write_x)s(
                   - %(load_x)s(x[y_idx * xs1])
                   - %(load_b)s(b[y_idx * bs0])
                   + row_max + log%(f)s(sum));
              }
            }
          }
        }
        """ % locals(), file=sio)

        return [Kernel(code=sio.getvalue(), name=kname, params=params,
                       flags=flags, objvar=k_var)]

    def c_code(self, node, nodename, inp, out, sub):
        itemsize_x = np.dtype(node.inputs[0].dtype).itemsize
        worksize_x = np.dtype(work_dtype(node.inputs[0].dtype)).itemsize
        itemsize_b = np.dtype(node.inputs[1].dtype).itemsize
        itemsize_y_idx = np.dtype(node.inputs[2].dtype).itemsize
        itemsize_nll = np.dtype(node.outputs[0].dtype).itemsize
        itemsize_sm = np.dtype(node.outputs[1].dtype).itemsize
        itemsize_am = np.dtype(node.outputs[2].dtype).itemsize
        x, b, y_idx = inp
        nll, sm, am = out
        fail = sub['fail']
        ctx = sub['params']
        k_var = "k_xent_sm_1hot_bias_%(nodename)s" % locals()
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                %(fail)s;
            }
        """ % locals()
        sio = StringIO()
        print("""
        if (PyGpuArray_DIMS(%(x)s)[0] !=
            PyGpuArray_DIMS(%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "dimension mismatch in x,y_idx arguments");
            %(fail)s;
        }
        if (PyGpuArray_DIMS(%(x)s)[1] != PyGpuArray_DIMS(%(b)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "dimension mismatch in x,b arguments");
            %(fail)s;
        }
        if (theano_prep_output(&%(nll)s, 1, PyGpuArray_DIMS(%(y_idx)s), %(x)s->ga.typecode, GA_C_ORDER, %(ctx)s)) %(fail)s
        if (theano_prep_output(&%(sm)s, 2, PyGpuArray_DIMS(%(x)s), %(x)s->ga.typecode, GA_C_ORDER, %(ctx)s)) %(fail)s
        if (theano_prep_output(&%(am)s, 1, PyGpuArray_DIMS(%(y_idx)s), %(y_idx)s->ga.typecode, GA_C_ORDER, %(ctx)s)) %(fail)s
        {
            size_t n_blocks = std::min(PyGpuArray_DIM(%(x)s, 0), (size_t)4096);
            size_t n_threads = std::min(PyGpuArray_DIM(%(x)s, 1), (size_t)256);
            size_t n_shared = n_threads * %(worksize_x)s;
     //TODO: launch more threads per row and do parallel sum and max reductions
            int err = k_xent_sm_1hot_bias_call(
                1, &n_blocks, &n_threads, n_shared,
                PyGpuArray_DIMS(%(x)s)[0],
                PyGpuArray_DIMS(%(x)s)[1],
                %(x)s->ga.data, %(x)s->ga.offset,
                PyGpuArray_STRIDE(%(x)s, 0) / %(itemsize_x)s,
                PyGpuArray_STRIDE(%(x)s, 1) / %(itemsize_x)s,
                %(b)s->ga.data, %(b)s->ga.offset,
                PyGpuArray_STRIDE(%(b)s, 0) / %(itemsize_b)s,
                %(y_idx)s->ga.data, %(y_idx)s->ga.offset,
                PyGpuArray_STRIDE(%(y_idx)s, 0) / %(itemsize_y_idx)s,
                %(nll)s->ga.data, %(nll)s->ga.offset,
                PyGpuArray_STRIDE(%(nll)s, 0) / %(itemsize_nll)s,
                %(sm)s->ga.data, %(sm)s->ga.offset,
                PyGpuArray_STRIDE(%(sm)s, 0) / %(itemsize_sm)s,
                PyGpuArray_STRIDE(%(sm)s, 1) / %(itemsize_sm)s,
                %(am)s->ga.data, %(am)s->ga.offset,
                PyGpuArray_STRIDE(%(am)s, 0) / %(itemsize_am)s);
            %(err_check)s
        }
        """ % locals(), file=sio)
        return sio.getvalue()

    def c_code_cache_version(self):
        return (14,)


gpu_crossentropy_softmax_argmax_1hot_with_bias = GpuCrossentropySoftmaxArgmax1HotWithBias()


class GpuCrossentropySoftmax1HotWithBiasDx(GpuKernelBase, Op):
    """
    Implement CrossentropySoftmax1HotWithBiasDx on the gpu.

    Gradient wrt x of the CrossentropySoftmax1Hot Op.

    """
    nin = 3
    nout = 1
    __props__ = ()
    _f16_ok = True

    def make_node(self, dnll, sm, y_idx):
        ctx_name = infer_context_name(dnll, sm, y_idx)
        dnll = as_gpuarray_variable(dnll, ctx_name)
        sm = as_gpuarray_variable(sm, ctx_name)
        y_idx = as_gpuarray_variable(y_idx, ctx_name)
        return Apply(self, [dnll, sm, y_idx], [sm.type()])

    def c_code_cache_version(self):
        return (14,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code(self, node, nodename, inp, out, sub):
        typecode_dx = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        itemsize_dnll = np.dtype(node.inputs[0].dtype).itemsize
        itemsize_sm = np.dtype(node.inputs[1].dtype).itemsize
        itemsize_y_idx = np.dtype(node.inputs[2].dtype).itemsize
        itemsize_dx = np.dtype(node.outputs[0].dtype).itemsize
        dtype_dnll = node.inputs[0].dtype
        dtype_sm = node.inputs[1].dtype
        dtype_y_idx = node.inputs[2].dtype
        dtype_dx = node.outputs[0].dtype
        type_intp = gpuarray.dtype_to_ctype(np.intp)
        dnll, sm, y_idx = inp
        dx, = out
        fail = sub['fail']
        ctx = sub['params']
        k_var = "kCrossEntropySoftmax1HotWithBiasDx_" + nodename
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: %(k_var)s: %%s.",
                             GpuKernel_error(&%(k_var)s, err));
                %(fail)s;
            }
        """ % locals()
        return """
        // Get `dnll.shape[0]` or set it to zero if `dnll` is a scalar.
        const ssize_t %(dnll)s_dims0 = (PyGpuArray_NDIM(%(dnll)s) > 0 ?
                                        PyGpuArray_DIMS(%(dnll)s)[0] :
                                        (ssize_t) 0);
        // Get `dnll.strides[0]` and set it to zero if `dnll` is a scalar
        // or a vector with just one element.
        const ssize_t %(dnll)s_strides0 = (%(dnll)s_dims0 > 1 ?
                                           PyGpuArray_STRIDES(%(dnll)s)[0] :
                                           (ssize_t) 0);
        if ((PyGpuArray_NDIM(%(dnll)s) > 1)
            || (PyGpuArray_NDIM(%(sm)s) != 2)
            || (PyGpuArray_NDIM(%(y_idx)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (%(dnll)s_dims0 !=
            PyGpuArray_DIMS(%(sm)s)[0] && %(dnll)s_dims0 > 1)
        {
            PyErr_Format(PyExc_ValueError,
                         "dnll.shape[0] == %%i, but sm.shape[0] == %%i",
                         %(dnll)s_dims0,
                         PyGpuArray_DIMS(%(sm)s)[0]);
            %(fail)s;
        }
        if (%(dnll)s_dims0 !=
            PyGpuArray_DIMS(%(y_idx)s)[0] && %(dnll)s_dims0 > 1)
        {
            PyErr_SetString(PyExc_ValueError,
                            "dnll.shape[0] != y_idx.shape[0]");
            %(fail)s;
        }
        if (PyGpuArray_DIMS(%(sm)s)[0] !=
            PyGpuArray_DIMS(%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "sm.shape[0] != y_idx.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (PyGpuArray_DIMS(%(dx)s)[0] !=
                PyGpuArray_DIMS(%(sm)s)[0])
            || (PyGpuArray_DIMS(%(dx)s)[1] !=
                PyGpuArray_DIMS(%(sm)s)[1]))
        {
            Py_XDECREF(%(dx)s);
            %(dx)s = pygpu_empty(2, PyGpuArray_DIMS(%(sm)s),
                                 %(typecode_dx)s, GA_C_ORDER,
                                 %(ctx)s, Py_None);
            if (!%(dx)s) {
                %(fail)s
            }
        }
        {
            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(dx)s)[0], (size_t)256), 1, 1};
            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(dx)s)[1], (size_t)256), 1, 1};
            ssize_t stride_DNLL0 = %(dnll)s_strides0 / %(itemsize_dnll)s;
            ssize_t stride_SM0 = PyGpuArray_STRIDES(%(sm)s)[0] / %(itemsize_sm)s;
            ssize_t stride_SM1 = PyGpuArray_STRIDES(%(sm)s)[1] / %(itemsize_sm)s;
            ssize_t stride_YIDX0 = PyGpuArray_STRIDES(%(y_idx)s)[0] / %(itemsize_y_idx)s;
            ssize_t stride_DX0 = PyGpuArray_STRIDES(%(dx)s)[0] / %(itemsize_dx)s;
            ssize_t stride_DX1 = PyGpuArray_STRIDES(%(dx)s)[1] / %(itemsize_dx)s;
            void *kernel_params[] = {
                (void *)&PyGpuArray_DIMS(%(dx)s)[0],
                (void *)&PyGpuArray_DIMS(%(dx)s)[1],
                (void *)%(dnll)s->ga.data, (void *)&%(dnll)s->ga.offset,
                (void *)&stride_DNLL0,
                (void *)%(sm)s->ga.data, (void *)&%(sm)s->ga.offset,
                (void *)&stride_SM0, (void *)&stride_SM1,
                (void *)%(y_idx)s->ga.data, (void *)&%(y_idx)s->ga.offset,
                (void *)&stride_YIDX0,
                (void *)%(dx)s->ga.data, (void *)&%(dx)s->ga.offset,
                (void *)&stride_DX0, (void *)&stride_DX1};
            int err = GpuKernel_call(&%(k_var)s, 3, n_blocks, threads_per_block, 0, kernel_params);
            %(err_check)s
        }
        assert(%(dx)s);
        """ % locals()

    def gpu_kernels(self, node, nodename):
        dtype_dnll = node.inputs[0].dtype
        dtype_sm = node.inputs[1].dtype
        dtype_y_idx = node.inputs[2].dtype
        dtype_dx = node.outputs[0].dtype
        work_dnll = work_dtype(dtype_dnll)
        load_dnll = load_w(dtype_dnll)
        load_sm = load_w(dtype_sm)
        write_dx = write_w(dtype_dx)
        flags = Kernel.get_flags(dtype_dnll, dtype_sm, dtype_y_idx, dtype_dx)
        wtype_dnll = gpuarray.dtype_to_ctype(work_dnll)
        type_dnll = gpuarray.dtype_to_ctype(dtype_dnll)
        type_sm = gpuarray.dtype_to_ctype(dtype_sm)
        type_y_idx = gpuarray.dtype_to_ctype(dtype_y_idx)
        type_dx = gpuarray.dtype_to_ctype(dtype_dx)
        kname = "kCrossEntropySoftmax1HotWithBiasDx"
        k_var = "kCrossEntropySoftmax1HotWithBiasDx_" + nodename
        params = [
            gpuarray.SIZE, gpuarray.SIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
        ]
        sio = StringIO()
        print("""#include "cluda.h"

        KERNEL void %(kname)s(
           const ga_size N, const ga_size K,
           GLOBAL_MEM const %(type_dnll)s* dnll, const ga_size offset_dnll, const ga_ssize dnll_s0,
           GLOBAL_MEM const %(type_sm)s* sm, const ga_size offset_sm, const ga_ssize sm_s0, const ga_ssize sm_s1,
           GLOBAL_MEM const %(type_y_idx)s* y_idx, const ga_size offset_y_idx, const ga_ssize y_idx_s0,
           GLOBAL_MEM %(type_dx)s* dx, const ga_size offset_dx, const ga_ssize dx_s0, const ga_ssize dx_s1)
        {
            dnll = (GLOBAL_MEM const %(type_dnll)s *)(((GLOBAL_MEM char *)dnll)+offset_dnll);
            sm = (GLOBAL_MEM const %(type_sm)s *)(((GLOBAL_MEM char *)sm)+offset_sm);
            y_idx = (GLOBAL_MEM const %(type_y_idx)s *)(((GLOBAL_MEM char *)y_idx)+offset_y_idx);
            dx = (GLOBAL_MEM %(type_dx)s *)(((GLOBAL_MEM char *)dx)+offset_dx);
            for (ga_int i = GID_0; i < N; i += GDIM_0)
            {
                %(wtype_dnll)s dnll_i = %(load_dnll)s(dnll[i * dnll_s0]);
                %(type_y_idx)s y_i = y_idx[i * y_idx_s0];
                for (ga_int j = LID_0; j < K; j += LDIM_0)
                {
                    if (y_i == j)
                    {
                        dx[i * dx_s0 + j * dx_s1] =
                            %(write_dx)s(dnll_i *
                              (%(load_sm)s(sm[i * sm_s0 + j * sm_s1]) - 1.0));
                    }
                    else
                    {
                        dx[i * dx_s0 + j * dx_s1] =
                            %(write_dx)s(dnll_i *
                              %(load_sm)s(sm[i * sm_s0 + j * sm_s1]));
                    }
                }
            }
        }
        """ % locals(), file=sio)
        return [Kernel(code=sio.getvalue(), name=kname, params=params,
                       flags=flags, objvar=k_var)]


gpu_crossentropy_softmax_1hot_with_bias_dx = GpuCrossentropySoftmax1HotWithBiasDx()


class GpuSoftmax(GpuKernelBase, Op):
    """
    Implement Softmax on the gpu.

    """
    __props__ = ()
    _f16_ok = True

    def make_node(self, x):
        x = as_gpuarray_variable(x, infer_context_name(x))
        return Apply(self, [x], [x.type()])

    def infer_shape(self, node, shape):
        return shape

    def c_code_cache_version(self):
        return (17,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code(self, node, nodename, inp, out, sub):
        dtype_x = node.inputs[0].dtype
        work_x = work_dtype(dtype_x)
        dtype_z = node.outputs[0].dtype
        itemsize_x = np.dtype(dtype_x).itemsize
        itemsize_z = np.dtype(dtype_z).itemsize
        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        x, = inp
        z, = out
        fail = sub['fail']
        ctx = sub['params']
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError, fmt_str, msg);
                %(fail)s;
            }
        """ % locals()
        return """
        if (PyGpuArray_NDIM(%(x)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if ((NULL == %(z)s) ||
            (PyGpuArray_DIMS(%(z)s)[0] !=
             PyGpuArray_DIMS(%(x)s)[0]) ||
            (PyGpuArray_DIMS(%(z)s)[1] !=
             PyGpuArray_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = pygpu_empty(2, PyGpuArray_DIMS(%(x)s),
                                %(typecode)s, GA_C_ORDER,
                                %(ctx)s, Py_None);
            if (!%(z)s) {
                %(fail)s
            }
        }
        {
            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)(32 * 1024)), 1, 1};
//TODO, detect the maximum number of thread per block.
            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)256), 1, 1}; // TODO: Read GA_CTX_PROP_MAXLSIZE0
            size_t shmem_sz = PyGpuArray_DIMS(%(x)s)[1] *
                                     2 * sizeof(npy_%(work_x)s);
            ssize_t stride_X0 = PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s;
            ssize_t stride_X1 = PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s;
            ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
            ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s;
            const char *fmt_str, *msg;
            void *kernel_params[] = {
                (void *)&PyGpuArray_DIMS(%(x)s)[0],
                (void *)&PyGpuArray_DIMS(%(x)s)[1],
                (void *)%(x)s->ga.data, (void *)&%(x)s->ga.offset,
                (void *)&stride_X0, (void *)&stride_X1,
                (void *)%(z)s->ga.data, (void *)&%(z)s->ga.offset,
                (void *)&stride_Z0, (void *)&stride_Z1};
            int err = GA_NO_ERROR;
            if (PyGpuArray_DIMS(%(x)s)[0] > 0)
            {
              //Those numbers are based on not too recent GPU
              //to make them compatible with more GPU.
              //TODO: read the information from the card.
              if(shmem_sz < (32 * 1024 - 500)){
                err = GpuKernel_call(&kSoftmax_%(nodename)s, 3,
                                     n_blocks, threads_per_block, shmem_sz,
                                     kernel_params);
                fmt_str = "gpuarray error: kSoftmax_%(nodename)s: %%s";
                msg = GpuKernel_error(&kSoftmax_%(nodename)s, err);
              }else{
                err = GpuKernel_call(&kSoftmax_fixed_shared%(nodename)s, 3,
                                     n_blocks, threads_per_block,
                                     threads_per_block[0] * sizeof(npy_%(work_x)s),
                                     kernel_params);
                fmt_str = "gpuarray error: kSoftmax_fixed_shared%(nodename)s: %%s";
                msg = GpuKernel_error(&kSoftmax_fixed_shared%(nodename)s, err);
              }
              %(err_check)s
            }
        }
        assert(%(z)s);
        """ % locals()

    def gpu_kernels(self, node, nodename):
        dtype_x = node.inputs[0].dtype
        dtype_sm = node.outputs[0].dtype
        load_x = load_w(dtype_x)
        write_sm = write_w(node.outputs[0].dtype)
        work_sm = work_dtype(dtype_sm)
        flags = Kernel.get_flags(dtype_x, dtype_sm)
        type_x = gpuarray.dtype_to_ctype(dtype_x)
        type_sm = gpuarray.dtype_to_ctype(dtype_sm)
        type_acc = gpuarray.dtype_to_ctype(work_sm)

        ctype = gpuarray.dtype_to_ctype(work_sm)

        params = [
            gpuarray.SIZE, gpuarray.SIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE
        ]
        kernels = []
        kname = "kSoftmax"
        k_var = "kSoftmax_" + nodename
        code = """#include "cluda.h"

        KERNEL void %(kname)s (const ga_size M, const ga_size N,
                               GLOBAL_MEM const %(type_x)s * x, const ga_size offset_x, const ga_ssize sx0, const ga_ssize sx1,
                               GLOBAL_MEM %(type_sm)s * sm, const ga_size offset_sm, const ga_ssize sm_s0, const ga_ssize sm_s1 GA_DECL_SHARED_PARAM(%(type_acc)s, buf))
        {
            GA_DECL_SHARED_BODY(%(type_acc)s, buf);
            LOCAL_MEM_ARG %(type_acc)s * buf2 = buf + N;
            x = (GLOBAL_MEM const %(type_x)s *)(((GLOBAL_MEM char *)x)+offset_x);
            sm = (GLOBAL_MEM %(type_sm)s *)(((GLOBAL_MEM char *)sm)+offset_sm);
            for (ga_int blockIDX = GID_0; blockIDX < M; blockIDX += GDIM_0) {
                for (ga_int tx = LID_0; tx< N; tx += LDIM_0) {
                    buf[tx] = %(load_x)s(x[blockIDX * sx0 + tx * sx1]);
                    buf2[tx] = buf[tx];
                }
                local_barrier();
                {
                    // This function trashes buf[1..GA_WARP_SIZE],
                    // leaving the reduction result in buf[0].
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < N; i += GA_WARP_SIZE)
                        {
                            buf[LID_0] = max(buf[LID_0], buf[i]);
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = max(buf[LID_0], buf[LID_0+_n]);
                        local_barrier();
                    }
                }
                %(ctype)s row_max = buf[0];
                local_barrier();
                for(ga_int __i=LID_0; __i<N; __i+=LDIM_0){
                    buf[__i] = exp(buf2[__i] - row_max);
                    buf2[__i] = buf[__i];
                }
                local_barrier();
                {
                    // This function trashes buf[1..GA_WARP_SIZE],
                    // leaving the reduction result in buf[0].
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < N; i += GA_WARP_SIZE)
                        {
                            buf[LID_0] = buf[LID_0] + buf[i];
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = buf[LID_0] + buf[LID_0+_n];
                        local_barrier();
                    }
                }
                %(ctype)s row_sum = buf[0];
                local_barrier();
                for(ga_int __i=LID_0; __i<N; __i+=LDIM_0) {
                    buf[__i] = buf2[__i] / row_sum;
                }
                local_barrier();
                for (ga_int tx = LID_0; tx< N; tx += LDIM_0) {
                    sm[blockIDX * sm_s0 + tx * sm_s1] = %(write_sm)s(buf[tx]);
                }
                local_barrier();
            }
        }
        """ % locals()
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        kname = "kSoftmax_fixed_shared"
        k_var = "kSoftmax_fixed_shared" + nodename
        code = """#include "cluda.h"

        KERNEL void %(kname)s (const ga_size M, const ga_size N,
                               GLOBAL_MEM const %(type_x)s * x, const ga_size offset_x, const ga_ssize sx0, const ga_ssize sx1,
                               GLOBAL_MEM %(type_sm)s * sm, const ga_size offset_sm, const ga_ssize sm_s0, const ga_ssize sm_s1 GA_DECL_SHARED_PARAM(%(type_acc)s, buf))
        {
            GA_DECL_SHARED_BODY(%(type_acc)s, buf);
            x = (GLOBAL_MEM const %(type_x)s *)(((GLOBAL_MEM char *)x)+offset_x);
            sm = (GLOBAL_MEM %(type_sm)s *)(((GLOBAL_MEM char *)sm)+offset_sm);
            for (ga_int blockIDX = GID_0; blockIDX < M; blockIDX += GDIM_0){
                GLOBAL_MEM const %(type_x)s *x_ptr = &x[blockIDX * sx0];
                GLOBAL_MEM %(type_sm)s *sm_ptr = &sm[blockIDX * sm_s0];
                {
                    // This function trashes buf[1..n_threads],
                    // leaving the reduction result in buf[0].
                    %(ctype)s red = %(load_x)s(x_ptr[LID_0 * sx1]);
                    #pragma unroll 16
                    for (ga_int i = LID_0 + LDIM_0; i<N; i += LDIM_0) {
                        red = max(red, %(load_x)s(x_ptr[i * sx1]));
                    }
                    buf[LID_0] = red;
                    local_barrier();
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < LDIM_0; i += GA_WARP_SIZE) {
                            buf[LID_0] = max(buf[LID_0], buf[i]);
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = max(buf[LID_0], buf[LID_0+_n]);
                        local_barrier();
                    }
                }
                %(ctype)s row_max = buf[0];
                local_barrier();
                {
                    // This function trashes buf[1..n_threads],
                    // leaving the reduction result in buf[0].
                    %(ctype)s red = exp(%(load_x)s(x_ptr[LID_0 * sx1]) - row_max);
                    #pragma unroll 16
                    for (ga_int i = LID_0 + LDIM_0; i<N; i += LDIM_0) {
                        red = red + exp(%(load_x)s(x_ptr[i * sx1]) - row_max);
                    }
                    buf[LID_0] = red;
                    local_barrier();
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < LDIM_0; i += GA_WARP_SIZE) {
                            buf[LID_0] = buf[LID_0] + buf[i];
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = buf[LID_0] + buf[LID_0+_n];
                        local_barrier();
                    }
                }
                %(ctype)s row_sum = buf[0];
                local_barrier();
                for (ga_int tx = LID_0; tx< N; tx += LDIM_0){
                    sm_ptr[tx * sm_s1] = %(write_sm)s(exp(%(load_x)s(x_ptr[tx * sx1]) - row_max) / row_sum);
                }
                local_barrier();
            }
        }
        """ % locals()
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels


gpu_softmax = GpuSoftmax()


class GpuSoftmaxWithBias(GpuKernelBase, Op):
    """
    Implement SoftmaxWithBias on the gpu.

    """
    nin = 2
    nout = 1
    __props__ = ()
    _f16_ok = True

    def make_node(self, x, b):
        ctx_name = infer_context_name(x, b)
        x = as_gpuarray_variable(x, ctx_name)
        b = as_gpuarray_variable(b, ctx_name)
        return Apply(self, [x, b], [x.type()])

    def infer_shape(self, node, shape):
        return [shape[0]]

    def c_code_cache_version(self):
        return (16,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code(self, node, nodename, inp, out, sub):
        dtype_x = node.inputs[0].dtype
        dtype_b = node.inputs[1].dtype
        dtype_z = node.outputs[0].dtype
        work_x = work_dtype(dtype_x)
        itemsize_x = np.dtype(dtype_x).itemsize
        itemsize_b = np.dtype(dtype_b).itemsize
        itemsize_z = np.dtype(dtype_z).itemsize
        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        x, b = inp
        z, = out
        fail = sub['fail']
        ctx = sub['params']
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError, fmt_str, msg);
                %(fail)s;
            }
        """ % locals()
        return """
        if (PyGpuArray_NDIM(%(x)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "rank error input");
            %(fail)s;
        }
        if (PyGpuArray_NDIM(%(b)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "rank error for the bias");
            %(fail)s;
        }
        if ((PyGpuArray_DIMS(%(x)s)[1] !=
            PyGpuArray_DIMS(%(b)s)[0]))
        {
            PyErr_Format(PyExc_ValueError,
                         "number of columns in x (%%ld)"
                         " does not match length of b (%%ld)",
                         (long int)PyGpuArray_DIMS(%(x)s)[1],
                         (long int)PyGpuArray_DIMS(%(b)s)[0]);
            %(fail)s;
        }
        if ((NULL == %(z)s)
            || (PyGpuArray_DIMS(%(z)s)[0] !=
                PyGpuArray_DIMS(%(x)s)[0])
            || (PyGpuArray_DIMS(%(z)s)[1] !=
                PyGpuArray_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = pygpu_empty(2, PyGpuArray_DIMS(%(x)s),
                                %(typecode)s, GA_C_ORDER,
                                %(ctx)s, Py_None);
            if (!%(z)s) {
                %(fail)s
            }
        }
        {
            size_t n_blocks[3] = {std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)(32*1024)), 1, 1};
//TODO, detect the maximum number of thread per block.
            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)256), 1, 1}; // TODO: Read GA_CTX_PROP_MAXLSIZE0
            size_t shmem_sz = PyGpuArray_DIMS(%(x)s)[1] *
                                     2 * sizeof(npy_%(work_x)s);
            ssize_t stride_X0 = PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s;
            ssize_t stride_X1 = PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s;
            ssize_t stride_B0 = PyGpuArray_STRIDES(%(b)s)[0] / %(itemsize_b)s;
            ssize_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
            ssize_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s;
            const char *fmt_str, *msg;
            void *kernel_params[] = {
                (void *)&PyGpuArray_DIMS(%(x)s)[0],
                (void *)&PyGpuArray_DIMS(%(x)s)[1],
                (void *)%(x)s->ga.data, (void *)&%(x)s->ga.offset,
                (void *)&stride_X0, (void *)&stride_X1,
                (void *)%(b)s->ga.data, (void *)&%(b)s->ga.offset,
                (void *)&stride_B0,
                (void *)%(z)s->ga.data, (void *)&%(z)s->ga.offset,
                (void *)&stride_Z0, (void *)&stride_Z1};
            int err = GA_NO_ERROR;
            if (PyGpuArray_DIMS(%(x)s)[0] > 0)
            {
              if(shmem_sz < (32 * 1024 - 500)){
                err = GpuKernel_call(&kSoftmaxWithBias_%(nodename)s, 3,
                                     n_blocks, threads_per_block, shmem_sz,
                                     kernel_params);
                fmt_str = "gpuarray error: kSoftmaxWithBias_%(nodename)s: %%s";
                msg = GpuKernel_error(&kSoftmaxWithBias_%(nodename)s, err);
              }else{
                err = GpuKernel_call(&kSoftmaxWithBias_fixed_shared%(nodename)s,
                                     3, n_blocks, threads_per_block,
                                     threads_per_block[0] * sizeof(npy_%(work_x)s),
                                     kernel_params);
                fmt_str = "gpuarray error: kSoftmaxWithBias_fixed_shared%(nodename)s: %%s";
                msg = GpuKernel_error(&kSoftmaxWithBias_fixed_shared%(nodename)s, err);
              }
              %(err_check)s
            }
        }
        assert(%(z)s);
        """ % locals()

    def gpu_kernels(self, node, nodename):
        dtype_x = node.inputs[0].dtype
        dtype_b = node.inputs[1].dtype
        dtype_sm = node.outputs[0].dtype
        load_x = load_w(node.inputs[0].dtype)
        load_b = load_w(node.inputs[1].dtype)
        write_sm = write_w(node.outputs[0].dtype)
        work_sm = work_dtype(node.outputs[0].dtype)
        flags = Kernel.get_flags(dtype_x, dtype_b, dtype_sm)
        type_x = gpuarray.dtype_to_ctype(dtype_x)
        type_b = gpuarray.dtype_to_ctype(dtype_b)
        type_sm = gpuarray.dtype_to_ctype(dtype_sm)
        type_acc = gpuarray.dtype_to_ctype(work_sm)

        ctype = gpuarray.dtype_to_ctype(work_sm)

        params = [
            gpuarray.SIZE, gpuarray.SIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE,
            gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SSIZE, gpuarray.SSIZE,
        ]
        kernels = []
        kname = "kSoftmaxWithBias"
        k_var = "kSoftmaxWithBias_" + nodename
        code = """#include "cluda.h"

        KERNEL void %(kname)s (const ga_size M, const ga_size N,
                       GLOBAL_MEM const %(type_x)s * x, const ga_size offset_x, const ga_ssize sx0, const ga_ssize sx1,
                       GLOBAL_MEM const %(type_b)s * b, const ga_size offset_b, const ga_ssize sb0,
                       GLOBAL_MEM %(type_sm)s * sm, const ga_size offset_sm, const ga_ssize sm_s0, const ga_ssize sm_s1 GA_DECL_SHARED_PARAM(%(type_acc)s, buf))
        {
            GA_DECL_SHARED_BODY(%(type_acc)s, buf);
            LOCAL_MEM_ARG %(type_acc)s * buf2 = buf + N;
            x = (GLOBAL_MEM const %(type_x)s *)(((GLOBAL_MEM char *)x)+offset_x);
            b = (GLOBAL_MEM const %(type_b)s *)(((GLOBAL_MEM char *)b)+offset_b);
            sm = (GLOBAL_MEM %(type_sm)s *)(((GLOBAL_MEM char *)sm)+offset_sm);
            for (ga_int blockIDX = GID_0; blockIDX < M; blockIDX += GDIM_0){
                for (ga_int tx = LID_0; tx< N; tx += LDIM_0){
                    buf[tx] = %(load_x)s(x[blockIDX * sx0 + tx * sx1]);
                    buf[tx] += %(load_b)s(b[tx * sb0]);
                    buf2[tx] = buf[tx];
                }
                local_barrier();
                {
                    // This function trashes buf[1..GA_WARP_SIZE],
                    // leaving the reduction result in buf[0].
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < N; i += GA_WARP_SIZE)
                        {
                            buf[LID_0] = max(buf[LID_0], buf[i]);
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = max(buf[LID_0], buf[LID_0+_n]);
                        local_barrier();
                    }
                }
                %(ctype)s row_max = buf[0];
                local_barrier();
                for(ga_int __i=LID_0; __i<N; __i+=LDIM_0){;
                    buf[__i] = exp(buf2[__i] - row_max);
                    buf2[__i] = buf[__i];
                }
                local_barrier();
                {
                    // This function trashes buf[1..GA_WARP_SIZE],
                    // leaving the reduction result in buf[0].
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < N; i += GA_WARP_SIZE)
                        {
                            buf[LID_0] = buf[LID_0] + buf[i];
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = buf[LID_0] + buf[LID_0+_n];
                        local_barrier();
                    }
                }
                %(ctype)s row_sum = buf[0];
                local_barrier();
                for(ga_int __i=LID_0; __i<N; __i+=LDIM_0){
                    buf[__i] = buf2[__i] / row_sum;
                }
                local_barrier();
                for (ga_int tx = LID_0; tx< N; tx += LDIM_0){
                    sm[blockIDX * sm_s0 + tx * sm_s1] = %(write_sm)s(buf[tx]);
                }
                local_barrier();
            }
        }
        """ % locals()
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        kname = "kSoftmaxWithBias_fixed_shared"
        k_var = "kSoftmaxWithBias_fixed_shared" + nodename
        code = """#include "cluda.h"

        KERNEL void %(kname)s (const ga_size M, const ga_size N,
                       GLOBAL_MEM const %(type_x)s * x, const ga_size offset_x, const ga_ssize sx0, const ga_ssize sx1,
                       GLOBAL_MEM const %(type_b)s * b, const ga_size offset_b, const ga_ssize sb0,
                       GLOBAL_MEM %(type_sm)s * sm, const ga_size offset_sm, const ga_ssize sm_s0, const ga_ssize sm_s1 GA_DECL_SHARED_PARAM(%(type_acc)s, buf))
        {
            GA_DECL_SHARED_BODY(%(type_acc)s, buf);
            x = (GLOBAL_MEM const %(type_x)s *)(((GLOBAL_MEM char *)x)+offset_x);
            b = (GLOBAL_MEM const %(type_b)s *)(((GLOBAL_MEM char *)b)+offset_b);
            sm = (GLOBAL_MEM %(type_sm)s *)(((GLOBAL_MEM char *)sm)+offset_sm);
            for (ga_int blockIDX = GID_0; blockIDX < M; blockIDX += GDIM_0){
                GLOBAL_MEM const %(type_x)s *x_ptr = &x[blockIDX * sx0];
                GLOBAL_MEM %(type_sm)s *sm_ptr = &sm[blockIDX * sm_s0];
                {
                    // This function trashes buf[1..n_threads],
                    // leaving the reduction result in buf[0].
                    %(ctype)s red = %(load_x)s(x_ptr[LID_0 * sx1]) + %(load_b)s(b[LID_0 * sb0]);
                    #pragma unroll 16
                    for (ga_int i = LID_0 + LDIM_0; i<N; i += LDIM_0) {
                        red = max(red, %(load_x)s(x_ptr[i * sx1]) + %(load_b)s(b[i * sb0]));
                    }
                    buf[LID_0] = red;
                    local_barrier();
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < LDIM_0; i += GA_WARP_SIZE) {
                            buf[LID_0] = max(buf[LID_0], buf[i]);
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = max(buf[LID_0], buf[LID_0+_n]);
                        local_barrier();
                    }
                }
                %(ctype)s row_max = buf[0];
                local_barrier();
                {
                    // This function trashes buf[1..n_threads],
                    // leaving the reduction result in buf[0].
                    %(ctype)s red = exp(%(load_x)s(x_ptr[LID_0 * sx1]) + %(load_b)s(b[LID_0 * sb0]) - row_max);
                    #pragma unroll 16
                    for (ga_int i = LID_0 + LDIM_0; i<N; i += LDIM_0) {
                    red = red + exp(%(load_x)s(x_ptr[i * sx1]) + %(load_b)s(b[i * sb0]) - row_max);
                    }
                    buf[LID_0] = red;
                    local_barrier();
                    if (LID_0 < GA_WARP_SIZE) {
                        for (ga_int i = LID_0 + GA_WARP_SIZE; i < LDIM_0; i += GA_WARP_SIZE) {
                            buf[LID_0] = buf[LID_0] + buf[i];
                        }
                    }
                    local_barrier();
                    //reduce so that LID_0 0 has the reduction of everything
                    for (ga_uint _n = GA_WARP_SIZE / 2; _n > 0; _n /= 2) {
                        if (LID_0 < _n && LID_0 + _n < N)
                            buf[LID_0] = buf[LID_0] + buf[LID_0+_n];
                        local_barrier();
                    }
                }
                %(ctype)s row_sum = buf[0];
                local_barrier();
                for (ga_int tx = LID_0; tx< N; tx += LDIM_0){
                    sm_ptr[tx * sm_s1] = %(write_sm)s(exp(%(load_x)s(x_ptr[tx * sx1]) + %(load_b)s(b[tx * sb0]) - row_max) / row_sum);
                }
                local_barrier();
            }
        }
        """ % locals()
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels


gpu_softmax_with_bias = GpuSoftmaxWithBias()
