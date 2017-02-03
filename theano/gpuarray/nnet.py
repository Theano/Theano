from __future__ import absolute_import, print_function, division
import os
import numpy as np

from theano import Op, Apply, config
from six import StringIO

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
                        infer_context_name)
from .type import GpuArrayType
from .kernel_codegen import (nvcc_kernel,
                             inline_softmax,
                             inline_softmax_fixed_shared)
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

    def get_params(self, node):
        return node.inputs[0].type.context

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>', 'gpuarray_helper.h']

    def c_header_dirs(self):
        return [os.path.dirname(__file__)]

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
        f = '' if dtype_x == 'float64' else 'f'
        sio = StringIO()
        print("""
        KERNEL void %(kname)s(const ga_size M, const ga_size N,
            const %(type_x)s* x_data, const ga_size offset_x,
            const ga_ssize xs0, const ga_ssize xs1,
            const %(type_b)s* b, const ga_size offset_b,
            const ga_ssize bs0,
            const %(type_y_idx)s* y_idx_data, const ga_size offset_y_idx,
            const ga_ssize y_idxs0,
            %(type_x)s* nll_data, const ga_size offset_nll,
            const ga_ssize nlls0,
            %(type_x)s* sm_data, const ga_size offset_sm,
            const ga_ssize sms0, const ga_ssize sms1,
            %(type_y_idx)s* am_data, const ga_size offset_am,
            const ga_ssize ams0)
        {
          x_data = (const %(type_x)s *)(((char *)x_data)+offset_x);
          b = (const %(type_b)s *)(((char *)b)+offset_b);
          y_idx_data = (const %(type_y_idx)s *)(((char *)y_idx_data)+offset_y_idx);
          nll_data = (%(type_x)s *)(((char *)nll_data)+offset_nll);
          sm_data = (%(type_x)s *)(((char *)sm_data)+offset_sm);
          am_data = (%(type_y_idx)s *)(((char *)am_data)+offset_am);

          for (int row = blockIdx.x; row < M; row += gridDim.x){

            const %(type_x)s* x = x_data + xs0 * row;
            %(type_x)s* sm = sm_data + sms0 * row;

            extern LOCAL_MEM %(work_x)s per_thread_values[];
            LOCAL_MEM %(work_x)s row_max, sum, sum_inv;
            LOCAL_MEM int row_max_threadIdx;

            %(work_x)s per_thread_row_max, per_thread_sum;
            int per_thread_row_max_j;

            // COMPUTE ROW MAX AND ARGMAX

            // compute separate per-thread maximums and argmaxes
            per_thread_row_max = NAN;
            per_thread_row_max_j = 0;

            for (int j = threadIdx.x; j < N; j += blockDim.x)
            {
              %(work_x)s row_ij = %(load_x)s(x[j * xs1]) + %(load_b)s(b[j * bs0]);
              per_thread_row_max_j = (row_ij > per_thread_row_max) ? j : per_thread_row_max_j;
              per_thread_row_max = fmax%(f)s(row_ij, per_thread_row_max);
            }
            per_thread_values[threadIdx.x] = per_thread_row_max;

            local_barrier();

            if (threadIdx.x == 0) {
              row_max = NAN;
              row_max_threadIdx = 0;
              for (int j = 0; j < blockDim.x; j++)
              {
                %(work_x)s per_thread_max = per_thread_values[j];
                row_max_threadIdx = (per_thread_max > row_max) ? j : row_max_threadIdx;
                row_max = fmax%(f)s(per_thread_max, row_max);
              }
            }

            local_barrier();

            // The thread with the higest max writes out which of its
            // values was the winner.
            if (threadIdx.x == row_max_threadIdx) am_data[row * ams0] = per_thread_row_max_j;

            // COMPUTE SOFTMAX
            per_thread_sum = 0.0;
            for (int j = threadIdx.x; j < N; j += blockDim.x)
            {
              %(work_x)s row_ij = %(load_x)s(x[j * xs1]) + %(load_b)s(b[j * bs0]);
              %(work_x)s sm_ij = exp%(f)s(row_ij - row_max);
              per_thread_sum += sm_ij;
              sm[j * sms1] = %(write_x)s(sm_ij);
            }

            per_thread_values[threadIdx.x] = per_thread_sum;

            local_barrier();

            if (threadIdx.x == 0) {
              sum = 0.0;
              for (int j = 0; j < blockDim.x; j++) {
                sum += per_thread_values[j];
              }
              sum_inv = 1.0 / sum;
            }

            local_barrier();

            for (int j = threadIdx.x; j < N; j += blockDim.x) {
              sm[j * sms1] = %(write_x)s(%(load_x)s(sm[j * sms1]) * sum_inv);
            }

            if (threadIdx.x == 0) {
              const %(type_y_idx)s y_idx = (int)y_idx_data[row * y_idxs0];
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
        params = [
            'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp'
            ]
        return [Kernel(code=sio.getvalue(), name=kname, params=params,
                       flags=flags, objvar=k_var)]

    def c_code(self, node, nodename, inp, out, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError('cuda only')
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
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            %(err_check)s
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
            %(sync)s
        }
        """ % locals(), file=sio)
        return sio.getvalue()

    def c_code_cache_version(self):
        return (12,)


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

    def get_params(self, node):
        return node.inputs[0].type.context

    def c_code_cache_version(self):
        return (12,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code(self, node, nodename, inp, out, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError("cuda only")
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
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            %(err_check)s
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
            %(sync)s
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
        sio = StringIO()
        print("""
        KERNEL void %(kname)s(
           const ga_size N, const ga_size K,
           const %(type_dnll)s* dnll, const ga_size offset_dnll,
           const ga_ssize dnll_s0,
           const %(type_sm)s* sm, const ga_size offset_sm,
           const ga_ssize sm_s0, const ga_ssize sm_s1,
           const %(type_y_idx)s* y_idx, const ga_size offset_y_idx,
           const ga_ssize y_idx_s0,
           %(type_dx)s* dx, const ga_size offset_dx,
           const ga_ssize dx_s0, const ga_ssize dx_s1)
        {
            dnll = (const %(type_dnll)s *)(((char *)dnll)+offset_dnll);
            sm = (const %(type_sm)s *)(((char *)sm)+offset_sm);
            y_idx = (const %(type_y_idx)s *)(((char *)y_idx)+offset_y_idx);
            dx = (%(type_dx)s *)(((char *)dx)+offset_dx);

            for (int i = blockIdx.x; i < N; i += gridDim.x)
            {
                %(wtype_dnll)s dnll_i = %(load_dnll)s(dnll[i * dnll_s0]);
                %(type_y_idx)s y_i = y_idx[i * y_idx_s0];

                for (int j = threadIdx.x; j < K; j += blockDim.x)
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
        params = [
            'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp'
            ]
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

    def get_params(self, node):
        return node.inputs[0].type.context

    def infer_shape(self, node, shape):
        return shape

    def c_code_cache_version(self):
        return (15,) + inline_softmax.code_version

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code(self, node, nodename, inp, out, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError("cuda only")
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
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            msg = "sync error";
            %(err_check)s
            """ % locals()
        else:
            sync = ""
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
            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)512), 1, 1};
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
              %(sync)s
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
        params = [
            'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp'
            ]
        kernels = []
        kname = "kSoftmax"
        k_var = "kSoftmax_" + nodename
        code = nvcc_kernel(
            kname,
            params=['const ga_size M', 'const ga_size N',
                    'const %s * x' % type_x, 'const ga_size offset_x',
                    'const ga_ssize sx0', 'const ga_ssize sx1',
                    '%s * sm' % type_sm, 'const ga_size offset_sm',
                    'const ga_ssize sm_s0', 'const ga_ssize sm_s1'],
            body=["extern __shared__ %s buf[]" % type_acc,
                  "%s * buf2 = buf + N" % type_acc,
                  "x = (const %s *)(((char *)x)+offset_x)" % type_x,
                  "sm = (%s *)(((char *)sm)+offset_sm)" % type_sm,
                  "for (int blockIDX = blockIdx.x; blockIDX < M;"
                  "     blockIDX += gridDim.x){",
                  "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                  "buf[tx] = %s(x[blockIDX * sx0 + tx * sx1])" % load_x,
                  "buf2[tx] = buf[tx]",
                  "}",
                  "__syncthreads()",
                  inline_softmax('N', 'buf', 'buf2', 'threadIdx.x',
                                 'blockDim.x', dtype=work_sm),
                  "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                  # This set all value correctly
                  "sm[blockIDX * sm_s0 + tx * sm_s1] = %s(buf[tx])" % write_sm,
                  "}",
                  "__syncthreads()",
                  "}",
                  ])
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        kname = "kSoftmax_fixed_shared"
        k_var = "kSoftmax_fixed_shared" + nodename
        code = nvcc_kernel(
            kname,
            params=['const ga_size M', 'const ga_size N',
                    'const %s * x' % type_x, 'const ga_size offset_x',
                    'const ga_ssize sx0', 'const ga_ssize sx1',
                    '%s * sm' % type_sm, 'const ga_size offset_sm',
                    'const ga_ssize sm_s0', 'const ga_ssize sm_s1'],
            body=["extern __shared__ %s buf[]" % type_acc,
                  "x = (const %s *)(((char *)x)+offset_x)" % type_x,
                  "sm = (%s *)(((char *)sm)+offset_sm)" % type_sm,
                  "for (int blockIDX = blockIdx.x; blockIDX < M;"
                  "     blockIDX += gridDim.x){",
                  "const %s *x_ptr = &x[blockIDX * sx0]" % type_x,
                  "%s *sm_ptr = &sm[blockIDX * sm_s0]" % type_sm,
                  inline_softmax_fixed_shared('N', 'buf', 'x_ptr', 'sx1',
                                              load_x,
                                              'sm_ptr', 'sm_s1', write_sm,
                                              'threadIdx.x', 'blockDim.x',
                                              dtype=work_sm),
                  "__syncthreads()",
                  "}",
                  ])
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

    def get_params(self, node):
        return node.inputs[0].type.context

    def infer_shape(self, node, shape):
        return [shape[0]]

    def c_code_cache_version(self):
        return (14,) + inline_softmax.code_version

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def c_code(self, node, nodename, inp, out, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError('cuda only')
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
        sync = ""
        if config.gpuarray.sync:
            sync = """
            err = GpuArray_sync(&%(z)s->ga);
            msg = "sync error";
            %(err_check)s
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
            size_t threads_per_block[3] = {std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)512), 1, 1};
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
              %(sync)s
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
        params = [
            'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp',
            gpuarray.GpuArray, 'uintp', 'intp', 'intp'
            ]
        kernels = []
        kname = "kSoftmaxWithBias"
        k_var = "kSoftmaxWithBias_" + nodename
        code = nvcc_kernel(
            kname,
            params=['const ga_size M', 'const ga_size N',
                    'const %s * x' % type_x, 'const ga_size offset_x',
                    'const ga_ssize sx0', 'const ga_ssize sx1',
                    'const %s * b' % type_b, 'const ga_size offset_b',
                    'const ga_ssize sb0',
                    '%s * sm' % type_sm, 'const ga_size offset_sm',
                    'const ga_ssize sm_s0', 'const ga_ssize sm_s1'],
            body=["extern __shared__ %s buf[]" % type_acc,
                  "%s * buf2 = buf + N" % type_acc,
                  "x = (const %s *)(((char *)x)+offset_x)" % type_x,
                  "b = (const %s *)(((char *)b)+offset_b)" % type_b,
                  "sm = (%s *)(((char *)sm)+offset_sm)" % type_sm,
                  "for (int blockIDX = blockIdx.x; blockIDX < M;"
                  "     blockIDX += gridDim.x){",
                  "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                  "buf[tx] = %s(x[blockIDX * sx0 + tx * sx1])" % load_x,
                  "buf[tx] += %s(b[tx * sb0])" % load_b,
                  "buf2[tx] = buf[tx]",
                  "}",
                  "__syncthreads()",
                  inline_softmax('N', 'buf', 'buf2',
                                 'threadIdx.x', 'blockDim.x', work_sm),
                  "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                  "sm[blockIDX * sm_s0 + tx * sm_s1] = %s(buf[tx])" % write_sm,
                  "}",
                  "__syncthreads()",
                  "}",
                  ])
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        kname = "kSoftmaxWithBias_fixed_shared"
        k_var = "kSoftmaxWithBias_fixed_shared" + nodename
        code = nvcc_kernel(
            kname,
            params=['const ga_size M', 'const ga_size N',
                    'const %s * x' % type_x, 'const ga_size offset_x',
                    'const ga_ssize sx0', 'const ga_ssize sx1',
                    'const %s * b' % type_b, 'const ga_size offset_b',
                    'const ga_ssize sb0',
                    '%s * sm' % type_sm, 'const ga_size offset_sm',
                    'const ga_ssize sm_s0', 'const ga_ssize sm_s1'],
            body=["extern __shared__ %s buf[]" % type_acc,
                  "x = (const %s *)(((char *)x)+offset_x)" % type_x,
                  "b = (const %s *)(((char *)b)+offset_b)" % type_b,
                  "sm = (%s *)(((char *)sm)+offset_sm)" % type_sm,
                  "for (int blockIDX = blockIdx.x; blockIDX < M;"
                  "     blockIDX += gridDim.x){",
                  "const %s *x_ptr = &x[blockIDX * sx0]" % type_x,
                  "%s *sm_ptr = &sm[blockIDX * sm_s0]" % type_sm,
                  inline_softmax_fixed_shared('N', 'buf', 'x_ptr', 'sx1',
                                              load_x,
                                              'sm_ptr', 'sm_s1', write_sm,
                                              'threadIdx.x', 'blockDim.x',
                                              'b', 'sb0', load_b, work_sm),
                  "__syncthreads()",
                  "}",
                  ])
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels

gpu_softmax_with_bias = GpuSoftmaxWithBias()
