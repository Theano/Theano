from theano import Op, Apply
from theano.compat.six import StringIO

from theano.sandbox.cuda.nvcc_compiler import NVCC_compiler

from theano.sandbox.cuda.kernel_codegen import (nvcc_kernel,
                                                inline_softmax,
                                                inline_softmax_fixed_shared)
try:
    import pygpu
    from pygpu import gpuarray, elemwise
except ImportError:
    pass

from theano.sandbox.gpuarray.basic_ops import as_gpuarray_variable


class GpuCrossentropySoftmaxArgmax1HotWithBias(Op):
    """
    Implement CrossentropySoftmaxArgmax1HotWithBias on the gpu.
    """
    nin = 3
    nout = 3

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, b, y_idx):
        #N.B. won't work when we don't cast y_idx to float anymore
        x = as_gpuarray_variable(x)
        b = as_gpuarray_variable(b)
        y_idx = as_gpuarray_variable(y_idx)
        nll = y_idx.type()
        sm = x.type()
        am = y_idx.type()
        return Apply(self, [x, b, y_idx], [nll, sm, am])

    def c_headers(self):
        return ['cuda.h', '<compyte/extension.h>', '<compyte/numpy_compat.h>']

    def c_support_code_apply(self, node):
        dtype0 = node.inputs[0].dtype
        dtype1 = node.inputs[1].dtype
        dtype2 = node.inputs[2].dtype
        return """
        __global__ void k_xent_sm_1hot_bias(int M, int N,
            const npy_%(dtype0)s* x_data, int xs0, int xs1,
            const npy_%(dtype1)s* b, int bs0,
            const npy_%(dtype2)s* y_idx_data, int y_idxs0,
            npy_%(dtype)s* nll_data, int nlls0,
            npy_%(dtype)s* sm_data, int sms0, int sms1,
            npy_%(dtype)s* am_data, int ams0)
        {
          for (int row = blockIdx.x; row < M; row += gridDim.x){

            const npy_%(dtype0)s* x = x_data + xs0 * row;
            const int y_idx = (int)y_idx_data[row * y_idxs0];
            npy_%(dtype0)s* sm = sm_data + sms0 * row;

            npy_%(dtype0)s sum = 0.0;
            int row_max_j = 0;
            npy_%(dtype0)s row_max = x[0] + b[0];
            for (int j = 1; j < N; ++j)
            {
                npy_%(dtype0)s row_ij = x[j*xs1] + b[j*bs0];
                //todo: store to shared memory
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }
            //compute the exp
            for (int j = 0; j < N; ++j)
            {
                npy_%(dtype0)s row_ij = x[j*xs1] + b[j*bs0];
                npy_%(dtype0)s sm_ij = exp(row_ij - row_max);
                sum += sm_ij;
                sm[j * sms1] = sm_ij;
            }
            npy_%(dtype0)s sum_inv = 1.0 / sum;
            for (int j = 0; j < N; ++j)
            {
                sm[j * sms1] *= sum_inv;
            }
            if ((y_idx >= N) || (y_idx < 0))
            {
                //TODO: set raise an error bit in a global var?
                nll_data[row*nlls0] = 0.0; // raise some suspicion at least...
            }
            else
            {
                nll_data[row*nlls0] = - x[y_idx*xs1]
                           - b[y_idx*bs0]
                           + row_max
                           + log(sum);
            }
            am_data[row*ams0] = row_max_j;
          }
        }

        CUdeviceptr (*cuda_get_ptr)(gpudata *g);
        """

    def c_init_code(self):
        return ['cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))compyte_get_extension("cuda_get_ptr");']

    def c_code(self, node, nodename, inp, out, sub):
        dtype = self.dtype
        typecode = pygpu.gpuarray.dtype_to_typecode(dtype)
        x, b, y_idx = inp
        nll, sm, am = out
        classname = self.__class__.__name__
        fail = sub['fail']
        sio = StringIO()
        print >> sio, """
        if (PyGpuArray_NDIM(%(y_idx)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "y_idx not 1d tensor");
            %(fail)s;
        }
        if (PyGpuArray_NDIM(%(x)s) != 2)
        {
            PyErr_SetString(PyExc_ValueError, "x not 2d tensor");
            %(fail)s;
        }
        if (PyGpuArray_NDIM(%(b)s) != 1)
        {
            PyErr_SetString(PyExc_ValueError, "b not 1d tensor");
            %(fail)s;
        }
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
        if ((NULL == %(nll)s) //initial condition
            || (PyGpuArray_DIMS(%(nll)s)[0] !=
                PyGpuArray_DIMS(%(y_idx)s)[0]))
        {
            Py_XDECREF(%(nll)s);
            %(nll)s = pygpu_empty(1, PyGpuArray_DIMS(%(y_idx)s),
                                %(typecode)s,
                                GA_C_ORDER,
                                pygpu_default_context(), Py_None);
            if (!%(nll)s) {
                %(fail)s
            }
        }
        if ((NULL == %(sm)s)
            || (PyGpuArray_DIMS(%(sm)s)[0] !=
                PyGpuArray_DIMS(%(x)s)[0])
            || (PyGpuArray_DIMS(%(sm)s)[1] !=
                PyGpuArray_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(sm)s);
            %(sm)s = pygpu_empty(2, PyGpuArray_DIMS(%(x)s),
                                %(typecode)s,
                                GA_C_ORDER,
                                pygpu_default_context(), Py_None);
            if(!%(sm)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc sm output");
                // no need to decref cnda_nll, the cleanup code should do it up
                %(fail)s;
            }
        }
        if ((NULL == %(am)s)
            || (PyGpuArray_DIMS(%(am)s)[0] !=
                PyGpuArray_DIMS(%(y_idx)s)[0]))
        {
            Py_XDECREF(%(am)s);
            %(am)s = pygpu_empty(1, PyGpuArray_DIMS(%(y_idx)s),
                                %(typecode)s,
                                GA_C_ORDER,
                                pygpu_default_context(), Py_None);
            if(!%(am)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc am output");
                // no need to decref nll and sm,
                // the cleanup code should do it up
                %(fail)s;
            }
        }
        {
            int n_blocks = std::min(PyGpuArray_DIMS(%(x)s)[0],
                                    256);
     //TODO: launch more threads per row and do parallel sum and max reductions
            int n_threads = 1;
            int n_shared_bytes = 0; //n_threads * sizeof(%(dtype)s);


            k_xent_sm_1hot_bias<<<n_blocks, n_threads, n_shared_bytes>>>(
                PyGpuArray_DIMS(%(x)s)[0],
                PyGpuArray_DIMS(%(x)s)[1],
                (dtype_%(x)s*)(((char *)cuda_get_ptr(%(x)s->ga.data)) +
                                   %(x)s->ga.offset);
                PyGpuArray_STRIDES(%(x)s)[0],
                PyGpuArray_STRIDES(%(x)s)[1],
                (dtype_%(b)s*)(((char *)cuda_get_ptr(%(b)s->ga.data)) +
                                   %(b)s->ga.offset);
                PyGpuArray_STRIDES(%(b)s)[0],
                (dtype_%(y_idx)s*)(((char *)cuda_get_ptr(%(y_idx)s->ga.data)) +
                                   %(y_idx)s->ga.offset);
                PyGpuArray_STRIDES(%(y_idx)s)[0],
                (dtype_%(nll)s*)(((char *)cuda_get_ptr(%(nll)s->ga.data)) +
                                   %(nll)s->ga.offset);
                PyGpuArray_STRIDES(%(nll)s)[0],
                (dtype_%(sm)s*)(((char *)cuda_get_ptr(%(sm)s->ga.data)) +
                                   %(sm)s->ga.offset);
                PyGpuArray_STRIDES(%(sm)s)[0],
                PyGpuArray_STRIDES(%(sm)s)[1],
                (dtype_%(am)s*)(((char *)cuda_get_ptr(%(am)s->ga.data)) +
                                   %(am)s->ga.offset);
                PyGpuArray_STRIDES(%(am)s)[0]);
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if (cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Cuda error: %(classname)s %(nodename)s: %%s.\\n"
                             "The kernel was launched with %%d threads,"
                             " %%d blocks and %%d shared memory\\n",
                             cudaGetErrorString(err),
                             n_threads, n_blocks, n_shared_bytes);
                // no need to decref output vars the cleanup code will do it
                %(fail)s;
            }
        }
        """ % locals()
        return sio.getvalue()

    def c_code_cache_version(self):
        #return ()
        return (4,)

    def c_compiler(self):
        return NVCC_compiler


gpu_crossentropy_softmax_argmax_1hot_with_bias = GpuCrossentropySoftmaxArgmax1HotWithBias()


class GpuCrossentropySoftmax1HotWithBiasDx(Op):
    """
    Implement CrossentropySoftmax1HotWithBiasDx on the gpu.
    """
    nin = 3
    nout = 1
    """Gradient wrt x of the CrossentropySoftmax1Hot Op"""
    def __init__(self, **kwargs):
        Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, dy, sm, y_idx):
        dy = as_gpuarray_variable(dy)
        sm = as_gpuarray_variable(sm)
        y_idx = as_gpuarray_variable(y_idx)
        return Apply(self, [dy, sm, y_idx], [sm.type()])

    def c_code_cache_version(self):
        #return ()
        return (6,)

    def c_headers(self):
        return ['cuda.h', '<compyte/extension.h>', '<compyte/numpy_compat.h>']

    def c_compiler(self):
        return NVCC_compiler

    def c_code(self, node, nodename, inp, out, sub):
        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        dnll, sm, y_idx = inp
        dx, = out
        fail = sub['fail']
        return """
        if ((PyGpuArray_NDIM(%(dnll)s) != 1)
            || (PyGpuArray_NDIM(%(sm)s) != 2)
            || (PyGpuArray_NDIM(%(y_idx)s) != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (PyGpuArray_DIMS(%(dnll)s)[0] !=
            PyGpuArray_DIMS(%(sm)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "dnll.shape[0] == %%i, but sm.shape[0] == %%i",
                         PyGpuArray_DIMS(%(dnll)s)[0],
                         PyGpuArray_DIMS(%(sm)s)[0]);
            %(fail)s;
        }
        if (PyGpuArray_DIMS(%(dnll)s)[0] !=
            PyGpuArray_DIMS(%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "dnll.shape[0] != y_idx.shape[0]");
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
                                 %(typecode)s,
                                 GA_C_ORDER,
                                 pygpu_default_context(), Py_None);
            if (!%(dx)s) {
                %(fail)s
            }
        }
        {
            int n_blocks = std::min(PyGpuArray_DIMS(%(dx)s)[0],
                                    256);
            int n_threads = std::min(PyGpuArray_DIMS(%(dx)s)[1],256);

            kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s
                <<<n_blocks, n_threads>>>(
                        PyGpuArray_DIMS(%(dx)s)[0],
                        PyGpuArray_DIMS(%(dx)s)[1],

                        (dtype_%(dnll)s*)(((char *)cuda_get_ptr(%(dnll)s->ga.data)) +
                                           %(dnll)s->ga.offset);
                        PyGpuArray_STRIDES(%(dnll)s)[0],

                        (dtype_%(sm)s*)(((char *)cuda_get_ptr(%(sm)s->ga.data)) +
                                           %(sm)s->ga.offset);
                        PyGpuArray_STRIDES(%(sm)s)[0],
                        PyGpuArray_STRIDES(%(sm)s)[1],

                        (dtype_%(y_idx)s*)(((char *)cuda_get_ptr(%(y_idx)s->ga.data)) +
                                           %(y_idx)s->ga.offset);
                        PyGpuArray_STRIDES(%(y_idx)s)[0],

                        (dtype_%(dx)s*)(((char *)cuda_get_ptr(%(dx)s->ga.data)) +
                                           %(dx)s->ga.offset);
                        PyGpuArray_STRIDES(%(dx)s)[0],
                        PyGpuArray_STRIDES(%(dx)s)[1]
                );
            CNDA_THREAD_SYNC;
            cudaError_t err = cudaGetLastError();
            if( cudaSuccess != err)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Cuda error: %%s: %%s.\\n"
                             "The kernel was launched with %%d threads and"
                             " %%d blocks\\n",
                             "kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s",
                             cudaGetErrorString(err), n_threads, n_blocks);
                %(fail)s;
            }
        }
        assert(%(dx)s);
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        dtype0 = node.inputs[0].dtype
        dtype1 = node.inputs[1].dtype
        dtype2 = node.inputs[2].dtype
        return """
        __global__ void kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s(
           int N, int K,
           const npy_%(dtype0)s* dnll, const int dnll_s0,
           const npy_%(dtype1)s* sm, const int sm_s0, const int sm_s1,
           const npy_%(dtype2)s* y_idx, const int y_idx_s0,
           npy_%(dtype1)s* dx, const int dx_s0, const int dx_s1)
        {
            for (int i = blockIdx.x; i < N; i += gridDim.x)
            {
                npy_%(dtype0)s dnll_i = dnll[i * dnll_s0];
                int y_i = (int)y_idx[i * y_idx_s0];

                for (int j = threadIdx.x; j < K; j += blockDim.x)
                {
                    if (y_i == j)
                    {
                        dx[i * dx_s0 + j * dx_s1] =
                            dnll_i * (sm[i * sm_s0 + j * sm_s1]-1.0);
                    }
                    else
                    {
                        dx[i * dx_s0 + j * dx_s1] =
                            dnll_i * sm[i * sm_s0 + j * sm_s1];
                    }
                    //dx[i * dx_s0 + j * dx_s1] =
                    //    dnll_i * sm[i * sm_s0 + j * sm_s1];
                    //dx[i*dx_s0+j*dx_s1] = 0;
                }
            }
        }

        CUdeviceptr (*cuda_get_ptr)(gpudata *g);
        """ % locals()

    def c_init_code(self):
        return ['cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))compyte_get_extension("cuda_get_ptr");']

gpu_crossentropy_softmax_1hot_with_bias_dx = GpuCrossentropySoftmax1HotWithBiasDx()
