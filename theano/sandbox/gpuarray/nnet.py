import numpy

from theano import Op, Apply, config
from theano.compat.six import StringIO
from theano.sandbox.gpuarray.comp import NVCC_compiler


try:
    import pygpu
    from pygpu import gpuarray, elemwise
except ImportError:
    pass

from theano.sandbox.gpuarray.basic_ops import as_gpuarray_variable
from theano.sandbox.gpuarray.type import GpuArrayType
from theano.sandbox.gpuarray.kernel_codegen import (nvcc_kernel,
                                                   inline_softmax,
                                                   inline_softmax_fixed_shared)



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
        nll = GpuArrayType(x.type.dtype,
                           y_idx.type.broadcastable)()
        sm = x.type()
        am = y_idx.type()
        return Apply(self, [x, b, y_idx], [nll, sm, am])

    def c_headers(self):
        return ['cuda.h', '<gpuarray/extension.h>', '<numpy_compat.h>']

    def c_support_code_apply(self, node, nodename):
        dtype_x = node.inputs[0].dtype
        dtype_b = node.inputs[1].dtype
        dtype_y_idx = node.inputs[2].dtype
        return """
        __global__ void k_xent_sm_1hot_bias_%(nodename)s(int M, int N,
            const npy_%(dtype_x)s* x_data, int xs0, int xs1,
            const npy_%(dtype_b)s* b, int bs0,
            const npy_%(dtype_y_idx)s* y_idx_data, int y_idxs0,
            npy_%(dtype_x)s* nll_data, int nlls0,
            npy_%(dtype_x)s* sm_data, int sms0, int sms1,
            npy_%(dtype_y_idx)s* am_data, int ams0)
        {
          for (int row = blockIdx.x; row < M; row += gridDim.x){

            const npy_%(dtype_x)s* x = x_data + xs0 * row;
            const npy_%(dtype_y_idx)s y_idx = y_idx_data[row * y_idxs0];
            npy_%(dtype_x)s* sm = sm_data + sms0 * row;

            npy_%(dtype_x)s sum = 0.0;
            int row_max_j = 0;
            npy_%(dtype_x)s row_max = x[0] + b[0];
            for (int j = 1; j < N; ++j)
            {
                npy_%(dtype_x)s row_ij = x[j*xs1] + b[j*bs0];
                //todo: store to shared memory
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }
            //compute the exp
            for (int j = 0; j < N; ++j)
            {
                npy_%(dtype_x)s row_ij = x[j*xs1] + b[j*bs0];
                npy_%(dtype_x)s sm_ij = exp(row_ij - row_max);
                sum += sm_ij;
                sm[j * sms1] = sm_ij;
            }
            npy_%(dtype_x)s sum_inv = 1.0 / sum;
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
        """ % locals()

    def c_init_code(self):
        return ['cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))gpuarray_get_extension("cuda_get_ptr");']

    def c_code(self, node, nodename, inp, out, sub):
        typecode_x = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
        typecode_b = pygpu.gpuarray.dtype_to_typecode(node.inputs[1].dtype)
        typecode_y_idx = pygpu.gpuarray.dtype_to_typecode(node.inputs[2].dtype)
        itemsize_x = numpy.dtype(node.inputs[0].dtype).itemsize
        itemsize_b = numpy.dtype(node.inputs[1].dtype).itemsize
        itemsize_y_idx = numpy.dtype(node.inputs[2].dtype).itemsize
        itemsize_nll = numpy.dtype(node.outputs[0].dtype).itemsize
        itemsize_sm = numpy.dtype(node.outputs[1].dtype).itemsize
        itemsize_am = numpy.dtype(node.outputs[2].dtype).itemsize
        x, b, y_idx = inp
        nll, sm, am = out
        dtype_x = node.inputs[0].dtype
        dtype_b = node.inputs[1].dtype
        dtype_y_idx = node.inputs[2].dtype
        dtype_nll = node.outputs[0].dtype
        dtype_sm = node.outputs[1].dtype
        dtype_am = node.outputs[2].dtype
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
                                %(typecode_x)s,
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
                                %(typecode_b)s,
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
                                %(typecode_y_idx)s,
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
            int n_blocks = PyGpuArray_DIMS(%(x)s)[0] < 256 ? PyGpuArray_DIMS(%(x)s)[0] : 256;
     //TODO: launch more threads per row and do parallel sum and max reductions
            int n_threads = 1;
            int n_shared_bytes = 0; //n_threads * sizeof(dtype);


            k_xent_sm_1hot_bias_%(nodename)s<<<n_blocks, n_threads, n_shared_bytes>>>(
                PyGpuArray_DIMS(%(x)s)[0],
                PyGpuArray_DIMS(%(x)s)[1],
                (npy_%(dtype_x)s*)(((char *)cuda_get_ptr(%(x)s->ga.data)) +
                                   %(x)s->ga.offset),
                PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s,
                (npy_%(dtype_b)s*)(((char *)cuda_get_ptr(%(b)s->ga.data)) +
                                   %(b)s->ga.offset),
                PyGpuArray_STRIDES(%(b)s)[0] / %(itemsize_b)s,
                (npy_%(dtype_y_idx)s*)(((char *)cuda_get_ptr(%(y_idx)s->ga.data)) +
                                   %(y_idx)s->ga.offset),
                PyGpuArray_STRIDES(%(y_idx)s)[0] / %(itemsize_y_idx)s,
                (npy_%(dtype_nll)s*)(((char *)cuda_get_ptr(%(nll)s->ga.data)) +
                                   %(nll)s->ga.offset),
                PyGpuArray_STRIDES(%(nll)s)[0] / %(itemsize_nll)s,
                (npy_%(dtype_sm)s*)(((char *)cuda_get_ptr(%(sm)s->ga.data)) +
                                   %(sm)s->ga.offset),
                PyGpuArray_STRIDES(%(sm)s)[0] / %(itemsize_sm)s,
                PyGpuArray_STRIDES(%(sm)s)[1] / %(itemsize_sm)s,
                (npy_%(dtype_am)s*)(((char *)cuda_get_ptr(%(am)s->ga.data)) +
                                   %(am)s->ga.offset),
                PyGpuArray_STRIDES(%(am)s)[0] / %(itemsize_am)s);
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
        return (5,)

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

    def make_node(self, dnll, sm, y_idx):
        dnll = as_gpuarray_variable(dnll)
        sm = as_gpuarray_variable(sm)
        y_idx = as_gpuarray_variable(y_idx)
        return Apply(self, [dnll, sm, y_idx], [sm.type()])

    def c_code_cache_version(self):
        #return ()
        return (6,)

    def c_headers(self):
        return ['cuda.h', '<gpuarray/extension.h>', '<numpy_compat.h>']

    def c_compiler(self):
        return NVCC_compiler

    def c_code(self, node, nodename, inp, out, sub):
        typecode_dx = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        itemsize_dnll = numpy.dtype(node.inputs[0].dtype).itemsize
        itemsize_sm = numpy.dtype(node.inputs[1].dtype).itemsize
        itemsize_y_idx = numpy.dtype(node.inputs[2].dtype).itemsize
        itemsize_dx = numpy.dtype(node.outputs[0].dtype).itemsize
        dtype_dnll = node.inputs[0].dtype
        dtype_sm = node.inputs[1].dtype
        dtype_y_idx = node.inputs[2].dtype
        dtype_dx = node.outputs[0].dtype
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
                                 %(typecode_dx)s,
                                 GA_C_ORDER,
                                 pygpu_default_context(), Py_None);
            if (!%(dx)s) {
                %(fail)s
            }
        }
        {
            int n_blocks = PyGpuArray_DIMS(%(dx)s)[0] < 256 ? PyGpuArray_DIMS(%(dx)s)[0] : 256;
            int n_threads = PyGpuArray_DIMS(%(dx)s)[1] < 256 ? PyGpuArray_DIMS(%(dx)s)[1] : 256;

            kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s
                <<<n_blocks, n_threads>>>(
                        PyGpuArray_DIMS(%(dx)s)[0],
                        PyGpuArray_DIMS(%(dx)s)[1],

                        (npy_%(dtype_dnll)s*)(((char *)cuda_get_ptr(%(dnll)s->ga.data)) +
                                           %(dnll)s->ga.offset),
                        PyGpuArray_STRIDES(%(dnll)s)[0] / %(itemsize_dnll)s,

                        (npy_%(dtype_sm)s*)(((char *)cuda_get_ptr(%(sm)s->ga.data)) +
                                           %(sm)s->ga.offset),
                        PyGpuArray_STRIDES(%(sm)s)[0] / %(itemsize_sm)s,
                        PyGpuArray_STRIDES(%(sm)s)[1] / %(itemsize_sm)s,

                        (npy_%(dtype_y_idx)s*)(((char *)cuda_get_ptr(%(y_idx)s->ga.data)) +
                                           %(y_idx)s->ga.offset),
                        PyGpuArray_STRIDES(%(y_idx)s)[0] / %(itemsize_y_idx)s,

                        (npy_%(dtype_dx)s*)(((char *)cuda_get_ptr(%(dx)s->ga.data)) +
                                           %(dx)s->ga.offset),
                        PyGpuArray_STRIDES(%(dx)s)[0] / %(itemsize_dx)s,
                        PyGpuArray_STRIDES(%(dx)s)[1] / %(itemsize_dx)s
                );
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
        dtype_dnll = node.inputs[0].dtype
        dtype_sm = node.inputs[1].dtype
        dtype_y_idx = node.inputs[2].dtype
        dtype_dx = node.outputs[0].dtype
        return """
        __global__ void kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s(
           int N, int K,
           const npy_%(dtype_dnll)s* dnll, const int dnll_s0,
           const npy_%(dtype_sm)s* sm, const int sm_s0, const int sm_s1,
           const npy_%(dtype_y_idx)s* y_idx, const int y_idx_s0,
           npy_%(dtype_dx)s* dx, const int dx_s0, const int dx_s1)
        {
            for (int i = blockIdx.x; i < N; i += gridDim.x)
            {
                npy_%(dtype_dnll)s dnll_i = dnll[i * dnll_s0];
                npy_%(dtype_y_idx)s y_i = y_idx[i * y_idx_s0];

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
        return ['cuda_get_ptr = (CUdeviceptr (*)(gpudata *g))gpuarray_get_extension("cuda_get_ptr");']

gpu_crossentropy_softmax_1hot_with_bias_dx = GpuCrossentropySoftmax1HotWithBiasDx()


class GpuSoftmax (Op):
    """
    Implement Softmax on the gpu.
    """
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        x = as_gpuarray_variable(x)
        return Apply(self, [x], [x.type()])

    def infer_shape(self, node, shape):
        return shape

    def c_code_cache_version(self):
        return (12,) + inline_softmax.code_version
        
    def c_headers(self):
        return ['cuda.h', '<gpuarray/extension.h>', '<numpy_compat.h>',
                '<gpuarray/ext_cuda.h>']

    def c_compiler(self):
        return NVCC_compiler
        
    def c_init_code(self):
        return ['setup_ext_cuda();']

    def c_code(self, node, nodename, inp, out, sub):
        dtype_x = node.inputs[0].dtype
        dtype_z = node.outputs[0].dtype
        itemsize_x = numpy.dtype(dtype_x).itemsize
        itemsize_z = numpy.dtype(dtype_z).itemsize
        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        x, = inp
        z, = out
        fail = sub['fail']
        if config.gpuarray.sync:
            cnda_thread_sync = "GpuArray_sync(&%(zz)s->ga);" % dict(zz=zz)
        else:
            cnda_thread_sync = ""  
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
                                %(typecode)s,
                                GA_C_ORDER,
                                pygpu_default_context(), Py_None);
            if (!%(z)s) {
                %(fail)s
            } 
        }
        {
            int n_blocks = std::min(PyGpuArray_DIMS(%(x)s)[0],
                                    (size_t)(32 * 1024));
//TODO, detect the maximum number of thread per block.
            int n_threads = std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)512);
            int n_shared_bytes = PyGpuArray_DIMS(%(x)s)[1] *
                                     2 * sizeof(npy_%(dtype_x)s);

            if (PyGpuArray_DIMS(%(x)s)[0] > 0)
            {
              //Those numbers are based on not too recent GPU
              //to make them compatible with more GPU.
              //TODO: read the information from the card.
              if(n_shared_bytes < (32 * 1024 - 500)){
                kSoftmax_%(nodename)s
                    <<<
                        n_blocks,
                        n_threads,
                        n_shared_bytes
                    >>>(
                            PyGpuArray_DIMS(%(x)s)[0],
                            PyGpuArray_DIMS(%(x)s)[1],

                            (npy_%(dtype_x)s*)(
                                    ((char *)cuda_get_ptr(%(x)s->ga.data)) +
                                    %(x)s->ga.offset),
                            PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                            PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s,

                            (npy_%(dtype_z)s*)(
                                    ((char *)cuda_get_ptr(%(z)s->ga.data)) +
                                    %(z)s->ga.offset),
                            PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                            PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s
                    );
              }else{
                kSoftmax_fixed_shared%(nodename)s
                    <<<
                        n_blocks,
                        n_threads,
                        n_threads * sizeof(npy_%(dtype_x)s)
                    >>>(
                            PyGpuArray_DIMS(%(x)s)[0],
                            PyGpuArray_DIMS(%(x)s)[1],

                            (npy_%(dtype_x)s*)(
                                    ((char *)cuda_get_ptr(%(x)s->ga.data)) +
                                    %(x)s->ga.offset),
                            PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                            PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s,

                            (npy_%(dtype_z)s*)(
                                    ((char *)cuda_get_ptr(%(z)s->ga.data)) +
                                    %(z)s->ga.offset),
                            PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                            PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s
                    );
              }
              %(cnda_thread_sync)s
              cudaError_t err = cudaGetLastError();
              if( cudaSuccess != err)
              {
                  PyErr_Format(PyExc_RuntimeError,
                               "Cuda error: %%s: %%s.\\n Used %%d blocks,"
                               " %%d threads %%d bytes of shared memory",
                               "kSoftmax[_fixed_shared]%(nodename)s",
                               cudaGetErrorString(err),
                               n_blocks, n_threads, n_shared_bytes);
                  %(fail)s;
              }
            }
        }
        assert(%(z)s);
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        dtype_x = node.inputs[0].dtype
        dtype_sm = node.outputs[0].dtype
        ret1 = nvcc_kernel("kSoftmax_%s" % nodename,
                params=['int M', 'int N',
                    'const npy_%(dtype_x)s * x', 'const int sx0', 'const int sx1',
                    'npy_%(dtype_sm)s * sm', 'const int sm_s0', 'const int sm_s1'],
                body=[
                    "extern __shared__ npy_%(dtype_sm)s buf[]",
                    "npy_%(dtype_sm)s * buf2 = buf + N",
                    "for (int blockIDX = blockIdx.x; blockIDX < M;"
                    "     blockIDX += gridDim.x){",
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                        "buf[tx] = x[blockIDX * sx0 + tx * sx1]",
                        "buf2[tx] = buf[tx]",
                      "}",
                      "__syncthreads()",
                      inline_softmax('N', 'buf', 'buf2',
                                     'threadIdx.x', 'blockDim.x', dtype_sm),
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                        # This set all value correctly
                        "sm[blockIDX * sm_s0 + tx * sm_s1] = buf[tx]",
                      "}",
                      "__syncthreads()",
                    "}",
                ])
        ret2 = nvcc_kernel("kSoftmax_fixed_shared%s" % nodename,
                params=['int M', 'int N',
                    'const npy_%(dtype_x)s * x', 'const int sx0', 'const int sx1',
                    'npy_%(dtype_sm)s * sm', 'const int sm_s0', 'const int sm_s1'],
                body=[
                    "extern __shared__ npy_%(dtype_sm)s buf[]",
                    "for (int blockIDX = blockIdx.x; blockIDX < M;"
                    "     blockIDX += gridDim.x){",
                      "const npy_%(dtype_x)s *x_ptr = &x[blockIDX * sx0]",
                      "npy_%(dtype_sm)s *sm_ptr = &sm[blockIDX * sm_s0]",
                      inline_softmax_fixed_shared('N', 'buf', 'x_ptr', 'sx1',
                                                  'sm_ptr', 'sm_s1',
                                                  'threadIdx.x', 'blockDim.x',
                                                  dtype=dtype_sm),
                      "__syncthreads()",
                    "}",
                    ])
        return (ret1 + "\n" + ret2) % locals()

gpu_softmax = GpuSoftmax()


class GpuSoftmaxWithBias (Op):
    """
    Implement SoftmaxWithBias on the gpu.
    """
    nin = 2
    nout = 1

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x, b):
        x = as_gpuarray_variable(x)
        b = as_gpuarray_variable(b)
        return Apply(self, [x, b], [x.type()])

    def infer_shape(self, node, shape):
        return  [shape[0]]
        
    def c_code_cache_version(self):
        return (11,) + inline_softmax.code_version
        
    def c_headers(self):
        return ['cuda.h', '<gpuarray/extension.h>', '<numpy_compat.h>',
                '<gpuarray/ext_cuda.h>']

    def c_compiler(self):
        return NVCC_compiler
        
    def c_init_code(self):
        return ['setup_ext_cuda();']
        
    def c_code(self, node, nodename, inp, out, sub):
        dtype_x = node.inputs[0].dtype
        dtype_b = node.inputs[1].dtype
        dtype_z = node.outputs[0].dtype
        itemsize_x = numpy.dtype(dtype_x).itemsize
        itemsize_b = numpy.dtype(dtype_b).itemsize
        itemsize_z = numpy.dtype(dtype_z).itemsize
        typecode = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        x, b = inp
        z, = out
        fail = sub['fail']
        if config.gpuarray.sync:
            cnda_thread_sync = "GpuArray_sync(&%(zz)s->ga);" % dict(zz=zz)
        else:
            cnda_thread_sync = "" 
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
                                %(typecode)s,
                                GA_C_ORDER,
                                pygpu_default_context(), Py_None);
            if (!%(z)s) {
                %(fail)s
            } 
        }
        {
            int n_blocks = std::min(PyGpuArray_DIMS(%(x)s)[0], (size_t)(32*1024));
//TODO, detect the maximum number of thread per block.
            int n_threads = std::min(PyGpuArray_DIMS(%(x)s)[1], (size_t)512);
            int n_shared_bytes = PyGpuArray_DIMS(%(x)s)[1] *
                                     2 * sizeof(npy_%(dtype_x)s);
            if (PyGpuArray_DIMS(%(x)s)[0] > 0)
            {
              if(n_shared_bytes < (32 * 1024 - 500)){
                kSoftmaxWithBias_%(nodename)s
                    <<<
                        n_blocks,
                        n_threads,
                        n_shared_bytes
                    >>>(
                        PyGpuArray_DIMS(%(x)s)[0],
                        PyGpuArray_DIMS(%(x)s)[1],

                        (npy_%(dtype_x)s*)(
                                    ((char *)cuda_get_ptr(%(x)s->ga.data)) +
                                    %(x)s->ga.offset),
                        PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                        PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s,

                        (npy_%(dtype_b)s*)(((char *)cuda_get_ptr(%(b)s->ga.data)) +
                                           %(b)s->ga.offset),
                        PyGpuArray_STRIDES(%(b)s)[0] / %(itemsize_b)s,

                        (npy_%(dtype_z)s*)(((char *)cuda_get_ptr(%(z)s->ga.data)) +
                                           %(z)s->ga.offset),
                        PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                        PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s
                    );
              }else{
                kSoftmaxWithBias_fixed_shared%(nodename)s
                    <<<
                        n_blocks,
                        n_threads,
                        n_threads * sizeof(npy_%(dtype_x)s)
                    >>>(
                        PyGpuArray_DIMS(%(x)s)[0],
                        PyGpuArray_DIMS(%(x)s)[1],

                        (npy_%(dtype_x)s*)(
                                    ((char *)cuda_get_ptr(%(x)s->ga.data)) +
                                    %(x)s->ga.offset),
                        PyGpuArray_STRIDES(%(x)s)[0] / %(itemsize_x)s,
                        PyGpuArray_STRIDES(%(x)s)[1] / %(itemsize_x)s,

                        (npy_%(dtype_b)s*)(
                                    ((char *)cuda_get_ptr(%(b)s->ga.data)) +
                                    %(b)s->ga.offset),
                        PyGpuArray_STRIDES(%(b)s)[0] / %(itemsize_b)s,

                        (npy_%(dtype_z)s*)(
                                    ((char *)cuda_get_ptr(%(z)s->ga.data)) +
                                    %(z)s->ga.offset),
                        PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                        PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s
                    );
              }
                %(cnda_thread_sync)s
                cudaError_t err = cudaGetLastError();
                if( cudaSuccess != err)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Cuda error: %%s: %%s.\\n",
                                 "kSoftmaxWithBias_%(nodename)s",
                                 cudaGetErrorString(err));
                    %(fail)s;
                }
            }
        }
        assert(%(z)s);
        """ % locals()

    def c_support_code_apply(self, node, nodename):
        dtype_x = node.inputs[0].dtype
        dtype_b = node.inputs[1].dtype
        dtype_sm = node.outputs[0].dtype
        ret1 = nvcc_kernel("kSoftmaxWithBias_%s" % nodename,
                params=['int M', 'int N',
                        'const npy_%(dtype_x)s * x', 'const int sx0', 'const int sx1',
                        'const npy_%(dtype_b)s * b', 'const int sb0',
                        'npy_%(dtype_sm)s * sm', 'const int sm_s0', 'const int sm_s1'],
                body=[
                    "extern __shared__ npy_%(dtype_sm)s buf[]",
                    "npy_%(dtype_sm)s * buf2 = buf + N",
                    "for (int blockIDX = blockIdx.x; blockIDX < M;"
                    "     blockIDX += gridDim.x){",
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                         "buf[tx] = x[blockIDX * sx0 + tx * sx1]",
                         "buf[tx] += b[tx * sb0]",
                         "buf2[tx] = buf[tx]",
                      "}",
                       "__syncthreads()",
                       inline_softmax('N', 'buf', 'buf2',
                                      'threadIdx.x', 'blockDim.x', dtype_sm),
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                         "sm[blockIDX * sm_s0 + tx * sm_s1] = buf[tx]",
                      "}",
                      "__syncthreads()",
                    "}",
                    ])
        ret2 = nvcc_kernel("kSoftmaxWithBias_fixed_shared%s" % nodename,
                           params=['int M', 'int N',
                                   'const npy_%(dtype_x)s * x',
                                   'const int sx0', 'const int sx1',
                                   'const npy_%(dtype_b)s * b', 'const int sb0',
                                   'npy_%(dtype_sm)s * sm',
                                   'const int sm_s0', 'const int sm_s1'],
                           body=[
                               "extern __shared__ npy_%(dtype_sm)s buf[]",
                               "for (int blockIDX = blockIdx.x; blockIDX < M;"
                               "     blockIDX += gridDim.x){",
                               "const npy_%(dtype_x)s *x_ptr = &x[blockIDX * sx0]",
                               "npy_%(dtype_sm)s *sm_ptr = &sm[blockIDX * sm_s0]",
                               inline_softmax_fixed_shared('N', 'buf',
                                                           'x_ptr', 'sx1',
                                                           'sm_ptr', 'sm_s1',
                                                           'threadIdx.x',
                                                           'blockDim.x',
                                                           'b', 'sb0',
                                                           dtype_sm),
                               "__syncthreads()",
                               "}",
                           ])
        return (ret1 + "\n" + ret2) % locals()

gpu_softmax_with_bias = GpuSoftmaxWithBias()
