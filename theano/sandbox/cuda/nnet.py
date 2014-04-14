import numpy
import theano

from theano import Op, Apply, tensor
from theano.tensor import as_tensor_variable
from theano.compat.six import StringIO

from theano.sandbox.cuda import GpuOp, CudaNdarray, as_cuda_ndarray_variable

from theano.sandbox.cuda.kernel_codegen import (nvcc_kernel,
                                                inline_softmax,
                                                inline_softmax_fixed_shared)


class GpuCrossentropySoftmaxArgmax1HotWithBias (GpuOp):
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
        nll = y_idx.type()
        sm = x.type()
        am = y_idx.type()
        return Apply(self, [x, b, y_idx], [nll, sm, am])

    def c_support_code(self):
        return """
        __global__ void k_xent_sm_1hot_bias(int M, int N,
            const float * x_data, int xs0, int xs1,
            const float * b, int bs0,
            const float * y_idx_data, int y_idxs0,
            float * nll_data, int nlls0,
            float * sm_data, int sms0, int sms1,
            float * am_data, int ams0)
        {
          for (int row = blockIdx.x; row < M; row += gridDim.x){

            const float * x = x_data + xs0 * row;
            const int y_idx = (int)y_idx_data[row * y_idxs0];
            float * sm = sm_data + sms0 * row;

            float sum = 0.0;
            int row_max_j = 0;
            float row_max = x[0] + b[0];
            for (int j = 1; j < N; ++j)
            {
                float row_ij = x[j*xs1] + b[j*bs0];
                //todo: store to shared memory
                row_max_j = (row_ij > row_max) ? j : row_max_j;
                row_max   = (row_ij > row_max) ? row_ij : row_max;
            }
            //compute the exp
            for (int j = 0; j < N; ++j)
            {
                float row_ij = x[j*xs1] + b[j*bs0];
                float sm_ij = exp(row_ij - row_max);
                sum += sm_ij;
                sm[j * sms1] = sm_ij;
            }
            float sum_inv = 1.0 / sum;
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

        """

    def c_code(self, node, nodename, inp, out, sub):
        x, b, y_idx = inp
        nll, sm, am = out
        classname = self.__class__.__name__
        fail = sub['fail']
        sio = StringIO()
        print >> sio, """
        if (%(y_idx)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "y_idx not 1d tensor");
            %(fail)s;
        }
        if (%(x)s->nd != 2)
        {
            PyErr_SetString(PyExc_ValueError, "x not 2d tensor");
            %(fail)s;
        }
        if (%(b)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "b not 1d tensor");
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(%(x)s)[0] !=
            CudaNdarray_HOST_DIMS(%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "dimension mismatch in x,y_idx arguments");
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(%(x)s)[1] != CudaNdarray_HOST_DIMS(%(b)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "dimension mismatch in x,b arguments");
            %(fail)s;
        }
        if ((NULL == %(nll)s) //initial condition
            || (CudaNdarray_HOST_DIMS(%(nll)s)[0] !=
                CudaNdarray_HOST_DIMS(%(y_idx)s)[0]))
        {
            Py_XDECREF(%(nll)s);
            %(nll)s = (CudaNdarray*)CudaNdarray_NewDims(1,
                CudaNdarray_HOST_DIMS(%(y_idx)s));
            if(!%(nll)s)
            {
                %(fail)s;
            }
        }
        if ((NULL == %(sm)s)
            || (CudaNdarray_HOST_DIMS(%(sm)s)[0] !=
                CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(sm)s)[1] !=
                CudaNdarray_HOST_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(sm)s);
            %(sm)s = (CudaNdarray*) CudaNdarray_NewDims(2,
                CudaNdarray_HOST_DIMS(%(x)s));
            if(!%(sm)s)
            {
                PyErr_SetString(PyExc_MemoryError,
                                "failed to alloc sm output");
                // no need to decref cnda_nll, the cleanup code should do it up
                %(fail)s;
            }
        }
        if ((NULL == %(am)s)
            || (CudaNdarray_HOST_DIMS(%(am)s)[0] !=
                CudaNdarray_HOST_DIMS(%(y_idx)s)[0]))
        {
            Py_XDECREF(%(am)s);
            %(am)s = (CudaNdarray*) CudaNdarray_NewDims(1,
                CudaNdarray_HOST_DIMS(%(y_idx)s));
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
            int n_blocks = std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                                    NUM_VECTOR_OP_BLOCKS);
     //TODO: launch more threads per row and do parallel sum and max reductions
            int n_threads = 1;
            int n_shared_bytes = 0; //n_threads * sizeof(float);

            k_xent_sm_1hot_bias<<<n_blocks, n_threads, n_shared_bytes>>>(
                CudaNdarray_HOST_DIMS(%(x)s)[0],
                CudaNdarray_HOST_DIMS(%(x)s)[1],
                CudaNdarray_DEV_DATA(%(x)s),
                CudaNdarray_HOST_STRIDES(%(x)s)[0],
                CudaNdarray_HOST_STRIDES(%(x)s)[1],
                CudaNdarray_DEV_DATA(%(b)s),
                CudaNdarray_HOST_STRIDES(%(b)s)[0],
                CudaNdarray_DEV_DATA(%(y_idx)s),
                CudaNdarray_HOST_STRIDES(%(y_idx)s)[0],
                CudaNdarray_DEV_DATA(%(nll)s),
                CudaNdarray_HOST_STRIDES(%(nll)s)[0],
                CudaNdarray_DEV_DATA(%(sm)s),
                CudaNdarray_HOST_STRIDES(%(sm)s)[0],
                CudaNdarray_HOST_STRIDES(%(sm)s)[1],
                CudaNdarray_DEV_DATA(%(am)s),
                CudaNdarray_HOST_STRIDES(%(am)s)[0]);
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

gpu_crossentropy_softmax_argmax_1hot_with_bias = GpuCrossentropySoftmaxArgmax1HotWithBias()


class GpuCrossentropySoftmax1HotWithBiasDx (GpuOp):
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
        return Apply(self, [dy, sm, y_idx], [sm.type()])

    def c_code_cache_version(self):
        #return ()
        return (6,)

    def c_code(self, node, nodename, inp, out, sub):
        dnll, sm, y_idx = inp
        dx, = out
        fail = sub['fail']
        return """
        if ((%(dnll)s->nd != 1)
            || (%(sm)s->nd != 2)
            || (%(y_idx)s->nd != 1))
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(%(dnll)s)[0] !=
            CudaNdarray_HOST_DIMS(%(sm)s)[0])
        {
            PyErr_Format(PyExc_ValueError,
                         "dnll.shape[0] == %%i, but sm.shape[0] == %%i",
                         CudaNdarray_HOST_DIMS(%(dnll)s)[0],
                         CudaNdarray_HOST_DIMS(%(sm)s)[0]);
            %(fail)s;
        }
        if (CudaNdarray_HOST_DIMS(%(dnll)s)[0] !=
            CudaNdarray_HOST_DIMS(%(y_idx)s)[0])
        {
            PyErr_SetString(PyExc_ValueError,
                            "dnll.shape[0] != y_idx.shape[0]");
            %(fail)s;
        }
        if ((NULL == %(dx)s)
            || (CudaNdarray_HOST_DIMS(%(dx)s)[0] !=
                CudaNdarray_HOST_DIMS(%(sm)s)[0])
            || (CudaNdarray_HOST_DIMS(%(dx)s)[1] !=
                CudaNdarray_HOST_DIMS(%(sm)s)[1]))
        {
            Py_XDECREF(%(dx)s);
            %(dx)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(dx)s)
                || CudaNdarray_alloc_contiguous(%(dx)s, 2,
                                                CudaNdarray_HOST_DIMS(%(sm)s)))
            {
                Py_XDECREF(%(dx)s);
                %(dx)s = NULL;
                %(fail)s;
            }
        }
        {
            int n_blocks = std::min(CudaNdarray_HOST_DIMS(%(dx)s)[0],
                                    NUM_VECTOR_OP_BLOCKS);
            int n_threads = std::min(CudaNdarray_HOST_DIMS(%(dx)s)[1],256);

            kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s
                <<<n_blocks, n_threads>>>(
                        CudaNdarray_HOST_DIMS(%(dx)s)[0],
                        CudaNdarray_HOST_DIMS(%(dx)s)[1],

                        CudaNdarray_DEV_DATA(%(dnll)s),
                        CudaNdarray_HOST_STRIDES(%(dnll)s)[0],

                        CudaNdarray_DEV_DATA(%(sm)s),
                        CudaNdarray_HOST_STRIDES(%(sm)s)[0],
                        CudaNdarray_HOST_STRIDES(%(sm)s)[1],

                        CudaNdarray_DEV_DATA(%(y_idx)s),
                        CudaNdarray_HOST_STRIDES(%(y_idx)s)[0],

                        CudaNdarray_DEV_DATA(%(dx)s),
                        CudaNdarray_HOST_STRIDES(%(dx)s)[0],
                        CudaNdarray_HOST_STRIDES(%(dx)s)[1]
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
        return """
        __global__ void kCrossEntropySoftmax1HotWithBiasDx_%(nodename)s(
           int N, int K,
           const float * dnll, const int dnll_s0,
           const float * sm, const int sm_s0, const int sm_s1,
           const float * y_idx, const int y_idx_s0,
           float * dx, const int dx_s0, const int dx_s1)
        {
            for (int i = blockIdx.x; i < N; i += gridDim.x)
            {
                float dnll_i = dnll[i * dnll_s0];
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
        """ % locals()

gpu_crossentropy_softmax_1hot_with_bias_dx = GpuCrossentropySoftmax1HotWithBiasDx()


class GpuSoftmax (GpuOp):
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
        return Apply(self, [x], [x.type()])

    def infer_shape(self, node, shape):
        return shape

    def c_code_cache_version(self):
        return (9,) + inline_softmax.code_version

    def c_code(self, node, nodename, inp, out, sub):
        x, = inp
        z, = out
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2)
        {
            PyErr_SetString(PyExc_ValueError, "rank error");
            %(fail)s;
        }
        if ((NULL == %(z)s) ||
            (CudaNdarray_HOST_DIMS(%(z)s)[0] !=
             CudaNdarray_HOST_DIMS(%(x)s)[0]) ||
            (CudaNdarray_HOST_DIMS(%(z)s)[1] !=
             CudaNdarray_HOST_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s)
                || CudaNdarray_alloc_contiguous(%(z)s, 2,
                                                CudaNdarray_HOST_DIMS(%(x)s)))
            {
                Py_XDECREF(%(z)s);
                %(z)s = NULL;
                %(fail)s;
            }
        }
        {
            int n_blocks = std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],
                                    32 * 1024);
//TODO, detect the maximum number of thread per block.
            int n_threads = std::min(CudaNdarray_HOST_DIMS(%(x)s)[1], 512);
            int n_shared_bytes = CudaNdarray_HOST_DIMS(%(x)s)[1] *
                                     2 * sizeof(float);

            if (CudaNdarray_HOST_DIMS(%(x)s)[0] > 0)
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
                            CudaNdarray_HOST_DIMS(%(x)s)[0],
                            CudaNdarray_HOST_DIMS(%(x)s)[1],

                            CudaNdarray_DEV_DATA(%(x)s),
                            CudaNdarray_HOST_STRIDES(%(x)s)[0],
                            CudaNdarray_HOST_STRIDES(%(x)s)[1],

                            CudaNdarray_DEV_DATA(%(z)s),
                            CudaNdarray_HOST_STRIDES(%(z)s)[0],
                            CudaNdarray_HOST_STRIDES(%(z)s)[1]
                    );
              }else{
                kSoftmax_fixed_shared%(nodename)s
                    <<<
                        n_blocks,
                        n_threads,
                        n_threads * sizeof(float)
                    >>>(
                            CudaNdarray_HOST_DIMS(%(x)s)[0],
                            CudaNdarray_HOST_DIMS(%(x)s)[1],

                            CudaNdarray_DEV_DATA(%(x)s),
                            CudaNdarray_HOST_STRIDES(%(x)s)[0],
                            CudaNdarray_HOST_STRIDES(%(x)s)[1],

                            CudaNdarray_DEV_DATA(%(z)s),
                            CudaNdarray_HOST_STRIDES(%(z)s)[0],
                            CudaNdarray_HOST_STRIDES(%(z)s)[1]
                    );
              }
              CNDA_THREAD_SYNC;
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
        ret1 = nvcc_kernel("kSoftmax_%s" % nodename,
                params=['int M', 'int N',
                    'const float * x', 'const int sx0', 'const int sx1',
                    'float * sm', 'const int sm_s0', 'const int sm_s1'],
                body=[
                    "extern __shared__ float buf[]",
                    "float * buf2 = buf + N",
                    "for (int blockIDX = blockIdx.x; blockIDX < M;"
                    "     blockIDX += gridDim.x){",
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                        "buf[tx] = x[blockIDX * sx0 + tx * sx1]",
                        "buf2[tx] = buf[tx]",
                      "}",
                      "__syncthreads()",
                      inline_softmax('N', 'buf', 'buf2',
                                     'threadIdx.x', 'blockDim.x'),
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                        # This set all value correctly
                        "sm[blockIDX * sm_s0 + tx * sm_s1] = buf[tx]",
                      "}",
                      "__syncthreads()",
                    "}",
                ])
        ret2 = nvcc_kernel("kSoftmax_fixed_shared%s" % nodename,
                params=['int M', 'int N',
                    'const float * x', 'const int sx0', 'const int sx1',
                    'float * sm', 'const int sm_s0', 'const int sm_s1'],
                body=[
                    "extern __shared__ float buf[]",
                    "for (int blockIDX = blockIdx.x; blockIDX < M;"
                    "     blockIDX += gridDim.x){",
                      "const float *x_ptr = &x[blockIDX * sx0]",
                      "float *sm_ptr = &sm[blockIDX * sm_s0]",
                      inline_softmax_fixed_shared('N', 'buf', 'x_ptr', 'sx1',
                                                  'sm_ptr', 'sm_s1',
                                                  'threadIdx.x', 'blockDim.x'),
                      "__syncthreads()",
                    "}",
                    ])
        return ret1 + "\n" + ret2

gpu_softmax = GpuSoftmax()


class GpuSoftmaxWithBias (GpuOp):
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
        return Apply(self, [x, b], [x.type()])

    def infer_shape(self, node, shape):
        return  [shape[0]]

    def c_code_cache_version(self):
        #return ()
        return (8,) + inline_softmax.code_version

    def c_code(self, node, nodename, inp, out, sub):
        x, b = inp
        z, = out
        fail = sub['fail']
        return """
        if (%(x)s->nd != 2)
        {
            PyErr_SetString(PyExc_ValueError, "rank error input");
            %(fail)s;
        }
        if (%(b)s->nd != 1)
        {
            PyErr_SetString(PyExc_ValueError, "rank error for the bias");
            %(fail)s;
        }
        if ((CudaNdarray_HOST_DIMS(%(x)s)[1] !=
            CudaNdarray_HOST_DIMS(%(b)s)[0]))
        {
            PyErr_Format(PyExc_ValueError,
                         "number of columns in x (%%ld)"
                         " does not match length of b (%%ld)",
                         (long int)CudaNdarray_HOST_DIMS(%(x)s)[1],
                         (long int)CudaNdarray_HOST_DIMS(%(b)s)[0]);
            %(fail)s;
        }
        if ((NULL == %(z)s)
            || (CudaNdarray_HOST_DIMS(%(z)s)[0] !=
                CudaNdarray_HOST_DIMS(%(x)s)[0])
            || (CudaNdarray_HOST_DIMS(%(z)s)[1] !=
                CudaNdarray_HOST_DIMS(%(x)s)[1]))
        {
            Py_XDECREF(%(z)s);
            %(z)s = (CudaNdarray*)CudaNdarray_New();
            if ((NULL == %(z)s)
                || CudaNdarray_alloc_contiguous(%(z)s, 2,
                       CudaNdarray_HOST_DIMS(%(x)s)))
            {
                Py_XDECREF(%(z)s);
                %(z)s = NULL;
                %(fail)s;
            }
        }
        {
            int n_blocks = std::min(CudaNdarray_HOST_DIMS(%(x)s)[0],32*1024);
//TODO, detect the maximum number of thread per block.
            int n_threads = std::min(CudaNdarray_HOST_DIMS(%(x)s)[1], 512);
            int n_shared_bytes = CudaNdarray_HOST_DIMS(%(x)s)[1] *
                                     2 * sizeof(float);
            if (CudaNdarray_HOST_DIMS(%(x)s)[0] > 0)
            {
              if(n_shared_bytes < (32 * 1024 - 500)){
                kSoftmaxWithBias_%(nodename)s
                    <<<
                        n_blocks,
                        n_threads,
                        n_shared_bytes
                    >>>(
                        CudaNdarray_HOST_DIMS(%(x)s)[0],
                        CudaNdarray_HOST_DIMS(%(x)s)[1],

                        CudaNdarray_DEV_DATA(%(x)s),
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],

                        CudaNdarray_DEV_DATA(%(b)s),
                        CudaNdarray_HOST_STRIDES(%(b)s)[0],

                        CudaNdarray_DEV_DATA(%(z)s),
                        CudaNdarray_HOST_STRIDES(%(z)s)[0],
                        CudaNdarray_HOST_STRIDES(%(z)s)[1]
                    );
              }else{
                kSoftmaxWithBias_fixed_shared%(nodename)s
                    <<<
                        n_blocks,
                        n_threads,
                        n_threads * sizeof(float)
                    >>>(
                        CudaNdarray_HOST_DIMS(%(x)s)[0],
                        CudaNdarray_HOST_DIMS(%(x)s)[1],

                        CudaNdarray_DEV_DATA(%(x)s),
                        CudaNdarray_HOST_STRIDES(%(x)s)[0],
                        CudaNdarray_HOST_STRIDES(%(x)s)[1],

                        CudaNdarray_DEV_DATA(%(b)s),
                        CudaNdarray_HOST_STRIDES(%(b)s)[0],

                        CudaNdarray_DEV_DATA(%(z)s),
                        CudaNdarray_HOST_STRIDES(%(z)s)[0],
                        CudaNdarray_HOST_STRIDES(%(z)s)[1]
                    );
              }
                CNDA_THREAD_SYNC;
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
        ret1 = nvcc_kernel("kSoftmaxWithBias_%s" % nodename,
                params=['int M', 'int N',
                        'const float * x', 'const int sx0', 'const int sx1',
                        'const float * b', 'const int sb0',
                        'float * sm', 'const int sm_s0', 'const int sm_s1'],
                body=[
                    "extern __shared__ float buf[]",
                    "float * buf2 = buf + N",
                    "for (int blockIDX = blockIdx.x; blockIDX < M;"
                    "     blockIDX += gridDim.x){",
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                         "buf[tx] = x[blockIDX * sx0 + tx * sx1]",
                         "buf[tx] += b[tx * sb0]",
                         "buf2[tx] = buf[tx]",
                      "}",
                       "__syncthreads()",
                       inline_softmax('N', 'buf', 'buf2',
                                      'threadIdx.x', 'blockDim.x'),
                      "for (int tx = threadIdx.x; tx< N; tx += blockDim.x){",
                         "sm[blockIDX * sm_s0 + tx * sm_s1] = buf[tx]",
                      "}",
                      "__syncthreads()",
                    "}",
                    ])
        ret2 = nvcc_kernel("kSoftmaxWithBias_fixed_shared%s" % nodename,
                           params=['int M', 'int N',
                                   'const float * x',
                                   'const int sx0', 'const int sx1',
                                   'const float * b', 'const int sb0',
                                   'float * sm',
                                   'const int sm_s0', 'const int sm_s1'],
                           body=[
                               "extern __shared__ float buf[]",
                               "for (int blockIDX = blockIdx.x; blockIDX < M;"
                               "     blockIDX += gridDim.x){",
                               "const float *x_ptr = &x[blockIDX * sx0]",
                               "float *sm_ptr = &sm[blockIDX * sm_s0]",
                               inline_softmax_fixed_shared('N', 'buf',
                                                           'x_ptr', 'sx1',
                                                           'sm_ptr', 'sm_s1',
                                                           'threadIdx.x',
                                                           'blockDim.x',
                                                           'b', 'sb0'),
                               "__syncthreads()",
                               "}",
                           ])
        return ret1 + "\n" + ret2

gpu_softmax_with_bias = GpuSoftmaxWithBias()


class GpuGroupDot(GpuOp):
    def __init__(self, n_groups):
        self.n_groups = n_groups

    def __eq__(self, other):
        return type(self) == type(other) and self.n_groups == other.n_groups

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n_groups)

    def __str__(self):
        return 'GpuGroupDot{%d}' % (self.n_groups)

    def make_node(self, vec, mat, bias, index):
        vec = as_cuda_ndarray_variable(vec)
        mat = as_cuda_ndarray_variable(mat)
        bias = as_cuda_ndarray_variable(bias)
        index = as_tensor_variable(index)

        assert vec.ndim == 2
        assert mat.ndim == 3
        assert bias.ndim == 2
        assert index.ndim == 1
        assert 'int' in index.dtype
        return theano.gof.Apply(self, [vec, mat, bias, index], [vec.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        shared = theano.sandbox.cuda.float32_shared_constructor

        self.W = shared(numpy.zeros((2, 2), dtype=node.inputs[1].dtype))
        self.b = shared(numpy.zeros((2,), dtype=node.inputs[2].dtype))
        self.h = shared(numpy.zeros((2), dtype=node.inputs[0].dtype))
        self.out = shared(numpy.zeros((2), dtype=node.outputs[0].dtype))

        out = tensor.dot(self.h, self.W) + self.b
        updates = [(self.out, out)]
        self.step = theano.function([], [], name='GpuGroupDotStep',
                                    updates=updates)

        return super(GpuGroupDot, self).make_thunk(node, storage_map,
                                                   compute_map, no_recycling)

    def perform(self, node, ins, _outs):
        state_below, matrix, biases, groups = ins
        if not (_outs[0][0] and _outs[0][0].shape == state_below.shape):
            nw_shape = (state_below.shape[0], biases.shape[1])
            _outs[0][0] = CudaNdarray.zeros(nw_shape)
        for pos in xrange(self.n_groups):
            self.W.set_value(matrix[pos], borrow=True)
            self.b.set_value(biases[pos], borrow=True)
            for jdx in xrange(groups.shape[0]):
                if groups[jdx] == pos:
                    self.h.set_value(state_below[jdx], borrow=True)
                    self.step()
                    _outs[0][0][jdx] = self.out.get_value(borrow=True,
                                                          return_internal_type=True)

    def c_code(self, node, name, inputs, outputs, sub):
        state_below, matrix, biases, groups = inputs
        out0, = outputs
        fail = sub['fail']
        groups_ctype = node.inputs[3].dtype + '_t'
        n_groups = self.n_groups
        return """
        if (%(out0)s == NULL ||
            CudaNdarray_HOST_DIMS(%(out0)s)[0] !=
               CudaNdarray_HOST_DIMS(%(state_below)s)[0] ||
            CudaNdarray_HOST_DIMS(%(out0)s)[1] !=
               CudaNdarray_HOST_DIMS(%(state_below)s)[1]) {
            Py_XDECREF(%(out0)s);
            int dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(state_below)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(biases)s)[1];
            // I know it seems from the rest of the code that this could be _NewDims()
            // but the code gives wrong results if we do that.
            // TODO: find out why.
            %(out0)s = (CudaNdarray *)CudaNdarray_ZEROS(2, dims);
            if (%(out0)s == NULL) { %(fail)s; }
        }
        {
        int fail = 0;
        npy_intp ng_shp = 0;
        CudaNdarray *proxy_W = (CudaNdarray *)CudaNdarray_new_nd(2);
        CudaNdarray *proxy_h = (CudaNdarray *)CudaNdarray_new_nd(1);
        CudaNdarray *proxy_o = (CudaNdarray *)CudaNdarray_new_nd(1);
        CudaNdarray *tmp = (CudaNdarray *)CudaNdarray_new_nd(1);
        if (!proxy_W || !proxy_h || !proxy_o || !tmp) { fail = 1; goto %(name)s_fail; }
        CudaNdarray_set_dim(proxy_W, 0, CudaNdarray_HOST_DIMS(%(matrix)s)[2]);
        CudaNdarray_set_dim(proxy_W, 1, CudaNdarray_HOST_DIMS(%(matrix)s)[1]);
        CudaNdarray_set_dim(proxy_h, 0, CudaNdarray_HOST_DIMS(%(state_below)s)[1]);
        CudaNdarray_set_dim(proxy_o, 0, CudaNdarray_HOST_DIMS(%(out0)s)[1]);
        CudaNdarray_set_dim(tmp, 0, CudaNdarray_HOST_DIMS(%(biases)s)[1]);
        CudaNdarray_set_stride(proxy_W, 0, CudaNdarray_HOST_STRIDES(%(matrix)s)[2]);
        CudaNdarray_set_stride(proxy_W, 1, CudaNdarray_HOST_STRIDES(%(matrix)s)[1]);
        CudaNdarray_set_stride(proxy_h, 0, CudaNdarray_HOST_STRIDES(%(state_below)s)[1]);
        CudaNdarray_set_stride(proxy_o, 0, CudaNdarray_HOST_STRIDES(%(out0)s)[1]);
        CudaNdarray_set_stride(tmp, 0, CudaNdarray_HOST_STRIDES(%(biases)s)[1]);
        ng_shp = PyArray_DIMS(%(groups)s)[0];
        for (npy_intp pos = 0; pos < %(n_groups)s; pos++) {
            CudaNdarray_set_device_data(proxy_W, CudaNdarray_DEV_DATA(%(matrix)s)+(pos * CudaNdarray_HOST_STRIDES(%(matrix)s)[0]), %(matrix)s);
            CudaNdarray_set_device_data(tmp, CudaNdarray_DEV_DATA(%(biases)s)+(pos * CudaNdarray_HOST_STRIDES(%(biases)s)[0]), %(biases)s);
            for (npy_intp jdx = 0; jdx < ng_shp; jdx++) {
               %(groups_ctype)s *p = (%(groups_ctype)s *)PyArray_GETPTR1(%(groups)s, jdx);
               if (*p == pos) {
                   CudaNdarray_set_device_data(proxy_o, CudaNdarray_DEV_DATA(%(out0)s)+(jdx * CudaNdarray_HOST_STRIDES(%(out0)s)[0]), %(out0)s);
                   CudaNdarray_set_device_data(proxy_h, CudaNdarray_DEV_DATA(%(state_below)s)+(jdx * CudaNdarray_HOST_STRIDES(%(state_below)s)[0]), %(state_below)s);
                   if (CudaNdarray_CopyFromCudaNdarray(proxy_o, tmp))
                       { fail = 1; goto %(name)s_fail; }
                   if (CudaNdarray_sgemv(1.0f, proxy_W, proxy_h,
                                         1.0f, proxy_o)) { fail = 1; goto %(name)s_fail; }
               }
            }
        }
  %(name)s_fail:
    Py_XDECREF(proxy_W);
    Py_XDECREF(proxy_h);
    Py_XDECREF(proxy_o);
    Py_XDECREF(tmp);
    if (fail)
      %(fail)s;
        }
        """ % locals()

    def grad(self, inputs, grads):
        state_below, matrix, biases, groups = inputs
        gout, = grads
        rval = GpuGroupDotGrad(n_groups=self.n_groups)(state_below,
                                                       matrix, biases,
                                                       groups, gout)
        return rval + [theano.gradient.grad_undefined(self, 3, groups)]


class GpuGroupDotGrad(GpuOp):
    def __init__(self, n_groups):
        self.n_groups = n_groups

    def __eq__(self, other):
        return type(self) == type(other) and self.n_groups == other.n_groups

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n_groups)

    def __str__(self):
        return 'GpuGroupDotGrad{%d}' % (self.n_groups)

    def make_node(self, vec, mat, bias, index, grad_on_out):
        vec = as_cuda_ndarray_variable(vec)
        mat = as_cuda_ndarray_variable(mat)
        bias = as_cuda_ndarray_variable(bias)
        index = as_tensor_variable(index)
        grad_on_out = as_cuda_ndarray_variable(grad_on_out)

        assert vec.ndim == 2
        assert mat.ndim == 3
        assert bias.ndim == 2
        assert index.ndim == 1
        assert 'int' in index.dtype
        return theano.gof.Apply(self,
                                [vec, mat, bias, index, grad_on_out],
                                [vec.type(), mat.type(), bias.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        shared = theano.sandbox.cuda.float32_shared_constructor

        self.W = shared(numpy.zeros((2, 3), dtype=node.inputs[1].dtype))
        self.h = shared(numpy.zeros((2,), dtype=node.inputs[0].dtype))
        self.grad_on_out = shared(numpy.zeros((3,),
                                              dtype=node.inputs[3].dtype))
        self.gW = shared(numpy.zeros((2, 3), dtype=node.outputs[1].dtype))
        self.gh = shared(numpy.zeros((2,), dtype=node.outputs[0].dtype))

        gW = tensor.outer(self.h, self.grad_on_out)
        gh = tensor.dot(self.grad_on_out, self.W.T)

        updates = [(self.gW, gW), (self.gh, gh)]
        self.step = theano.function([], [], updates=updates,
                                    name='GpuGroupDotGradStep')

        return super(GpuGroupDotGrad, self).make_thunk(node, storage_map,
                                                       compute_map,
                                                       no_recycling)

    def perform(self, node, ins, _outs):
        state_below, matrix, biases, groups, grad_on_out = ins

        if not (_outs[0][0] and _outs[0][0].shape == state_below.shape):
            _outs[0][0] = CudaNdarray.zeros(state_below.shape)

        if not (_outs[1][0] and _outs[1][0].shape == matrix.shape):
            _outs[1][0] = CudaNdarray.zeros(matrix.shape)

        if not (_outs[2][0] and _outs[2][0].shape == biases.shape):
            _outs[2][0] = CudaNdarray.zeros(biases.shape)

        for pos in xrange(self.n_groups):
            mask = groups == pos
            if mask.sum() != 0:
                self.W.set_value(matrix[pos], borrow=True)
                for jdx in xrange(groups.shape[0]):
                    if groups[jdx] == pos:
                        self.h.set_value(state_below[jdx], borrow=True)
                        self.grad_on_out.set_value(grad_on_out[jdx],
                                                   borrow=True)
                        self.step()
                        _outs[0][0][jdx] = self.gh.get_value(borrow=True,
                                                             return_internal_type=True)
                        _outs[1][0][pos] += self.gW.get_value(borrow=True,
                                                              return_internal_type=True)
                        _outs[2][0][pos] += grad_on_out[jdx]

    def c_code(self, node, name, inputs, outputs, sub):
        state_below, matrix, biases, groups, grad_on_out = inputs
        out0, out1, out2 = outputs
        fail = sub['fail']
        n_groups = self.n_groups
        groups_ctype = node.inputs[3].dtype + '_t'
        return """
        if (%(out0)s == NULL || CudaNdarray_NDIM(%(out0)s) != 2 ||
            CudaNdarray_HOST_DIMS(%(out0)s)[0] !=
               CudaNdarray_HOST_DIMS(%(state_below)s)[0] ||
            CudaNdarray_HOST_DIMS(%(out0)s)[1] !=
               CudaNdarray_HOST_DIMS(%(state_below)s)[1]) {
            Py_XDECREF(%(out0)s);
            int dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(state_below)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(state_below)s)[1];
            %(out0)s = (CudaNdarray *)CudaNdarray_NewDims(2, dims);
            if (%(out0)s == NULL) { %(fail)s; }
        }
        if (%(out1)s == NULL || CudaNdarray_NDIM(%(out1)s) != 3 ||
            CudaNdarray_HOST_DIMS(%(out1)s)[0] !=
               CudaNdarray_HOST_DIMS(%(matrix)s)[0] ||
            CudaNdarray_HOST_DIMS(%(out1)s)[1] !=
               CudaNdarray_HOST_DIMS(%(matrix)s)[1] ||
            CudaNdarray_HOST_DIMS(%(out1)s)[2] !=
               CudaNdarray_HOST_DIMS(%(matrix)s)[2]) {
            Py_XDECREF(%(out1)s);
            int dims[3];
            dims[0] = CudaNdarray_HOST_DIMS(%(matrix)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(matrix)s)[1];
            dims[2] = CudaNdarray_HOST_DIMS(%(matrix)s)[2];
            %(out1)s = (CudaNdarray *)CudaNdarray_ZEROS(3, dims);
            if (%(out1)s == NULL) { %(fail)s; }
        } else {
          cudaError_t err = cudaMemset(CudaNdarray_DEV_DATA(%(out1)s), 0,
                                       CudaNdarray_SIZE(%(out1)s) * sizeof(real));
          if (err) {
            // Clear error flag
            cudaGetLastError();
            PyErr_SetString(PyExc_RuntimeError,
                            "GpuGroupDotGrad: cudaMemset failed for output 1");
          %(fail)s;
          }
        }
        if (%(out2)s == NULL || CudaNdarray_NDIM(%(out2)s) != 2 ||
            CudaNdarray_HOST_DIMS(%(out2)s)[0] !=
               CudaNdarray_HOST_DIMS(%(biases)s)[0] ||
            CudaNdarray_HOST_DIMS(%(out2)s)[1] !=
               CudaNdarray_HOST_DIMS(%(biases)s)[1]) {
            Py_XDECREF(%(out2)s);
            int dims[2];
            dims[0] = CudaNdarray_HOST_DIMS(%(biases)s)[0];
            dims[1] = CudaNdarray_HOST_DIMS(%(biases)s)[1];
            %(out2)s = (CudaNdarray *)CudaNdarray_ZEROS(2, dims);
            if (%(out2)s == NULL) { %(fail)s; }
        } else {
          cudaError_t err = cudaMemset(CudaNdarray_DEV_DATA(%(out2)s), 0,
                                       CudaNdarray_SIZE(%(out2)s) * sizeof(real));
          if (err) {
            // Clear error flag
            cudaGetLastError();
            PyErr_SetString(PyExc_RuntimeError,
                            "GpuGroupDotGrad: cudaMemset failed for output 2");
          %(fail)s;
          }
        }
        {
        int fail = 0;
        npy_intp ng_shp = 0;
        CudaNdarray *proxy_W = (CudaNdarray *)CudaNdarray_new_nd(2);
        CudaNdarray *proxy_h = (CudaNdarray *)CudaNdarray_new_nd(1);
        CudaNdarray *proxy_grad = (CudaNdarray *)CudaNdarray_new_nd(1);
        CudaNdarray *proxy_gW = (CudaNdarray *)CudaNdarray_new_nd(2);
        CudaNdarray *proxy_gh = (CudaNdarray *)CudaNdarray_new_nd(1);
        CudaNdarray *proxy_o2 = (CudaNdarray *)CudaNdarray_new_nd(1);
        if (!proxy_W || !proxy_h || !proxy_grad || !proxy_gW || !proxy_gh || !proxy_o2)
          { fail = 1; goto %(name)s_fail; }
        CudaNdarray_set_dim(proxy_W, 0, CudaNdarray_HOST_DIMS(%(matrix)s)[1]);
        CudaNdarray_set_dim(proxy_W, 1, CudaNdarray_HOST_DIMS(%(matrix)s)[2]);
        CudaNdarray_set_dim(proxy_h, 0, CudaNdarray_HOST_DIMS(%(state_below)s)[1]);
        CudaNdarray_set_dim(proxy_grad, 0, CudaNdarray_HOST_DIMS(%(grad_on_out)s)[1]);
        CudaNdarray_set_dim(proxy_gW, 0, CudaNdarray_HOST_DIMS(%(out1)s)[1]);
        CudaNdarray_set_dim(proxy_gW, 1, CudaNdarray_HOST_DIMS(%(out1)s)[2]);
        CudaNdarray_set_dim(proxy_gh, 0, CudaNdarray_HOST_DIMS(%(out0)s)[1]);
        CudaNdarray_set_dim(proxy_o2, 0, CudaNdarray_HOST_DIMS(%(out2)s)[1]);
        CudaNdarray_set_stride(proxy_W, 0, CudaNdarray_HOST_STRIDES(%(matrix)s)[1]);
        CudaNdarray_set_stride(proxy_W, 1, CudaNdarray_HOST_STRIDES(%(matrix)s)[2]);
        CudaNdarray_set_stride(proxy_h, 0, CudaNdarray_HOST_STRIDES(%(state_below)s)[1]);
        CudaNdarray_set_stride(proxy_grad, 0, CudaNdarray_HOST_STRIDES(%(grad_on_out)s)[1]);
        CudaNdarray_set_stride(proxy_gW, 0, CudaNdarray_HOST_STRIDES(%(out1)s)[1]);
        CudaNdarray_set_stride(proxy_gW, 1, CudaNdarray_HOST_STRIDES(%(out1)s)[2]);
        CudaNdarray_set_stride(proxy_gh, 0, CudaNdarray_HOST_STRIDES(%(out0)s)[1]);
        CudaNdarray_set_stride(proxy_o2, 0, CudaNdarray_HOST_STRIDES(%(out2)s)[1]);
        ng_shp = PyArray_DIMS(%(groups)s)[0];
        for (npy_intp pos = 0; pos < %(n_groups)s; pos++) {
            CudaNdarray_set_device_data(proxy_W, CudaNdarray_DEV_DATA(%(matrix)s)+(pos * CudaNdarray_HOST_STRIDES(%(matrix)s)[0]), %(matrix)s);
            CudaNdarray_set_device_data(proxy_gW, CudaNdarray_DEV_DATA(%(out1)s)+(pos * CudaNdarray_HOST_STRIDES(%(out1)s)[0]), %(out1)s);
            CudaNdarray_set_device_data(proxy_o2, CudaNdarray_DEV_DATA(%(out2)s)+(pos * CudaNdarray_HOST_STRIDES(%(out2)s)[0]), %(out2)s);
            for (npy_intp jdx = 0; jdx < ng_shp; jdx++) {
               %(groups_ctype)s *p = (%(groups_ctype)s *)PyArray_GETPTR1(%(groups)s, jdx);
               if (*p == pos) {
                   CudaNdarray_set_device_data(proxy_h, CudaNdarray_DEV_DATA(%(state_below)s)+(jdx * CudaNdarray_HOST_STRIDES(%(state_below)s)[0]), %(state_below)s);
                   CudaNdarray_set_device_data(proxy_grad, CudaNdarray_DEV_DATA(%(grad_on_out)s)+(jdx * CudaNdarray_HOST_STRIDES(%(grad_on_out)s)[0]), %(grad_on_out)s);
                   CudaNdarray_set_device_data(proxy_gh, CudaNdarray_DEV_DATA(%(out0)s)+(jdx * CudaNdarray_HOST_STRIDES(%(out0)s)[0]), %(out0)s);
                   if (CudaNdarray_sgemv(1.0f, proxy_W, proxy_grad, 0.0f, proxy_gh))
                     { fail = 1; goto %(name)s_fail; }
                   if (CudaNdarray_sger(1.0f, proxy_h, proxy_grad, proxy_gW))
                     { fail = 1; goto %(name)s_fail; }
                   if (CudaNdarray_inplace_elemwise((PyObject *)proxy_o2, (PyObject *)proxy_grad, IADD))
                     { fail = 1; goto %(name)s_fail; }
               }
            }
        }
  %(name)s_fail:
    Py_XDECREF(proxy_W);
    Py_XDECREF(proxy_h);
    Py_XDECREF(proxy_grad);
    Py_XDECREF(proxy_gW);
    Py_XDECREF(proxy_gh);
    Py_XDECREF(proxy_o2);
    if (fail)
      %(fail)s;
        }
        """ % locals()
