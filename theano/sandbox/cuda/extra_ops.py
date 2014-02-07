import theano
import copy
from theano import Op, Apply
from theano.gof import local_optimizer
from theano.sandbox.cuda import cuda_available, GpuOp

from theano.tensor.extra_ops import CumsumOp

if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host
    from theano.sandbox.cuda.opt import register_opt as register_gpu_opt


class GpuCumsum(CumsumOp, GpuOp):
    def __init__(self, axis=None):
        self.axis = axis

    def make_node(self, x):
        assert x.dtype == 'float32'
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError('x must be cudandarray', x)

        out_type = x.type()

        if self.axis is None:
            out_type = CudaNdarrayType(broadcastable=(False,), dtype=x.dtype)

        return theano.Apply(self, [x], [out_type])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_ = copy.copy(node)
        assert node.op is node_.op
        if node_.op.max_threads_dim0 is None:
            cuda = theano.sandbox.cuda
            device_id = cuda.use.device_number
            if device_id is None:
                cuda.use("gpu",
                         force=False,
                         default_to_move_computation_to_gpu=False,
                         move_shared_float32_to_gpu=False,
                         enable_cuda=False,
                         test_driver=True)
                device_id = cuda.use.device_number
            cuda_ndarray = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray
            prop = cuda_ndarray.device_properties(device_id)
            node_.op.max_threads_dim0 = prop['maxThreadsDim0']
        return super(GpuCumsum, node_.op).make_thunk(node_, storage_map,
                                                     compute_map, no_recycling)

    def c_code_cache_version(self):
        #return (1,)
        return ()

    def c_support_code_apply(self, node, nodename):
        mode = self.mode
        return """
        static __global__ void k_cumsum_1D_%(nodename)s(float* g_idata,
                                                        float* g_odata,
                                                        int n)
        {
            extern __shared__ float temp[2*blockDim.x];

            int stride = 1;

            temp[2*threadIdx.x]   = g_idata[2*threadIdx.x];
            temp[2*threadIdx.x+1] = g_idata[2*threadIdx.x+1];

            for (int d = n/2; d > 0; d /= 2)
            {
                CNDA_THREAD_SYNC;

                if (threadIdx.x < d)
                {
                    int ai = stride*(2*threadIdx.x+1)-1;
                    int bi = stride*(2*threadIdx.x+2)-1;
                    temp[bi] += temp[ai];
                }

                stride *= 2;
            }

            if (threadIdx.x == 0) { temp[n - 1] = 0; } // NOt sure about that

            for (int d = 1; d < n; d *= 2)
            {
                CNDA_THREAD_SYNC;

                if (threadIdx.x < d)
                {
                    int ai = stride*(2*threadIdx.x+1)-1;
                    int bi = stride*(2*threadIdx.x+2)-1;

                    float t   = temp[ai];
                    temp[ai]  = temp[bi];
                    temp[bi] += t;
                }
            }

            CNDA_THREAD_SYNC;
            g_odata[2*threadIdx.x]   = temp[2*threadIdx.x];
            g_odata[2*threadIdx.x+1] = temp[2*threadIdx.x+1];
        }
        """ % locals()

    def c_code(self, node, name, inames, onames, sub):
        x, = inames
        z, = onames
        axis = self.axis
        fail = sub['fail']

        sub = sub.copy()
        max_threads_dim0 = self.max_threads_dim0
        if max_threads_dim0 is None:
            raise NotImplementedError("GpuConv.c_code should not be called "
                                      "directly. It should be called by "
                                      "make_thunk() that add some information "
                                      "related to the selected GPU.")
        sub.update(locals())

        #Right now, only the 1D case implementation exists.

        code = """
            npy_intp shape[1] = { CudaNdarray_SIZE(%(x)s) };
            if(! (%(z)s && CudaNdarray_HOST_DIMS(%(z)s)[0] == shape[0]) )
            {
                Py_XDECREF(%(z)s);
                %(z)s = (CudaNdarray*) CudaNdarray_NewDims(1, shape);
            }

            if (!%(z)s)
                %(fail)s;
            {
                dim3 dim_block( min(shape[0], %(max_threads_dim0)s) );
                dim3 dim_grid(1);

                if (dim_block.x < shape[0])
                    dim_grid.x = (shape[0]-1 / dim_block.x) + 1;  # Ceil


                void (*f)(float*, float*, int);
                f = k_cumsum_1D_%(name)s;

                f<<<dim_grid,dim_block>>>(CudaNdarray_DEV_DATA(%(x)s),
                                          CudaNdarray_DEV_DATA(%(z)s),
                                          shape[0]);

                CNDA_THREAD_SYNC;
                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Cuda error: %%s: %%s. (grid: %%i x %%i;"
                                 " block: %%i x %%i x %%i; shared: %%i)\\n",
                        "k_cumsum_1D_%(name)s",
                        cudaGetErrorString(sts),
                        dim_grid.x,
                        dim_grid.y,
                        dim_block.x,
                        dim_block.y,
                        dim_block.z,
                        0);
                    %(fail)s;
                }
            }
        """ % locals()

        return code
