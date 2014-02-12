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
        self.max_threads_dim0 = None

    def make_node(self, x):
        assert x.dtype == 'float32'
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError('x must be cudandarray', x)

        out_type = x.type()

        if self.axis is None and x.ndim > 1:
            out_type = CudaNdarrayType(broadcastable=(False,), dtype=x.dtype)()

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
        axis = self.axis
        return """
        __global__
        void k_finalCumSum_1D_%(nodename)s(float* output, float* blockSum, int numElements) {
            int globalThreadID = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

            if (globalThreadID < ceil(numElements/2.0)) {
                const float currentBlockSum = blockSum[blockIdx.x];

                output[globalThreadID*2]     += currentBlockSum;
                output[globalThreadID*2 + 1] += currentBlockSum;
            }
        }


        __global__
        void k_cumadd_%(nodename)s(float* input, float* output, int beforeLastElementIdx, int lastElementIdx) {
            output[lastElementIdx] = input[lastElementIdx] + output[beforeLastElementIdx];
        }


        __global__
        void k_blockCumSum_1D_%(nodename)s(float* input, float* output, int numElements, float* blockSum) {
            int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;

            if (globalThreadID < ceil(numElements/2.0)) {
                extern __shared__ float partialCumSum[];

                // Load data in shared memory
                partialCumSum[threadIdx.x*2]     = input[globalThreadID*2];
                partialCumSum[threadIdx.x*2 + 1] = input[globalThreadID*2 + 1];

                // Reduction Phase
                int stride;
                for (stride = 1; stride <= blockDim.x; stride *= 2) {
                    __syncthreads();
                    int index = (threadIdx.x + 1) * (stride * 2) - 1;
                    if(index < blockDim.x*2) {
                        partialCumSum[index] += partialCumSum[index - stride];
                    }
                }

                // Reverse Phase
                for (; stride > 0; stride /= 2) {
                    __syncthreads();
                    int index = (threadIdx.x + 1) * (stride * 2) - 1;
                    if(index + stride < blockDim.x*2) {
                        partialCumSum[index + stride] += partialCumSum[index];
                    }
                }

                // Wtite the final output to global memory
                __syncthreads();
                output[globalThreadID*2]     = partialCumSum[threadIdx.x*2];
                output[globalThreadID*2 + 1] = partialCumSum[threadIdx.x*2 + 1];

                if (blockSum != NULL){
                    if (threadIdx.x == blockDim.x - 1) {
                        blockSum[blockIdx.x] = partialCumSum[threadIdx.x*2 + 1];
                    }
                }
            }
        }

        void cumSum_1D_%(nodename)s(CudaNdarray* input, CudaNdarray* output, npy_intp* shape, int maxThreads) {

            if (shape[0] <= 1) {
                CudaNdarray_CopyFromCudaNdarray(output, input);
                return;
            }

            int numElements = shape[0] - (shape[0] %% 2);
            int blockSize = ceil( min(numElements, 2*maxThreads) / 2.0);
            int dimGridX = ceil(numElements / (2.0*blockSize));
            npy_intp shapeBlockSum[1] = { dimGridX };
            CudaNdarray* deviceBlockSum = (CudaNdarray*) CudaNdarray_NewDims(1, shapeBlockSum);

            dim3 dimBlock(blockSize, 1, 1);
            dim3 dimGrid(dimGridX, 1, 1);
            int sharedBytes = (2*blockSize) * sizeof(float);

            cudaThreadSynchronize();
            k_blockCumSum_1D_%(nodename)s<<<dimGrid, dimBlock, sharedBytes>>>
            (
                CudaNdarray_DEV_DATA(input),
                CudaNdarray_DEV_DATA(output),
                numElements,
                CudaNdarray_DEV_DATA(deviceBlockSum)
            );

            if (dimGridX > 1) {
                // Do a cumsum over the blockSum (recursive).
                cumSum_1D_%(nodename)s(deviceBlockSum, deviceBlockSum, shapeBlockSum, maxThreads);

                dim3 dimGrid(dimGridX-1, 1, 1);
                dim3 dimBlock(blockSize, 1, 1);
                k_finalCumSum_1D_%(nodename)s<<<dimGrid, dimBlock>>>
                (
                    CudaNdarray_DEV_DATA(output),
                    CudaNdarray_DEV_DATA(deviceBlockSum),
                    numElements
                );
            }

            // If shape[0] is odd, the last element is compute manually
            if (shape[0] != numElements){
                cudaThreadSynchronize();
                k_cumadd_%(nodename)s<<<dim3(1,1,1), dim3(1,1,1)>>>
                (
                    CudaNdarray_DEV_DATA(input),
                    CudaNdarray_DEV_DATA(output),
                    shape[0]-2,
                    shape[0]-1
                );
            }

            cudaFree(CudaNdarray_DEV_DATA(deviceBlockSum));
            cudaThreadSynchronize();
        }
        """ % locals()

    def c_code(self, node, nodename, inames, onames, sub):
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
            if(! (%(z)s && CudaNdarray_HOST_DIMS(%(z)s)[0] == shape[0]) ) {
                Py_XDECREF(%(z)s);
                %(z)s = (CudaNdarray*) CudaNdarray_NewDims(1, shape);
            }

            if (!%(z)s) {
                %(fail)s;
            }

            { // Namespace for kernel calls //
                cumSum_1D_%(nodename)s(%(x)s, %(z)s, shape, %(max_threads_dim0)s);

                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Cuda error: %%s: %%s.\\n",
                        "cumSum_1D_%(nodename)s",
                        cudaGetErrorString(sts));
                    %(fail)s;
                }
            }
        """ % locals()

        return code


def gpu_cumsum(x, axis=None):
    return GpuCumsum(axis)(x)

from theano.sandbox.cuda import GpuFlatten

@local_optimizer([CumsumOp])
def use_gpu_cumsum(node):
    if type(node.op) is CumsumOp and node.inputs[0].dtype == 'float32':

        x = gpu_from_host(node.inputs[0])
        if node.op.axis is None and x.ndim > 1:
            x = GpuFlatten()(x)

        return [host_from_gpu(gpu_cumsum(x, axis=node.op.axis))]

if cuda_available:
    register_gpu_opt()(use_gpu_cumsum)
