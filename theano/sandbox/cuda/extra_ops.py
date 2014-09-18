import theano
import copy
from theano import Op, Apply
from theano.gof import local_optimizer
from theano.sandbox.cuda import cuda_available, GpuOp

from theano.tensor.extra_ops import CumsumOp
from theano.sandbox.cuda import GpuFlatten

if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host, HostFromGpu
    from theano.sandbox.cuda.opt import register_opt as register_gpu_opt


class GpuCumsum(CumsumOp, GpuOp):
    SUPPORTED_NDIMS = 2
    __props__ = ('axis', 'max_threads_dim0', 'max_grid_size1')

    def __init__(self, axis):
        """
        ``axis`` can not be None. If you want the array flatten, do it before.
        """
        self.axis = axis
        self.max_threads_dim0 = None
        self.max_grid_size1 = None

    def perform(self, node, inp, out):
        return Op.perform(self, node, inp, out)

    def make_node(self, x):
        assert x.dtype == 'float32'
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError('x must be a CudaNdarrayType', x)

        if x.ndim > GpuCumsum.SUPPORTED_NDIMS:
            raise NotImplementedError('Only cumsum on 1D and 2D array are supported right now!')

        if self.axis >= x.ndim:
            raise ValueError('axis(={1}) out of bounds'.format(self.axis))

        return theano.Apply(self, [x], [x.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_ = copy.copy(node)
        assert node.op is node_.op
        if node_.op.max_threads_dim0 is None or node_.op.max_grid_size1 is None:
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
            node_.op.max_grid_size1 = prop['maxGridSize1']

        return super(GpuCumsum, node_.op).make_thunk(node_, storage_map,
                                                     compute_map, no_recycling)

    def __str__(self):
        return "%s{%s}" % (self.__class__.__name__, self.axis)

    def c_code_cache_version(self):
        return (5,)

    def c_support_code_apply(self, node, nodename):
        return """
        __device__
        void k_reductionPhase_%(nodename)s(float* partialCumSum) {
            // Traverse down from leaves to root building partial sums at internal nodes in the tree.
            for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
                __syncthreads();
                unsigned int index = (threadIdx.x + 1) * (stride * 2) - 1;
                if(index < blockDim.x*2) {
                    partialCumSum[index] += partialCumSum[index - stride];
                }
            }
        }

        __device__
        void k_reversePhase_%(nodename)s(float* partialCumSum) {
            // Traverse back up the tree building the scan from the partial sums
            for (unsigned int stride = exp2(ceil(log2((float)blockDim.x))); stride > 0; stride /= 2) {
                __syncthreads();
                unsigned int index = (threadIdx.x + 1) * (stride * 2) - 1;
                if(index + stride < blockDim.x*2) {
                    partialCumSum[index + stride] += partialCumSum[index];
                }
            }
        }

        __device__
        void k_fetchData_%(nodename)s(float* partialCumSum, float* input, int globalThreadID, dim3 dataStrides, int dataOffset) {
            // blockIdx.y represents the # of the current independent cumsum
            int idx_even = (globalThreadID*2    ) * dataStrides.x + (blockIdx.y + dataOffset) * dataStrides.y;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides.x + (blockIdx.y + dataOffset) * dataStrides.y;
            partialCumSum[threadIdx.x*2]     = input[idx_even];
            partialCumSum[threadIdx.x*2 + 1] = input[idx_odd];
        }

        __device__
        void k_pushData_%(nodename)s(float* partialCumSum, float* output, int globalThreadID, dim3 dataStrides, int dataOffset) {
            __syncthreads();
            // blockIdx.y represents the # of the current independent cumsum
            int idx_even = (globalThreadID*2    ) * dataStrides.x + (blockIdx.y + dataOffset) * dataStrides.y;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides.x + (blockIdx.y + dataOffset) * dataStrides.y;
            output[idx_even] = partialCumSum[threadIdx.x*2];
            output[idx_odd]  = partialCumSum[threadIdx.x*2 + 1];
        }

        __global__
        void k_cumadd_%(nodename)s(float* input, float* output, dim3 inputStrides, dim3 outputStrides, int dataOffset, int beforeLastElementIdx, int lastElementIdx) {
            int dataOffsetY_input = (blockIdx.y + dataOffset) * inputStrides.y;
            int dataOffsetY_output = (blockIdx.y + dataOffset) * outputStrides.y;

            int idx_last_input = lastElementIdx*inputStrides.x + dataOffsetY_input;
            int idx_last_output = lastElementIdx*outputStrides.x + dataOffsetY_output;

            int idx_beforelast = beforeLastElementIdx*outputStrides.x + dataOffsetY_output;
            output[idx_last_output] = input[idx_last_input] + output[idx_beforelast];
        }

        __global__
        void k_finalCumSum_%(nodename)s(float* output, float* blockSum, int numElements, dim3 dataStrides, int dataOffset) {
            int globalThreadID = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

            // Check if current has data to process.
            if (globalThreadID >= ceil(numElements/2.0)) {
                return;
            }

            const float currentBlockSum = blockSum[blockIdx.x*gridDim.y + blockIdx.y + dataOffset];

            int dataOffsetY = (blockIdx.y + dataOffset) * (int)dataStrides.y;
            int idx_even = (globalThreadID*2    ) * dataStrides.x + dataOffsetY;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides.x + dataOffsetY;
            output[idx_even] += currentBlockSum;
            output[idx_odd] += currentBlockSum;
        }

        __global__
        void k_blockCumSum_%(nodename)s(float* input, float* output, int numElements, dim3 inputStrides, dim3 outputStrides, int dataOffset, float* blockSum) {
            // Regarding blockIdx and threadIdx, 'Cumsum' is always performed along the X axis.
            // The Y axis will contain all the independent cumsums of the 2D case.

            int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;

            // Check if current thread has data to process.
            if (globalThreadID >= ceil(numElements/2.0)) {
                return;
            }

            extern __shared__ float partialCumSum[];

            // Load data in shared memory
            k_fetchData_%(nodename)s(partialCumSum, input, globalThreadID, inputStrides, dataOffset);

            // Use a dichotomy approach to compute the cumsum (i.e. balanced binary tree).
            // The tree is sweeped from the leaves to the root and from the root to the leaves.
            // Similar to http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
            k_reductionPhase_%(nodename)s(partialCumSum);
            k_reversePhase_%(nodename)s(partialCumSum);

            // Write the final output to global memory
            k_pushData_%(nodename)s(partialCumSum, output, globalThreadID, outputStrides, dataOffset);

            if (blockSum != NULL){
                if (threadIdx.x == blockDim.x - 1) {
                    blockSum[blockIdx.x*gridDim.y + blockIdx.y + dataOffset] = partialCumSum[threadIdx.x*2 + 1];
                }
            }
        }

        int cumSum_%(nodename)s(CudaNdarray* input, CudaNdarray* output, int maxThreads, int axis, int maxGridY) {
            int shape[2] = { 1, 1 };
            dim3 inputStrides(0,0,0);
            dim3 outputStrides(0,0,0);

            switch (CudaNdarray_NDIM(input))
            {
            case 1:
                shape[0] = CudaNdarray_HOST_DIMS(input)[0];
                inputStrides.x = CudaNdarray_HOST_STRIDES(input)[0];
                outputStrides.x = CudaNdarray_HOST_STRIDES(output)[0];
                break;
            case 2:
                shape[0] = CudaNdarray_HOST_DIMS(input)[0];
                shape[1] = CudaNdarray_HOST_DIMS(input)[1];
                inputStrides.x = CudaNdarray_HOST_STRIDES(input)[0];
                inputStrides.y = CudaNdarray_HOST_STRIDES(input)[1];
                outputStrides.x = CudaNdarray_HOST_STRIDES(output)[0];
                outputStrides.y = CudaNdarray_HOST_STRIDES(output)[1];
                break;
            default:
                printf("Only 1D and 2D cumsum is implemented yet.\\n");
                return -1;
            }

            if (shape[axis] <= 1) {
                CudaNdarray_CopyFromCudaNdarray(output, input);
                return 0;
            }

            if (axis == 1) {
                int tmp = inputStrides.x;
                inputStrides.x = inputStrides.y;
                inputStrides.y = tmp;

                tmp = outputStrides.x;
                outputStrides.x = outputStrides.y;
                outputStrides.y = tmp;
            }

            int numElements = shape[axis] - (shape[axis] %% 2);
            int blockSize = ceil( min(numElements, 2*maxThreads) / 2.0);
            int dimGridX = ceil(numElements / (2.0*blockSize));  // Nb. of elements to perform cumsum on.
            int dimGridY = shape[1-axis];                        // Nb. of independent cumsums.
            const int shapeBlockSum[2] = { dimGridX, dimGridY };

            CudaNdarray* deviceBlockSum = (CudaNdarray*) CudaNdarray_NewDims(2, shapeBlockSum);

            for (int dataOffset = 0; dataOffset < dimGridY; dataOffset += maxGridY){
                int localDimGridY = min(dimGridY - dataOffset, maxGridY);
                dim3 dimBlock(blockSize, 1, 1);
                dim3 dimGrid(dimGridX, localDimGridY, 1);
                int sharedBytes = (2*blockSize) * sizeof(float);

                k_blockCumSum_%(nodename)s<<<dimGrid, dimBlock, sharedBytes>>>
                (
                    CudaNdarray_DEV_DATA(input),
                    CudaNdarray_DEV_DATA(output),
                    numElements,
                    inputStrides,
                    outputStrides,
                    dataOffset,
                    CudaNdarray_DEV_DATA(deviceBlockSum)
                );

                if (dimGridX > 1) {
                    // Do a cumsum over the blockSum (recursive).
                    if (cumSum_%(nodename)s(deviceBlockSum, deviceBlockSum, maxThreads, 0, maxGridY) == -1){
                        return -1;
                    }

                    // Since there are more than one block (i.e. `dimGridX > 1`)
                    //  report partial cumsums of previous blocks to subsequents ones.
                    dim3 dimGrid(dimGridX, dimGridY, 1);
                    dim3 dimBlock(blockSize, 1, 1);
                    k_finalCumSum_%(nodename)s<<<dimGrid, dimBlock>>>
                    (
                        CudaNdarray_DEV_DATA(output),
                        CudaNdarray_DEV_DATA(deviceBlockSum),
                        numElements,
                        outputStrides,
                        dataOffset
                    );
                }

                // If shape[axis] is odd, the last element is compute manually
                if (shape[axis] != numElements){
                    dim3 dimGrid(1, localDimGridY, 1);
                    dim3 dimBlock(1, 1, 1);
                    k_cumadd_%(nodename)s<<<dimGrid, dimBlock>>>
                    (
                        CudaNdarray_DEV_DATA(input),
                        CudaNdarray_DEV_DATA(output),
                        inputStrides,
                        outputStrides,
                        dataOffset,
                        shape[axis]-2,
                        shape[axis]-1
                    );
                }
            }

            cudaFree(CudaNdarray_DEV_DATA(deviceBlockSum));
            CNDA_THREAD_SYNC;
            return 0;
        }
        """ % locals()

    def c_code(self, node, nodename, inames, onames, sub):
        x, = inames
        z, = onames
        axis = self.axis if self.axis is not None else 0
        fail = sub['fail']

        max_threads_dim0 = self.max_threads_dim0
        max_grid_size1 = self.max_grid_size1
        if max_threads_dim0 is None or max_grid_size1 is None:
            raise NotImplementedError("GpuCumsum.c_code should not be called "
                                      "directly. It should be called by "
                                      "make_thunk() that add some information "
                                      "related to the selected GPU.")

        code = """
            const int* shape = CudaNdarray_HOST_DIMS(%(x)s);
            bool needAllocation = !%(z)s || CudaNdarray_NDIM(%(x)s) != CudaNdarray_NDIM(%(z)s);

            // If output is already allocated, check if its shape matches the input's one.
            if (!needAllocation) {
                for (int i= 0; i < CudaNdarray_NDIM(%(x)s); ++i) {
                    if (CudaNdarray_HOST_DIMS(%(x)s)[i] != CudaNdarray_HOST_DIMS(%(z)s)[i]) {
                        needAllocation = true;
                    }
                }
            }

            if (needAllocation){
                Py_XDECREF(%(z)s);
                %(z)s = (CudaNdarray*) CudaNdarray_NewDims(CudaNdarray_NDIM(%(x)s), shape);
            }

            if (!%(z)s) {
                %(fail)s;
            }

            { // Namespace for kernel calls //
                if (cumSum_%(nodename)s(%(x)s, %(z)s, %(max_threads_dim0)s, %(axis)s, %(max_grid_size1)s) == -1){
                    %(fail)s;
                }

                cudaError_t sts = cudaGetLastError();
                if (cudaSuccess != sts)
                {
                    PyErr_Format(PyExc_RuntimeError,
                                 "Cuda error: %%s: %%s.\\n",
                        "cumSum_%(nodename)s",
                        cudaGetErrorString(sts));
                    %(fail)s;
                }
            }
        """ % locals()

        return code


@local_optimizer([CumsumOp])
def use_gpu_cumsum(node):
    if type(node.op) is CumsumOp \
       and node.inputs[0].dtype == 'float32' \
       and node.inputs[0].owner \
       and isinstance(node.inputs[0].owner.op, HostFromGpu):

        axis = node.op.axis
        x = node.inputs[0]

        if axis is not None and x.ndim > GpuCumsum.SUPPORTED_NDIMS:
            return None

        x = gpu_from_host(x)

        if axis is None and x.ndim > 1:
            x = GpuFlatten()(x)

        # ``gpu_cumsum`` assume array has been flattened if needed.
        if axis is None:
            axis = 0

        return [host_from_gpu(GpuCumsum(axis)(x))]

if cuda_available:
    register_gpu_opt()(use_gpu_cumsum)
