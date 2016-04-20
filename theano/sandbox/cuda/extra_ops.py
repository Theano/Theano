from __future__ import absolute_import, print_function, division
import theano
import copy
from theano import Op
from theano.gof import local_optimizer
from theano.sandbox.cuda import cuda_available, GpuOp
from theano.sandbox.cuda.basic_ops import gpu_flatten
from theano.tensor.extra_ops import CumsumOp

if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host, HostFromGpu
    from theano.sandbox.cuda import register_opt as register_gpu_opt


class GpuCumsum(CumsumOp, GpuOp):
    """

    Parameters
    ----------
    axis
        Can not be None. If you want the array flatten, do it before.

    """

    SUPPORTED_NDIMS = 3
    __props__ = ('axis', 'max_threads_dim0', 'max_grid_size1', 'max_grid_size2')

    def __init__(self, axis):
        self.axis = axis
        self.max_threads_dim0 = None
        self.max_grid_size1 = None
        self.max_grid_size2 = None

    # We must reuse the same method, not reimplement and call it.
    # Otherwise DebugMode will print many warnings.
    perform = Op.perform

    def make_node(self, x):
        assert x.dtype == 'float32'
        if not isinstance(x.type, CudaNdarrayType):
            raise TypeError('x must be a CudaNdarrayType', x)

        if x.ndim > GpuCumsum.SUPPORTED_NDIMS:
            raise NotImplementedError('Only cumsum on 1D, 2D and 3D array are supported right now!')

        if self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError('axis(={1}) out of bounds'.format(self.axis))

        return theano.Apply(self, [x], [x.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_ = copy.copy(node)
        assert node.op is node_.op
        if node_.op.max_threads_dim0 is None or node_.op.max_grid_size1 is None or node_.op.max_grid_size2 is None:
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
            node_.op.max_grid_size2 = prop['maxGridSize2']

        return super(GpuCumsum, node_.op).make_thunk(node_, storage_map,
                                                     compute_map, no_recycling)

    def __str__(self):
        return "%s{%s}" % (self.__class__.__name__, self.axis)

    def c_code_cache_version(self):
        return (9,)

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
        void k_fetchData_%(nodename)s(float* partialCumSum, float* input, int globalThreadID, dim3 dataStrides, int offsetY, int offsetZ) {
            // blockIdx.y and blockIdx.z represents the current independent cumsum
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;
            int offset = idY * dataStrides.y + idZ * dataStrides.z;
            int idx_even = (globalThreadID*2    ) * dataStrides.x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides.x + offset;
            partialCumSum[threadIdx.x*2]     = input[idx_even];
            partialCumSum[threadIdx.x*2 + 1] = input[idx_odd];
        }

        __device__
        void k_pushData_%(nodename)s(float* partialCumSum, float* output, int globalThreadID, dim3 dataStrides, int offsetY, int offsetZ) {
            __syncthreads();
            // blockIdx.y and blockIdx.z represents the current independent cumsum
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;
            int offset = idY * dataStrides.y + idZ * dataStrides.z;
            int idx_even = (globalThreadID*2    ) * dataStrides.x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides.x + offset;
            output[idx_even] = partialCumSum[threadIdx.x*2];
            output[idx_odd]  = partialCumSum[threadIdx.x*2 + 1];
        }

        __global__
        void k_cumadd_%(nodename)s(float* input, float* output, dim3 inputStrides, dim3 outputStrides, int offsetY, int offsetZ, int beforeLastElementIdx, int lastElementIdx) {
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;

            int dataOffsetY_input = idY * inputStrides.y + idZ * inputStrides.z;
            int dataOffsetY_output = idY * outputStrides.y + idZ * outputStrides.z;

            int idx_last_input = lastElementIdx*inputStrides.x + dataOffsetY_input;
            int idx_last_output = lastElementIdx*outputStrides.x + dataOffsetY_output;

            int idx_beforelast = beforeLastElementIdx*outputStrides.x + dataOffsetY_output;
            output[idx_last_output] = input[idx_last_input] + output[idx_beforelast];
        }

        __global__
        void k_finalCumSum_%(nodename)s(float* output, float* blockSum, int nbElementsPerCumsum, dim3 dataStrides, int offsetY, int offsetZ) {
            int globalThreadID = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

            // Check if current has data to process.
            if (globalThreadID >= ceil(nbElementsPerCumsum/2.0)) {
                return;
            }

            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;

            const float currentBlockSum = blockSum[blockIdx.x*(gridDim.y*gridDim.z) + idY*gridDim.z + idZ];

            int offset = idY * dataStrides.y + idZ * dataStrides.z;
            int idx_even = (globalThreadID*2    ) * dataStrides.x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides.x + offset;
            output[idx_even] += currentBlockSum;
            output[idx_odd] += currentBlockSum;
        }

        __global__
        void k_blockCumSum_%(nodename)s(float* input, float* output, int nbElementsPerCumsum, dim3 inputStrides, dim3 outputStrides, int offsetY, int offsetZ, float* blockSum) {
            // Regarding blockIdx and threadIdx, 'Cumsum' is always performed along the X axis.
            // The Y and Z axis of the grid will contain all independent cumsums of the 2D/3D case.

            int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;

            // Check if current thread has data to process.
            if (globalThreadID >= ceil(nbElementsPerCumsum/2.0)) {
                return;
            }

            extern __shared__ float partialCumSum[];

            // Load data in shared memory
            k_fetchData_%(nodename)s(partialCumSum, input, globalThreadID, inputStrides, offsetY, offsetZ);

            // Use a dichotomy approach to compute the cumsum (i.e. balanced binary tree).
            // The tree is sweeped from the leaves to the root and from the root to the leaves.
            // Similar to http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
            k_reductionPhase_%(nodename)s(partialCumSum);
            k_reversePhase_%(nodename)s(partialCumSum);

            // Write the final output to global memory
            k_pushData_%(nodename)s(partialCumSum, output, globalThreadID, outputStrides, offsetY, offsetZ);

            if (blockSum != NULL){
                if (threadIdx.x == blockDim.x - 1) {
                    blockSum[blockIdx.x*(gridDim.y*gridDim.z) + (blockIdx.y + offsetY)*gridDim.z + blockIdx.z + offsetZ] = partialCumSum[threadIdx.x*2 + 1];
                }
            }
        }

        int cumSum_%(nodename)s(CudaNdarray* input, CudaNdarray* output, int axis, int maxThreads, int maxGridY, int maxGridZ) {
            int shape[3] = { 1, 1, 1 };
            dim3 inputStrides(0, 0, 0);
            dim3 outputStrides(0, 0, 0);

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
            case 3:
                shape[0] = CudaNdarray_HOST_DIMS(input)[0];
                shape[1] = CudaNdarray_HOST_DIMS(input)[1];
                shape[2] = CudaNdarray_HOST_DIMS(input)[2];
                inputStrides.x = CudaNdarray_HOST_STRIDES(input)[0];
                inputStrides.y = CudaNdarray_HOST_STRIDES(input)[1];
                inputStrides.z = CudaNdarray_HOST_STRIDES(input)[2];
                outputStrides.x = CudaNdarray_HOST_STRIDES(output)[0];
                outputStrides.y = CudaNdarray_HOST_STRIDES(output)[1];
                outputStrides.z = CudaNdarray_HOST_STRIDES(output)[2];
                break;
            default:
                return -1;
            }

            if (shape[axis] <= 1) {
                CudaNdarray_CopyFromCudaNdarray(output, input);
                return 0;
            }

            // Perform cumsum on array of even size.
            int nbElementsPerCumsum = shape[axis] - (shape[axis] %% 2);

            // Determine how many elements can be processed in one block.
            int dimBlockX = ceil( min(nbElementsPerCumsum, 2*maxThreads) / 2.0);

            // Determine how many blocks are needed in total.
            int dimGridX = ceil(nbElementsPerCumsum / (2.0*dimBlockX));  // Nb. of blocks needed per cumsum.
            int dimGridY;  // Nb. of independent cumsums (width).
            int dimGridZ;  // Nb. of independent cumsums (height).

            int tmp;
            switch (axis)
            {
            case 0:
                dimGridY = shape[1];
                dimGridZ = shape[2];
                break;
            case 1:
                dimGridY = shape[0];
                dimGridZ = shape[2];

                tmp = inputStrides.x;
                inputStrides.x = inputStrides.y;
                inputStrides.y = tmp;

                tmp = outputStrides.x;
                outputStrides.x = outputStrides.y;
                outputStrides.y = tmp;
                break;
            case 2:
                dimGridY = shape[1];
                dimGridZ = shape[0];

                tmp = inputStrides.x;
                inputStrides.x = inputStrides.z;
                inputStrides.z = tmp;

                tmp = outputStrides.x;
                outputStrides.x = outputStrides.z;
                outputStrides.z = tmp;
                break;
            default:
                return -1;
            }

            const int shapeBlockSum[2] = { dimGridX, dimGridY*dimGridZ };
            CudaNdarray* deviceBlockSum = (CudaNdarray*) CudaNdarray_NewDims(2, shapeBlockSum);

            // Perform `maxGridY`*`maxGridZ` cumsums in parallel.
            for (int offsetY = 0; offsetY < dimGridY; offsetY += maxGridY){
                int localDimGridY = min(dimGridY - offsetY, maxGridY);

                for (int offsetZ = 0; offsetZ < dimGridZ; offsetZ += maxGridZ){
                    int localDimGridZ = min(dimGridZ - offsetZ, maxGridZ);

                    dim3 dimGrid(dimGridX, localDimGridY, localDimGridZ);
                    dim3 dimBlock(dimBlockX, 1, 1);  // One cumsum per block.
                    int sharedBytes = (2*dimBlockX) * sizeof(float);

                    k_blockCumSum_%(nodename)s<<<dimGrid, dimBlock, sharedBytes>>>
                    (
                        CudaNdarray_DEV_DATA(input),
                        CudaNdarray_DEV_DATA(output),
                        nbElementsPerCumsum,
                        inputStrides,
                        outputStrides,
                        offsetY,
                        offsetZ,
                        CudaNdarray_DEV_DATA(deviceBlockSum)
                    );

                    if (dimGridX > 1) {
                        // Do a cumsum over the blockSum (recursive).
                        if (cumSum_%(nodename)s(deviceBlockSum, deviceBlockSum, 0, maxThreads, maxGridY, maxGridZ) == -1){
                            Py_DECREF(deviceBlockSum);
                            return -1;
                        }

                        // Since there are more than one block (i.e. `dimGridX > 1`)
                        //  report partial cumsums of previous blocks to subsequents ones.
                        dim3 dimGrid(dimGridX, localDimGridY, localDimGridZ);
                        dim3 dimBlock(dimBlockX, 1, 1);
                        k_finalCumSum_%(nodename)s<<<dimGrid, dimBlock>>>
                        (
                            CudaNdarray_DEV_DATA(output),
                            CudaNdarray_DEV_DATA(deviceBlockSum),
                            nbElementsPerCumsum,
                            outputStrides,
                            offsetY,
                            offsetZ
                        );
                    }

                    // If shape[axis] is odd, the last element is compute manually
                    if (shape[axis] != nbElementsPerCumsum){
                        dim3 dimGrid(1, localDimGridY, localDimGridZ);
                        dim3 dimBlock(1, 1, 1);
                        k_cumadd_%(nodename)s<<<dimGrid, dimBlock>>>
                        (
                            CudaNdarray_DEV_DATA(input),
                            CudaNdarray_DEV_DATA(output),
                            inputStrides,
                            outputStrides,
                            offsetY,
                            offsetZ,
                            shape[axis]-2,
                            shape[axis]-1
                        );
                    }
                }
            }
            Py_DECREF(deviceBlockSum);
            CNDA_THREAD_SYNC;
            return 0;
        }
        """ % locals()

    def c_code(self, node, nodename, inames, onames, sub):
        x, = inames
        z, = onames

        # We assume array has been already flattened if needed.
        axis = self.axis if self.axis is not None else 0
        fail = sub['fail']

        max_threads_dim0 = self.max_threads_dim0
        max_grid_size1 = self.max_grid_size1
        max_grid_size2 = self.max_grid_size2
        if max_threads_dim0 is None or max_grid_size1 is None or max_grid_size2 is None:
            raise NotImplementedError("GpuCumsum.c_code should not be called "
                                      "directly. It should be called by "
                                      "make_thunk() that add some information "
                                      "related to the selected GPU.")

        code = """
            const int* shape = CudaNdarray_HOST_DIMS(%(x)s);
            bool needAllocation = !%(z)s || CudaNdarray_NDIM(%(x)s) != CudaNdarray_NDIM(%(z)s);

            int axis = %(axis)s;
            if (axis < 0) {
                // Convert negative axis to positive axis.
                axis += CudaNdarray_NDIM(%(x)s);
            }

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
                if (cumSum_%(nodename)s(%(x)s, %(z)s, axis, %(max_threads_dim0)s, %(max_grid_size1)s, %(max_grid_size2)s) == -1){
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


def values_eq_approx_high_tol(a, b):
    """
    This fct is needed to don't have DebugMode raise useless
    error due to rounding error.

    This happen with big input size due to change in the order of
    operation.

    """
    rtol = None
    if a.size > 100000:
        # For float32 the default rtol is 1e-5
        rtol = 5e-5
    return CudaNdarrayType.values_eq_approx(a, b, rtol=rtol)


@register_gpu_opt()
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
            x = gpu_flatten(x)

        # ``gpu_cumsum`` assume array has been flattened if needed.
        if axis is None:
            axis = 0

        ret = host_from_gpu(GpuCumsum(axis)(x))
        ret.tag.values_eq_approx = values_eq_approx_high_tol
        return [ret]
