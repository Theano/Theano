from __future__ import absolute_import, print_function, division
import theano
import numpy
from theano import Op, Apply, config
from theano.tensor.extra_ops import CumsumOp

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
                        infer_context_name)
from .opt import register_opt as register_gpu_opt, op_lifter
from .type import GpuArrayType


class GpuCumsum(CumsumOp, GpuKernelBase):
    """
    Parameters
    ----------
    axis
        Can not be None. If you want the array flattened, do it before.
    """
    SUPPORTED_NDIMS = 3
    __props__ = ('axis',)

    def __init__(self, axis):
        self.axis = axis

    def __str__(self):
        return "%s{%s}" % (self.__class__.__name__, self.axis)

    def c_code_cache_version(self):
        return (1,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']


    def make_node(self, x): 
        assert x.type.dtype == 'float32', "Only float32 supported for GpuCumSum"
        x = as_gpuarray_variable(x, infer_context_name(x))

        if x.ndim > GpuCumsum.SUPPORTED_NDIMS:
            raise NotImplementedError('Only cumsum on 1D, 2D and\
                                       3D arrays are supported right now!')

        if self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError('axis(={1}) out of bounds'.format(self.axis))
        return Apply(self, [x], [x.type()])


    # copied from neighbour.py
    def perform(self, node, inp, out, ctx):
        # Disable the perform method from the CPU version
        Op.perform(self, node, inp, out, ctx)


    def gpu_kernels(self, node, nodename):
        kernels = []
        # cumadd
        kname = "k_cumadd"   
        k_var = "k_cumadd_" + nodename
        dtype_x = node.inputs[0].dtype
        flags = Kernel.get_flags(dtype_x)
        code = """
        KERNEL void %(kname)s(float* input, float* output, ssize_t inputStrides_x,
                              ssize_t inputStrides_y, ssize_t inputStrides_z,
                              ssize_t outputStrides_x, ssize_t outputStrides_y,
                              ssize_t outputStrides_z, const int offsetY, const int offsetZ, 
                              const int beforeLastElementIdx, const int lastElementIdx){
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;

            int dataOffsetY_input = idY * inputStrides_y + idZ * inputStrides_z;
            int dataOffsetY_output = idY * outputStrides_y + idZ * outputStrides_z;
            int idx_last_input = lastElementIdx*inputStrides_x + dataOffsetY_input;
            int idx_last_output = lastElementIdx*outputStrides_x + dataOffsetY_output;
            int idx_beforelast = beforeLastElementIdx*outputStrides_x + dataOffsetY_output;
            output[idx_last_output] = input[idx_last_input] + output[idx_beforelast];
            }
        """ % locals()
        params = [gpuarray.GpuArray, gpuarray.GpuArray, gpuarray.SSIZE, 
                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE, 
                  gpuarray.SSIZE, gpuarray.SSIZE, 
                  'intc', 'intc',
                  'intc', 'intc',
                  ]
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        # blockCumSum
        kname = "k_blockCumSum" 
        k_var = "k_blockCumSum_" + nodename
        params = [gpuarray.GpuArray, gpuarray.GpuArray, gpuarray.SIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
                  'int32', 'int32', gpuarray.GpuArray,]
        code="""
        // helper functions
        WITHIN_KERNEL
        void k_reductionPhase_%(nodename)s(float* partialCumSum) {
            // Traverse down from leaves to root building partial sums at internal nodes in the tree.
            for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
                local_barrier();
                unsigned int index = (threadIdx.x + 1) * (stride * 2) - 1;
                if(index < blockDim.x*2) {
                    partialCumSum[index] += partialCumSum[index - stride];
                }
            }
        }

        WITHIN_KERNEL
        void k_fetchData_%(nodename)s(float* partialCumSum, float* input, int globalThreadID,
                                      ssize_t dataStrides_x, ssize_t dataStrides_y, ssize_t dataStrides_z,
                                      int offsetY, int offsetZ) {
            // blockIdx.y and blockIdx.z represents the current independent cumsum
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ; int offset = idY * dataStrides_y + idZ * dataStrides_z;
            int idx_even = (globalThreadID*2    ) * dataStrides_x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides_x + offset;
            partialCumSum[threadIdx.x*2]     = input[idx_even];
            partialCumSum[threadIdx.x*2 + 1] = input[idx_odd];
        }

        WITHIN_KERNEL
        void k_reversePhase_%(nodename)s(float* partialCumSum) {
            // Traverse back up the tree building the scan from the partial sums
            for (unsigned int stride = exp2(ceil(log2((float)blockDim.x))); stride > 0; stride /= 2) {
                local_barrier();
                unsigned int index = (threadIdx.x + 1) * (stride * 2) - 1;
                if(index + stride < blockDim.x*2) {
                    partialCumSum[index + stride] += partialCumSum[index];
                }
            }
        }

        WITHIN_KERNEL
        void k_pushData_%(nodename)s(float* partialCumSum, float* output, int globalThreadID,
                                     ssize_t dataStrides_x, ssize_t dataStrides_y, ssize_t dataStrides_z,
                                     int offsetY, int offsetZ) {
            local_barrier();
            // blockIdx.y and blockIdx.z represents the current independent cumsum
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;
            int offset = idY * dataStrides_y + idZ * dataStrides_z;
            int idx_even = (globalThreadID*2    ) * dataStrides_x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides_x + offset;
            output[idx_even] = partialCumSum[threadIdx.x*2];
            output[idx_odd]  = partialCumSum[threadIdx.x*2 + 1];
        }

        KERNEL void k_blockCumSum(float* input, float* output,
                                        size_t nbElementsPerCumsum, ssize_t inputStrides_x,
                                        ssize_t inputStrides_y,  ssize_t inputStrides_z,
                                        ssize_t outputStrides_x, ssize_t outputStrides_y,
                                        ssize_t outputStrides_z, int offsetY,
                                        int offsetZ, float* blockSum) {
            // Regarding blockIdx and threadIdx, 'Cumsum' is always performed along the X axis.
            // The Y and Z axis of the grid will contain all independent cumsums of the 2D/3D case.

            int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;

            // Check if current thread has data to process.
            if (globalThreadID >= ceil(nbElementsPerCumsum/2.0)) {
                return;
            }

            extern __shared__ float partialCumSum[];

            // Load data in shared memory
            k_fetchData_%(nodename)s(partialCumSum, input, globalThreadID, inputStrides_x, inputStrides_y, inputStrides_z, offsetY, offsetZ);

            // Use a dichotomy approach to compute the cumsum (i.e. balanced binary tree).
            // The tree is sweeped from the leaves to the root and from the root to the leaves.
            // Similar to http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
            k_reductionPhase_%(nodename)s(partialCumSum);
            k_reversePhase_%(nodename)s(partialCumSum);

            // Write the final output to global memory
            k_pushData_%(nodename)s(partialCumSum, output, globalThreadID, outputStrides_x, outputStrides_y, outputStrides_x,, offsetY, offsetZ);

            if (blockSum != NULL){
                if (threadIdx.x == blockDim.x - 1) {
                    blockSum[blockIdx.x*(gridDim.y*gridDim.z) + (blockIdx.y + offsetY)*gridDim.z + blockIdx.z + offsetZ] = partialCumSum[threadIdx.x*2 + 1];
                }
            }
        }
        """
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        # k_finalCumSum
        kname = "k_finalCumSum" 
        k_var = "k_finalCumSum_" + nodename
        code = """
        KERNEL void k_finalCumSum(float* output, float* blockSum, size_t nbElementsPerCumsum, 
                                               ssize_t dataStrides_x,  ssize_t dataStrides_y,  ssize_t dataStrides_z,
                                               int offsetY, int offsetZ) {
            int globalThreadID = (blockIdx_x + 1) * blockDim_x + threadIdx_x;

            // Check if current has data to process.
            if (globalThreadID >= ceil(nbElementsPerCumsum/2.0)) {
                return;
            }

            int idY = blockIdx_y + offsetY;
            int idZ = blockIdx_z + offsetZ;

            const float currentBlockSum = blockSum[blockIdx_x*(gridDim_y*gridDim_z) + idY*gridDim.z + idZ];

            int offset = idY * dataStrides_y + idZ * dataStrides_z;
            int idx_even = (globalThreadID*2    ) * dataStrides_x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides_x + offset;
            output[idx_even] += currentBlockSum;
            output[idx_odd] += currentBlockSum;
        }
        """
        params = [gpuarray.GpuArray, gpuarray.GpuArray, gpuarray.SIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
                  'int32', 'int32',]
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels


    def c_code(self, node, name, inp, out, sub):
        if node.inputs[0].type.context.kind != 'cuda':
            raise NotImplementedError("cuda only")
        x, = inp
        z, = out
        axis = self.axis if self.axis is not None else 0
        fail = sub['fail']

        code = """
            const size_t* shape = PyGpuArray_DIMS(%(x)s);
            bool needAllocation = !%(z)s || PyGpuArray_NDIM(%(x)s) != PyGpuArray_NDIM(%(z)s);

            int axis = %(axis)s;
            if (axis < 0) {
                // Convert negative axis to positive axis.
                axis += PyGpuArray_NDIM(%(x)s);
            }

            if (theano_prep_output(&%(z)s, PyGpuArray_NDIM(%(x)s), PyGpuArray_DIMS(%(x)s), %(type)s, GA_C_ORDER, %(ctx)s) == 0){
                %(fail)s;
            }

            { // Namespace for kernel calls //
                size_t max_threads_dim0; 
                size_t max_grid_size1; 
                size_t max_grid_size2; 
                int err;
                err = %(ctx)s->ops->property(%(ctx)s->ctx, NULL, NULL, GA_CTX_PROP_MAXLSIZE0, &max_threads_dim0);
                if (err != GA_NO_ERROR){
                    PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_threads_dims0");
                    %(fail)s;
                }
                err = %(ctx)s->ops->property(%(ctx)s->ctx, NULL, NULL, GA_CTX_PROP_MAXGSIZE1, &max_grid_size1);
                if (err != GA_NO_ERROR){
                    PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_grid_size1");
                    %(fail)s;
                }
                err = %(ctx)s->ops->property(%(ctx)s->ctx, NULL, NULL, GA_CTX_PROP_MAXGSIZE2, &max_grid_size2);
                if (err != GA_NO_ERROR){
                    PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_grid_size2");
                    %(fail)s;
                }
                if (cumSum_%(nodename)s(%(x)s, %(z)s, axis, max_threads_dim0, max_grid_size1, max_grid_size2) == -1){
                    %(fail)s;
                }
            }
        """ % locals()

        return code

        def c_support_code_apply(self, node, nodename):
            code = """int cumSum_%(nodename)s(float* input, float* output, int axis, size_t maxThreads, size_t maxGridY, size_t maxGridZ) {
                size_t shape[3] = { 1, 1, 1 };
                ssize_t inputStrides_x;
                ssize_t inputStrides_y;
                ssize_t inputStrides_z;
                ssize_t outputStrides_x;
                ssize_t outputStrides_y;
                ssize_t outputStrides_z;
                switch (PYArray_NDIM(input))
                {
                case 1:
                    shape[0] = PyArray_DIMS(input)[0];
                    inputStrides_x = PyGpuArray_STRIDES(input)[0];
                    outputStrides_x = PyGpuArray_STRIDES(output)[0];
                    break;
                case 2:
                    shape[0] = PyArray_DIMS(input)[0];
                    shape[1] = PyArray_DIMS(input)[1];
                    inputStrides_x = PyGpuArray_STRIDES(input)[0];
                    inputStrides_y = PyGpuArray_STRIDES(input)[1];
                    outputStrides_x = PyGpuArray_STRIDES(output)[0];
                    outputStrides_y = PyGpuArray_STRIDES(output)[1];
                    break;
                case 3:
                    shape[0] = PyArray_DIMS(input)[0];
                    shape[1] = PyArray_DIMS(input)[1];
                    shape[2] = PyArray_DIMS(input)[2];
                    inputStrides_x = PyGpuArray_STRIDES(input)[0];
                    inputStrides_y = PyGpuArray_STRIDES(input)[1];
                    inputStrides_z = PyGpuArray_STRIDES(input)[2];
                    outputStrides_x = PyGpuArray_STRIDES(output)[0];
                    outputStrides_y = PyGpuArray_STRIDES(output)[1];
                    outputStrides_z = PyGpuArray_STRIDES(output)[2];
                    break;
                default:
                    return -1;
                }
                if (shape[axis] <= 1) {
                    output = pygpu_copy(input, GA_ANY_ORDER);
                    return 0;
                }
                // Perform cumsum on array of even size.
                size_t nbElementsPerCumsum = shape[axis] - (shape[axis] %% 2);
                // Determine how many elements can be processed in one block.
                size_t dimBlockX = ceil( min(nbElementsPerCumsum, 2*maxThreads) / 2.0);
                // Determine how many blocks are needed in total.
                size_t dimGridX = ceil(nbElementsPerCumsum / (2.0*dimBlockX));  // Nb. of blocks needed per cumsum.
                size_t dimGridY;  // Nb. of independent cumsums (width).
                size_t dimGridZ;  // Nb. of independent cumsums (height).
                ssize_t tmp;
                switch (axis)
                {
                case 0:
                    dimGridY = shape[1];
                    dimGridZ = shape[2];
                    break;
                case 1:
                    dimGridY = shape[0];
                    dimGridZ = shape[2];
                    tmp = inputStrides_x;
                    inputStrides_x = inputStrides_y;
                    inputStrides_y = tmp;
                    tmp = outputStrides_x;
                    outputStrides_x = outputStrides_y;
                    outputStrides_y = tmp;
                    break;
                case 2:
                    dimGridY = shape[1];
                    dimGridZ = shape[0];
                    tmp = inputStrides_x;
                    inputStrides_x = inputStrides_z;
                    inputStrides_z = tmp;
                    tmp = outputStrides_x;
                    outputStrides_x = outputStrides_z;
                    outputStrides_z = tmp;
                    break;
                default:
                    return -1;
                }
                const size_t shapeBlockSum[2] = { dimGridX, dimGridY*dimGridZ };
                PyGpuArrayObject* deviceBlockSum = pygpu_empty(2, shapeBlockSum, output->typecode,
                                                               GA_C_ORDER, input->context->ctx, Py_None);
                if (deviceBlockSum == NULL){
                    return -1;
                }
                // Perform `maxGridY`*`maxGridZ` cumsums in parallel.
                for (size_t offsetY = 0; offsetY < dimGridY; offsetY += maxGridY){
                    size_t localDimGridY = min(dimGridY - offsetY, maxGridY);
                    for (size_t offsetZ = 0; offsetZ < dimGridZ; offsetZ += maxGridZ){
                        size_t localDimGridZ = min(dimGridZ - offsetZ, maxGridZ);
                        size_t dimGrid[3] = {dimGridX, localDimGridY, localDimGridZ};
                        size_t dimBlock[3] = {dimBlockX, 1, 1};  // One cumsum per block.
                        size_t sharedBytes = (2*dimBlockX) * sizeof(float);
                        void* kernel_params[] = {(void*) input->ga.data,
                                                 (void*) output->ga.data,
                                                 (void*) &nbElementsPerCumsum, 
                                                 (void*) &inputStrides_x,
                                                 (void*) &inputStrides_y,
                                                 (void*) &inputStrides_z,
                                                 (void*) &outputStrides_x,
                                                 (void*) &outputStrides_y,
                                                 (void*) &outputStrides_z,
                                                 (void*) &offsetY,
                                                 (void*) &offsetZ,
                                                 (void*) deviceBlockSum->ga.data;
                            };
                        int err = GpuKernel_call(k_blockCumSum_%(nodename)s, 3, dimBlock, dimGrid, sharedBytes, kernel_params);

                        if (dimGridX > 1) {
                            // Do a cumsum over the blockSum (recursive).
                            if (cumSum_%(nodename)s(deviceBlockSum, deviceBlockSum, 0, maxThreads, maxGridY, maxGridZ) == -1){
                                Py_DECREF(deviceBlockSum);
                                return -1;
                            }
                            // Since there are more than one block (i.e. `dimGridX > 1`)
                            //  report partial cumsums of previous blocks to subsequents ones.
                            size_t dimGrid[3] = {dimGridX, localDimGridY, localDimGridZ};
                            size_t dimBlock[3] = {dimBlockX, 1, 1};
                            void* kernel_params[] = {(void*) output->ga.data,
                                                     (void*) deviceBlockSum->ga.data, 
                                                     (void*) &nbElementsPerCumsum,
                                                     (void*) &outputStrides_x,
                                                     (void*) &outputStrides_y,
                                                     (void*) &outputStrides_z,
                                                     (void*) &offsetY,
                                                     (void*) &offsetZ
                                };
                            int err = GpuKernel_call(k_finalCumSum_%(nodename)s, 3, dimBlock, dimGrid, sharedBytes, kernel_params);
                        }
                        // If shape[axis] is odd, the last element is compute manually
                        if (shape[axis] != nbElementsPerCumsum){
                            size_t dimGrid[3] = {1, localDimGridY, localDimGridZ};
                            size_t dimBlock[3] = {1, 1, 1};
                            void* kernel_params[] = {(void*) input->ga.data,
                                                     (void*) output->ga.data,
                                                     (void*) &inputStrides_x,
                                                     (void*) &inputStrides_y,
                                                     (void*) &inputStrides_z,
                                                     (void*) &outputStrides_x,
                                                     (void*) &outputStrides_y,
                                                     (void*) &outputStrides_z,
                                                     (void*) &offsetY,
                                                     (void*) &offsetZ,
                                                     (void*) &(shape[axis]-2),
                                                     (void*) &(shape[axis]-1)
                            };
                            int err = GpuKernel_call(k_cumadd_%(nodename)s, 3, dimBlock, dimGrid, sharedBytes, kernel_params);
                        }
                    }
                }
                Py_XDECREF(deviceBlockSum);
                return 0;
            }
            """
            return "\n".join(super(GpuKernelBase, self).c_support_code_apply(node, name), code)


@op_lifter([CumsumOp])
def use_gpu_cumsumop(node, ctx_name):
    return GpuCumsum(node.op.axis)

register_gpu_opt()(use_gpu_cumsumop)
