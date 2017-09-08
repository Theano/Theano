from __future__ import absolute_import, print_function, division
from theano import Apply, Op
from theano.tensor.extra_ops import CumOp

try:
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel, GpuReshape, infer_context_name, gpuarray_helper_inc_dir)
from .opt import register_opt, op_lifter, register_opt2
from .type import gpu_context_type
from theano.gof import ParamsType
import theano.scalar as scalar


class GpuCumOp(GpuKernelBase, Op):
    """
    Parameters
    ----------
    axis
        Can not be None. If you want the array flattened, do it before.
    """
    SUPPORTED_NDIMS = 3
    __props__ = ('axis', 'mode')
    params_type = ParamsType(axis=scalar.int32,
                             context=gpu_context_type)

    def __init__(self, axis, mode='add'):
        assert axis is not None
        self.axis = int(axis)
        self.mode = mode

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        return self.axis == other.axis and self.mode == other.mode

    def __hash__(self):
        return hash(self.axis) ^ hash(self.mode)

    def c_code_cache_version(self):
        return (7,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>', '<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [gpuarray_helper_inc_dir()]

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)

    def make_node(self, x):
        assert x.type.dtype == 'float32', "Only float32 supported for GpuCumOp"

        context_name = infer_context_name(x)

        x = as_gpuarray_variable(x, context_name)

        if x.ndim > GpuCumOp.SUPPORTED_NDIMS:
            raise NotImplementedError('Only cum op on 1D, 2D and\
                                       3D arrays are supported right now!')

        if self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError('axis(={0}) out of bounds'.format(self.axis))
        return Apply(self, [x], [x.type()])

    def gpu_kernels(self, node, nodename):
        kernels = []
        # cumadd
        kname = "k_cumadd"
        op = {'mul': '*', 'add': '+'}[self.mode]
        k_var = "k_cumadd_" + nodename
        dtype_x = node.inputs[0].dtype
        flags = Kernel.get_flags(dtype_x)
        code = """#include "cluda.h"

        KERNEL void %(kname)s(float* input, ga_size input_offset,
                              float* output, ga_size output_offset,
                              ga_ssize inputStrides_x, ga_ssize inputStrides_y, ga_ssize inputStrides_z,
                              ga_ssize outputStrides_x, ga_ssize outputStrides_y, ga_ssize outputStrides_z,
                              const int offsetY, const int offsetZ,
                              const int beforeLastElementIdx, const int lastElementIdx){
            input = (float *)(((char *)input) + input_offset);
            output = (float *)(((char *)output) + output_offset);
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;

            int dataOffsetY_input = idY * inputStrides_y + idZ * inputStrides_z;
            int dataOffsetY_output = idY * outputStrides_y + idZ * outputStrides_z;
            int idx_last_input = lastElementIdx*inputStrides_x + dataOffsetY_input;
            int idx_last_output = lastElementIdx*outputStrides_x + dataOffsetY_output;
            int idx_beforelast = beforeLastElementIdx*outputStrides_x + dataOffsetY_output;
            output[idx_last_output] = input[idx_last_input] %(op)s output[idx_beforelast];
            }
        """ % locals()
        params = [gpuarray.GpuArray, gpuarray.SIZE,
                  gpuarray.GpuArray, gpuarray.SIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE,
                  'intc', 'intc',
                  'intc', 'intc',
                  ]
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        # blockCumOp
        kname = "k_blockCumOp"
        k_var = "k_blockCumOp_" + nodename
        params = [gpuarray.GpuArray, gpuarray.SIZE,
                  gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
                  'int32', 'int32', gpuarray.GpuArray, gpuarray.SIZE]
        code = """#include "cluda.h"

        // helper functions
        WITHIN_KERNEL
        void k_reductionPhase(float* partialCumOp) {
            // Traverse down from leaves to root building partial sums at internal nodes in the tree.
            for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
                local_barrier();
                unsigned int index = (threadIdx.x + 1) * (stride * 2) - 1;
                if (index < blockDim.x*2) {
                    partialCumOp[index] %(op)s= partialCumOp[index - stride];
                }
            }
        }

        WITHIN_KERNEL
        void k_fetchData(float* partialCumOp, float* input, int globalThreadID,
                         ga_ssize dataStrides_x, ga_ssize dataStrides_y, ga_ssize dataStrides_z,
                         int offsetY, int offsetZ) {
            // blockIdx.y and blockIdx.z represents the current independent cum op
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ; int offset = idY * dataStrides_y + idZ * dataStrides_z;
            int idx_even = (globalThreadID*2    ) * dataStrides_x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides_x + offset;
            partialCumOp[threadIdx.x*2]     = input[idx_even];
            partialCumOp[threadIdx.x*2 + 1] = input[idx_odd];
        }

        WITHIN_KERNEL
        void k_reversePhase(float* partialCumOp) {
            // Traverse back up the tree building the scan from the partial sums
            for (unsigned int stride = exp2(ceil(log2((float)blockDim.x))); stride > 0; stride /= 2) {
                local_barrier();
                unsigned int index = (threadIdx.x + 1) * (stride * 2) - 1;
                if (index + stride < blockDim.x*2) {
                    partialCumOp[index + stride] %(op)s= partialCumOp[index];
                }
            }
        }

        WITHIN_KERNEL
        void k_pushData(float* partialCumOp, float* output, int globalThreadID,
                        ga_ssize dataStrides_x, ga_ssize dataStrides_y, ga_ssize dataStrides_z,
                        int offsetY, int offsetZ) {
            local_barrier();
            // blockIdx.y and blockIdx.z represents the current independent cum op
            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;
            int offset = idY * dataStrides_y + idZ * dataStrides_z;
            int idx_even = (globalThreadID*2    ) * dataStrides_x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides_x + offset;
            output[idx_even] = partialCumOp[threadIdx.x*2];
            output[idx_odd]  = partialCumOp[threadIdx.x*2 + 1];
        }

        KERNEL void k_blockCumOp(float* input, ga_size input_offset,
                                 float* output, ga_size output_offset,
                                 size_t nbElementsPerCumOp, ga_ssize inputStrides_x,
                                 ga_ssize inputStrides_y,  ga_ssize inputStrides_z,
                                 ga_ssize outputStrides_x, ga_ssize outputStrides_y,
                                 ga_ssize outputStrides_z, int offsetY,
                                 int offsetZ, float* blockSum, ga_size blockSum_offset) {
            input = (float *)(((char *)input) + input_offset);
            output = (float *)(((char *)output) + output_offset);
            blockSum = (float *)(((char *)blockSum) + blockSum_offset);

            // Regarding blockIdx and threadIdx, 'CumOp' is always performed along the X axis.
            // The Y and Z axis of the grid will contain all independent cumops of the 2D/3D case.

            int globalThreadID = blockIdx.x * blockDim.x + threadIdx.x;

            // Check if current thread has data to process.
            if (globalThreadID >= (nbElementsPerCumOp+1)/2) {
                return;
            }

            extern __shared__ float partialCumOp[];

            // Load data in shared memory
            k_fetchData(partialCumOp, input, globalThreadID, inputStrides_x, inputStrides_y, inputStrides_z, offsetY, offsetZ);

            // Use a dichotomy approach to compute the cum op (i.e. balanced binary tree).
            // The tree is sweeped from the leaves to the root and from the root to the leaves.
            // Similar to http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf
            k_reductionPhase(partialCumOp);
            k_reversePhase(partialCumOp);

            // Write the final output to global memory
            k_pushData(partialCumOp, output, globalThreadID, outputStrides_x, outputStrides_y, outputStrides_z, offsetY, offsetZ);

            if (blockSum != NULL){
                if (threadIdx.x == blockDim.x - 1) {
                    blockSum[blockIdx.x*(gridDim.y*gridDim.z) + (blockIdx.y + offsetY)*gridDim.z + blockIdx.z + offsetZ] = partialCumOp[threadIdx.x*2 + 1];
                }
            }
        }
        """ % locals()
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        # k_finalCumOp
        kname = "k_finalCumOp"
        k_var = "k_finalCumOp_" + nodename
        code = """#include "cluda.h"

        KERNEL void k_finalCumOp(float* output, ga_size output_offset,
                                 float* blockSum, ga_size blockSum_offset,
                                 size_t nbElementsPerCumOp,
                                 ga_ssize dataStrides_x,  ga_ssize dataStrides_y,  ga_ssize dataStrides_z,
                                 int offsetY, int offsetZ) {

            output = (float *)(((char *)output) + output_offset);
            blockSum = (float *)(((char *)blockSum) + blockSum_offset);

            int globalThreadID = (blockIdx.x + 1) * blockDim.x + threadIdx.x;

            // Check if current has data to process.
            if (globalThreadID >= (nbElementsPerCumOp+1)/2)
                return;

            int idY = blockIdx.y + offsetY;
            int idZ = blockIdx.z + offsetZ;

            const float currentBlockSum = blockSum[blockIdx.x*(gridDim.y*gridDim.z) + idY*gridDim.z + idZ];

            int offset = idY * dataStrides_y + idZ * dataStrides_z;
            int idx_even = (globalThreadID*2    ) * dataStrides_x + offset;
            int idx_odd  = (globalThreadID*2 + 1) * dataStrides_x + offset;
            output[idx_even] %(op)s= currentBlockSum;
            output[idx_odd] %(op)s= currentBlockSum;
        }
        """ % locals()
        params = [gpuarray.GpuArray, gpuarray.SIZE,
                  gpuarray.GpuArray, gpuarray.SIZE, gpuarray.SIZE,
                  gpuarray.SSIZE, gpuarray.SSIZE, gpuarray.SSIZE,
                  'int32', 'int32', ]
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels

    def c_code(self, node, nodename, inp, out, sub):
        if node.inputs[0].type.context.kind != b'cuda':
            raise NotImplementedError("cuda only")
        return """
            const size_t* shape = PyGpuArray_DIMS(%(x)s);
            bool needAllocation = !%(z)s || PyGpuArray_NDIM(%(x)s) != PyGpuArray_NDIM(%(z)s);

            int axis = %(params)s->axis;
            if (axis < 0) {
                // Convert negative axis to positive axis.
                axis += PyGpuArray_NDIM(%(x)s);
            }

            if (theano_prep_output(&%(z)s, PyGpuArray_NDIM(%(x)s), PyGpuArray_DIMS(%(x)s),
                                   %(x)s->ga.typecode, GA_C_ORDER, %(params)s->context) != 0) {
                %(fail)s;
            }

            { // Namespace for kernel calls //
                size_t max_threads_dim0;
                size_t max_grid_size1;
                size_t max_grid_size2;
                int err;
                err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXLSIZE0, &max_threads_dim0);
                if (err != GA_NO_ERROR){
                    PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_threads_dims0");
                    %(fail)s;
                }
                err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXGSIZE1, &max_grid_size1);
                if (err != GA_NO_ERROR){
                    PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_grid_size1");
                    %(fail)s;
                }
                err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXGSIZE2, &max_grid_size2);
                if (err != GA_NO_ERROR){
                    PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_grid_size2");
                    %(fail)s;
                }
                if (cumOp_%(nodename)s(%(x)s, %(z)s, axis, max_threads_dim0, max_grid_size1, max_grid_size2) == -1){
                    %(fail)s;
                }
            }
        """ % dict(x=inp[0], z=out[0], nodename=nodename, fail=sub['fail'], params=sub['params'])

    def c_support_code_struct(self, node, nodename):
        code = """

        int cumOp_%(nodename)s(PyGpuArrayObject* input, PyGpuArrayObject* output, int axis, size_t maxThreads, size_t maxGridY, size_t maxGridZ) {
            size_t shape[3] = { 1, 1, 1 };
            ssize_t inputStrides_x;
            ssize_t inputStrides_y;
            ssize_t inputStrides_z;
            ssize_t outputStrides_x;
            ssize_t outputStrides_y;
            ssize_t outputStrides_z;
            switch (PyGpuArray_NDIM(input))
            {
            case 1:
                shape[0] = PyGpuArray_DIMS(input)[0];
                inputStrides_x = PyGpuArray_STRIDES(input)[0] / sizeof(float);
                outputStrides_x = PyGpuArray_STRIDES(output)[0] / sizeof(float);
                break;
            case 2:
                shape[0] = PyGpuArray_DIMS(input)[0];
                shape[1] = PyGpuArray_DIMS(input)[1];
                inputStrides_x = PyGpuArray_STRIDES(input)[0] / sizeof(float);
                inputStrides_y = PyGpuArray_STRIDES(input)[1] / sizeof(float);
                outputStrides_x = PyGpuArray_STRIDES(output)[0] / sizeof(float);
                outputStrides_y = PyGpuArray_STRIDES(output)[1] / sizeof(float);
                break;
            case 3:
                shape[0] = PyGpuArray_DIMS(input)[0];
                shape[1] = PyGpuArray_DIMS(input)[1];
                shape[2] = PyGpuArray_DIMS(input)[2];
                inputStrides_x = PyGpuArray_STRIDES(input)[0] / sizeof(float);
                inputStrides_y = PyGpuArray_STRIDES(input)[1] / sizeof(float);
                inputStrides_z = PyGpuArray_STRIDES(input)[2] / sizeof(float);
                outputStrides_x = PyGpuArray_STRIDES(output)[0] / sizeof(float);
                outputStrides_y = PyGpuArray_STRIDES(output)[1] / sizeof(float);
                outputStrides_z = PyGpuArray_STRIDES(output)[2] / sizeof(float);
                break;
            default:
                PyErr_SetString(PyExc_RuntimeError, "Unsupported Axis");
                return -1;
            }
            if (shape[axis] <= 1) {
                int err = pygpu_move(output, input);
                return err;
            }
            // Perform cum op on array of even size.
            size_t nbElementsPerCumOp = shape[axis] - (shape[axis] %% 2);
            // Determine how many elements can be processed in one block.
            size_t dimBlockX = ((nbElementsPerCumOp > 2*maxThreads ? 2*maxThreads : nbElementsPerCumOp)+1)/2;
            // Determine how many blocks are needed in total.
            size_t dimGridX = (nbElementsPerCumOp+2*dimBlockX-1) / (2*dimBlockX);  // Nb. of blocks needed per cum op.
            size_t dimGridY;  // Nb. of independent cum ops (width).
            size_t dimGridZ;  // Nb. of independent cum ops (height).
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
                PyErr_SetString(PyExc_RuntimeError, "Unsupported Axis");
                return -1;
            }

            const size_t shapeBlockSum[2] = { dimGridX, dimGridY*dimGridZ };
            PyGpuArrayObject* deviceBlockSum = pygpu_empty(2, shapeBlockSum, output->ga.typecode,
                                                           GA_C_ORDER, input->context, Py_None);
            if (deviceBlockSum == NULL){
                return -1;
            }
            // Perform `maxGridY`*`maxGridZ` cum ops in parallel.
            for (size_t offsetY = 0; offsetY < dimGridY; offsetY += maxGridY){
                size_t localDimGridY = (dimGridY - offsetY < maxGridY) ? (dimGridY - offsetY) : (maxGridY);

                for (size_t offsetZ = 0; offsetZ < dimGridZ; offsetZ += maxGridZ){
                    size_t localDimGridZ = (dimGridZ - offsetZ < maxGridZ) ? (dimGridZ - offsetZ) : (maxGridZ);
                    size_t dimGrid[3] = {dimGridX, localDimGridY, localDimGridZ};
                    size_t dimBlock[3] = {dimBlockX, 1, 1};  // One cum op per block.
                    size_t sharedBytes = (2*dimBlockX) * sizeof(float);

                    int err = k_blockCumOp_call(3, dimGrid, dimBlock, sharedBytes, input->ga.data, input->ga.offset, output->ga.data, output->ga.offset, nbElementsPerCumOp, inputStrides_x, inputStrides_y, inputStrides_z, outputStrides_x, outputStrides_y, outputStrides_z, offsetY, offsetZ, deviceBlockSum->ga.data, deviceBlockSum->ga.offset);
                    if (err != GA_NO_ERROR){
                        PyErr_SetString(PyExc_RuntimeError, "blockCumOp call failed");
                        return -1;
                    }

                    if (dimGridX > 1) {
                        // Do a cum op over the blockSum (recursive).
                        if (cumOp_%(nodename)s(deviceBlockSum, deviceBlockSum, 0, maxThreads, maxGridY, maxGridZ) == -1){
                            Py_DECREF(deviceBlockSum);
                            return -1;
                        }
                        // Since there are more than one block (i.e. `dimGridX > 1`)
                        //  report partial cum ops of previous blocks to subsequents ones.
                        size_t dimGrid[3] = {dimGridX, localDimGridY, localDimGridZ};
                        size_t dimBlock[3] = {dimBlockX, 1, 1};

                        int err = k_finalCumOp_call(3, dimGrid, dimBlock, sharedBytes, output->ga.data, output->ga.offset, deviceBlockSum->ga.data, deviceBlockSum->ga.offset, nbElementsPerCumOp, outputStrides_x, outputStrides_y, outputStrides_z, offsetY, offsetZ);
                        if (err != GA_NO_ERROR){
                            PyErr_SetString(PyExc_RuntimeError, "finalCumOp call failed");
                            return -1;
                        }
                    }
                    // If shape[axis] is odd, the last element is compute manually
                    if (shape[axis] != nbElementsPerCumOp){
                        size_t dimGrid[3] = {1, localDimGridY, localDimGridZ};
                        size_t dimBlock[3] = {1, 1, 1};

                        int err = k_cumadd_call(3, dimGrid, dimBlock, sharedBytes, input->ga.data, input->ga.offset, output->ga.data, output->ga.offset, inputStrides_x, inputStrides_y, inputStrides_z, outputStrides_x, outputStrides_y, outputStrides_z, offsetY, offsetZ, shape[axis] - 2, shape[axis] - 1);
                        if (err != GA_NO_ERROR){
                            PyErr_SetString(PyExc_RuntimeError, "cumadd call failed");
                            return -1;
                        }

                    }
                }
            }
            Py_XDECREF(deviceBlockSum);
            return 0;
        }
        """ % locals()
        return super(GpuCumOp, self).c_support_code_struct(node, nodename) + code


# GpuCumsumOp exists only to serve backward compatibility.
# Once an object is created, it will be converted to CumOp object.
class GpuCumsumOp(GpuKernelBase, Op):
    SUPPORTED_NDIMS = 3
    __props__ = ("axis",)

    def __new__(typ, *args, **kwargs):
        obj = object.__new__(GpuCumOp, *args, **kwargs)
        obj.mode = 'add'
        return obj


@register_opt('fast_compile')
@op_lifter([CumOp])
@register_opt2([CumOp], 'fast_compile')
def local_gpua_cumop(op, ctx_name, inputs, outputs):
    if inputs[0].dtype != 'float32':
        return False
    axis = op.axis
    x = inputs[0]
    if axis is not None and x.ndim > GpuCumOp.SUPPORTED_NDIMS:
        return False

    x = as_gpuarray_variable(x, ctx_name)

    if axis is None and x.ndim > 1:
        x = GpuReshape(1)(x, (-1,))

    # ``gpu_cumop`` assume array has been flattened if needed.
    if axis is None:
        axis = 0

    return GpuCumOp(axis, op.mode)(x)
