from __future__ import absolute_import, print_function, division
import numpy

from theano import Op, Apply, config
from theano.tensor.nnet.neighbours import Images2Neibs
import theano.tensor as T

try:
    import pygpu
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
                        infer_context_name)
from .opt import register_opt as register_gpu_opt, op_lifter
from .type import GpuArrayType


class GpuImages2Neibs(GpuKernelBase, Images2Neibs, Op):
    """
    Images2Neibs for the GPU.

    """
    def __init__(self, mode='valid'):
        if mode not in ['valid', 'ignore_borders', 'wrap_centered']:
            raise NotImplementedError("Only the mode valid, ignore_borders"
                                      " and wrap_centered"
                                      " have been implemented for the op"
                                      " GpuImages2Neibs")
        self.mode = mode

    def make_node(self, ten4, neib_shape, neib_step):
        ten4 = as_gpuarray_variable(ten4, infer_context_name(ten4))
        neib_shape = T.as_tensor_variable(neib_shape)
        neib_step = T.as_tensor_variable(neib_step)

        assert ten4.ndim == 4
        assert neib_shape.ndim == 1
        assert neib_step.ndim == 1
        assert "int" in neib_shape.dtype
        assert "int" in neib_step.dtype

        return Apply(self, [ten4, neib_shape, neib_step],
                     [GpuArrayType(broadcastable=(False, False),
                                   dtype=ten4.type.dtype,
                                   context_name=ten4.type.context_name)()])

    def get_params(self, node):
        return node.inputs[0].type.context

    def c_code_cache_version(self):
        return (11,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def gpu_kernels(self, node, nodename):
        dtype_ten4 = node.inputs[0].dtype
        dtype_z = node.outputs[0].dtype
        flags = Kernel.get_flags(dtype_ten4, dtype_z)
        type_ten4 = gpuarray.dtype_to_ctype(dtype_ten4)
        type_z = gpuarray.dtype_to_ctype(dtype_z)
        mode = self.mode
        kernels = []
        kname = "k_multi_warp_less"
        k_var = "k_multi_warp_less_" + nodename
        code = """
// a version that uses less registers but doesn't work in all cases.
        KERNEL void %(kname)s(
            const int nb_batch,
            const int nb_stack,
            const int height,
            const int width,
            const int c,
            const int d,
            const int step_x,
            const int step_y,
            const int grid_c,
            const int grid_d,
            const size_t stride0, const size_t stride1,
            const size_t stride2, const size_t stride3,
            const %(type_ten4)s * global_ten4, const size_t offset_ten4,
            const size_t out_s0, const size_t out_s1,
            %(type_z)s * global_out, const size_t offset_out
        )
        {
            const int wrap_centered_idx_shift_x = c/2;
            const int wrap_centered_idx_shift_y = d/2;
            global_ten4 = (const %(type_ten4)s *)(((char *)global_ten4)+offset_ten4);
            global_out = (%(type_z)s *)(((char *)global_out)+offset_out);

            for(int tblock = blockIdx.x*blockDim.z+threadIdx.z;
                tblock<nb_batch*nb_stack*grid_c*grid_d;
                tblock+=gridDim.x*blockDim.z){
                const int b = tblock%%grid_d;
                int left = tblock/grid_d;
                const int a = left%%grid_c;
                left = left/grid_c;
                const int s = left%%nb_stack;
                left = left/nb_stack;
                const int n = left;

                if(n>nb_batch)continue;
                if(s>nb_stack)continue;
                if(a>grid_c)continue;
                if(b>grid_d)continue;
                            int z_row = b + grid_d*(a + grid_c*
                                                    (s + nb_stack*n));
                            int i = threadIdx.y;     // loop over c
                            {
                                int ten4_2 = i + a * step_x;
                                if("%(mode)s"=="wrap_centered"){
                                    ten4_2 -= wrap_centered_idx_shift_x;
                                    if ( ten4_2 < 0 )
                                        ten4_2 += height;
                                    else if (ten4_2 >= height)
                                        ten4_2 -= height;
                                }
                                int j = threadIdx.x;  // loop over d
                                {
                                    int ten4_3 = j + b * step_y;
                                    if("%(mode)s"=="wrap_centered"){
                                        ten4_3 -= wrap_centered_idx_shift_y;
                                        if ( ten4_3 < 0 )
                                            ten4_3 += width;
                                        else if (ten4_3 >= width)
                                            ten4_3 -= width;
                                    }

                                    int ten4_idx = stride3*ten4_3 +
                                                   stride2*ten4_2 +
                                                   stride1*s + stride0*n;

                                    int z_col = j + d * i;
                                    int z_idx = z_col * out_s1 +
                                                z_row * out_s0;
                                    global_out[z_idx] = global_ten4[ten4_idx];
                                }
                            }
            }
        }""" % locals()
        params = [
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc',
            'uintp', 'uintp', 'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp',
            'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp',
            ]
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))

        kname = "k_multi_warp"
        k_var = "k_multi_warp_" + nodename
        code = """
        KERNEL void %(kname)s(
            const int nb_batch,
            const int nb_stack,
            const int height,
            const int width,
            const int c,
            const int d,
            const int step_x,
            const int step_y,
            const int grid_c,
            const int grid_d,
            const size_t stride0, const size_t stride1,
            const size_t stride2, const size_t stride3,
            const %(type_ten4)s * global_ten4, const size_t offset_ten4,
            const size_t out_s0, const size_t out_s1,
            %(type_z)s * global_out, const size_t offset_out
        )
        {
            const int wrap_centered_idx_shift_x = c/2;
            const int wrap_centered_idx_shift_y = d/2;
            global_ten4 = (const %(type_ten4)s *)(((char *)global_ten4)+offset_ten4);
            global_out = (%(type_z)s *)(((char *)global_out)+offset_out);

            for(int tblock = blockIdx.x*blockDim.z+threadIdx.z;
                tblock<nb_batch*nb_stack*grid_c*grid_d;
                tblock+=gridDim.x*blockDim.z){
                const int b = tblock%%grid_d;
                int left = tblock/grid_d;
                const int a = left%%grid_c;
                left = left/grid_c;
                const int s = left%%nb_stack;
                left = left/nb_stack;
                const int n = left;

                if(n>nb_batch)continue;
                if(s>nb_stack)continue;
                if(a>grid_c)continue;
                if(b>grid_d)continue;
                            int z_row = b + grid_d*(a + grid_c*
                                                    (s + nb_stack*n));
                            // loop over c
                            for (int i = threadIdx.y; i < c; i+=blockDim.y)
                            {
                                int ten4_2 = i + a * step_x;
                                if("%(mode)s"=="wrap_centered"){
                                    ten4_2 -= wrap_centered_idx_shift_x;
                                    if ( ten4_2 < 0 )
                                        ten4_2 += height;
                                    else if (ten4_2 >= height)
                                        ten4_2 -= height;
                                }
                                // loop over d
                                for (int j = threadIdx.x; j < d; j+=blockDim.x)
                                {
                                    int ten4_3 = j + b * step_y;
                                    if("%(mode)s"=="wrap_centered"){
                                        ten4_3 -= wrap_centered_idx_shift_y;
                                        if ( ten4_3 < 0 )
                                            ten4_3 += width;
                                        else if (ten4_3 >= width)
                                            ten4_3 -= width;
                                    }

                                    int ten4_idx = stride3*ten4_3 +
                                                   stride2*ten4_2 +
                                                   stride1*s + stride0*n;

                                    int z_col = j + d * i;
                                    int z_idx = z_col * out_s1 +
                                                z_row * out_s0;
                                    global_out[z_idx] = global_ten4[ten4_idx];
                                }
                            }
            }
        }
        """ % locals()
        params = [
            'intc', 'intc', 'intc', 'intc', 'intc', 'intc',
            'intc', 'intc', 'intc', 'intc',
            'uintp', 'uintp', 'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp',
            'uintp', 'uintp',
            gpuarray.GpuArray, 'uintp',
            ]
        kernels.append(Kernel(code=code, name=kname, params=params,
                              flags=flags, objvar=k_var))
        return kernels

    def c_code(self, node, name, inp, out, sub):
        if node.inputs[0].type.context.kind != 'cuda':
            raise NotImplementedError("cuda only")
        dtype_ten4 = node.inputs[0].dtype
        dtype_neib_shape = node.inputs[1].dtype
        dtype_neib_step = node.inputs[2].dtype
        dtype_z = node.outputs[0].dtype
        itemsize_ten4 = numpy.dtype(dtype_ten4).itemsize
        itemsize_z = numpy.dtype(dtype_z).itemsize
        typecode_z = pygpu.gpuarray.dtype_to_typecode(node.outputs[0].dtype)
        ten4, neib_shape, neib_step = inp
        z, = out
        fail = sub['fail']
        ctx = sub['params']
        mode = self.mode
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: *fptr: %%s.",
                             GpuKernel_error(fptr, err));
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
        int grid_c = -1;
        int grid_d = -1;

        {
            if (PyGpuArray_NDIM(%(ten4)s) != 4)
            {
                PyErr_Format(PyExc_TypeError,
                             "GpuImages2Neibs: pvals wrong rank");
                %(fail)s;
            }
            if (PyArray_NDIM(%(neib_shape)s) != 1)
            {
                PyErr_Format(PyExc_TypeError,
                             "GpuImages2Neibs: unis wrong rank");
                %(fail)s;
            }

            if (PyArray_DIMS(%(neib_shape)s)[0] != 2)
            {
                PyErr_Format(PyExc_ValueError,
                             "GpuImages2Neibs: neib_shape has to contain two"
                             " elements");
                %(fail)s;
            }

            const int c = *(npy_%(dtype_neib_shape)s*) PyArray_GETPTR1(
                                                     %(neib_shape)s, 0);
            const int d = *(npy_%(dtype_neib_shape)s*) PyArray_GETPTR1(
                                                     %(neib_shape)s, 1);
            const npy_intp step_x = (npy_intp) *(npy_%(dtype_neib_step)s*)
                                         PyArray_GETPTR1(%(neib_step)s, 0);
            const npy_intp step_y = (npy_intp) *(npy_%(dtype_neib_step)s*)
                                         PyArray_GETPTR1(%(neib_step)s, 1);

            if ( "%(mode)s" == "wrap_centered") {
                if (c%%2!=1 || d%%2!=1){
                    PyErr_Format(PyExc_TypeError,
        "GpuImages2Neibs: in mode wrap_centered need patch with odd shapes");
                    %(fail)s;
                }
                if ( PyGpuArray_DIMS(%(ten4)s)[2] < c ||
                     PyGpuArray_DIMS(%(ten4)s)[3] < d)
                {
                    PyErr_Format(PyExc_TypeError,
                                 "GpuImages2Neibs: in wrap_centered mode,"
                                 " don't support image shapes smaller then"
                                 " the patch shapes: neib_shape=(%%d,%%d),"
                                 " ten4[2:]=[%%d,%%d]",
                                 c, d, PyGpuArray_DIMS(%(ten4)s)[2],
                                 PyGpuArray_DIMS(%(ten4)s)[3]);
                    %(fail)s;
                }
                grid_c = ceil_intdiv(((PyGpuArray_DIMS(%(ten4)s))[2]),
                                     (size_t)step_x);
                grid_d = ceil_intdiv(((PyGpuArray_DIMS(%(ten4)s))[3]),
                                     (size_t)step_y);


            }else if ( "%(mode)s" == "valid") {
                if ( ((PyGpuArray_DIMS(%(ten4)s))[2] < c) ||
                     ((((PyGpuArray_DIMS(%(ten4)s))[2]-c) %% step_x)!=0))
                {
                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
                                 " neib_shape[0]=%%d, neib_step[0]=%%d and"
                                 " ten4.shape[2]=%%d not consistent",
                                 c, step_x,
                                 PyGpuArray_DIMS(%(ten4)s)[2]);
                    %(fail)s;
                }
                if ( ((PyGpuArray_DIMS(%(ten4)s))[3] < d) ||
                     ((((PyGpuArray_DIMS(%(ten4)s))[3]-d) %% step_y)!=0))
                {
                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
                                 " neib_shape[1]=%%d, neib_step[1]=%%d and"
                                 " ten4.shape[3]=%%d not consistent",
                                 d, step_y,
                                 PyGpuArray_DIMS(%(ten4)s)[3]);
                    %(fail)s;
                }
                //number of patch in height
                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]-c)/step_x);
                //number of patch in width
                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]-d)/step_y);
            }else if ( "%(mode)s" == "ignore_borders") {
                //number of patch in height
                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]-c)/step_x);
                //number of patch in width
                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]-d)/step_y);
            }else{
                PyErr_Format(PyExc_TypeError,
                             "GpuImages2Neibs:: unknown mode '%(mode)s'");
                 %(fail)s;
            }

            // new dimensions for z
            const int z_dim1 = c * d;
            const int z_dim0 =  grid_c
                                * grid_d
                                * PyGpuArray_DIMS(%(ten4)s)[1]
                                * PyGpuArray_DIMS(%(ten4)s)[0];

            if ((NULL == %(z)s)
                || (PyGpuArray_DIMS(%(z)s)[0] != z_dim0)
                || (PyGpuArray_DIMS(%(z)s)[1] != z_dim1))
            {
                Py_XDECREF(%(z)s);
                size_t dims[2];
                dims[0] = z_dim0;
                dims[1] = z_dim1;
                %(z)s = pygpu_empty(2, dims, %(typecode_z)s,
                                    GA_C_ORDER, %(ctx)s, Py_None);
                if (!%(z)s)
                {
                    PyErr_SetString(PyExc_MemoryError, "GpuImages2Neibs:"
                                    " failed to alloc z output");
                    %(fail)s;
                }
            }

        }

        { // NESTED SCOPE

            const int nb_batch = PyGpuArray_DIMS(%(ten4)s)[0];
            const int nb_stack = PyGpuArray_DIMS(%(ten4)s)[1];
            const int height = PyGpuArray_DIMS(%(ten4)s)[2];
            const int width = PyGpuArray_DIMS(%(ten4)s)[3];

            const int c = *(npy_%(dtype_neib_shape)s*) PyArray_GETPTR1(
                                                     %(neib_shape)s, 0);
            const int d = *(npy_%(dtype_neib_shape)s*) PyArray_GETPTR1(
                                                     %(neib_shape)s, 1);
            const npy_intp step_x = (npy_intp) *(npy_%(dtype_neib_step)s*)
                                         PyArray_GETPTR1(%(neib_step)s, 0);
            const npy_intp step_y = (npy_intp) *(npy_%(dtype_neib_step)s*)
                                         PyArray_GETPTR1(%(neib_step)s, 1);

            size_t threads_per_block[3] = {d, c, 1};
            //Their is a max of 512 threads per blocks
            while(threads_per_block[0]*threads_per_block[1]>512 && threads_per_block[1]>1)threads_per_block[1]--;
            while(threads_per_block[0]*threads_per_block[1]>512 && threads_per_block[0]>1)threads_per_block[0]--;

            //Make bigger block to have better memory access pattern and
            //a higher core utilisation. for smaller patch size

            while(c*d*(threads_per_block[2]+1) < 128 && threads_per_block[2]<64 &&
                  threads_per_block[2]<PyGpuArray_DIMS(%(z)s)[0]){
                threads_per_block[2]++;
            }
            int nb_block;
            if (PyGpuArray_DIMS(%(z)s)[0] %% threads_per_block[2] == 0)
                nb_block = PyGpuArray_DIMS(%(z)s)[0] / threads_per_block[2];
            else
                nb_block = (PyGpuArray_DIMS(%(z)s)[0] / threads_per_block[2]) + 1;
            size_t n_blocks[3] = {std::min(32*1024,nb_block), 1, 1};

            GpuKernel *fptr;
            if(threads_per_block[0]==d && threads_per_block[1]==c){
                fptr = &k_multi_warp_less_%(name)s;
            }else{
                fptr = &k_multi_warp_%(name)s;
            }

            size_t stride_A0 = PyGpuArray_STRIDES(%(ten4)s)[0] / %(itemsize_ten4)s;
            size_t stride_A1 = PyGpuArray_STRIDES(%(ten4)s)[1] / %(itemsize_ten4)s;
            size_t stride_A2 = PyGpuArray_STRIDES(%(ten4)s)[2] / %(itemsize_ten4)s;
            size_t stride_A3 = PyGpuArray_STRIDES(%(ten4)s)[3] / %(itemsize_ten4)s;
            size_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s;
            size_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s;
            void *kernel_params[] = {(void *)&nb_batch,
                                     (void *)&nb_stack,
                                     (void *)&height, (void *)&width,
                                     (void *)&c, (void *)&d,
                                     (void *)&step_x, (void *)&step_y,
                                     (void *)&grid_c, (void *)&grid_d,
                                     (void *)&stride_A0,
                                     (void *)&stride_A1,
                                     (void *)&stride_A2,
                                     (void *)&stride_A3,
                                     (void *)%(ten4)s->ga.data,
                                     (void *)&%(ten4)s->ga.offset,
                                     (void *)&stride_Z0,
                                     (void *)&stride_Z1,
                                     (void *)%(z)s->ga.data,
                                     (void *)&%(z)s->ga.offset};
            int err = GpuKernel_call(fptr, 3, threads_per_block, n_blocks, 0, kernel_params);
            %(err_check)s
            %(sync)s
        } // END NESTED SCOPE
        """ % locals()

    def perform(self, node, inp, out, ctx):
        # Disable the perform method from the CPU version
        Op.perform(self, node, inp, out, ctx)


@op_lifter([Images2Neibs])
def use_gpu_images2neibs(node, context_name):
    if node.op.mode in ['valid', 'ignore_borders', 'wrap_centered']:
        return GpuImages2Neibs(node.op.mode)

register_gpu_opt()(use_gpu_images2neibs)
