import numpy

from theano import Op, Apply, config
from theano.gof import local_optimizer
from theano.sandbox.neighbours import Images2Neibs
import theano.tensor as T

try:
    import pygpu
    from pygpu import gpuarray, elemwise
except ImportError:
    pass

from theano.sandbox.gpuarray.basic_ops import (as_gpuarray_variable,
                                               host_from_gpu, gpu_from_host)
from theano.sandbox.gpuarray.opt import register_opt as register_gpu_opt
from theano.sandbox.gpuarray.opt import op_lifter as op_lifter
from theano.sandbox.gpuarray.type import GpuArrayType
from theano.sandbox.gpuarray.comp import NVCC_compiler


class GpuImages2Neibs(Images2Neibs, Op):
    def __init__(self, mode='valid'):
        if mode not in ['valid', 'ignore_borders', 'wrap_centered']:
            raise NotImplementedError("Only the mode valid, ignore_borders"
                                      " and wrap_centered"
                                      " have been implemented for the op"
                                      " GpuImages2Neibs")
        self.mode = mode

    def make_node(self, ten4, neib_shape, neib_step):

        assert ten4.ndim == 4
        assert neib_shape.ndim == 1
        assert neib_step.ndim == 1
        assert "int" in neib_shape.dtype
        assert "int" in neib_step.dtype

        ten4 = as_gpuarray_variable(ten4)
        neib_shape = T.as_tensor_variable(neib_shape)
        neib_step = T.as_tensor_variable(neib_step)

        return Apply(self, [ten4, neib_shape, neib_step],
                     [GpuArrayType(broadcastable=(False, False),
                                   dtype=ten4.type.dtype)()])

    def c_code_cache_version(self):
        return (9,1)

    def c_headers(self):
        return ['cuda.h', '<gpuarray/extension.h>', '<numpy_compat.h>',
                '<gpuarray/ext_cuda.h>']

    def c_compiler(self):
        return NVCC_compiler

    def c_init_code(self):
        return ['setup_ext_cuda();']

    def c_support_code_apply(self, node, nodename):
        dtype_ten4 = node.inputs[0].dtype
        dtype_z = node.outputs[0].dtype
        mode = self.mode
        return """
//a version that use less register but don't work in all case.
        static __global__ void k_multi_warp_less_%(nodename)s(
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
            const int stride0, const int stride1,
            const int stride2, const int stride3,
            npy_%(dtype_ten4)s * global_ten4,
            const int out_s0, const int out_s1,
            npy_%(dtype_z)s * global_out
        )
        {
            const int wrap_centered_idx_shift_x = c/2;
            const int wrap_centered_idx_shift_y = d/2;

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
        }

        static __global__ void k_multi_warp_%(nodename)s(
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
            const int stride0, const int stride1,
            const int stride2, const int stride3,
            npy_%(dtype_ten4)s * global_ten4,
            const int out_s0, const int out_s1,
            npy_%(dtype_z)s * global_out
        )
        {
            const int wrap_centered_idx_shift_x = c/2;
            const int wrap_centered_idx_shift_y = d/2;

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

    def c_code(self, node, name, inp, out, sub):
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
        mode = self.mode
        if config.gpuarray.sync:
            cnda_thread_sync = "GpuArray_sync(&%(z)s->ga);" % dict(z=z)
        else:
            cnda_thread_sync = ""
        return """
#ifndef CEIL_INTDIV
#define CEIL_INTDIV(a, b) ((a/b) + ((a %% b) ? 1: 0))
#endif

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
                grid_c = CEIL_INTDIV(((PyGpuArray_DIMS(%(ten4)s))[2]),
                                     step_x);
                grid_d = CEIL_INTDIV(((PyGpuArray_DIMS(%(ten4)s))[3]),
                                     step_y);


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
                                    GA_C_ORDER, pygpu_default_context(),
                                    Py_None);
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

            dim3 n_threads(d,c,1);
            //Their is a max of 512 threads per blocks
            while(n_threads.x*n_threads.y>512 && n_threads.y>1)n_threads.y--;
            while(n_threads.x*n_threads.y>512 && n_threads.x>1)n_threads.x--;

            //Make bigger block to have better memory access pattern and
            //a higher core utilisation. for smaller patch size

            while(c*d*(n_threads.z+1) < 128 && n_threads.z<64 &&
                  n_threads.z<PyGpuArray_DIMS(%(z)s)[0]){
                n_threads.z++;
            }
            int nb_block;
            if (PyGpuArray_DIMS(%(z)s)[0] %% n_threads.z == 0)
                nb_block = PyGpuArray_DIMS(%(z)s)[0] / n_threads.z;
            else
                nb_block = (PyGpuArray_DIMS(%(z)s)[0] / n_threads.z) + 1;
            dim3 n_blocks(std::min(32*1024,nb_block));
            int n_shared = 0;

            void (*f)(int, int, int ,int,
                      int, int, int ,int,
                      int, int,
                      int, int, int, int,
                      npy_%(dtype_ten4)s*,
                      int, int,
                      npy_%(dtype_z)s*);
            if(n_threads.x==d && n_threads.y==c){
                f = k_multi_warp_less_%(name)s;
            }else{
                f = k_multi_warp_%(name)s;
            }

            f<<<n_blocks, n_threads, n_shared>>>(
                nb_batch,
                nb_stack,
                height, width,
                c, d, step_x, step_y,
                grid_c, grid_d,
                PyGpuArray_STRIDES(%(ten4)s)[0] / %(itemsize_ten4)s,
                PyGpuArray_STRIDES(%(ten4)s)[1] / %(itemsize_ten4)s,
                PyGpuArray_STRIDES(%(ten4)s)[2] / %(itemsize_ten4)s,
                PyGpuArray_STRIDES(%(ten4)s)[3] / %(itemsize_ten4)s,
                (npy_%(dtype_ten4)s*)(
                                ((char *)cuda_get_ptr(%(ten4)s->ga.data)) +
                                %(ten4)s->ga.offset),
                PyGpuArray_STRIDES(%(z)s)[0] / %(itemsize_z)s,
                PyGpuArray_STRIDES(%(z)s)[1] / %(itemsize_z)s,
                (npy_%(dtype_z)s*)(((char *)cuda_get_ptr(%(z)s->ga.data)) +
                                   %(z)s->ga.offset)
            );
            %(cnda_thread_sync)s
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError, "GpuImages2Neibs:"
                             " Cuda error: %%s: %%s. (grid: %%i x %%i;"
                             " block: %%i x %%i x %%i; shared: %%i)\\n",
                    "k_multi_warp_%(name)s",
                    cudaGetErrorString(sts),
                    n_blocks.x,
                    n_blocks.y,
                    n_threads.x,
                    n_threads.y,
                    n_threads.z,
                    n_shared);
                %(fail)s;
            }

        } // END NESTED SCOPE
        """ % locals()

@op_lifter([Images2Neibs])
def use_gpu_images2neibs(node):
    if node.op.mode in ['valid', 'ignore_borders', 'wrap_centered']:
        return GpuImages2Neibs(node.op.mode)

register_gpu_opt()(use_gpu_images2neibs)
