from __future__ import absolute_import, print_function, division

from theano import Op, Apply
from theano.gof import ParamsType
from theano.tensor.nnet.neighbours import Images2Neibs
import theano.tensor as T

try:
    from pygpu import gpuarray
except ImportError:
    pass

from .basic_ops import (as_gpuarray_variable, GpuKernelBase, Kernel,
                        infer_context_name)
from .type import GpuArrayType, gpu_context_type


class GpuImages2Neibs(GpuKernelBase, Images2Neibs, Op):
    """
    Images2Neibs for the GPU.

    """
    params_type = ParamsType(mode=Images2Neibs.BORDER_MODE, context=gpu_context_type)

    def get_params(self, node):
        return self.params_type.get_params(self, context=node.inputs[0].type.context)

    def make_node(self, ten4, neib_shape, neib_step=None):
        ten4 = as_gpuarray_variable(ten4, infer_context_name(ten4))
        neib_shape = T.as_tensor_variable(neib_shape)
        if neib_step is None:
            neib_step = neib_shape
        else:
            neib_step = T.as_tensor_variable(neib_step)

        assert ten4.ndim == 4
        assert neib_shape.ndim == 1
        assert neib_step.ndim == 1
        assert neib_shape.dtype in T.integer_dtypes
        assert neib_step.dtype in T.integer_dtypes

        return Apply(self, [ten4, neib_shape, neib_step],
                     [GpuArrayType(broadcastable=(False, False),
                                   dtype=ten4.type.dtype,
                                   context_name=ten4.type.context_name)()])

    def c_code_cache_version(self):
        return (14,)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray/types.h>']

    def gpu_kernels(self, node, nodename):
        dtype_ten4 = node.inputs[0].dtype
        dtype_z = node.outputs[0].dtype
        flags = Kernel.get_flags(dtype_ten4, dtype_z)
        type_ten4 = gpuarray.dtype_to_ctype(dtype_ten4)
        type_z = gpuarray.dtype_to_ctype(dtype_z)
        # `BORDER_MODE`'s c_support_code() contains C constants definitions that are useful here.
        mode_constants = self.BORDER_MODE.c_support_code()
        kernels = []
        kname = "k_multi_warp_less"
        k_var = "k_multi_warp_less_" + nodename
        code = """#include "cluda.h"

        // a version that uses less registers but doesn't work in all cases.
        %(mode_constants)s
        KERNEL void %(kname)s(
            const ga_int mode,
            const ga_int nb_batch,
            const ga_int nb_stack,
            const ga_int height,
            const ga_int width,
            const ga_int c,
            const ga_int d,
            const ga_int step_x,
            const ga_int step_y,
            const ga_int grid_c,
            const ga_int grid_d,
            const ga_size stride0, const ga_size stride1,
            const ga_size stride2, const ga_size stride3,
            GLOBAL_MEM const %(type_ten4)s * global_ten4, const ga_size offset_ten4,
            const ga_size out_s0, const ga_size out_s1,
            GLOBAL_MEM %(type_z)s * global_out, const ga_size offset_out
        )
        {
            const ga_int wrap_centered_half_idx_shift_x = c/2;
            const ga_int wrap_centered_half_idx_shift_y = d/2;
            global_ten4 = (GLOBAL_MEM const %(type_ten4)s *)(((GLOBAL_MEM char *)global_ten4)+offset_ten4);
            global_out = (GLOBAL_MEM %(type_z)s *)(((GLOBAL_MEM char *)global_out)+offset_out);

            for(ga_int tblock = GID_0*LDIM_2+LID_2;
                tblock<nb_batch*nb_stack*grid_c*grid_d;
                tblock+=GDIM_0*LDIM_2){
                const ga_int b = tblock%%grid_d;
                ga_int left = tblock/grid_d;
                const ga_int a = left%%grid_c;
                left = left/grid_c;
                const ga_int s = left%%nb_stack;
                left = left/nb_stack;
                const ga_int n = left;

                if(n>nb_batch)continue;
                if(s>nb_stack)continue;
                if(a>grid_c)continue;
                if(b>grid_d)continue;
                            ga_int z_row = b + grid_d*(a + grid_c*
                                                    (s + nb_stack*n));
                            ga_int i = LID_1;     // loop over c
                            {
                                ga_int ten4_2 = i + a * step_x;
                                if(mode == MODE_WRAP_CENTERED) {
                                    ten4_2 -= wrap_centered_half_idx_shift_x;
                                    if ( ten4_2 < 0 )
                                        ten4_2 += height;
                                    else if (ten4_2 >= height)
                                        ten4_2 -= height;
                                } else if (mode == MODE_HALF) {
                                    ten4_2 -= wrap_centered_half_idx_shift_x;
                                } else if (mode == MODE_FULL) {
                                    ten4_2 -= c - 1;
                                }
                                ga_int j = LID_0;  // loop over d
                                {
                                    ga_int ten4_3 = j + b * step_y;
                                    if(mode == MODE_WRAP_CENTERED){
                                        ten4_3 -= wrap_centered_half_idx_shift_y;
                                        if ( ten4_3 < 0 )
                                            ten4_3 += width;
                                        else if (ten4_3 >= width)
                                            ten4_3 -= width;
                                    } else if (mode == MODE_HALF) {
                                        ten4_3 -= wrap_centered_half_idx_shift_y;
                                    } else if (mode == MODE_FULL) {
                                        ten4_3 -= d - 1;
                                    }

                                    ga_int z_col = j + d * i;
                                    ga_int z_idx = z_col * out_s1 +
                                                z_row * out_s0;
                                    if(ten4_2 < 0 || ten4_2 >= height || ten4_3 < 0 || ten4_3 >= width){
                                        global_out[z_idx] = 0;
                                    } else {
                                        ga_int ten4_idx = stride3*ten4_3 +
                                                       stride2*ten4_2 +
                                                       stride1*s + stride0*n;
                                        global_out[z_idx] = global_ten4[ten4_idx];
                                    }
                                }
                            }
            }
        }""" % dict(kname=kname, type_ten4=type_ten4, type_z=type_z, mode_constants=mode_constants)
        params = [
            'intc',
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
        code = """#include "cluda.h"

        %(mode_constants)s
        KERNEL void %(kname)s(
            const ga_int mode,
            const ga_int nb_batch,
            const ga_int nb_stack,
            const ga_int height,
            const ga_int width,
            const ga_int c,
            const ga_int d,
            const ga_int step_x,
            const ga_int step_y,
            const ga_int grid_c,
            const ga_int grid_d,
            const ga_size stride0, const ga_size stride1,
            const ga_size stride2, const ga_size stride3,
            GLOBAL_MEM const %(type_ten4)s * global_ten4, const ga_size offset_ten4,
            const ga_size out_s0, const ga_size out_s1,
            GLOBAL_MEM %(type_z)s * global_out, const ga_size offset_out
        )
        {
            const ga_int wrap_centered_half_idx_shift_x = c/2;
            const ga_int wrap_centered_half_idx_shift_y = d/2;
            global_ten4 = (GLOBAL_MEM const %(type_ten4)s *)(((GLOBAL_MEM char *)global_ten4)+offset_ten4);
            global_out = (GLOBAL_MEM %(type_z)s *)(((GLOBAL_MEM char *)global_out)+offset_out);

            for(ga_int tblock = GID_0*LDIM_2+LID_2;
                tblock<nb_batch*nb_stack*grid_c*grid_d;
                tblock+=GDIM_0*LDIM_2){
                const ga_int b = tblock%%grid_d;
                ga_int left = tblock/grid_d;
                const ga_int a = left%%grid_c;
                left = left/grid_c;
                const ga_int s = left%%nb_stack;
                left = left/nb_stack;
                const ga_int n = left;

                if(n>nb_batch)continue;
                if(s>nb_stack)continue;
                if(a>grid_c)continue;
                if(b>grid_d)continue;
                            ga_int z_row = b + grid_d*(a + grid_c*
                                                    (s + nb_stack*n));
                            // loop over c
                            for (ga_int i = LID_1; i < c; i+=LDIM_1)
                            {
                                ga_int ten4_2 = i + a * step_x;
                                if(mode == MODE_WRAP_CENTERED) {
                                    ten4_2 -= wrap_centered_half_idx_shift_x;
                                    if ( ten4_2 < 0 )
                                        ten4_2 += height;
                                    else if (ten4_2 >= height)
                                        ten4_2 -= height;
                                } else if (mode == MODE_HALF) {
                                    ten4_2 -= wrap_centered_half_idx_shift_x;
                                } else if (mode == MODE_FULL) {
                                    ten4_2 -= c - 1;
                                }
                                // loop over d
                                for (ga_int j = LID_0; j < d; j+=LDIM_0)
                                {
                                    ga_int ten4_3 = j + b * step_y;
                                    if(mode == MODE_WRAP_CENTERED) {
                                        ten4_3 -= wrap_centered_half_idx_shift_y;
                                        if ( ten4_3 < 0 )
                                            ten4_3 += width;
                                        else if (ten4_3 >= width)
                                            ten4_3 -= width;
                                    } else if (mode == MODE_HALF) {
                                        ten4_3 -= wrap_centered_half_idx_shift_y;
                                    } else if (mode == MODE_FULL) {
                                        ten4_3 -= d - 1;
                                    }

                                    ga_int z_col = j + d * i;
                                    ga_int z_idx = z_col * out_s1 +
                                                z_row * out_s0;
                                    if(ten4_2 < 0 || ten4_2 >= height || ten4_3 < 0 || ten4_3 >= width){
                                        global_out[z_idx] = 0;
                                    } else {
                                        ga_int ten4_idx = stride3*ten4_3 +
                                                       stride2*ten4_2 +
                                                       stride1*s + stride0*n;
                                        global_out[z_idx] = global_ten4[ten4_idx];
                                    }
                                }
                            }
            }
        }
        """ % dict(kname=kname, type_ten4=type_ten4, type_z=type_z, mode_constants=mode_constants)
        params = [
            'intc',
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

    def c_support_code(self):
        return """
        template <typename T>
        static T ceil_intdiv(T a, T b)
        {
            return (a/b) + ((a % b) ? 1: 0);
        }
        """

    def c_code(self, node, name, inp, out, sub):
        err_check = """
            if (err != GA_NO_ERROR) {
                PyErr_Format(PyExc_RuntimeError,
                             "gpuarray error: *fptr: %%s.",
                             GpuKernel_error(fptr, err));
                %(fail)s;
            }
        """ % dict(fail=sub['fail'])

        # NB: To reduce C code variability:
        # For itemsize_ten4, I use GpuArray_ITEMSIZE(&ten4->ga) instead of np.dtype(node.inputs[0].dtype).itemsize
        # For itemsize_z, I use itemsize_ten4, as ten4 and z have same type properties (deduced from make_node)
        # For typecode_z, I use ten4->ga.typecode (for same reason as above)
        return """
        int grid_c = -1;
        int grid_d = -1;
        size_t itemsize_ten4 = GpuArray_ITEMSIZE(&%(ten4)s->ga);
        size_t itemsize_z = itemsize_ten4;
        int typecode_z = %(ten4)s->ga.typecode;

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

            if (step_x <=0 || step_y <=0)
            {
                PyErr_Format(PyExc_ValueError,
                             "neib_step wrong step ; values <= 0. Got %%lld %%lld.",
                             (long long) step_x, (long long) step_y);
                %(fail)s;
            }

            if (c <=0 || d <=0)
            {
                PyErr_Format(PyExc_ValueError,
                             "neib_shape values <= 0. Got %%lld %%lld.",
                             (long long)c, (long long)d);
                %(fail)s;
            }

            if (%(params)s->mode == MODE_WRAP_CENTERED) {
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


            } else if (%(params)s->mode == MODE_VALID) {
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
            } else if (%(params)s->mode == MODE_IGNORE_BORDERS) {
                //number of patch in height
                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]-c)/step_x);
                //number of patch in width
                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]-d)/step_y);
            } else if (%(params)s->mode == MODE_HALF) {
                if ( ((PyGpuArray_DIMS(%(ten4)s))[2] < c) ||
                     ((((PyGpuArray_DIMS(%(ten4)s))[2]-(c%%2)) %% step_x)!=0))
                {
                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
                                 " neib_shape[0]=%%d, neib_step[0]=%%d and"
                                 " ten4.shape[2]=%%d not consistent",
                                 c, step_x,
                                 PyGpuArray_DIMS(%(ten4)s)[2]);
                    %(fail)s;
                }
                if ( ((PyGpuArray_DIMS(%(ten4)s))[3] < d) ||
                     ((((PyGpuArray_DIMS(%(ten4)s))[3]-(d%%2)) %% step_y)!=0))
                {
                    PyErr_Format(PyExc_TypeError, "GpuImages2Neibs:"
                                 " neib_shape[1]=%%d, neib_step[1]=%%d and"
                                 " ten4.shape[3]=%%d not consistent",
                                 d, step_y,
                                 PyGpuArray_DIMS(%(ten4)s)[3]);
                    %(fail)s;
                }
                //number of patch in height
                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]-(c%%2))/step_x);
                //number of patch in width
                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]-(d%%2))/step_y);
            } else if (%(params)s->mode == MODE_FULL) {
                if ( ((PyGpuArray_DIMS(%(ten4)s))[2] < c) ||
                 ( (((PyGpuArray_DIMS(%(ten4)s))[2]+c-2) %% step_x)!=0))
                {
                    PyErr_Format(PyExc_TypeError,
                                 "neib_shape[0]=%%ld, neib_step[0]=%%ld and"
                                 " ten4.shape[2]=%%ld not consistent",
                                 (long int)c, (long int)step_x,
                                 (long int)(PyGpuArray_DIMS(%(ten4)s)[2]));
                    %(fail)s;
                }
                if ( ((PyGpuArray_DIMS(%(ten4)s))[3] < d) ||
                     ( (((PyGpuArray_DIMS(%(ten4)s))[3]+d-2) %% step_y)!=0))
                {
                    PyErr_Format(PyExc_TypeError,
                                 "neib_shape[1]=%%ld, neib_step[1]=%%ld and"
                                 " ten4.shape[3]=%%ld not consistent",
                                 (long int)d, (long int)step_y,
                                 (long int)(PyGpuArray_DIMS(%(ten4)s)[3]));
                    %(fail)s;
                }
                //number of patch in height
                grid_c = 1+(((PyGpuArray_DIMS(%(ten4)s))[2]+c-2)/step_x);
                //number of patch in width
                grid_d = 1+(((PyGpuArray_DIMS(%(ten4)s))[3]+d-2)/step_y);
            } else {
                PyErr_Format(PyExc_TypeError,
                             "GpuImages2Neibs:: unknown mode %%d", %(params)s->mode);
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
                %(z)s = pygpu_empty(2, dims, typecode_z,
                                    GA_C_ORDER, %(params)s->context, Py_None);
                if (!%(z)s)
                {
                    PyErr_SetString(PyExc_MemoryError, "GpuImages2Neibs:"
                                    " failed to alloc z output");
                    %(fail)s;
                }
            }

        }

        { // NESTED SCOPE

            const int mode = %(params)s->mode;
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
            //get the max threads per blocks
            size_t max_threads_dim;
            int err = gpucontext_property(%(params)s->context->ctx, GA_CTX_PROP_MAXLSIZE0, &max_threads_dim);
            if (err != GA_NO_ERROR){
                PyErr_SetString(PyExc_RuntimeError, "Could not fetch max_threads_dims");
                %(fail)s;
            }
            while(threads_per_block[0]*threads_per_block[1]>max_threads_dim && threads_per_block[1]>1)threads_per_block[1]--;
            while(threads_per_block[0]*threads_per_block[1]>max_threads_dim && threads_per_block[0]>1)threads_per_block[0]--;

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
            /*
            printf("%%zu %%zu %%zu %%zu %%zu %%zu %%zu\\n",
                   max_threads_dim, threads_per_block[0], threads_per_block[1], threads_per_block[2],
                   n_blocks[0], n_blocks[1], n_blocks[2]);
            */
            size_t stride_A0 = PyGpuArray_STRIDES(%(ten4)s)[0] / itemsize_ten4;
            size_t stride_A1 = PyGpuArray_STRIDES(%(ten4)s)[1] / itemsize_ten4;
            size_t stride_A2 = PyGpuArray_STRIDES(%(ten4)s)[2] / itemsize_ten4;
            size_t stride_A3 = PyGpuArray_STRIDES(%(ten4)s)[3] / itemsize_ten4;
            size_t stride_Z0 = PyGpuArray_STRIDES(%(z)s)[0] / itemsize_z;
            size_t stride_Z1 = PyGpuArray_STRIDES(%(z)s)[1] / itemsize_z;
            void *kernel_params[] = {(void *)&mode,
                                     (void *)&nb_batch,
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
            err = GpuKernel_call(fptr, 3, n_blocks, threads_per_block, 0, kernel_params);
            %(err_check)s
        } // END NESTED SCOPE
        """ % dict(ten4=inp[0], neib_shape=inp[1], neib_step=inp[2], z=out[0],
                   dtype_neib_shape=node.inputs[1].dtype,
                   dtype_neib_step=node.inputs[2].dtype,
                   err_check=err_check,
                   name=name,
                   params=sub['params'],
                   fail=sub['fail'])

    def perform(self, node, inp, out, params):
        # Disable the perform method from the CPU version
        Op.perform(self, node, inp, out, params)
