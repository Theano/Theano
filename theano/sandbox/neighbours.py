import theano
from theano import Op, Apply
import theano.tensor as T
from theano.gof import local_optimizer
from theano.sandbox.cuda import cuda_available, GpuOp

if cuda_available:
    from theano.sandbox.cuda import CudaNdarrayType
    from theano.sandbox.cuda.basic_ops import host_from_gpu, gpu_from_host
    from theano.sandbox.cuda.opt import register_opt as register_gpu_opt


class BadOldCode(Exception):
    """
    We create a specific Exception to be sure it does not get caught by
    mistake.
    """
    pass


class Images2Neibs(Op):
    def __init__(self, mode='valid'):
        """
        Modes:
            valid : Reshapes the input as a a 2D tensor where each row is a pooling example.
                Requires an input that is a multiple of the pooling factor (in each direction)
            ignore_borders : Same as valid, but will ignore the borders if the shape(s) of the input
                is not a multiple of the pooling factor(s)
            wrap_centered : ?? TODO comment
        """
        if mode not in ['valid', 'wrap_centered', 'ignore_borders']:
            raise NotImplementedError("Only the mode valid, ignore_borders"
                                      " and wrap_centered have been"
                                      " implemented for the op Images2Neibs")
        self.mode = mode

    def __eq__(self, other):
        return type(self) == type(other) and self.mode == other.mode

    def __hash__(self):
        return hash(type(self)) ^ hash(self.mode)

    def __str__(self):
        return self.__class__.__name__ + "{%s}" % self.mode

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "mode"):
            self.mode = 'valid'

    def make_node(self, ten4, neib_shape, neib_step=None):
        """
        :param ten4:     a list of lists of images
                         ten4 is of shape (list 1 dim, list 2 dim,
                                           row, col)
        :param neigb:    (r,c) where r is the height of the neighborhood
                        in rows and c is the width of the neighborhood
                        in columns
        :param neib_step: (dr,dc) where dr is the number of rows to
                          skip between patch and dc is the number of
                          columns. When None, this is the same as
                          neib_shape(patch are disjoint)

        output:
            a 2D matrix, written using the following pattern

            idx = 0
            for i in xrange(list 1 dim)
                for j in xrange(list 2 dim)
                    for k in <image column coordinates>
                        for l in <image row coordinates>
                            output[idx,:] = flattened version of ten4[i,j,l:l+r,k:k+c]
                            idx += 1
            (note: the op isn't necessarily implemented internally with these
            for loops, they're just the easiest way to describe the output pattern)
        """
        ten4 = T.as_tensor_variable(ten4)
        neib_shape = T.as_tensor_variable(neib_shape)
        if neib_step is None:
            neib_step = neib_shape
        else:
            neib_step = T.as_tensor_variable(neib_step)

        assert ten4.ndim == 4
        assert neib_shape.ndim == 1
        assert neib_step.ndim == 1

        return Apply(self, [ten4, neib_shape, neib_step],
                     [T.matrix(dtype=ten4.type.dtype)])

    def grad(self, inp, grads):
        x, neib_shape, neib_step = inp
        gz, = grads
        if self.mode in ['valid', 'ignore_borders']:
            raise BadOldCode("The Images2Neibs grad is not implemented."
                            " It was in the past, but returned the wrong"
                            " answer!")
            # This is the reverse of the op, not the grad!
            return [neibs2images(gz, neib_shape, x.shape, mode=self.mode),
                    None, None]
        else:
            raise NotImplementedError()

    def c_code_cache_version(self):
        return (5,)

    def c_code(self, node, name, inp, out, sub):
        ten4, neib_shape, neib_step = inp
        z, = out

        fail = sub['fail']
        mode = self.mode
        return """
#ifndef CEIL_INTDIV
#define CEIL_INTDIV(a, b) ((a/b) + ((a %% b) ? 1: 0))
#endif

        int grid_c = -1; //number of patch in height
        int grid_d = -1; //number of patch in width
        {
        if (%(ten4)s->nd != 4)
        {
            PyErr_Format(PyExc_TypeError, "ten4 wrong rank");
            %(fail)s;
        }
        if (%(neib_shape)s->nd != 1)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong rank");
            %(fail)s;
        }
        if ( (%(neib_shape)s->dimensions)[0] != 2)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong shape ; has to"
                                          " contain 2 elements");
            %(fail)s;
        }
        if (%(neib_step)s->nd != 1)
        {
            PyErr_Format(PyExc_TypeError, "neib_step wrong rank");
            %(fail)s;
        }
        if ( (%(neib_step)s->dimensions)[0] != 2)
        {
            PyErr_Format(PyExc_TypeError,
                         "neib_step wrong step ; has to contain 2 elements");
            %(fail)s;
        }

        // (c,d) = neib_shape
        const npy_intp c = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
        const npy_intp d = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);
        // (step_x,step_y) = neib_step
        const npy_intp step_x = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 0);
        const npy_intp step_y = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 1);

        if ( "%(mode)s" == "wrap_centered") {
            if (c%%2!=1 || d%%2!=1){
                PyErr_Format(PyExc_TypeError, "Images2Neibs: in mode wrap_centered need patch with odd shapes");
                %(fail)s;
            }
            if ( (%(ten4)s->dimensions)[2] < c || (%(ten4)s->dimensions)[3] < d)
            {
                PyErr_Format(PyExc_TypeError, "Images2Neibs: in wrap_centered mode, don't support image shapes smaller then the patch shapes: neib_shape=(%%ld,%%ld), ten4[2:]=[%%ld,%%ld]",
                             (long int)c, (long int)d, (long int)(%(ten4)s->dimensions[2]), (long int)(%(ten4)s->dimensions[3]));
                %(fail)s;
            }
            grid_c = CEIL_INTDIV(((%(ten4)s->dimensions)[2]),step_x);
            grid_d = CEIL_INTDIV(((%(ten4)s->dimensions)[3]),step_y);

        }else if ( "%(mode)s" == "valid") {
            if ( ((%(ten4)s->dimensions)[2] < c) ||( (((%(ten4)s->dimensions)[2]-c) %% step_x)!=0))
            {
                PyErr_Format(PyExc_TypeError, "neib_shape[0]=%%ld, neib_step[0]=%%ld and ten4.shape[2]=%%ld not consistent",
                             (long int)c, (long int)step_x, (long int)(%(ten4)s->dimensions[2]));
                %(fail)s;
            }
            if ( ((%(ten4)s->dimensions)[3] < d) ||( (((%(ten4)s->dimensions)[3]-d) %% step_y)!=0))
            {
                PyErr_Format(PyExc_TypeError, "neib_shape[1]=%%ld, neib_step[1]=%%ld and ten4.shape[3]=%%ld not consistent",
                             (long int)d, (long int)step_y, (long int)(%(ten4)s->dimensions[3]));
                %(fail)s;
            }
            grid_c = 1+(((%(ten4)s->dimensions)[2]-c)/step_x); //number of patch in height
            grid_d = 1+(((%(ten4)s->dimensions)[3]-d)/step_y); //number of patch in width
        }else if ( "%(mode)s" == "ignore_borders") {
            grid_c = 1+(((%(ten4)s->dimensions)[2]-c)/step_x); //number of patch in height
            grid_d = 1+(((%(ten4)s->dimensions)[3]-d)/step_y); //number of patch in width
        }else{
            PyErr_Format(PyExc_TypeError, "Images2Neibs: unknow mode '%(mode)s'");
            %(fail)s;
        }

        // new dimensions for z
        const npy_intp z_dim1 = c * d;
        const npy_intp z_dim0 =  grid_c
                            * grid_d
                            * (%(ten4)s->dimensions)[1]
                            * (%(ten4)s->dimensions)[0];

        if ((NULL == %(z)s)
            || ((%(z)s->dimensions)[0] != z_dim0 )
            || ((%(z)s->dimensions)[1] != z_dim1 )
        )
        {
            Py_XDECREF(%(z)s);
            npy_intp dims[2];
            dims[0] = z_dim0;
            dims[1] = z_dim1;

            %(z)s = (PyArrayObject*) PyArray_EMPTY(2,
                dims,
                type_num_%(ten4)s,
                0);

            if (!%(z)s)
            {
                PyErr_SetString(PyExc_MemoryError, "failed to alloc z output");
                %(fail)s;
            }
        }
        }

        { // NESTED SCOPE

        const int nb_batch = (%(ten4)s->dimensions)[0];
        const int nb_stack = (%(ten4)s->dimensions)[1];
        const int height = (%(ten4)s->dimensions)[2];
        const int width = (%(ten4)s->dimensions)[3];

        // (c,d) = neib_shape
        const npy_intp c = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
        const npy_intp d = (npy_intp) *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);
        // (step_x,step_y) = neib_step
        const npy_intp step_x = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 0);
        const npy_intp step_y = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 1);

        const int wrap_centered_idx_shift_x = c/2;
        const int wrap_centered_idx_shift_y = d/2;
        // Oh this is messed up...
        for (int n = 0; n < nb_batch; n++)              // loop over batches
            for (int s = 0; s < nb_stack; s++)          // loop over stacks
                for (int a = 0; a < grid_c; a++)        // loop over the number of patch in height
                    for (int b = 0; b < grid_d; b++)    // loop over the number of patch in width
                    {
                        int z_row = b + grid_d*(a + grid_c*(s + nb_stack*n));
                        for (int i = 0; i < c; i++)     // loop over c
                        {
                            int ten4_2 = i + a * step_x;
                            if ( "%(mode)s" == "wrap_centered" ){
                                ten4_2 -= wrap_centered_idx_shift_x;
                                if ( ten4_2 < 0 ) ten4_2 += height;
                                else if (ten4_2 >= height) ten4_2 -= height;
                            }
                            for (int j = 0; j < d; j++)  // loop over d
                            {

                                int ten4_3 = j + b * step_y;
                                if ( "%(mode)s" == "wrap_centered" ){
                                    ten4_3 -= wrap_centered_idx_shift_y;
                                    if ( ten4_3 < 0 ) ten4_3 += width;
                                    else if (ten4_3 >= width) ten4_3 -= width;
                                }
                                int z_col = j + d * i;

                                dtype_%(z)s* curr_z = (dtype_%(z)s*) PyArray_GETPTR2(%(z)s, z_row, z_col);
                                *curr_z = *( (dtype_%(ten4)s*) PyArray_GETPTR4(%(ten4)s, n, s, ten4_2, ten4_3));

                                //printf("\\n(%%i,%%i,%%i,%%i) --> (%%i,%%i)",n,s, ten4_2, ten4_3, z_row, z_col);
                                //printf("%%f ", *curr_z);
                            }
                        }
                    }
        } // END NESTED SCOPE
        """ % locals()


def images2neibs(ten4, neib_shape, neib_step=None, mode='valid'):
    return Images2Neibs(mode)(ten4, neib_shape, neib_step)


def neibs2images(neibs, neib_shape, original_shape, mode='valid'):
    """
    Inverse of images2neib.

    neibs : matrix like the one obtained by images2neib
    neib_shape : neib_shape that was used in images2neib
    original_shape : original shape of the 4d tensor given to images2neib

    Return a 4d tensor of shape `original_shape`.
    """
    neibs = T.as_tensor_variable(neibs)
    neib_shape = T.as_tensor_variable(neib_shape)
    original_shape = T.as_tensor_variable(original_shape)

    new_neib_shape = T.stack(original_shape[-1] // neib_shape[1],
                             neib_shape[1])
    output_2d = images2neibs(neibs.dimshuffle('x', 'x', 0, 1),
                             new_neib_shape, mode=mode)

    if mode == 'ignore_borders':
        valid_shape = list(original_shape)
        valid_shape[2] = (valid_shape[2] // neib_shape[0]) * neib_shape[0]
        valid_shape[3] = (valid_shape[3] // neib_shape[1]) * neib_shape[1]
        output_4d = output_2d.reshape(valid_shape)
        #padding the borders with zeros
        for d in [2, 3]:
            pad_shape = list(output_4d.shape)
            pad_shape[d] = original_shape[d] - valid_shape[d]
            output_4d = T.concatenate([output_4d, T.zeros(pad_shape)], axis=d)
    else:
        output_4d = output_2d.reshape(original_shape)

    return output_4d


# This is work in progress
class GpuImages2Neibs(Images2Neibs, GpuOp):
    def __init__(self, mode='valid'):
        if mode not in ['valid', 'wrap_centered']:
            raise NotImplementedError("Only the mode valid and wrap_centered"
                                      " have been implemented for the op"
                                      " GpuImages2Neibs")
        self.mode = mode

    def make_node(self, ten4, neib_shape, neib_step):
        assert ten4.dtype == 'float32'
        if not isinstance(ten4.type, CudaNdarrayType):
            raise TypeError('ten4 must be cudandarray', ten4)

        assert ten4.ndim == 4
        assert neib_shape.ndim == 1
        assert neib_step.ndim == 1

        return Apply(self, [ten4, neib_shape, neib_step],
                     [CudaNdarrayType(broadcastable=(False, False),
                                      dtype=ten4.type.dtype)()])

    def c_code_cache_version(self):
        return (8,)

    def c_support_code_apply(self, node, nodename):
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
            const int stride0, const int stride1, const int stride2, const int stride3,
            float * global_ten4,
            const int out_s0, const int out_s1,
            float * global_out
        )
        {
            const int wrap_centered_idx_shift_x = c/2;
            const int wrap_centered_idx_shift_y = d/2;

            for(int tblock = blockIdx.x*blockDim.z+threadIdx.z;tblock<nb_batch*nb_stack*grid_c*grid_d;tblock+=gridDim.x*blockDim.z){
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
                            int z_row = b + grid_d*(a + grid_c*(s + nb_stack*n));
                            int i = threadIdx.y;     // loop over c
                            {
                                int ten4_2 = i + a * step_x;
                                if("%(mode)s"=="wrap_centered"){
                                    ten4_2 -= wrap_centered_idx_shift_x;
                                    if ( ten4_2 < 0 ) ten4_2 += height;
                                    else if (ten4_2 >= height) ten4_2 -= height;
                                }
                                int j = threadIdx.x;  // loop over d
                                {
                                    int ten4_3 = j + b * step_y;
                                    if("%(mode)s"=="wrap_centered"){
                                        ten4_3 -= wrap_centered_idx_shift_y;
                                        if ( ten4_3 < 0 ) ten4_3 += width;
                                        else if (ten4_3 >= width) ten4_3 -= width;
                                    }

                                    //int ten4_idx = ten4_3 + width*(ten4_2 + height*(s +nb_stack*n));
                                    //int ten4_idx = stride3*ten4_3 + stride2*(ten4_2 + stride1*(s + stride0*n));
                                    int ten4_idx = stride3*ten4_3 + stride2*ten4_2 + stride1*s + stride0*n;

                                    int z_col = j + d * i;
                                    int z_idx = z_col * out_s1 + z_row * out_s0;
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
            const int stride0, const int stride1, const int stride2, const int stride3,
            float * global_ten4,
            const int out_s0, const int out_s1,
            float * global_out
        )
        {
            const int wrap_centered_idx_shift_x = c/2;
            const int wrap_centered_idx_shift_y = d/2;

            for(int tblock = blockIdx.x*blockDim.z+threadIdx.z;tblock<nb_batch*nb_stack*grid_c*grid_d;tblock+=gridDim.x*blockDim.z){
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
                            int z_row = b + grid_d*(a + grid_c*(s + nb_stack*n));
                            for (int i = threadIdx.y; i < c; i+=blockDim.y)     // loop over c
                            {
                                int ten4_2 = i + a * step_x;
                                if("%(mode)s"=="wrap_centered"){
                                    ten4_2 -= wrap_centered_idx_shift_x;
                                    if ( ten4_2 < 0 ) ten4_2 += height;
                                    else if (ten4_2 >= height) ten4_2 -= height;
                                }
                                for (int j = threadIdx.x; j < d; j+=blockDim.x)  // loop over d
                                {
                                    int ten4_3 = j + b * step_y;
                                    if("%(mode)s"=="wrap_centered"){
                                        ten4_3 -= wrap_centered_idx_shift_y;
                                        if ( ten4_3 < 0 ) ten4_3 += width;
                                        else if (ten4_3 >= width) ten4_3 -= width;
                                    }

                                    //int ten4_idx = ten4_3 + width*(ten4_2 + height*(s +nb_stack*n));
                                    //int ten4_idx = stride3*ten4_3 + stride2*(ten4_2 + stride1*(s + stride0*n));
                                    int ten4_idx = stride3*ten4_3 + stride2*ten4_2 + stride1*s + stride0*n;

                                    int z_col = j + d * i;
                                    int z_idx = z_col * out_s1 + z_row * out_s0;
                                    global_out[z_idx] = global_ten4[ten4_idx];
                                }
                            }
            }
        }

        """ % locals()

    def c_code(self, node, name, inp, out, sub):
        ten4, neib_shape, neib_step = inp
        z, = out
        fail = sub['fail']
        mode = self.mode
        return """
#ifndef CEIL_INTDIV
#define CEIL_INTDIV(a, b) ((a/b) + ((a %% b) ? 1: 0))
#endif

        int grid_c = -1;
        int grid_d = -1;

        {
            if (%(ten4)s->nd != 4)
            {
                PyErr_Format(PyExc_TypeError, "pvals wrong rank");
                %(fail)s;
            }
            if (%(neib_shape)s->nd != 1)
            {
                PyErr_Format(PyExc_TypeError, "unis wrong rank");
                %(fail)s;
            }

            if (%(neib_shape)s->dimensions[0] != 2)
            {
                PyErr_Format(PyExc_ValueError, "neib_shape has to contain two elements");
                %(fail)s;
            }

            const int c = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
            const int d = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);
            const npy_intp step_x = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 0);
            const npy_intp step_y = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 1);

            if ( "%(mode)s" == "wrap_centered") {
                if (c%%2!=1 || d%%2!=1){
                    PyErr_Format(PyExc_TypeError, "Images2Neibs: in mode wrap_centered need patch with odd shapes");
                    %(fail)s;
                }
                if ( CudaNdarray_HOST_DIMS(%(ten4)s)[2] < c || CudaNdarray_HOST_DIMS(%(ten4)s)[3] < d)
                {
                    PyErr_Format(PyExc_TypeError, "Images2Neibs: in wrap_centered mode, don't support image shapes smaller then the patch shapes: neib_shape=(%%d,%%d), ten4[2:]=[%%d,%%d]",
                                 c, d, CudaNdarray_HOST_DIMS(%(ten4)s)[2], CudaNdarray_HOST_DIMS(%(ten4)s)[3]);
                    %(fail)s;
                }
                grid_c = CEIL_INTDIV(((CudaNdarray_HOST_DIMS(%(ten4)s))[2]),
                                     step_x);
                grid_d = CEIL_INTDIV(((CudaNdarray_HOST_DIMS(%(ten4)s))[3]),
                                     step_y);


            }else if ( "%(mode)s" == "valid") {
                if ( ((CudaNdarray_HOST_DIMS(%(ten4)s))[2] < c) ||( (((CudaNdarray_HOST_DIMS(%(ten4)s))[2]-c) %% step_x)!=0))
                {
                    PyErr_Format(PyExc_TypeError, "neib_shape[0]=%%d, neib_step[0]=%%d and ten4.shape[2]=%%d not consistent",
                                 c, step_x, CudaNdarray_HOST_DIMS(%(ten4)s)[2]);
                    %(fail)s;
                }
                if ( ((CudaNdarray_HOST_DIMS(%(ten4)s))[3] < d) ||( (((CudaNdarray_HOST_DIMS(%(ten4)s))[3]-d) %% step_y)!=0))
                {
                    PyErr_Format(PyExc_TypeError, "neib_shape[1]=%%d, neib_step[1]=%%d and ten4.shape[3]=%%d not consistent",
                                 d, step_y, CudaNdarray_HOST_DIMS(%(ten4)s)[3]);
                    %(fail)s;
                }
                grid_c = 1+(((CudaNdarray_HOST_DIMS(%(ten4)s))[2]-c)/step_x); //number of patch in height
                grid_d = 1+(((CudaNdarray_HOST_DIMS(%(ten4)s))[3]-d)/step_y); //number of patch in width
            }else{
                PyErr_Format(PyExc_TypeError, "Images2Neibs: unknow mode '%(mode)s'");
                 %(fail)s;
            }

            // new dimensions for z
            const int z_dim1 = c * d;
            const int z_dim0 =  grid_c
                                * grid_d
                                * CudaNdarray_HOST_DIMS(%(ten4)s)[1]
                                * CudaNdarray_HOST_DIMS(%(ten4)s)[0];

            if ((NULL == %(z)s)
                || (CudaNdarray_HOST_DIMS(%(z)s)[0] != z_dim0)
                || (CudaNdarray_HOST_DIMS(%(z)s)[1] != z_dim1))
            {
                Py_XDECREF(%(z)s);
                npy_intp dims[2];
                dims[0] = z_dim0;
                dims[1] = z_dim1;
                %(z)s = (CudaNdarray*)CudaNdarray_NewDims(2, dims);
                if (!%(z)s)
                {
                    PyErr_SetString(PyExc_MemoryError,
                                    "failed to alloc z output");
                    %(fail)s;
                }
            }

        }

        { // NESTED SCOPE

            const int nb_batch = CudaNdarray_HOST_DIMS(%(ten4)s)[0];
            const int nb_stack = CudaNdarray_HOST_DIMS(%(ten4)s)[1];
            const int height = CudaNdarray_HOST_DIMS(%(ten4)s)[2];
            const int width = CudaNdarray_HOST_DIMS(%(ten4)s)[3];

            const int c = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 0);
            const int d = *(dtype_%(neib_shape)s*) PyArray_GETPTR1(%(neib_shape)s, 1);
            const npy_intp step_x = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 0);
            const npy_intp step_y = (npy_intp) *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 1);

            dim3 n_threads(d,c,1);
            //Their is a max of 512 threads per blocks
            while(n_threads.x*n_threads.y>512 && n_threads.y>1)n_threads.y--;
            while(n_threads.x*n_threads.y>512 && n_threads.x>1)n_threads.x--;

            //Make bigger block to have better memory access pattern and a higher core utilisation.
            //for smaller patch size
            while(c*d*(n_threads.z+1) < 128 && n_threads.z<64 && n_threads.z<CudaNdarray_HOST_DIMS(%(z)s)[0]){
                n_threads.z++;
            }
            int nb_block;
            if (CudaNdarray_HOST_DIMS(%(z)s)[0] %% n_threads.z == 0)
                nb_block = CudaNdarray_HOST_DIMS(%(z)s)[0] / n_threads.z;
            else
                nb_block = (CudaNdarray_HOST_DIMS(%(z)s)[0] / n_threads.z) + 1;
            dim3 n_blocks(std::min(32*1024,nb_block));
            int n_shared = 0;

            void (*f)(int, int, int ,int,
                      int, int, int ,int,
                      int, int,
                      int, int, int, int,
                      float*,
                      int, int,
                      float*);
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
                CudaNdarray_HOST_STRIDES(%(ten4)s)[0],
                CudaNdarray_HOST_STRIDES(%(ten4)s)[1],
                CudaNdarray_HOST_STRIDES(%(ten4)s)[2],
                CudaNdarray_HOST_STRIDES(%(ten4)s)[3],
                CudaNdarray_DEV_DATA(%(ten4)s),
                CudaNdarray_HOST_STRIDES(%(z)s)[0],
                CudaNdarray_HOST_STRIDES(%(z)s)[1],
                CudaNdarray_DEV_DATA(%(z)s)
            );
            CNDA_THREAD_SYNC;
            cudaError_t sts = cudaGetLastError();
            if (cudaSuccess != sts)
            {
                PyErr_Format(PyExc_RuntimeError,
                             "Cuda error: %%s: %%s. (grid: %%i x %%i;"
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


def gpu_images2neibs(ten4, neib_shape, neib_step=None, mode='valid'):
    return GpuImages2Neibs(mode)(ten4, neib_shape, neib_step)


@local_optimizer()
def use_gpu_images2neibs(node):
    if type(node.op) is Images2Neibs:
        return [host_from_gpu(gpu_images2neibs(gpu_from_host(node.inputs[0]),
                                               node.inputs[1], node.inputs[2],
                                               mode=node.op.mode))]

if cuda_available:
    register_gpu_opt()(use_gpu_images2neibs)
