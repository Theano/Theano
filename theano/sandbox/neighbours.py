"""
TODO: implement Images2Neibs.{perform,infer_shape}() methods

"""
from theano import Op, Apply
import theano.tensor as T
from theano.gradient import grad_not_implemented
from theano.gradient import grad_undefined


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
                            output[idx,:]
                                 = flattened version of ten4[i,j,l:l+r,k:k+c]
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
            if (neib_shape is neib_step or
                neib_shape == neib_step or
                # Theano Constant == do not compare the data
                # the equals function do that.
                (hasattr(neib_shape, "equals") and
                 neib_shape.equals(neib_step))):
                return [neibs2images(gz, neib_shape, x.shape, mode=self.mode),
                        grad_undefined(self, 1, neib_shape),
                        grad_undefined(self, 2, neib_step)]
        return [grad_not_implemented(self, 0, x),
                grad_undefined(self, 1, neib_shape),
                grad_undefined(self, 2, neib_step)]

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
        if (PyArray_NDIM(%(ten4)s) != 4)
        {
            PyErr_Format(PyExc_TypeError, "ten4 wrong rank");
            %(fail)s;
        }
        if (PyArray_NDIM(%(neib_shape)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong rank");
            %(fail)s;
        }
        if ( (PyArray_DIMS(%(neib_shape)s))[0] != 2)
        {
            PyErr_Format(PyExc_TypeError, "neib_shape wrong shape ; has to"
                                          " contain 2 elements");
            %(fail)s;
        }
        if (PyArray_NDIM(%(neib_step)s) != 1)
        {
            PyErr_Format(PyExc_TypeError, "neib_step wrong rank");
            %(fail)s;
        }
        if ( (PyArray_DIMS(%(neib_step)s))[0] != 2)
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
            if ( (PyArray_DIMS(%(ten4)s))[2] < c || (PyArray_DIMS(%(ten4)s))[3] < d)
            {
                PyErr_Format(PyExc_TypeError, "Images2Neibs: in wrap_centered mode, don't support image shapes smaller then the patch shapes: neib_shape=(%%ld,%%ld), ten4[2:]=[%%ld,%%ld]",
                             (long int)c, (long int)d, (long int)(PyArray_DIMS(%(ten4)s)[2]), (long int)(PyArray_DIMS(%(ten4)s)[3]));
                %(fail)s;
            }
            grid_c = CEIL_INTDIV(((PyArray_DIMS(%(ten4)s))[2]),step_x);
            grid_d = CEIL_INTDIV(((PyArray_DIMS(%(ten4)s))[3]),step_y);

        }else if ( "%(mode)s" == "valid") {
            if ( ((PyArray_DIMS(%(ten4)s))[2] < c) ||( (((PyArray_DIMS(%(ten4)s))[2]-c) %% step_x)!=0))
            {
                PyErr_Format(PyExc_TypeError, "neib_shape[0]=%%ld, neib_step[0]=%%ld and ten4.shape[2]=%%ld not consistent",
                             (long int)c, (long int)step_x, (long int)(PyArray_DIMS(%(ten4)s)[2]));
                %(fail)s;
            }
            if ( ((PyArray_DIMS(%(ten4)s))[3] < d) ||( (((PyArray_DIMS(%(ten4)s))[3]-d) %% step_y)!=0))
            {
                PyErr_Format(PyExc_TypeError, "neib_shape[1]=%%ld, neib_step[1]=%%ld and ten4.shape[3]=%%ld not consistent",
                             (long int)d, (long int)step_y, (long int)(PyArray_DIMS(%(ten4)s)[3]));
                %(fail)s;
            }
            grid_c = 1+(((PyArray_DIMS(%(ten4)s))[2]-c)/step_x); //number of patch in height
            grid_d = 1+(((PyArray_DIMS(%(ten4)s))[3]-d)/step_y); //number of patch in width
        }else if ( "%(mode)s" == "ignore_borders") {
            grid_c = 1+(((PyArray_DIMS(%(ten4)s))[2]-c)/step_x); //number of patch in height
            grid_d = 1+(((PyArray_DIMS(%(ten4)s))[3]-d)/step_y); //number of patch in width
        }else{
            PyErr_Format(PyExc_TypeError, "Images2Neibs: unknow mode '%(mode)s'");
            %(fail)s;
        }

        // new dimensions for z
        const npy_intp z_dim1 = c * d;
        const npy_intp z_dim0 =  grid_c
                            * grid_d
                            * (PyArray_DIMS(%(ten4)s))[1]
                            * (PyArray_DIMS(%(ten4)s))[0];

        if ((NULL == %(z)s)
            || ((PyArray_DIMS(%(z)s))[0] != z_dim0 )
            || ((PyArray_DIMS(%(z)s))[1] != z_dim1 )
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

        const int nb_batch = (PyArray_DIMS(%(ten4)s))[0];
        const int nb_stack = (PyArray_DIMS(%(ten4)s))[1];
        const int height = (PyArray_DIMS(%(ten4)s))[2];
        const int width = (PyArray_DIMS(%(ten4)s))[3];

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
    elif mode == 'valid':
        # TODO: we do not implement all mode with this code.
        # Add a check for the good cases.
        output_4d = output_2d.reshape(original_shape)
    else:
        raise NotImplementedError("neibs2images do not support mode=%s" % mode)

    return output_4d
