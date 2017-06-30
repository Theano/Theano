"""
TODO: implement Images2Neibs.infer_shape() methods

"""
from __future__ import absolute_import, print_function, division

import numpy as np

import theano
from theano import Op, Apply
from theano.gof import EnumList
import theano.tensor as T
from theano.gradient import grad_not_implemented
from theano.gradient import grad_undefined


class Images2Neibs(Op):
    """
    Reshapes the input as a 2D tensor where each row is an pooling
    example.

    Parameters
    ----------
    mode : {'valid', 'ignore_borders', 'wrap_centered'}
        - 'valid' :
            Requires an input that is a multiple of the pooling factor
            (in each direction).
        - 'half' :
            Equivalent to 'valid' if we pre-pad with zeros the input on
            each side by (neib_shape[0]//2, neib_shape[1]//2)
        - 'full' :
            Equivalent to 'valid' if we pre-pad with zeros the input on
            each side by (neib_shape[0] - 1, neib_shape[1] - 1)
        - 'ignore_borders' :
            Same as valid, but will ignore the borders if the shape(s)
            of the input is not a multiple of the pooling factor(s).
        - 'wrap_centered' :
            ?? TODO comment

    """

    __props__ = ("mode",)
    BORDER_MODE = EnumList(('MODE_VALID', 'valid'),
                           ('MODE_HALF', 'half'),
                           ('MODE_FULL', 'full'),
                           ('MODE_WRAP_CENTERED', 'wrap_centered'),
                           ('MODE_IGNORE_BORDERS', 'ignore_borders'))
    params_type = BORDER_MODE

    def get_params(self, node):
        return self.mode

    def __init__(self, mode='valid'):
        implemented_modes = self.BORDER_MODE.get_aliases()
        if mode not in implemented_modes:
            raise NotImplementedError("Only modes %s have been implemented for %s"
                                      % (', '.join(implemented_modes), type(self).__name__))
        self.mode = mode

    def __str__(self):
        return self.__class__.__name__ + "{%s}" % self.mode

    def __setstate__(self, d):
        self.__dict__.update(d)
        if not hasattr(self, "mode"):
            self.mode = 'valid'

    def make_node(self, ten4, neib_shape, neib_step=None):
        """
        Parameters
        ----------
        ten4 : a list of lists of images
            ten4 is of shape (list 1 dim, list 2 dim, row, col).
        neib_shape
            (r,c) where r is the height of the neighborhood in rows and c is
            the width of the neighborhood in columns.
        neib_step
            (dr,dc) where dr is the number of rows to skip between patch and dc
            is the number of columns. When None, this is the same as neib_shape
            (patch are disjoint).

        Returns
        -------
        matrix
            A 2D matrix, written using the following pattern::

                idx = 0
                for i in xrange(list 1 dim)
                    for j in xrange(list 2 dim)
                        for k in <image column coordinates>
                            for l in <image row coordinates>
                                output[idx,:]
                                     = flattened version of ten4[i,j,l:l+r,k:k+c]
                                idx += 1

            .. note:: The op isn't necessarily implemented internally with these
                for loops, they're just the easiest way to describe the output
                pattern.

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

        if self.mode in ['valid']:
            # Iterate over neighborhood positions, summing contributions.
            def pos2map(pidx, pgz, prior_result, neib_shape, neib_step):
                '''
                Helper function that adds gradient contribution from a single
                neighborhood position i,j.
                pidx = Index of position within neighborhood.
                pgz  = Gradient of shape (batch_size*num_channels*neibs)
                prior_result  = Shape (batch_size, num_channnels, rows, cols)
                neib_shape = Number of rows, cols in a neighborhood.
                neib_step  = Step sizes from image2neibs.
                '''
                nrows, ncols = neib_shape
                rstep, cstep = neib_step
                batch_size, num_channels, rows, cols = prior_result.shape
                i = pidx // ncols
                j = pidx - (i * ncols)
                # This position does not touch some img pixels in valid mode.
                result_indices = prior_result[:, :,
                                              i:(rows - nrows + i + 1):rstep,
                                              j:(cols - ncols + j + 1):cstep]
                newshape = (batch_size, num_channels) + \
                           ((rows - nrows) // rstep + 1,) + \
                           ((cols - ncols) // cstep + 1,)
                return T.inc_subtensor(result_indices, pgz.reshape(newshape))
            indices = T.arange(neib_shape[0] * neib_shape[1])
            pgzs = gz.dimshuffle((1, 0))
            result, _ = theano.scan(fn=pos2map,
                                    sequences=[indices, pgzs],
                                    outputs_info=T.zeros(x.shape),
                                    non_sequences=[neib_shape, neib_step])
            grad_input = result[-1]
            return [grad_input,
                    grad_undefined(self, 1, neib_shape),
                    grad_undefined(self, 2, neib_step)]

        return [grad_not_implemented(self, 0, x),
                grad_undefined(self, 1, neib_shape),
                grad_undefined(self, 2, neib_step)]

    def c_code_cache_version(self):
        return (10,)

    def perform(self, node, inp, out_, params):
        ten4, neib_shape, neib_step = inp
        z, = out_
        # GpuImages2Neibs should not run this perform in DebugMode
        if type(self) != Images2Neibs:
            raise theano.gof.utils.MethodNotDefined()

        def CEIL_INTDIV(a, b):
            if a % b:
                return (a // b) + 1
            else:
                return a // b

        grid_c = -1  # number of patch in height
        grid_d = -1  # number of patch in width
        assert ten4.ndim == 4
        assert neib_shape.ndim == 1
        assert neib_shape.shape[0] == 2
        assert neib_step.ndim == 1
        assert neib_step.shape[0] == 2
        c, d = neib_shape
        step_x, step_y = neib_step
        mode = self.mode
        if step_x <= 0 or step_y <= 0:
            raise ValueError(
                "neib_step wrong step ; values <= 0. Got " + str(neib_step))
        if c <= 0 or d <= 0:
            raise ValueError(
                "neib_shape values <=0. Got " + str(neib_shape))

        if mode == "wrap_centered":
            if (c % 2 != 1) or (d % 2 != 1):
                raise TypeError(
                    "Images2Neibs:"
                    " in mode wrap_centered need patch with odd shapes")

            if (ten4.shape[2] < c) or (ten4.shape[3] < d):
                raise TypeError(
                    "Images2Neibs: in wrap_centered mode, don't support"
                    " image shapes smaller then the patch shapes:"
                    " neib_shape=(%d,%d), ten4[2:]=[%d,%d]" %
                    (c, d, ten4.shape[2], ten4.shape[3]))
            grid_c = CEIL_INTDIV(ten4.shape[2], step_x)
            grid_d = CEIL_INTDIV(ten4.shape[3], step_y)
        elif mode == "valid":
            if (ten4.shape[2] < c) or (((ten4.shape[2] - c) % step_x) != 0):
                raise TypeError(
                    "neib_shape[0]=%d, neib_step[0]=%d and"
                    " ten4.shape[2]=%d not consistent" %
                    (c, step_x, ten4.shape[2]))
            if (ten4.shape[3] < d) or (((ten4.shape[3] - d) % step_y) != 0):
                raise TypeError(
                    "neib_shape[1]=%d, neib_step[1]=%d and"
                    " ten4.shape[3]=%d not consistent" %
                    (d, step_y, ten4.shape[3]))
            # number of patch in height
            grid_c = 1 + ((ten4.shape[2] - c) // step_x)
            # number of patch in width
            grid_d = 1 + ((ten4.shape[3] - d) // step_y)
        elif mode == "ignore_borders":
            # number of patch in height
            grid_c = 1 + ((ten4.shape[2] - c) // step_x)
            # number of patch in width
            grid_d = 1 + ((ten4.shape[3] - d) // step_y)
        elif mode == "half":
            # This is equivalent to 'valid' with padding (c // 2, d // 2) on both sides
            # Thus the expanded image will have size (h + 2 * (c // 2), w + 2 * (d // 2))
            # Plugging these in the equation for 'valid' we get
            # h + 2 * (c // 2) - c  = h - (c % 2)
            # w + 2 * (d // 2) - c  = w - (d % 2)
            if (ten4.shape[2] < c) or (((ten4.shape[2] - (c % 2)) % step_x) != 0):
                raise TypeError(
                    "neib_shape[0]=%d, neib_step[0]=%d and"
                    " ten4.shape[2]=%d not consistent" %
                    (c, step_x, ten4.shape[2]))
            if (ten4.shape[3] < d) or (((ten4.shape[3] - (d % 2)) % step_y) != 0):
                raise TypeError(
                    "neib_shape[1]=%d, neib_step[1]=%d and"
                    " ten4.shape[3]=%d not consistent" %
                    (d, step_y, ten4.shape[3]))
            # number of patch in height
            grid_c = 1 + ((ten4.shape[2] - (c % 2)) // step_x)
            # number of patch in width
            grid_d = 1 + ((ten4.shape[3] - (d % 2)) // step_y)
        elif mode == "full":
            # This is equivalent to 'valid' with padding (c - 1, d - 1) on both sides
            # Thus the expanded image will have size (h + 2 * (c - 1), w + 2 * (d - 1))
            # Plugging these in the equation for 'valid' we get
            # h + 2 * (c - 1) - c  = h + c - 2
            # w + 2 * (d - 1) - c  = w + d - 2
            if (ten4.shape[2] < c) or (((ten4.shape[2] + c - 2) % step_x) != 0):
                raise TypeError(
                    "neib_shape[0]=%d, neib_step[0]=%d and"
                    " ten4.shape[2]=%d not consistent" %
                    (c, step_x, ten4.shape[2]))
            if (ten4.shape[3] < d) or (((ten4.shape[3] + d - 2) % step_y) != 0):
                raise TypeError(
                    "neib_shape[1]=%d, neib_step[1]=%d and"
                    " ten4.shape[3]=%d not consistent" %
                    (d, step_y, ten4.shape[3]))
            # number of patch in height
            grid_c = 1 + ((ten4.shape[2] + c - 2) // step_x)
            # number of patch in width
            grid_d = 1 + ((ten4.shape[3] + d - 2) // step_y)
        else:
            raise TypeError("Images2Neibs: unknow mode '%s'" % mode)
        z_dim0 = grid_c * grid_d * ten4.shape[1] * ten4.shape[0]
        z_dim1 = c * d
        z[0] = np.empty((z_dim0, z_dim1), dtype=node.outputs[0].dtype)

        nb_batch = ten4.shape[0]
        nb_stack = ten4.shape[1]
        height = ten4.shape[2]
        width = ten4.shape[3]

        wrap_centered_half_idx_shift_x = c // 2
        wrap_centered_half_idx_shift_y = d // 2
        for n in range(nb_batch):
            for s in range(nb_stack):
                # loop over the number of patch in height
                for a in range(grid_c):
                    # loop over the number of patch in width
                    for b in range(grid_d):
                        z_row = b + grid_d * (a + grid_c * (s + nb_stack * n))
                        for i in range(c):
                            ten4_2 = i + a * step_x
                            if mode == "wrap_centered":
                                ten4_2 -= wrap_centered_half_idx_shift_x
                                if ten4_2 < 0:
                                    ten4_2 += height
                                elif ten4_2 >= height:
                                    ten4_2 -= height
                            elif mode == "half":
                                ten4_2 -= wrap_centered_half_idx_shift_x
                            elif mode == "full":
                                ten4_2 -= c - 1
                            if ten4_2 < 0 or ten4_2 >= height:
                                z[0][z_row, d * i: d * i + d] = 0
                            else:
                                for j in range(d):
                                    ten4_3 = j + b * step_y
                                    if mode == "wrap_centered":
                                        ten4_3 -= wrap_centered_half_idx_shift_y
                                        if ten4_3 < 0:
                                            ten4_3 += width
                                        elif ten4_3 >= width:
                                            ten4_3 -= width
                                    elif mode == "half":
                                        ten4_3 -= wrap_centered_half_idx_shift_y
                                    elif mode == "full":
                                        ten4_3 -= d - 1
                                    z_col = j + d * i
                                    if ten4_3 < 0 or ten4_3 >= width:
                                        z[0][z_row, z_col] = 0
                                    else:
                                        z[0][z_row, z_col] = ten4[n, s, ten4_2, ten4_3]

    def infer_shape(self, node, input_shape):
        in_shape = input_shape[0]
        c, d = node.inputs[1]
        step_x, step_y = node.inputs[2]
        if self.mode == 'wrap_centered':
            grid_c = T.ceil_intdiv(in_shape[2], step_x)
            grid_d = T.ceil_intdiv(in_shape[3], step_y)
        elif self.mode == 'valid':
            grid_c = 1 + ((in_shape[2] - c) // step_x)
            grid_d = 1 + ((in_shape[3] - d) // step_y)
        elif self.mode == 'ignore_borders':
            grid_c = 1 + ((in_shape[2] - c) // step_x)
            grid_d = 1 + ((in_shape[3] - d) // step_y)
        elif self.mode == 'half':
            grid_c = 1 + ((in_shape[2] - (c % 2)) // step_x)
            grid_d = 1 + ((in_shape[3] - (d % 2)) // step_y)
        elif self.mode == 'full':
            grid_c = 1 + ((in_shape[2] + c - 2) // step_x)
            grid_d = 1 + ((in_shape[3] + d - 2) // step_y)
        else:
            raise TypeError("Images2Neibs: unknow mode '%s'" % self.mode)
        z_dim0 = grid_c * grid_d * in_shape[1] * in_shape[0]
        z_dim1 = c * d
        return [(z_dim0, z_dim1)]

    def c_code(self, node, name, inp, out, sub):
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
        const dtype_%(neib_step)s step_x = *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 0);
        const dtype_%(neib_step)s step_y = *(dtype_%(neib_step)s*) PyArray_GETPTR1(%(neib_step)s, 1);

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

        if (%(mode)s == MODE_WRAP_CENTERED) {
            if (c%%2!=1 || d%%2!=1){
                PyErr_Format(PyExc_TypeError,
                             "Images2Neibs: in mode wrap_centered"
                             " need patch with odd shapes");
                %(fail)s;
            }
            if ( (PyArray_DIMS(%(ten4)s))[2] < c ||
                 (PyArray_DIMS(%(ten4)s))[3] < d)
            {
                PyErr_Format(PyExc_TypeError,
                    "Images2Neibs: in wrap_centered mode, don't support image"
                    " shapes smaller then the patch shapes:"
                    " neib_shape=(%%ld,%%ld), ten4[2:]=[%%ld,%%ld]",
                    (long int)c, (long int)d,
                    (long int)(PyArray_DIMS(%(ten4)s)[2]),
                    (long int)(PyArray_DIMS(%(ten4)s)[3]));
                %(fail)s;
            }
            grid_c = CEIL_INTDIV(((PyArray_DIMS(%(ten4)s))[2]),step_x);
            grid_d = CEIL_INTDIV(((PyArray_DIMS(%(ten4)s))[3]),step_y);

        } else if (%(mode)s == MODE_VALID) {
            if ( ((PyArray_DIMS(%(ten4)s))[2] < c) ||
                 ( (((PyArray_DIMS(%(ten4)s))[2]-c) %% step_x)!=0))
            {
                PyErr_Format(PyExc_TypeError,
                             "neib_shape[0]=%%ld, neib_step[0]=%%ld and"
                             " ten4.shape[2]=%%ld not consistent",
                             (long int)c, (long int)step_x,
                             (long int)(PyArray_DIMS(%(ten4)s)[2]));
                %(fail)s;
            }
            if ( ((PyArray_DIMS(%(ten4)s))[3] < d) ||
                 ( (((PyArray_DIMS(%(ten4)s))[3]-d) %% step_y)!=0))
            {
                PyErr_Format(PyExc_TypeError,
                             "neib_shape[1]=%%ld, neib_step[1]=%%ld and"
                             " ten4.shape[3]=%%ld not consistent",
                             (long int)d, (long int)step_y,
                             (long int)(PyArray_DIMS(%(ten4)s)[3]));
                %(fail)s;
            }
            //number of patch in height
            grid_c = 1+(((PyArray_DIMS(%(ten4)s))[2]-c)/step_x);
            //number of patch in width
            grid_d = 1+(((PyArray_DIMS(%(ten4)s))[3]-d)/step_y);
        } else if (%(mode)s == MODE_IGNORE_BORDERS) {
            //number of patch in height
            grid_c = 1+(((PyArray_DIMS(%(ten4)s))[2]-c)/step_x);
            //number of patch in width
            grid_d = 1+(((PyArray_DIMS(%(ten4)s))[3]-d)/step_y);
        } else if (%(mode)s == MODE_HALF) {
            if ( ((PyArray_DIMS(%(ten4)s))[2] < c) ||
                 ( (((PyArray_DIMS(%(ten4)s))[2]-(c%%2)) %% step_x)!=0))
            {
                PyErr_Format(PyExc_TypeError,
                             "neib_shape[0]=%%ld, neib_step[0]=%%ld and"
                             " ten4.shape[2]=%%ld not consistent",
                             (long int)c, (long int)step_x,
                             (long int)(PyArray_DIMS(%(ten4)s)[2]));
                %(fail)s;
            }
            if ( ((PyArray_DIMS(%(ten4)s))[3] < d) ||
                 ( (((PyArray_DIMS(%(ten4)s))[3]-(d%%2)) %% step_y)!=0))
            {
                PyErr_Format(PyExc_TypeError,
                             "neib_shape[1]=%%ld, neib_step[1]=%%ld and"
                             " ten4.shape[3]=%%ld not consistent",
                             (long int)d, (long int)step_y,
                             (long int)(PyArray_DIMS(%(ten4)s)[3]));
                %(fail)s;
            }
            //number of patch in height
            grid_c = 1+(((PyArray_DIMS(%(ten4)s))[2]-(c%%2))/step_x);
            //number of patch in width
            grid_d = 1+(((PyArray_DIMS(%(ten4)s))[3]-(d%%2))/step_y);
        } else if (%(mode)s == MODE_FULL) {
            if ( ((PyArray_DIMS(%(ten4)s))[2] < c) ||
                 ( (((PyArray_DIMS(%(ten4)s))[2]+c-2) %% step_x)!=0))
            {
                PyErr_Format(PyExc_TypeError,
                             "neib_shape[0]=%%ld, neib_step[0]=%%ld and"
                             " ten4.shape[2]=%%ld not consistent",
                             (long int)c, (long int)step_x,
                             (long int)(PyArray_DIMS(%(ten4)s)[2]));
                %(fail)s;
            }
            if ( ((PyArray_DIMS(%(ten4)s))[3] < d) ||
                 ( (((PyArray_DIMS(%(ten4)s))[3]+d-2) %% step_y)!=0))
            {
                PyErr_Format(PyExc_TypeError,
                             "neib_shape[1]=%%ld, neib_step[1]=%%ld and"
                             " ten4.shape[3]=%%ld not consistent",
                             (long int)d, (long int)step_y,
                             (long int)(PyArray_DIMS(%(ten4)s)[3]));
                %(fail)s;
            }
            //number of patch in height
            grid_c = 1+(((PyArray_DIMS(%(ten4)s))[2]+c-2)/step_x);
            //number of patch in width
            grid_d = 1+(((PyArray_DIMS(%(ten4)s))[3]+d-2)/step_y);
        } else {
            PyErr_Format(PyExc_TypeError,
                         "Images2Neibs: unknow mode %%d", %(mode)s);
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
                PyArray_TYPE((PyArrayObject*) py_%(ten4)s),
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

        const int wrap_centered_half_idx_shift_x = c/2;
        const int wrap_centered_half_idx_shift_y = d/2;
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
                            if (%(mode)s == MODE_WRAP_CENTERED) {
                                ten4_2 -= wrap_centered_half_idx_shift_x;
                                if ( ten4_2 < 0 ) ten4_2 += height;
                                else if (ten4_2 >= height) ten4_2 -= height;
                            } else if (%(mode)s == MODE_HALF) {
                                ten4_2 -= wrap_centered_half_idx_shift_x;
                            } else if (%(mode)s == MODE_FULL) {
                                ten4_2 -= c - 1;
                            }
                            if (ten4_2 < 0 | ten4_2 >= height) {
                                dtype_%(z)s* curr_z = (dtype_%(z)s*) PyArray_GETPTR2(%(z)s, z_row, d * i);
                                memset(curr_z, 0, d*sizeof(*curr_z));
                            } else {
                                for (int j = 0; j < d; j++)  // loop over d
                                {
                                    int ten4_3 = j + b * step_y;
                                    if (%(mode)s == MODE_WRAP_CENTERED) {
                                        ten4_3 -= wrap_centered_half_idx_shift_y;
                                        if ( ten4_3 < 0 ) ten4_3 += width;
                                        else if (ten4_3 >= width) ten4_3 -= width;
                                    } else if (%(mode)s == MODE_HALF) {
                                        ten4_3 -= wrap_centered_half_idx_shift_y;
                                    } else if (%(mode)s == MODE_FULL) {
                                        ten4_3 -= d - 1;
                                    }
                                    int z_col = j + d * i;
                                    dtype_%(z)s* curr_z = (dtype_%(z)s*) PyArray_GETPTR2(%(z)s, z_row, z_col);
                                    if (ten4_3 < 0 | ten4_3 >= width) {
                                        *curr_z = 0;
                                    } else {
                                        *curr_z = *( (dtype_%(ten4)s*) PyArray_GETPTR4(%(ten4)s, n, s, ten4_2, ten4_3));
                                    }
                                }
                            }
                        }
                    }
        } // END NESTED SCOPE
        """ % dict(ten4=inp[0], neib_shape=inp[1], neib_step=inp[2], z=out[0],
                   fail=sub['fail'], mode=sub['params'])


def images2neibs(ten4, neib_shape, neib_step=None, mode='valid'):
    """
    Function :func:`images2neibs <theano.tensor.nnet.neighbours.images2neibs>`
    allows to apply a sliding window operation to a tensor containing
    images or other two-dimensional objects.
    The sliding window operation loops over points in input data and stores
    a rectangular neighbourhood of each point.
    It is possible to assign a step of selecting patches (parameter `neib_step`).

    Parameters
    ----------
    ten4 : A 4d tensor-like
        A 4-dimensional tensor which represents a list of lists of images.
        It should have shape (list 1 dim, list 2 dim, row, col). The first
        two dimensions can be useful to store different channels and batches.
    neib_shape : A 1d tensor-like of 2 values
        A tuple containing two values: height and width of the neighbourhood.
        It should have shape (r,c) where r is the height of the neighborhood
        in rows and c is the width of the neighborhood in columns.
    neib_step : A 1d tensor-like of 2 values
        (dr,dc) where dr is the number of rows to skip between patch and dc is
        the number of columns. The parameter should be a tuple of two elements:
        number of rows and number of columns to skip each iteration.
        Basically, when the step is 1, the neighbourhood of every first element
        is taken and every possible rectangular subset is returned.
        By default it is equal to `neib_shape` in other words, the patches are
        disjoint. When the step is greater than `neib_shape`, some elements are
        omitted. When None, this is the same as neib_shape (patch are disjoint).
    mode : {'valid', 'ignore_borders', 'wrap_centered', 'half'}
        ``valid``
            Requires an input that is a multiple of the
            pooling factor (in each direction).
        ``half``
            Equivalent to 'valid' if we pre-pad with zeros the input on
            each side by (neib_shape[0]//2, neib_shape[1]//2)
        ``full``
            Equivalent to 'valid' if we pre-pad with zeros the input on
            each side by (neib_shape[0] - 1, neib_shape[1] - 1)
        ``ignore_borders``
            Same as valid, but will ignore the borders if the shape(s) of
            the input is not a multiple of the pooling factor(s).
        ``wrap_centered``
            ?? TODO comment

    Returns
    -------
    object
        Reshapes the input as a 2D tensor where each row is an
        pooling example. Pseudo-code of the output:

          .. code-block:: python

             idx = 0
             for i in xrange(list 1 dim):
                 for j in xrange(list 2 dim):
                     for k in <image column coordinates>:
                         for l in <image row coordinates>:
                             output[idx,:]
                                  = flattened version of ten4[i,j,l:l+r,k:k+c]
                             idx += 1

          .. note:: The operation isn't necessarily implemented internally with
             these for loops, they're just the easiest way to describe the
             output pattern.

    Notes
    -----
    .. note::
        Currently the step size should be chosen in the way that the
        corresponding dimension :math:`i` (width or height) is equal
        to :math:`n * step\_size_i + neib\_shape_i` for some :math:`n`.

    Examples
    --------

    .. code-block:: python

        # Defining variables
        images = T.tensor4('images')
        neibs = images2neibs(images, neib_shape=(5, 5))

        # Constructing theano function
        window_function = theano.function([images], neibs)

        # Input tensor (one image 10x10)
        im_val = np.arange(100.).reshape((1, 1, 10, 10))

        # Function application
        neibs_val = window_function(im_val)

    .. note:: The underlying code will construct a 2D tensor of disjoint
       patches 5x5. The output has shape 4x25.

    """
    return Images2Neibs(mode)(ten4, neib_shape, neib_step)


def neibs2images(neibs, neib_shape, original_shape, mode='valid'):
    """
    Function :func:`neibs2images <theano.sandbox.neighbours.neibs2images>`
    performs the inverse operation of
    :func:`images2neibs <theano.sandbox.neigbours.neibs2images>`. It inputs
    the output of :func:`images2neibs <theano.sandbox.neigbours.neibs2images>`
    and reconstructs its input.

    Parameters
    ----------
    neibs : 2d tensor
        Like the one obtained by
        :func:`images2neibs <theano.sandbox.neigbours.neibs2images>`.
    neib_shape
        `neib_shape` that was used in
        :func:`images2neibs <theano.sandbox.neigbours.neibs2images>`.
    original_shape
        Original shape of the 4d tensor given to
        :func:`images2neibs <theano.sandbox.neigbours.neibs2images>`

    Returns
    -------
    object
        Reconstructs the input of
        :func:`images2neibs <theano.sandbox.neigbours.neibs2images>`,
        a 4d tensor of shape `original_shape`.

    Notes
    -----
    Currently, the function doesn't support tensors created with
    `neib_step` different from default value. This means that it may be
    impossible to compute the gradient of a variable gained by
    :func:`images2neibs <theano.sandbox.neigbours.neibs2images>` w.r.t.
    its inputs in this case, because it uses
    :func:`images2neibs <theano.sandbox.neigbours.neibs2images>` for
    gradient computation.

    Examples
    --------
    Example, which uses a tensor gained in example for
    :func:`images2neibs <theano.sandbox.neigbours.neibs2images>`:

    .. code-block:: python

        im_new = neibs2images(neibs, (5, 5), im_val.shape)
        # Theano function definition
        inv_window = theano.function([neibs], im_new)
        # Function application
        im_new_val = inv_window(neibs_val)

    .. note:: The code will output the initial image array.

    """
    neibs = T.as_tensor_variable(neibs)
    neib_shape = T.as_tensor_variable(neib_shape)
    original_shape = T.as_tensor_variable(original_shape)

    new_neib_shape = T.stack([original_shape[-1] // neib_shape[1],
                              neib_shape[1]])
    output_2d = images2neibs(neibs.dimshuffle('x', 'x', 0, 1),
                             new_neib_shape, mode=mode)

    if mode == 'ignore_borders':
        # We use set_subtensor to accept original_shape we can't infer
        # the shape and still raise error when it don't have the right
        # shape.
        valid_shape = original_shape
        valid_shape = T.set_subtensor(
            valid_shape[2],
            (valid_shape[2] // neib_shape[0]) * neib_shape[0])
        valid_shape = T.set_subtensor(
            valid_shape[3],
            (valid_shape[3] // neib_shape[1]) * neib_shape[1])
        output_4d = output_2d.reshape(valid_shape, ndim=4)
        # padding the borders with zeros
        for d in [2, 3]:
            pad_shape = list(output_4d.shape)
            pad_shape[d] = original_shape[d] - valid_shape[d]
            output_4d = T.concatenate([output_4d, T.zeros(pad_shape)], axis=d)
    elif mode == 'valid':
        # TODO: we do not implement all mode with this code.
        # Add a check for the good cases.
        output_4d = output_2d.reshape(original_shape, ndim=4)
    else:
        raise NotImplementedError("neibs2images do not support mode=%s" % mode)

    return output_4d
