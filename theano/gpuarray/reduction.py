import os
import numpy
import theano
from theano.gof import Op, Apply
from theano.tensor.var import TensorConstant

from .basic_ops import (infer_context_name, as_gpuarray_variable)
from .type import GpuArrayType

try:
    import pygpu
except ImportError as e:
    pass


class GpuMaxAndArgmax(Op):
    """
    GPU version of MaxAndArgmax

    """
    __props__ = ()
    argmax_dtype = "int64"

    def make_node(self, X, axis=None):
        context_name = infer_context_name(X)
        if axis is None:
            axis = range(X.type.ndim)
        elif isinstance(axis, TensorConstant) and isinstance(axis.data, (list, numpy.ndarray)):
            axis = [int(i) for i in axis.data]
        elif not isinstance(axis, list):
            raise TypeError("Axis must be a list. Got %s" % axis)
        # Make axis entries non-negative, and verify that axes are valid.
        for idx in xrange(len(axis)):
            if axis[idx] < 0:
                axis[idx] += X.type.ndim
            if axis[idx] < 0 or axis[idx] >= X.type.ndim:
                raise ValueError('Invalid axis: %s (the number of dimensions of the '
                                 'input is: %s)' % (axis[idx], X.type.ndim))
        # Sort axes and make them unique.
        axis_set = set(axis)  # used to build "broadcastable" variable below.
        axis = list(axis_set)
        axis.sort()
        axis = theano.tensor.as_tensor_variable(axis)
        inputs = [as_gpuarray_variable(X, context_name), axis]
        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the max / argmax.
        broadcastable = [b for i, b in enumerate(X.type.broadcastable)
                         if i not in axis_set]
        outputs = [GpuArrayType(X.type.dtype, broadcastable, context_name=context_name, name='max')(),
                   GpuArrayType(self.argmax_dtype, broadcastable, context_name=context_name, name='argmax')()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        # NB: I must rewrite this method with pygpu functions instead of numpy functions.
        x, axes = inputs
        max, max_idx = outputs
        X = numpy.asarray(x)
        axes = tuple(axes)
        max[0] = theano._asarray(numpy.max(X, axes), dtype=node.outputs[0].dtype)
        # Numpy does not support multiple axes for argmax
        # Work around
        keep_axes = numpy.array([i for i in range(X.ndim) if i not in axes], dtype='int64')
        # Not-reduced axes in front
        transposed_x = numpy.transpose(X, numpy.concatenate((keep_axes, axes)))
        kept_shape = transposed_x.shape[:len(keep_axes)]
        reduced_shape = transposed_x.shape[len(keep_axes):]
        new_shape = kept_shape + (numpy.prod(reduced_shape),)
        reshaped_x = transposed_x.reshape(new_shape)

        max_idx[0] = theano._asarray(numpy.argmax(reshaped_x, axis=-1), dtype='int64')

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), os.path.dirname(__file__)]

    def c_code(self, node, name, input_names, output_names, sub):
        # Recall: X, axes = input_names
        # Recall: max, argmax = output_names
        # Recall: fail = sub['fail']
        max_typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
        argmax_typecode = pygpu.gpuarray.dtype_to_typecode(self.argmax_dtype)
        axes_ctype = 'int64_t'
        assert node.inputs[1].ndim == 1
        ret = """
        GpuArray temp;
        GpuArray* %(name)s_input = &%(X)s->ga;
        size_t %(name)s_input_ndim = PyGpuArray_NDIM(%(X)s);

        unsigned  %(name)s_redux_len = PyArray_DIM(%(axes)s, 0);
        unsigned* %(name)s_axes_to_reduce = (unsigned*)malloc(%(name)s_redux_len * sizeof(unsigned));
        for (unsigned i = 0; i < %(name)s_redux_len; ++i) {
            %(name)s_axes_to_reduce[i] = (unsigned) (*(%(axes_ctype)s*)PyArray_GETPTR1(%(axes)s, i));
        }

        size_t  %(name)s_output_ndim = %(name)s_input_ndim - %(name)s_redux_len;
        size_t* %(name)s_output_dims = NULL;
        if (%(name)s_output_ndim == 0) {
            /* Current backend function GpuArray_maxandargmax does not work when
             * all axes need to be reduced. So to handle this case, we create a view
             * of the input as a matrix with 1 row and as many columns as elements
             * in the input, so that the 2nd dimenson of the matrix will be reduced. */
            size_t total_size = 1;
            for (size_t i = 0; i < %(name)s_input_ndim; ++i) {
                total_size *= PyGpuArray_DIM(%(X)s, i);
            }
            size_t newdims[2] = {1, total_size};
            %(name)s_input = &temp;
            if (GA_NO_ERROR !=
                GpuArray_reshape(%(name)s_input, &%(X)s->ga, 2, newdims, GA_ANY_ORDER, 0)
            ) {
                %(fail)s
            }
            %(name)s_redux_len = 1;
            %(name)s_axes_to_reduce[0] = 1;
        } else {
            %(name)s_output_dims = (size_t*)malloc(%(name)s_output_ndim * sizeof(size_t));
            if (%(name)s_redux_len == 1) {
                for (unsigned i = 0; i < %(name)s_axes_to_reduce[0]; ++i) {
                    %(name)s_output_dims[i] = PyGpuArray_DIM(%(X)s, i);
                }
                for (unsigned i = %(name)s_axes_to_reduce[0] + 1; i < %(name)s_input_ndim; ++i) {
                    %(name)s_output_dims[i-1] = PyGpuArray_DIM(%(X)s, i);
                }
            } else {
                int64_t current_input_pos = -1;
                int64_t current_output_pos = -1;
                for (unsigned i = 0; i < %(name)s_redux_len; ++i) {
                    for (++current_input_pos; current_input_pos < %(name)s_axes_to_reduce[i]; ++current_input_pos) {
                        %(name)s_output_dims[++current_output_pos] = PyGpuArray_DIM(%(X)s, current_input_pos);
                    }
                }
                for (++current_input_pos; current_input_pos < %(name)s_input_ndim; ++current_input_pos) {
                    %(name)s_output_dims[++current_output_pos] = PyGpuArray_DIM(%(X)s, current_input_pos);
                }
            }
        }
        if (theano_prep_output(&%(max)s, %(name)s_output_ndim, %(name)s_output_dims, %(max_typecode)s, GA_C_ORDER, %(X)s->context)) {
            %(fail)s
        }
        if (theano_prep_output(&%(argmax)s, %(name)s_output_ndim, %(name)s_output_dims, %(argmax_typecode)s, GA_C_ORDER, %(X)s->context)) {
            %(fail)s
        }
        if (GA_NO_ERROR !=
            GpuArray_maxandargmax(&%(max)s->ga, &%(argmax)s->ga, %(name)s_input, %(name)s_redux_len, %(name)s_axes_to_reduce)
        ) {
            %(fail)s
        }
        """
        if theano.config.gpuarray.sync:
            ret += """
            GpuArray_sync(&%(max)s->ga);
            GpuArray_sync(&%(argmax)s->ga);
            """
        return ret % {'X': input_names[0], 'axes': input_names[1], 'max': output_names[0], 'argmax': output_names[1],
                      'axes_ctype': axes_ctype, 'max_typecode': max_typecode, 'argmax_typecode': argmax_typecode,
                      'name': name, 'fail': sub['fail']}

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        return """
        free(%(name)s_output_dims);
        free(%(name)s_axes_to_reduce);
        """ % {'name': name, 'X': inputs[0]}


gpu_maxandargmax = GpuMaxAndArgmax()
