import os
import numpy
from six import integer_types
import theano
from theano.gof import Variable, Op, Apply
from theano.tensor.type_other import NoneConst
from theano.tensor.var import TensorConstant
# from theano.tensor import as_tensor_variable
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
        # Check axis and convert it to Python variable.
        if (isinstance(axis, (integer_types, numpy.integer)) or
           (isinstance(axis, numpy.ndarray) and axis.ndim == 0)):
            axis = [int(axis)]
        elif isinstance(axis, (tuple, list, numpy.ndarray)):
            axis = [int(a) for a in axis]
        elif isinstance(axis, Variable):
            if NoneConst.equals(axis):
                axis = None
            elif not isinstance(axis, TensorConstant):
                raise TypeError("MaxAndArgmax needs a constant axis. Got %s" % axis)
            else:
                assert (axis.dtype.startswith("int") or axis.dtype.startswith("uint"))
                if (isinstance(axis.data, (integer_types, numpy.integer)) or
                   (isinstance(axis.data, numpy.ndarray) and axis.data.ndim == 0)):
                    axis = [int(axis.data)]
                elif isinstance(axis.data, (list, numpy.ndarray)):
                    axis = [int(i) for i in axis.data]
        # Make axis entries non-negative, and verify that axes are valid.
        if isinstance(axis, list):
            for idx in xrange(len(axis)):
                if axis[idx] < 0:
                    axis[idx] += X.type.ndim
                if axis[idx] < 0 or axis[idx] >= X.type.ndim:
                    raise ValueError('Invalid axis: %s (the number of dimensions of the '
                                     'input is: %s)' % (axis[idx], X.type.ndim))
        # Sort axes and make them unique.
        axis_set = set()  # used to build "broadcastable" variable below.
        all_axes = []
        if isinstance(axis, list):
            axis_set = set(axis)
            all_axes = list(axis_set)
            all_axes.sort()
            if all_axes == range(X.type.ndim):
                axis = None
        else:
            all_axes = range(X.ndim)
            axis_set = set(all_axes)
        if axis is None:
            axis = NoneConst.clone()
        else:
            axis = theano.tensor.as_tensor_variable(all_axes)
            # assert axis.ndim == 1
        inputs = [as_gpuarray_variable(X, context_name), axis]
        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the max / argmax.
        broadcastable = [b for i, b in enumerate(X.type.broadcastable)
                         if i not in axis_set]
        outputs = [GpuArrayType(X.type.dtype, broadcastable, context_name=context_name, name='max')(),
                   GpuArrayType(self.argmax_dtype, broadcastable, context_name=context_name, name='argmax')()]
        return Apply(self, inputs, outputs)

    def perform(self, node, inputs, outputs):
        X, axes = inputs
        max, max_idx = outputs
        if axes is None:
            axes = tuple(range(X.ndim))
        else:
            axes = tuple(axes)
            # axes = tuple(int(ax) for ax in axes)
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
        # axes_ctype = pygpu.gpuarray.dtype_to_ctype(node.inputs[1].dtype)
        axes_ctype = 'int64_t'
        ret = """
        GpuArray* %(name)s_input = &%(X)s->ga;
        size_t %(name)s_input_ndim = PyGpuArray_NDIM(%(X)s);
        """
        if NoneConst.equals(node.inputs[1]):
            ret += """
            unsigned  %(name)s_redux_len = %(name)s_input_ndim;
            unsigned* %(name)s_axes_to_reduce = new unsigned[%(name)s_redux_len];
            for(unsigned i = 0; i < %(name)s_redux_len; ++i) {
                %(name)s_axes_to_reduce[i] = i;
            }
            """
        else:
            assert node.inputs[1].ndim == 1
            ret += """
            unsigned  %(name)s_redux_len = PyArray_DIM(%(axes)s, 0);
            unsigned* %(name)s_axes_to_reduce = new unsigned[%(name)s_redux_len];
            for(unsigned i = 0; i < %(name)s_redux_len; ++i) {
                %(name)s_axes_to_reduce[i] =
                    (unsigned)( ((%(axes_ctype)s*)PyArray_DATA(%(axes)s)) [i * (PyArray_STRIDES(%(axes)s)[0] / sizeof(%(axes_ctype)s))] );
            }
            """
        ret += """
        size_t  %(name)s_output_ndim = %(name)s_input_ndim - %(name)s_redux_len;
        size_t* %(name)s_output_dims = NULL;
        if(%(name)s_output_ndim == 0) {
            /* Current backend function GpuArray_maxandargmax does not work when
             * all axes need to be reduced. So to handle this case, we create a view
             * of the input as a matrix with 1 row and as many columns as elements
             * in the input, so that the 2nd dimenson of the matrix will be reduced. */
            size_t total_size = 1;
            for(size_t i = 0; i < %(name)s_input_ndim; ++i) {
                total_size *= PyGpuArray_DIM(%(X)s, i);
            }
            size_t newdims[2] = {1, total_size};
            %(name)s_input = new GpuArray;
            if(GA_NO_ERROR !=
                GpuArray_reshape(%(name)s_input, &%(X)s->ga, 2, newdims, GA_ANY_ORDER, 0)
            ) {
                %(fail)s
            }
            %(name)s_redux_len = 1;
            %(name)s_axes_to_reduce[0] = 1;
        } else {
            %(name)s_output_dims = new size_t[%(name)s_output_ndim];
            if(%(name)s_redux_len == 1) {
                for(unsigned i = 0; i < %(name)s_axes_to_reduce[0]; ++i) {
                    %(name)s_output_dims[i] = PyGpuArray_DIM(%(X)s, i);
                }
                for(unsigned i = %(name)s_axes_to_reduce[0] + 1; i < %(name)s_input_ndim; ++i) {
                    %(name)s_output_dims[i-1] = PyGpuArray_DIM(%(X)s, i);
                }
            } else {
                int64_t current_input_pos = -1;
                int64_t current_output_pos = -1;
                for(unsigned i = 0; i < %(name)s_redux_len; ++i) {
                    for(++current_input_pos; current_input_pos < %(name)s_axes_to_reduce[i]; ++current_input_pos) {
                        %(name)s_output_dims[++current_output_pos] = PyGpuArray_DIM(%(X)s, current_input_pos);
                    }
                }
                for(++current_input_pos; current_input_pos < %(name)s_input_ndim; ++current_input_pos) {
                    %(name)s_output_dims[++current_output_pos] = PyGpuArray_DIM(%(X)s, current_input_pos);
                }
            }
        }
        if(theano_prep_output(&%(max)s, %(name)s_output_ndim, %(name)s_output_dims, %(max_typecode)s, GA_C_ORDER, %(X)s->context)) {
            %(fail)s
        }
        if(theano_prep_output(&%(argmax)s, %(name)s_output_ndim, %(name)s_output_dims, %(argmax_typecode)s, GA_C_ORDER, %(X)s->context)) {
            %(fail)s
        }
        if(GA_NO_ERROR !=
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
        delete[] %(name)s_output_dims;
        if(%(name)s_input != &%(X)s->ga) delete %(name)s_input;
        delete[] %(name)s_axes_to_reduce;
        """ % {'name': name, 'X': inputs[0]}


gpu_maxandargmax = GpuMaxAndArgmax()
