from __future__ import print_function, absolute_import, division
from theano.gof import Op, Apply
from theano.gof.type import Generic

from .basic_ops import (infer_context_name, as_gpuarray_variable, gpuarray_helper_inc_dir)
from .type import GpuArrayType

try:
    import pygpu
except ImportError as e:
    pass


class GpuMaxAndArgmax(Op):
    """
    GPU version of MaxAndArgmax

    """
    params_type = Generic()
    __props__ = ('axis',)
    argmax_dtype = "int64"

    def __init__(self, axis):
        assert isinstance(axis, (list, tuple))
        self.axis = tuple(axis)

    def get_params(self, node):
        return self.axis

    def make_node(self, X):
        context_name = infer_context_name(X)
        # We keep the original broadcastable flags for dimensions on which
        # we do not perform the max / argmax.
        all_axes = set(self.axis)
        broadcastable = [b for i, b in enumerate(X.type.broadcastable)
                         if i not in all_axes]
        inputs = [as_gpuarray_variable(X, context_name)]
        outputs = [GpuArrayType(X.type.dtype, broadcastable, context_name=context_name)(),
                   GpuArrayType(self.argmax_dtype, broadcastable, context_name=context_name)()]
        return Apply(self, inputs, outputs)

    def c_headers(self):
        return ['<numpy_compat.h>', '<gpuarray_helper.h>']

    def c_header_dirs(self):
        return [pygpu.get_include(), gpuarray_helper_inc_dir()]

    def c_code(self, node, name, input_names, output_names, sub):
        # Recall: X = input_names[0]
        # Recall: axes = sub['params']
        # Recall: max, argmax = output_names
        # Recall: fail = sub['fail']
        max_typecode = pygpu.gpuarray.dtype_to_typecode(node.inputs[0].dtype)
        argmax_typecode = pygpu.gpuarray.dtype_to_typecode(self.argmax_dtype)
        ret = """
        #if PY_MAJOR_VERSION >= 3
            #ifndef PyInt_AS_LONG
                #define PyInt_AS_LONG PyLong_AS_LONG
            #endif
        #endif

        int err = 0;

        unsigned  %(name)s_redux_len = PyTuple_GET_SIZE(%(axes)s);
        unsigned* %(name)s_axes_to_reduce = (unsigned*)malloc(%(name)s_redux_len * sizeof(unsigned));
        for (unsigned i = 0; i < %(name)s_redux_len; ++i) {
            PyObject* axis_object = PyTuple_GET_ITEM(%(axes)s, i);
            %(name)s_axes_to_reduce[i] = (unsigned) PyInt_AS_LONG(axis_object);
        }

        size_t  %(name)s_input_ndim = PyGpuArray_NDIM(%(X)s);
        size_t  %(name)s_output_ndim = %(name)s_input_ndim - %(name)s_redux_len;
        size_t* %(name)s_output_dims = (size_t*)malloc(%(name)s_output_ndim * sizeof(size_t));
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

        if (theano_prep_output(&%(max)s, %(name)s_output_ndim, %(name)s_output_dims, %(max_typecode)s, GA_C_ORDER, %(X)s->context)) {
            PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to prepare max output.");
            %(fail)s
        }
        if (theano_prep_output(&%(argmax)s, %(name)s_output_ndim, %(name)s_output_dims, %(argmax_typecode)s, GA_C_ORDER, %(X)s->context)) {
            PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to prepare argmax output.");
            %(fail)s
        }

        if (%(name)s_input_ndim == 0) {
            /* GpuArray_maxandargmax can't handle a 0-d array
             * because it expects that 1 <= redux_len <= input_ndim.
             * As input_ndim == 0, then 1 <= redux_len <= 0 is false.
             * To handle this case we copy input to max and we set argmax to 0.
             */
            if (GA_NO_ERROR != GpuArray_setarray(&%(max)s->ga, &%(X)s->ga)) {
                PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to copy input to max when input is a scalar.");
                %(fail)s
            }
            if (GA_NO_ERROR != GpuArray_memset(&%(argmax)s->ga, 0)) {
                PyErr_SetString(PyExc_RuntimeError, "GpuMaxAndArgmax: unable to set argmax to 0 when input is a scalar.");
                %(fail)s
            }
        } else if (GA_NO_ERROR != (err =
            GpuArray_maxandargmax(&%(max)s->ga, &%(argmax)s->ga, &%(X)s->ga, %(name)s_redux_len, %(name)s_axes_to_reduce)
        )) {
            PyErr_Format(PyExc_RuntimeError,
                "GpuMaxAndArgmax: unable to compute gpuarray maxandargmax: error %%d: %%s (%%s).",
                err, gpuarray_error_str(err), GpuArray_error(&%(X)s->ga, err));
            %(fail)s
        }
        """
        return ret % {'X': input_names[0], 'axes': sub['params'], 'max': output_names[0], 'argmax': output_names[1],
                      'max_typecode': max_typecode, 'argmax_typecode': argmax_typecode,
                      'name': name, 'fail': sub['fail']}

    def c_code_cleanup(self, node, name, inputs, outputs, sub):
        return """
        free(%(name)s_output_dims);
        free(%(name)s_axes_to_reduce);
        """ % {'name': name, 'X': inputs[0]}

    def c_code_cache_version(self):
        return (2,)
