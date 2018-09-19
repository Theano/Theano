from __future__ import absolute_import, print_function, division
import numpy as np
from six.moves import xrange

import theano
from theano.tensor import basic
from theano.tensor import nlinalg  # noqa
from theano import gof, scalar
from theano.gof import Generic, ParamsType, EnumList
from theano import gradient
from theano.gradient import DisconnectedType, disconnected_type
from theano.scalar import int32 as int_t
tensor = basic


class CpuContiguous(theano.Op):
    """
    Check to see if the input is c-contiguous,
    if it is, do nothing, else return a contiguous array.
    """

    __props__ = ()
    view_map = {0: [0]}
    check_input = False

    def make_node(self, x):
        x_ = theano.tensor.as_tensor_variable(x)
        return theano.Apply(self, [x_], [x_.type()])

    def perform(self, node, inputs, output_storage):
        x, = inputs
        y = output_storage[0]
        # if the ouput is contiguous do nothing, else copy
        # the input
        if not x.flags['C_CONTIGUOUS']:
            x = x.copy()
        assert x.flags['C_CONTIGUOUS']
        y[0] = x

    def grad(self, inputs, dout):
        return [theano.tensor.as_tensor_variable(dout[0])]

    def c_code(self, node, name, inames, onames, sub):
        x, = inames
        y, = onames
        code = """
            if (!PyArray_CHKFLAGS(%(x)s, NPY_ARRAY_C_CONTIGUOUS)){
                // check to see if output is contiguous first
                if (%(y)s != NULL &&
                    PyArray_CompareLists(PyArray_DIMS(%(y)s), PyArray_DIMS(%(x)s), PyArray_NDIM(%(x)s)) &&
                    PyArray_CHKFLAGS(%(y)s, NPY_ARRAY_C_CONTIGUOUS)){
                    PyArray_CopyInto(%(y)s, %(x)s);
                }
                else{
                    Py_XDECREF(%(y)s);
                    %(y)s = PyArray_GETCONTIGUOUS(%(x)s);
                }
            }
            else{
                Py_XINCREF(%(x)s);
                Py_XDECREF(%(y)s);
                %(y)s = %(x)s;
            }
            """ % locals()
        return code

    def c_code_cache_version(self):
        return (1,)

cpu_contiguous = CpuContiguous()


class SearchsortedOp(theano.Op):
    """Wrapper of numpy.searchsorted.

    For full documentation, see :func:`searchsorted`.

    See Also
    --------
    searchsorted : numpy-like function to use the SearchsortedOp

    """

    params_type = Generic()
    __props__ = ("side", )
    check_input = False

    def __init__(self, side='left'):
        if side == 'left' or side == 'right':
            self.side = side
        else:
            raise ValueError('\'%(side)s\' is an invalid value for keyword \'side\''
                             % locals())

    def get_params(self, node):
        return self.side

    def make_node(self, x, v, sorter=None):
        x = basic.as_tensor(x, ndim=1)
        v = basic.as_tensor(v)
        out_type = v.type.clone(dtype='int64')
        if sorter is None:
            return theano.Apply(self, [x, v], [out_type()])
        else:
            sorter = basic.as_tensor(sorter, ndim=1)
            if (theano.configdefaults.python_int_bitwidth() == 32 and
                    sorter.dtype == 'int64'):
                raise TypeError(
                    "numpy.searchsorted with Python 32bit do not support a"
                    " sorter of int64.")
            if sorter.type not in basic.int_vector_types:
                raise TypeError('sorter must be an integer vector',
                                sorter.type)
            return theano.Apply(self, [x, v, sorter], [out_type()])

    def infer_shape(self, node, shapes):
        return [shapes[1]]

    def perform(self, node, inputs, output_storage, params):
        x = inputs[0]
        v = inputs[1]
        if len(node.inputs) == 3:
            sorter = inputs[2]
        else:
            sorter = None
        z = output_storage[0]

        z[0] = np.searchsorted(x, v, side=params, sorter=sorter).astype(
            node.outputs[0].dtype)

    def c_support_code_struct(self, node, name):
        return """
            int right_%(name)s;
        """ % locals()

    def c_init_code_struct(self, node, name, sub):
        side = sub['params']
        fail = sub['fail']
        return """
            PyObject* tmp_%(name)s = PyUnicode_FromString("right");
            if (tmp_%(name)s == NULL)
                %(fail)s;
            right_%(name)s = PyUnicode_Compare(%(side)s, tmp_%(name)s);
            Py_DECREF(tmp_%(name)s);
        """ % locals()

    def c_code(self, node, name, inames, onames, sub):
        sorter = None
        if len(node.inputs) == 3:
            x, v, sorter = inames
        else:
            x, v = inames
        if not sorter:
            sorter = "NULL"
        z, = onames
        fail = sub['fail']

        return """
            Py_XDECREF(%(z)s);
            %(z)s = (PyArrayObject*) PyArray_SearchSorted(%(x)s, (PyObject*) %(v)s,
                                                          right_%(name)s ? NPY_SEARCHLEFT : NPY_SEARCHRIGHT, (PyObject*) %(sorter)s);
            if (!%(z)s)
                %(fail)s;
            if (PyArray_TYPE(%(z)s) != NPY_INT64){
                PyObject * tmp = PyArray_Cast(%(z)s, NPY_INT64);
                Py_XDECREF(%(z)s);
                %(z)s = (PyArrayObject*) tmp;
            }
        """ % locals()

    def c_code_cache_version(self):
        return (2,)

    def grad(self, inputs, output_gradients):
        num_ins = len(inputs)
        if num_ins == 3:
            x, v, sorter = inputs
        else:
            x, v = inputs

        x_grad = gradient._float_zeros_like(x)
        v_grad = gradient._float_zeros_like(v)
        if num_ins == 3:
            return [x_grad, v_grad, disconnected_type()]
        else:
            return [x_grad, v_grad]


def searchsorted(x, v, side='left', sorter=None):
    """Find indices where elements should be inserted to maintain order.

    Wrapping of numpy.searchsorted. Find the indices into a sorted array
    `x` such that, if the corresponding elements in `v` were inserted
    before the indices, the order of `x` would be preserved.

    Parameters
    ----------
    x: 1-D tensor (array-like)
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        which sorts it.
    v: tensor (array-like)
        Contains the values to be inserted into `x`.
    side: {'left', 'right'}, optional.
        If 'left' (default), the index of the first suitable
        location found is given. If 'right', return the last such index. If
        there is no suitable index, return either 0 or N (where N is the length
        of `x`).
    sorter: 1-D tensor of integers (array-like), optional
        Contains indices that sort array `x` into ascending order.
        They are typically the result of argsort.

    Returns
    -------
    indices : tensor of integers (int64)
        Array of insertion points with the same shape as `v`.

    See Also
    --------
    `numpy.searchsorted <https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.searchsorted.html>`_

    Notes
    -----
    * Binary search is used to find the required insertion points.
    * This Op is working **only on CPU** currently.

    Examples
    --------
    >>> from theano import tensor
    >>> x = tensor.dvector()
    >>> idx = x.searchsorted(3)
    >>> idx.eval({x: [1,2,3,4,5]})
    array(2)
    >>> tensor.extra_ops.searchsorted([1,2,3,4,5], 3).eval()
    array(2)
    >>> tensor.extra_ops.searchsorted([1,2,3,4,5], 3, side='right').eval()
    array(3)
    >>> tensor.extra_ops.searchsorted([1,2,3,4,5], [-10, 10, 2, 3]).eval()
    array([0, 5, 1, 2])

    .. versionadded:: 0.9

    """
    return SearchsortedOp(side=side)(x, v, sorter)


class CumOp(theano.Op):
    # See function cumsum/cumprod for docstring

    __props__ = ("axis", "mode")
    check_input = False
    params_type = ParamsType(c_axis=int_t,
                             mode=EnumList(('MODE_ADD', 'add'),
                                           ('MODE_MUL', 'mul')))

    def __init__(self, axis=None, mode='add'):
        if mode not in ('add', 'mul'):
            raise ValueError('%s: Unknown mode "%s"' % (type(self).__name__, mode))
        self.axis = axis
        self.mode = mode

    c_axis = property(lambda self: np.MAXDIMS if self.axis is None else self.axis)

    def make_node(self, x):
        x = basic.as_tensor_variable(x)
        out_type = x.type()

        if self.axis is None:
            out_type = theano.tensor.vector(dtype=x.dtype)  # Flatten
        elif self.axis >= x.ndim or self.axis < -x.ndim:
            raise ValueError('axis(={0}) out of bounds'.format(self.axis))

        return theano.Apply(self, [x], [out_type])

    def perform(self, node, inputs, output_storage, params):
        x = inputs[0]
        z = output_storage[0]
        z[0] = {'add': np.cumsum, 'mul': np.cumprod}[self.mode](x, axis=self.axis)

    def grad(self, inputs, output_gradients):
        x, = inputs
        gi, = output_gradients

        if self.axis is None:
            if self.mode == 'add':
                return [cumsum(gi[::-1])[::-1].reshape(x.shape)]
            elif self.mode == 'mul':
                fx = cumprod(x, axis=self.axis)
                return [cumsum(
                    (fx * gi)[::-1])[::-1].reshape(x.shape) / x]
            else:
                raise NotImplementedError(
                    '%s: unknown gradient for mode "%s"' %
                    (type(self).__name__, self.mode))

        reverse_slicing = [slice(None, None, None)] * gi.ndim
        reverse_slicing[self.axis] = slice(None, None, -1)
        reverse_slicing = tuple(reverse_slicing)
        # We need to reverse the gradients along ``self.axis``,
        #  compute cumsum, then reverse again
        if self.mode == 'add':
            return [cumsum(gi[reverse_slicing], self.axis)[reverse_slicing]]
        elif self.mode == 'mul':
            fx = cumprod(x, axis=self.axis)
            return [cumsum(
                (fx * gi)[reverse_slicing], self.axis)[reverse_slicing] / x]
        else:
            raise NotImplementedError(
                '%s: unknown gradient for mode "%s"' %
                (type(self).__name__, self.mode))

    def infer_shape(self, node, shapes):
        if self.axis is None:
            return [(tensor.prod(shapes[0]),)]  # Flatten

        return shapes

    def c_code(self, node, name, inames, onames, sub):
        x, = inames
        z, = onames
        axis = self.axis
        fail = sub['fail']
        params = sub['params']

        code = """
                int axis = %(params)s->c_axis;
                if (axis == 0 && PyArray_NDIM(%(x)s) == 1)
                    axis = NPY_MAXDIMS;
                npy_intp shape[1] = { PyArray_SIZE(%(x)s) };
                if(axis == NPY_MAXDIMS && !(%(z)s && PyArray_DIMS(%(z)s)[0] == shape[0]))
                {
                    Py_XDECREF(%(z)s);
                    %(z)s = (PyArrayObject*) PyArray_SimpleNew(1, shape, PyArray_TYPE((PyArrayObject*) py_%(x)s));
                }

                else if(axis != NPY_MAXDIMS && !(%(z)s && PyArray_CompareLists(PyArray_DIMS(%(z)s), PyArray_DIMS(%(x)s), PyArray_NDIM(%(x)s))))
                {
                    Py_XDECREF(%(z)s);
                    %(z)s = (PyArrayObject*) PyArray_SimpleNew(PyArray_NDIM(%(x)s), PyArray_DIMS(%(x)s), PyArray_TYPE(%(x)s));
                }

                if (!%(z)s)
                    %(fail)s;
                {

                    PyObject * t = NULL;
                    if(%(params)s->mode == MODE_ADD)
                        t = PyArray_CumSum(
                            %(x)s, axis,
                            PyArray_TYPE(%(x)s), %(z)s);
                    else if(%(params)s->mode == MODE_MUL)
                        t = PyArray_CumProd(
                            %(x)s, axis,
                            PyArray_TYPE(%(x)s), %(z)s);

                    if (!t){
                       %(fail)s;
                    }
                    // Because PyArray_CumSum/CumProd returns a newly created reference on t.
                    Py_XDECREF(t);
                }
            """ % locals()

        return code

    def c_code_cache_version(self):
        return (8,)

    def __str__(self):
        return "%s{%s, %s}" % (self.__class__.__name__, self.axis, self.mode)


def cumsum(x, axis=None):
    """Return the cumulative sum of the elements along a given axis.

    Wraping of numpy.cumsum.

    Parameters
    ----------
    x
        Input tensor variable.
    axis
        The axis along which the cumulative sum is computed.
        The default (None) is to compute the cumsum over the flattened array.


    .. versionadded:: 0.7

    """
    return CumOp(axis=axis, mode='add')(x)


def cumprod(x, axis=None):
    """Return the cumulative product of the elements along a given axis.

    Wraping of numpy.cumprod.

    Parameters
    ----------
    x
        Input tensor variable.

    axis
        The axis along which the cumulative product is computed.
        The default (None) is to compute the cumprod over the flattened array.


    .. versionadded:: 0.7

    """
    return CumOp(axis=axis, mode='mul')(x)


# CumsumOp and CumprodOp are for compatibility with old version,
# just in case unpickling a theano function with old Ops.
class CumsumOp(theano.Op):
    __props__ = ("axis",)

    def __new__(typ, *args, **kwargs):
        obj = object.__new__(CumOp, *args, **kwargs)
        obj.mode = 'add'
        return obj


class CumprodOp(theano.Op):
    __props__ = ("axis",)

    def __new__(typ, *args, **kwargs):
        obj = object.__new__(CumOp, *args, **kwargs)
        obj.mode = 'mul'
        return obj


class DiffOp(theano.Op):
    # See function diff for docstring

    __props__ = ("n", "axis")

    def __init__(self, n=1, axis=-1):
        self.n = n
        self.axis = axis
        # numpy return a view in that case.
        # TODO, make an optimization that remove this op in this case.
        if n == 0:
            self.view_map = {0: [0]}

    def make_node(self, x):
        x = basic.as_tensor_variable(x)
        return theano.Apply(self, [x], [x.type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        z[0] = np.diff(x, n=self.n, axis=self.axis)

    def grad(self, inputs, outputs_gradients):
        inputs = inputs[0]

        if inputs.ndim != 1:
            raise NotImplementedError("Grad is not implemented for inputs with"
                                      "number of dimension other than 1.")

        z = outputs_gradients[0]

        def _grad_helper(z):
            pre = basic.concatenate([[0.], z])
            app = basic.concatenate([z, [0.]])
            return pre - app

        for k in range(self.n):
            z = _grad_helper(z)
        return [z]

    def infer_shape(self, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        out_shape = list(i0_shapes)
        out_shape[self.axis] = out_shape[self.axis] - self.n
        return [out_shape]


def diff(x, n=1, axis=-1):
    """Calculate the n-th order discrete difference along given axis.

    The first order difference is given by out[i] = a[i + 1] - a[i]
    along the given axis, higher order differences are calculated by
    using diff recursively. Wraping of numpy.diff.

    Parameters
    ----------
    x
        Input tensor variable.

    n
        The number of times values are differenced, default is 1.

    axis
        The axis along which the difference is taken, default is the last axis.


    .. versionadded:: 0.6

    """
    return DiffOp(n=n, axis=axis)(x)


def bincount(x, weights=None, minlength=None, assert_nonneg=False):
    """Count number of occurrences of each value in array of ints.

    The number of bins (of size 1) is one larger than the largest
    value in x. If minlength is specified, there will be at least
    this number of bins in the output array (though it will be longer
    if necessary, depending on the contents of x). Each bin gives the
    number of occurrences of its index value in x. If weights is
    specified the input array is weighted by it, i.e. if a value n
    is found at position i, out[n] += weight[i] instead of out[n] += 1.

    Parameters
    ----------
    x : 1 dimension, nonnegative ints
    weights : array of the same shape as x with corresponding weights.
        Optional.
    minlength : A minimum number of bins for the output array.
        Optional.
    assert_nonneg : A flag that inserts an assert_op to check if
        every input x is nonnegative.
        Optional.


    .. versionadded:: 0.6

    """
    if x.ndim != 1:
        raise TypeError("Inputs must be of dimension 1.")

    if assert_nonneg:
        from theano.tensor.opt import Assert
        assert_op = Assert('Input to bincount has negative values!')
        x = assert_op(x, theano.tensor.all(x >= 0))

    max_value = theano.tensor.cast(x.max() + 1, 'int64')

    if minlength is not None:
        max_value = theano.tensor.maximum(max_value, minlength)

    # Note: we do not use inc_subtensor(out[x], ...) in the following lines,
    # since out[x] raises an exception if the indices (x) are int8.
    if weights is None:
        out = theano.tensor.zeros([max_value], dtype=x.dtype)
        out = theano.tensor.advanced_inc_subtensor1(out, 1, x)
    else:
        out = theano.tensor.zeros([max_value], dtype=weights.dtype)
        out = theano.tensor.advanced_inc_subtensor1(out, weights, x)
    return out


def squeeze(x):
    """
    Remove broadcastable dimensions from the shape of an array.

    It returns the input array, but with the
    broadcastable dimensions removed. This is
    always `x` itself or a view into `x`.

    .. versionadded:: 0.6

    Parameters
    ----------
    x
        Input data, tensor variable.

    Returns
    -------
    object
        `x` without its broadcastable dimensions.

    """
    view = x.dimshuffle([i for i in range(x.ndim)
                         if not x.broadcastable[i]])
    return view


def compress(condition, x, axis=None):
    """
    Return selected slices of an array along given axis.

    It returns the input tensor, but with selected slices along a given axis
    retained. If no axis is provided, the tensor is flattened.
    Corresponds to numpy.compress

    .. versionadded:: 0.7

    Parameters
    ----------
    x
        Input data, tensor variable.
    condition
         1 dimensional array of non-zero and zero values
         corresponding to indices of slices along a selected axis.

    Returns
    -------
    object
        `x` with selected slices.

    """
    indices = theano.tensor.basic.flatnonzero(condition)
    return x.take(indices, axis=axis)


class RepeatOp(theano.Op):
    # See the repeat function for docstring

    __props__ = ("axis",)

    def __init__(self, axis=None):
        self.axis = axis

    def make_node(self, x, repeats):
        x = basic.as_tensor_variable(x)
        repeats = basic.as_tensor_variable(repeats)

        if repeats.dtype not in tensor.integer_dtypes:
            raise TypeError("repeats.dtype must be an integer.")

        # Some dtypes are not supported by numpy's implementation of repeat.
        # Until another one is available, we should fail at graph construction
        # time, not wait for execution.
        ptr_bitwidth = theano.configdefaults.local_bitwidth()
        if ptr_bitwidth == 64:
            numpy_unsupported_dtypes = ('uint64',)
        if ptr_bitwidth == 32:
            numpy_unsupported_dtypes = ('uint32', 'int64', 'uint64')

        if repeats.dtype in numpy_unsupported_dtypes:
            raise TypeError(
                ("dtypes %s are not supported by numpy.repeat "
                 "for the 'repeats' parameter, "
                 % str(numpy_unsupported_dtypes)), repeats.dtype)

        if self.axis is None:
            broadcastable = [False]
        else:
            try:
                const_reps = basic.get_scalar_constant_value(repeats)
            except basic.NotScalarConstantError:
                const_reps = None
            if const_reps == 1:
                broadcastable = x.broadcastable
            else:
                broadcastable = list(x.broadcastable)
                broadcastable[self.axis] = False

        out_type = theano.tensor.TensorType(x.dtype, broadcastable)

        return theano.Apply(self, [x, repeats], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        repeats = inputs[1]
        z = output_storage[0]
        z[0] = np.repeat(x, repeats=repeats, axis=self.axis)

    def connection_pattern(self, node):

        return [[True], [False]]

    def grad(self, inputs, gout):
        (x, repeats) = inputs
        (gz,) = gout
        if repeats.ndim == 0:
            if self.axis is None:
                axis = x.ndim
            else:
                if self.axis >= 0:
                    axis = self.axis + 1
                else:
                    axis = self.axis + x.ndim + 1

            shape = [x.shape[k] for k in range(x.ndim)]
            shape.insert(axis, repeats)

            return [gz.reshape(shape, x.ndim + 1).sum(axis=axis),
                    DisconnectedType()()]
        elif repeats.ndim == 1:
            # For this implementation, we would need to specify the length
            # of repeats in order to split gz in the right way to sum
            # the good part.
            raise NotImplementedError()
        else:
            raise ValueError()

    def infer_shape(self, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        repeats = node.inputs[1]
        out_shape = list(i0_shapes)

        # uint64 shape are not supported.
        dtype = None
        if repeats.dtype in ['uint8', 'uint16', 'uint32']:
            dtype = 'int64'
        if self.axis is None:
            if repeats.ndim == 0:
                if len(i0_shapes) == 0:
                    out_shape = [repeats]
                else:
                    res = 1
                    for d in i0_shapes:
                        res = res * d
                    out_shape = (res * repeats, )
            else:
                out_shape = [theano.tensor.sum(repeats, dtype=dtype)]
        else:
            if repeats.ndim == 0:
                out_shape[self.axis] = out_shape[self.axis] * repeats
            else:
                out_shape[self.axis] = theano.tensor.sum(repeats, dtype=dtype)
        return [out_shape]


def repeat(x, repeats, axis=None):
    """Repeat elements of an array.

    It returns an array which has the same shape as `x`, except
    along the given axis. The axis is used to speficy along which
    axis to repeat values. By default, use the flattened input
    array, and return a flat output array.

    The number of repetitions for each element is `repeat`.
    `repeats` is broadcasted to fit the length of the given `axis`.

    Parameters
    ----------
    x
        Input data, tensor variable.
    repeats
        int, scalar or tensor variable
    axis : int, optional

    See Also
    --------
    tensor.tile

    .. versionadded:: 0.6

    """
    repeats = tensor.as_tensor_variable(repeats)

    if repeats.ndim > 1:
        raise ValueError('The dimension of repeats should not exceed 1.')

    if repeats.ndim == 1 and not repeats.broadcastable[0]:
            return RepeatOp(axis=axis)(x, repeats)
    else:
        if repeats.ndim == 1:
            repeats = repeats[0]

        if x.dtype == 'uint64':
            raise TypeError("theano.tensor.repeat don't support dtype uint64")

        if axis is None:
            axis = 0
            x = x.flatten()
        else:
            if axis >= x.ndim:
                raise ValueError('Axis should not exceed x.ndim-1.')
            if axis < 0:
                axis = x.ndim + axis

        shape = [x.shape[i] for i in xrange(x.ndim)]

        # shape_ is the shape of the intermediate tensor which has
        # an additional dimension comparing to x. We use alloc to
        # allocate space for this intermediate tensor to replicate x
        # along that additional dimension.
        shape_ = shape[:]
        shape_.insert(axis + 1, repeats)

        # shape is now the shape of output, where shape[axis] becomes
        # shape[axis]*repeats.
        shape[axis] = shape[axis] * repeats

        # dims_ is the dimension of that intermediate tensor.
        dims_ = list(np.arange(x.ndim))
        dims_.insert(axis + 1, 'x')

        # After the original tensor is duplicated along the additional
        # dimension, we reshape it to the expected output shape, and
        # return the output z.
        z = tensor.alloc(x.dimshuffle(*dims_), *shape_).reshape(shape)
        return z


class Bartlett(gof.Op):
    # See function bartlett for docstring
    __props__ = ()

    def make_node(self, M):
        M = tensor.as_tensor_variable(M)
        if M.ndim != 0:
            raise TypeError('%s only works on scalar input'
                            % self.__class__.__name__)
        elif M.dtype not in theano.tensor.integer_dtypes:
            # dtype is a theano attribute here
            raise TypeError('%s only works on integer input'
                            % self.__class__.__name__)
        return gof.Apply(self, [M], [tensor.dvector()])

    def perform(self, node, inputs, out_):
        M = inputs[0]
        out, = out_
        out[0] = np.bartlett(M)

    def infer_shape(self, node, in_shapes):
        temp = node.inputs[0]
        M = tensor.switch(tensor.lt(temp, 0),
                          tensor.cast(0, temp.dtype),
                          temp)
        return [[M]]

    def grad(self, inputs, output_grads):
        return [None for i in inputs]
bartlett_ = Bartlett()


# I create a function only to have the doc show well.
def bartlett(M):
    """
    An instance of this class returns the Bartlett spectral window in the
    time-domain. The Bartlett window is very similar to a triangular window,
    except that the end points are at zero. It is often used in signal
    processing for tapering a signal, without generating too much ripple in
    the frequency domain.

    .. versionadded:: 0.6

    Parameters
    ----------
    M : integer scalar
        Number of points in the output window. If zero or less,
        an empty vector is returned.

    Returns
    -------
    vector of doubles
        The triangular window, with the maximum value normalized to one
        (the value one appears only if the number of samples is odd), with
        the first and last samples equal to zero.

    """
    return bartlett_(M)


class FillDiagonal(gof.Op):
    # See function fill_diagonal for docstring
    __props__ = ()

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, a, val):
        a = tensor.as_tensor_variable(a)
        val = tensor.as_tensor_variable(val)
        if a.ndim < 2:
            raise TypeError('%s: first parameter must have at least'
                            ' two dimensions' % self.__class__.__name__)
        elif val.ndim != 0:
            raise TypeError('%s: second parameter must be a scalar'
                            % self.__class__.__name__)
        val = tensor.cast(val, dtype=scalar.upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError('%s: type of second parameter must be the same as'
                            ' the first\'s' % self.__class__.__name__)
        return gof.Apply(self, [a, val], [a.type()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0].copy()
        val = inputs[1]
        if a.ndim == 2:
            # numpy.fill_diagonal up to date(including 1.6.2) have a
            # bug for tall matrix.
            # For 2-d arrays, we accept rectangular ones.
            step = a.shape[1] + 1
            end = a.shape[1] * a.shape[1]
            # Write the value out into the diagonal.
            a.flat[:end:step] = val
        else:
            np.fill_diagonal(a, val)

        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Notes
        -----
        The gradient is currently implemented for matrices only.

        """
        a, val = inp
        grad = cost_grad[0]
        if (a.dtype.startswith('complex')):
            return [None, None]
        elif a.ndim > 2:
            raise NotImplementedError('%s: gradient is currently implemented'
                                      ' for matrices only' %
                                      self.__class__.__name__)
        wr_a = fill_diagonal(grad, 0)  # valid for any number of dimensions
        # diag is only valid for matrices
        wr_val = theano.tensor.nlinalg.diag(grad).sum()
        return [wr_a, wr_val]
fill_diagonal_ = FillDiagonal()


# I create a function only to have the doc show well.
def fill_diagonal(a, val):
    """
    Returns a copy of an array with all
    elements of the main diagonal set to a specified scalar value.

    .. versionadded:: 0.6

    Parameters
    ----------
    a
        Rectangular array of at least two dimensions.
    val
        Scalar value to fill the diagonal whose type must be
        compatible with that of array 'a' (i.e. 'val' cannot be viewed
        as an upcast of 'a').

    Returns
    -------
    array
        An array identical to 'a' except that its main diagonal
        is filled with scalar 'val'. (For an array 'a' with a.ndim >=
        2, the main diagonal is the list of locations a[i, i, ..., i]
        (i.e. with indices all identical).)

    Support rectangular matrix and tensor with more than 2 dimensions
    if the later have all dimensions are equals.



    """
    return fill_diagonal_(a, val)


class FillDiagonalOffset(gof.Op):
    # See function fill_diagonal_offset for docstring
    __props__ = ()

    def infer_shape(self, node, in_shapes):
        return [in_shapes[0]]

    def make_node(self, a, val, offset):
        a = tensor.as_tensor_variable(a)
        val = tensor.as_tensor_variable(val)
        offset = tensor.as_tensor_variable(offset)
        if a.ndim != 2:
            raise TypeError('%s: first parameter must have exactly'
                            ' two dimensions' % self.__class__.__name__)
        elif val.ndim != 0:
            raise TypeError('%s: second parameter must be a scalar'
                            % self.__class__.__name__)
        elif offset.ndim != 0:
            raise TypeError('%s: third parameter must be a scalar'
                            % self.__class__.__name__)
        val = tensor.cast(val, dtype=scalar.upcast(a.dtype, val.dtype))
        if val.dtype != a.dtype:
            raise TypeError('%s: type of second parameter must be the same'
                            ' as the first\'s' % self.__class__.__name__)
        elif offset.dtype not in theano.tensor.integer_dtypes:
            raise TypeError('%s: type of third parameter must be as integer'
                            ' use theano.tensor.cast( input, \'int32/int64\')'
                            % self.__class__.__name__)

        return gof.Apply(self, [a, val, offset], [a.type()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0].copy()
        val = inputs[1]
        offset = inputs[2]
        height, width = a.shape

        """
        Notes
        -----
        The fill_diagonal only support rectangular matrix. The output
        of tall matrix is "wrapped", which is an option in numpy 1.9.0
        but was regarded as a bug in numpy 1.6.2. Here I implement the
        fill_diagonal_offset with unwrapped output, so fill_diagonal_offset
        supports tall matrix.(This make a little difference between the output
        of fill_diagonal and fill_diagonal_offset only in the case of tall
        matrix)

        """
        if offset >= 0:
            start = offset
            num_of_step = min(min(width, height), width - offset)
        else:
            start = - offset * a.shape[1]
            num_of_step = min(min(width, height), height + offset)
        step = a.shape[1] + 1
        end = start + step * num_of_step
        # Write the value out into the diagonal.
        a.flat[start:end:step] = val

        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Notes
        -----
        The gradient is currently implemented for matrices only.
        """
        a, val, offset = inp
        grad = cost_grad[0]
        height, width = grad.shape

        if (a.dtype.startswith('complex')):
            return [None, None]

        # only valid for matrices
        wr_a = fill_diagonal_offset(grad, 0, offset)

        offset_abs = basic.abs_(offset)
        pos_offset_flag = basic.ge(offset, 0)
        neg_offset_flag = basic.lt(offset, 0)
        min_wh = basic.minimum(width, height)

        start = offset * pos_offset_flag + offset_abs * width * neg_offset_flag
        num_of_step = basic.minimum(min_wh, width * pos_offset_flag +
                                    height * neg_offset_flag - offset_abs)

        step = a.shape[1] + 1
        end = start + step * num_of_step

        # input of slice should be integer
        start = basic.cast(start, 'int32')
        step = basic.cast(step, 'int32')
        end = basic.cast(end, 'int32')

        wr_val = grad.flatten()[start:end:step].sum()

        wr_offset = theano.gradient.grad_undefined(
            self, 2, offset,
            "offset is not defined for non-integer offset so"
            " fill_diagonal_offset(a,val,offset+eps) is undefined")

        return [wr_a, wr_val, wr_offset]

fill_diagonal_offset_ = FillDiagonalOffset()


def fill_diagonal_offset(a, val, offset):
    """
    Returns a copy of an array with all
    elements of the main diagonal set to a specified scalar value.

    Parameters
    ----------
    a
        Rectangular array of two dimensions.
    val
        Scalar value to fill the diagonal whose type must be
        compatible with that of array 'a' (i.e. 'val' cannot be viewed
        as an upcast of 'a').
    offset
        Scalar value Offset of the diagonal from the main
        diagonal. Can be positive or negative integer.

    Returns
    -------
    array
        An array identical to 'a' except that its offset diagonal
        is filled with scalar 'val'. The output is unwrapped.

    """
    return fill_diagonal_offset_(a, val, offset)


def to_one_hot(y, nb_class, dtype=None):
    """
    Return a matrix where each row correspond to the one hot
    encoding of each element in y.

    Parameters
    ----------
    y
        A vector of integer value between 0 and nb_class - 1.
    nb_class : int
        The number of class in y.
    dtype : data-type
        The dtype of the returned matrix. Default floatX.

    Returns
    -------
    object
        A matrix of shape (y.shape[0], nb_class), where each row ``i`` is
        the one hot encoding of the corresponding ``y[i]`` value.

    """
    ret = theano.tensor.zeros((y.shape[0], nb_class),
                              dtype=dtype)
    ret = theano.tensor.set_subtensor(ret[theano.tensor.arange(y.shape[0]), y],
                                      1)
    return ret


class Unique(theano.Op):
    """
    Wraps numpy.unique. This op is not implemented on the GPU.

    Examples
    --------
    >>> import numpy as np
    >>> import theano

    >>> x = theano.tensor.vector()
    >>> f = theano.function([x], Unique(True, True, False)(x))
    >>> f([1, 2., 3, 4, 3, 2, 1.])
    [array([ 1.,  2.,  3.,  4.]), array([0, 1, 2, 3]), array([0, 1, 2, 3, 2, 1, 0])]

    >>> y = theano.tensor.matrix()
    >>> g = theano.function([y], Unique(True, True, False)(y))
    >>> g([[1, 1, 1.0], (2, 3, 3.0)])
    [array([ 1.,  2.,  3.]), array([0, 3, 4]), array([0, 0, 0, 1, 2, 2])]

    """

    __props__ = ("return_index", "return_inverse", "return_counts",
                 "axis")

    def __init__(self, return_index=False, return_inverse=False,
                 return_counts=False, axis=None):
        self.return_index = return_index
        self.return_inverse = return_inverse
        self.return_counts = return_counts
        self.axis = axis
        numpy_ver = [int(n) for n in np.__version__.split('.')[:2]]
        if self.axis is not None and bool(numpy_ver < [1, 13]):
            raise RuntimeError(
                "Numpy version = " + np.__version__ +
                ". Option 'axis={}' works starting"
                " from version 1.13.0.".format(axis))

    def make_node(self, x):
        x = basic.as_tensor_variable(x)
        self_axis = self.axis
        if self_axis is None:
            broadcastable = [False]
        else:
            if self_axis < 0:
                self_axis += len(x.broadcastable)
            if self_axis < 0 or self_axis >= len(x.broadcastable):
                raise RuntimeError(
                    "Unique axis `{}` is outside of input ndim = "
                    "{}.".format(self.axis, len(x.broadcastable))
                    )
            broadcastable = [b if axis != self_axis else False
                             for axis, b in enumerate(x.broadcastable)]
        outputs = [basic.TensorType(broadcastable=broadcastable,
                                    dtype=x.dtype)()]
        typ = basic.TensorType(broadcastable=[False], dtype='int64')
        if self.return_index:
            outputs.append(typ())
        if self.return_inverse:
            outputs.append(typ())
        if self.return_counts:
            outputs.append(typ())
        return theano.Apply(self, [x], outputs)

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage
        param = {}
        if self.return_index:
            param['return_index'] = True
        if self.return_inverse:
            param['return_inverse'] = True
        if self.return_counts:
            param['return_counts'] = True
        if self.axis is not None:
            param['axis'] = self.axis
        outs = np.unique(x, **param)
        if ((not self.return_inverse) and
                (not self.return_index) and
                (not self.return_counts)):
            z[0][0] = outs
        else:
            for i in range(len(outs)):
                z[i][0] = outs[i]

    def infer_shape(self, node, i0_shapes):
        ret = node.fgraph.shape_feature.default_infer_shape(node, i0_shapes)
        if self.axis is not None:
            self_axis = self.axis
            ndim = len(i0_shapes[0])
            if self_axis < 0:
                self_axis += ndim
            if self_axis < 0 or self_axis >= ndim:
                raise RuntimeError(
                    "Unique axis `{}` is outside of input ndim = "
                    "{}.".format(self.axis, ndim)
                    )
            ret[0] = tuple([node.fgraph.shape_feature.shape_ir(i,
                                                               node.outputs[0])
                            for i in xrange(ndim)])
        if self.return_inverse:
            if self.axis is None:
                shape = (basic.prod(i0_shapes[0]), )
            else:
                shape = (i0_shapes[0][self_axis], )
            if self.return_index:
                ret[2] = shape
                return ret
            ret[1] = shape
            return ret
        return ret

    def __setstate__(self, state):
        self.__dict__.update(state)
        # For backwards compatibility with pickled instances of Unique that
        # did not have the axis parameter specified
        if 'axis' not in state:
            self.axis = None


class UnravelIndex(gof.Op):
    __props__ = ('ndim', 'order')

    def __init__(self, ndim, order='C'):
        assert order in ('C', 'F')
        if not isinstance(ndim, int) or ndim < 1:
            raise ValueError('ndim must be an integer greater than 0')
        self.ndim = int(ndim)
        self.order = order

    def make_node(self, indices, dims):
        indices = basic.as_tensor_variable(indices)
        dims = basic.as_tensor_variable(dims)

        if indices.dtype not in basic.int_dtypes:
            raise TypeError("'%s' object cannot be interpreted as an index" % str(indices.dtype))
        if dims.dtype not in basic.int_dtypes:
            raise TypeError("'%s' object cannot be interpreted as an index" % str(dims.dtype))
        if dims.ndim != 1:
            raise TypeError("dims must be a 1D array")

        return gof.Apply(
            self, [indices, dims],
            [basic.TensorType(dtype='int64', broadcastable=(False,) * indices.ndim)()
             for i in xrange(self.ndim)])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]] * len(node.outputs)

    def perform(self, node, inp, out):
        indices, dims = inp
        res = np.unravel_index(indices, dims)
        assert len(res) == len(out)
        for i in xrange(len(out)):
            ret = theano._asarray(res[i], node.outputs[0].dtype)
            if ret.base is not None:
                # NumPy will return a view when it can.
                # But we don't want that.
                ret = ret.copy()
            out[i][0] = ret


def unravel_index(indices, dims, order='C', ndim=None):
    """
    Converts a flat index or array of flat indices into a tuple
    of coordinate arrays.

    This method is similar to the NumPy version, except for the
    additional ``ndim`` parameter. This parameter is required if
    the length of ``dims`` cannot be determined automatically.

    Parameters
    ----------
    indices : Theano or NumPy array
        An integer array whose elements are indices into the flattened
        version of an array of dimensions ``dims``.
    dims : tuple of ints
        The shape of the array to use for unraveling ``indices``.
    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.
    ndim : int, optional
        Specifies the number of dimensions, i.e., the length of
        ``dims``. This is required if the dimensions cannot be determined
        automatically from ``dims`` itself.

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has the same shape as the ``indices``
        array.

    See Also
    --------
    ravel_multi_index

    """
    if ndim is None:
        try:
            ndim = basic.get_vector_length(dims)
        except ValueError:
            raise ValueError(
                "The length of the provided dimension list (%s) cannot "
                "be automatically determined, so Theano is not able "
                "to know what the number of dimensions of the unraveled "
                "index will be. You can provide the 'ndim' keyword "
                "argument to 'unravel_index' to avoid this problem." % str(dims))

    res = UnravelIndex(ndim=ndim, order=order)(indices, dims)
    if ndim == 1:
        return (res,)
    else:
        return tuple(res)


class RavelMultiIndex(gof.Op):
    __props__ = ('mode', 'order')

    def __init__(self, mode='raise', order='C'):
        assert mode in ('raise', 'wrap', 'clip')
        assert order in ('C', 'F')
        self.mode = mode
        self.order = order

    def make_node(self, *inp):
        multi_index = [basic.as_tensor_variable(i) for i in inp[:-1]]
        dims = basic.as_tensor_variable(inp[-1])

        for i in multi_index:
            if i.dtype not in basic.int_dtypes:
                raise TypeError("'%s' object cannot be interpreted as an index" % str(i.dtype))
        if dims.dtype not in basic.int_dtypes:
            raise TypeError("'%s' object cannot be interpreted as an index" % str(dims.dtype))
        if dims.ndim != 1:
            raise TypeError("dims must be a 1D array")

        return gof.Apply(
            self, multi_index + [dims],
            [basic.TensorType(dtype='int64', broadcastable=(False,) * multi_index[0].ndim)()])

    def infer_shape(self, node, input_shapes):
        return [input_shapes[0]]

    def perform(self, node, inp, out):
        multi_index, dims = inp[:-1], inp[-1]
        res = np.ravel_multi_index(multi_index, dims,
                                   mode=self.mode, order=self.order)
        out[0][0] = theano._asarray(res, node.outputs[0].dtype)


def ravel_multi_index(multi_index, dims, mode='raise', order='C'):
    """
    Converts a tuple of index arrays into an array of flat
    indices, applying boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of Theano or NumPy arrays
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from ``multi_index`` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled.  Can specify
        either one mode or a tuple of modes, one mode per index.
        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range
        In 'clip' mode, a negative index which would normally
        wrap will clip to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as
        indexing in row-major (C-style) or column-major
        (Fortran-style) order.

    Returns
    -------
    raveled_indices : Theano array
        An array of indices into the flattened version of an array
        of dimensions ``dims``.

    See Also
    --------
    unravel_index

    """
    if not isinstance(multi_index, (tuple, list)):
        raise TypeError('multi_index must be a tuple or a list.')
    args = tuple(multi_index) + (dims,)
    return RavelMultiIndex(mode=mode, order=order)(*args)
