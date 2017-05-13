from __future__ import absolute_import, print_function, division
import numpy as np
import theano
from theano.tensor.basic import mul, arange


def _variable_is_none(var):
    return isinstance(var, theano.Constant) and var.data is None


def _check_tensor_is_scalar(var):
    '''
    Checks if a tensor variable is scalar, raise ValueError otherwise
    '''
    msg = '%(var)s is expected to be 0d tensor, got %(ndim)d'
    if var.ndim != 0:
        raise ValueError(
            msg % (var, var.ndim))


class SortOp(theano.Op):
    """
    This class is a wrapper for numpy sort function.

    """

    __props__ = ("kind", "order")

    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __str__(self):
        return self.__class__.__name__ + "{%s, %s}" % (self.kind,
                                                       str(self.order))

    def make_node(self, input, axis=-1):
        input = theano.tensor.as_tensor_variable(input)
        axis = theano.tensor.as_tensor_variable(axis)
        out_type = input.type()
        return theano.Apply(self, [input, axis], [out_type])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = np.sort(a, axis, self.kind, self.order)

    def infer_shape(self, node, inputs_shapes):
        if _variable_is_none(node.inputs[1]):
            # That means axis = None,
            # So the array is flattened before being sorted
            return [(mul(*inputs_shapes[0]),)]
        # axis should not be None
        # So there should be the same number of dimensions
        # in the input and output
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        a, axis = inputs
        indices = self.__get_argsort_indices(a, axis)
        inp_grad = output_grads[0][tuple(indices)]
        axis_grad = theano.gradient.grad_undefined(
            self, 1, axis,
            "The gradient of sort is not defined "
            "with respect to the integer axes itself")
        return [inp_grad, axis_grad]

    def __get_expanded_dim(self, a, axis, i):
        index_shape = [1] * a.ndim
        index_shape[i] = a.shape[i]
        # it's a way to emulate
        # numpy.ogrid[0: a.shape[0], 0: a.shape[1], 0: a.shape[2]]
        index_val = arange(a.shape[i]).reshape(index_shape)
        return index_val

    def __get_argsort_indices(self, a, axis):
        """
        Calculates indices which can be used to reverse sorting operation of
        "a" tensor along "axis".

        Returns
        -------
        1d array if axis is None
        list of lenght len(a.shape) otherwise

        """

        # The goal is to get gradient wrt input from gradient
        # wrt sort(input, axis)
        idx = argsort(a, axis, kind=self.kind, order=self.order)
        # rev_idx is the reverse of previous argsort operation
        rev_idx = argsort(idx, axis, kind=self.kind, order=self.order)
        indices = []
        axis_data = theano.tensor.switch(theano.tensor.ge(axis.data, 0),
                                         axis.data, a.ndim + axis.data)
        for i in range(a.ndim):
            index_val = theano.tensor.switch(theano.tensor.eq(i, axis_data),
                                             rev_idx,
                                             self.__get_expanded_dim(a,
                                                                     axis, i))
            indices.append(index_val)
        return indices
    """
    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
    """


def sort(a, axis=-1, kind='quicksort', order=None):
    """

    Parameters
    ----------
    a : Tensor
        Tensor to be sorted
    axis : Tensor
        Axis along which to sort. If None, the array is flattened before
        sorting.
    kind : {'quicksort', 'mergesort', 'heapsort'}, optional
        Sorting algorithm. Default is 'quicksort'.
    order : list, optional
        When `a` is a structured array, this argument specifies which
        fields to compare first, second, and so on. This list does not
        need to include all of the fields.

    Returns
    -------
    array
        A sorted copy of an array.

    """
    if axis is None:
        a = a.flatten()
        axis = 0
    return SortOp(kind, order)(a, axis)


class ArgSortOp(theano.Op):
    """
    This class is a wrapper for numpy argsort function.

    """

    __props__ = ("kind", "order")

    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __str__(self):
        return (self.__class__.__name__ +
                "{%s, %s}" % (self.kind, str(self.order)))

    def make_node(self, input, axis=-1):
        input = theano.tensor.as_tensor_variable(input)
        axis = theano.tensor.as_tensor_variable(axis)
        bcast = input.type.broadcastable
        return theano.Apply(self, [input, axis], [theano.tensor.TensorType(
            dtype="int64", broadcastable=bcast)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = theano._asarray(np.argsort(a, axis, self.kind, self.order),
                               dtype=node.outputs[0].dtype)

    def infer_shape(self, node, inputs_shapes):
        if _variable_is_none(node.inputs[1]):
            return [(mul(*inputs_shapes[0]),)]
        # axis should not be None, so there should be the same number of
        # dimensions in the input and output
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        # No grad defined for intergers.
        inp, axis = inputs
        inp_grad = inp.zeros_like()
        axis_grad = theano.gradient.grad_undefined(
            self, 1, axis,
            "argsort is not defined for non-integer axes so"
            " argsort(x, axis+eps) is undefined")
        return [inp_grad, axis_grad]
    """
    def R_op(self, inputs, eval_points):
        # R_op can receive None as eval_points.
        # That mean there is no diferientiable path through that input
        # If this imply that you cannot compute some outputs,
        # return None for those.
        if eval_points[0] is None:
            return eval_points
        return self.grad(inputs, eval_points)
    """


def argsort(a, axis=-1, kind='quicksort', order=None):
    """
    Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the algorithm
    specified by the kind keyword.  It returns an array of indices of
    the same shape as a that index data along the given axis in sorted
    order.

    """
    if axis is None:
        a = a.flatten()
        axis = 0
    return ArgSortOp(kind, order)(a, axis)


if hasattr(np, 'argpartition'):
    # numpy >= 1.8 implementation
    def _topk_py_impl(op, x, k, axis, idx_dtype):
        ndim = x.ndim
        if abs(k) == 1:
            i = (k + 1) // 2
            fn_max = [np.min, np.max][i]
            fn_argmax = [np.argmin, np.argmax][i]
            if not op.return_indices:
                return np.expand_dims(fn_max(x, axis=axis), axis)
            elif op.return_values:
                zi = np.expand_dims(
                    fn_argmax(x, axis=axis).astype(idx_dtype), axis)
                idx2 = tuple(np.arange(s).reshape((s,) + (1,) * (ndim - i - 1)) if i != axis else zi for i, s in enumerate(x.shape))
                zv = x[idx2]
                return zv, zi.astype(idx_dtype)
            else:
                zi = np.expand_dims(
                    fn_argmax(x, axis=axis).astype(idx_dtype), axis)
                return zi.astype(idx_dtype)

        asize = x.shape[axis]
        if asize == abs(k):
            if not op.return_indices:
                return x.copy()
            else:
                l = axis
                r = ndim - l
                reps = list(x.shape)
                reps[axis] = 1
                zi = np.arange(abs(k), dtype=idx_dtype)
                zi = zi.reshape((1,) * l + (k,) + (1,) * (r - 1))
                zi = np.tile(zi, reps)
                if op.return_values:
                    return x.copy(), zi
                else:
                    return zi

        idx = [slice(None)] * ndim
        if k > 0:
            idx[axis] = slice(-k, None)
        elif k < 0:
            idx[axis] = slice(-k)
        else:
            raise ValueError('k cannot be zero')
        if not op.return_indices:
            zv = np.partition(x, -k, axis=axis)[idx]
            return zv
        elif op.return_values:
            zi = np.argpartition(x, -k, axis=axis)[idx]
            idx2 = tuple(np.arange(s).reshape((s,)+(1,)*(ndim-i-1)) if i != axis else zi for i, s in enumerate(x.shape))
            zv = x[idx2]
            return zv, zi.astype(idx_dtype)
        else:
            zi = np.argpartition(x, -k, axis=axis)[idx]
            return zi
else:
    def _topk_py_impl(op, x, k, axis, idx_dtype):
        # TODO better compatibility?
        raise NotImplementedError('TopKOp: need numpy.argpartition() method (numpy >= 1.8)')


class TopKOp(theano.Op):
    """
    Operations related to finding k-largest elements.

    The outputs of this Op depends on ``returns_values`` and ``return_indices``,
    if both ``True``, will return two outputs, corresponding to k-largest values
    and indices. If only one is ``True``, this Op shall have only one output. Can't
    be both ``False``.

    Parameters
    ----------
    axis: integer
        The axis to perform the operation. Must be in range ``[-ndim, ndim)``, where
        ``ndim`` is the dimensionality of input tensor.

    return_values: bool
        Defaults to ``True``

        If ``True``, one output of the Op will return k-largest array values.

    return_indices: bool
        Defaults to ``False``

        If ``True``, one output of the Op will return the indices on the given axis.

    Notes
    -----
    - ``return_values`` and ``return_indices`` cannot be both ``False``

    See Also
    --------
    topk
    argtopk
    argtopk_and_topk

    """

    # TODO more params
    '''
    sorted: bool
        Defaults to ``False``

        If True, the result array would be incremental-sorted. Mutually exclusive with ``sparse``

    sparse: bool
        Defaults to ``False``

        if ``True``, the output array will always have the same shape as input.
        The non-top-k values will be replaced by zero.

    only_top_kth: bool
        Defaults to ``False``

        If ``True``, will only find the exact top k-th element. The Op behaves
        like a reduction.
    '''

    # TODO c_code


    __props__ = ('axis', 'return_values', 'return_indices')

    def __init__(self, axis=-1, return_indices=False, return_values=True):
        assert isinstance(axis, int)
        assert return_indices or return_values
        self.axis = axis
        self.return_indices = return_indices
        self.return_values = return_values

    def __str__(self):
        return '%(op)s{axis=%(axis)d}' % dict(
            op=self.__class__.__name__, axis=self.axis)

    def make_node(self, inp, k, idx_dtype='int64'):
        # numpy always uses float64 as output dtype for arg*() routines
        # however, we add this option as memory is more precious on gpu
        inp = theano.tensor.as_tensor_variable(inp)
        k = theano.tensor.as_tensor_variable(k)
        bcast = inp.type.broadcastable
        outs = []
        if self.return_values:
            outs.append(inp.type())
        if self.return_indices:
            outs.append(
                theano.tensor.TensorType(dtype=idx_dtype, broadcastable=bcast)())
        return theano.Apply(self, [inp, k], outs)

    def perform(self, node, inputs, output_storage):
        x, k = inputs
        ndim = x.ndim
        axis = self.axis
        assert -ndim <= axis < ndim
        axis %= ndim
        if not self.return_indices:
            pzv = output_storage[0]
            pzv[0] = _topk_py_impl(self, x, k, axis, None)
        elif self.return_values:
            pzv = output_storage[0]
            pzi = output_storage[1]
            pzv[0], pzi[0] = _topk_py_impl(self, x, k, axis, node.outputs[1].dtype)
        else:
            pzi = output_storage[0]
            pzi[0] = _topk_py_impl(self, x, k, axis, node.outputs[0].dtype)


    def infer_shape(self, node, inp_shapes):
        _check_tensor_is_scalar(node.inputs[1])
        shp = list(inp_shapes[0])
        if not isinstance(self.axis, int):
            raise TypeError(
                '"axis" parameter must be integer, got "%s"' % type(self.axis))
        ndim = node.inputs[0].ndim
        if ndim == 0:
            raise ValueError('Cannot take 0d tensor as input')
        if not -ndim <= self.axis < ndim:
            raise IndexError(
                '"axis" parameter out of range,'
                ' expected integer within [%d, %d]' % (-ndim, ndim - 1))
        shp[self.axis] = np.abs(node.inputs[1])
        shp = tuple(shp)
        return [shp for i in [self.return_values, self.return_indices] if i]

def topk(x, k, axis=-1):
    """
    Returns the k-largest elements along an axis.

    Parameters
    ----------

    x: tensor instance

    k: integer constant/variable
        Must not be 0. If negative, gives k-smallest elements instead.

    axis: integer or ``None``
        Upon which axis shall the operation be performed on. If ``None``,
        works on flattened array.

    Notes
    -----
    - The returned values may not be sorted.

    """
    if axis is None:
        x = theano.tensor.flatten(x)
        axis = -1
    return TopKOp(axis=axis)(x, k)


def argtopk(x, k, axis=-1, idx_dtype='int64'):
    """
    Returns the indices of k-largest elements along an axis.

    Parameters
    ----------

    x: tensor instance

    k: integer constant/variable
        Must not be 0. If negative, gives k-smallest elements instead.

    axis: integer or ``None``
        Upon which axis shall the operation be performed on. If ``None``,
        works on flattened array.

    idx_dtype: string
        Specify output dtype, defaults to ``int64``, must be integer type.

    Notes
    -----
    - The corresponding values of returned indices may not be sorted.

    """
    if axis is None:
        x = theano.tensor.flatten(x)
        axis = -1
    return TopKOp(axis=axis, return_indices=True, return_values=False)(x, k, idx_dtype=idx_dtype)


def topk_and_argtopk(x, k, axis=-1, idx_dtype='int64'):
    '''
    Returns the results of both topk() and argtopk() in one Op.

    See the respective documentation for details.

    '''
    if axis is None:
        x = theano.tensor.flatten(x)
        axis = -1
    return TopKOp(axis=axis, return_indices=True)(x, k, idx_dtype=idx_dtype)

