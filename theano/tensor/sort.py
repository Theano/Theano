from __future__ import absolute_import, print_function, division
import numpy as np
import theano
from theano.tensor.basic import mul, arange
from theano.gradient import grad_undefined
from theano.tensor.subtensor import set_subtensor


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
        axis_grad = grad_undefined(
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
        list of length len(a.shape) otherwise

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
        axis_grad = grad_undefined(
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


def _topk_py_impl(op, x, k, axis, idx_dtype):
    ndim = x.ndim
    assert -ndim <= axis < ndim
    axis %= ndim
    if k == 0:
        raise ValueError('topk: kth cannot be zero')
    elif k > x.shape[axis]:
        raise ValueError(
            'topk: kth cannot be larger than the size of specified axis %d' % axis)
    if abs(k) == 1:
        # negative k means min instead of max
        fn_max = [None, np.max, np.min][k]
        fn_argmax = [None, np.argmax, np.argmin][k]
        if not op.return_indices:
            return np.expand_dims(fn_max(x, axis=axis), axis)
        elif op.return_values:
            zi = np.expand_dims(
                fn_argmax(x, axis=axis), axis)
            idx2 = tuple(
                np.arange(s).reshape(
                    (s,) + (1,) * (ndim - i - 1)
                    ) if i != axis else zi for i, s in enumerate(x.shape))
            zv = x[idx2]
            return zv, zi.astype(idx_dtype)
        else:
            zi = np.expand_dims(
                fn_argmax(x, axis=axis), axis)
            return zi.astype(idx_dtype)

    if x.shape[axis] == abs(k):
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
    idx[axis] = (slice(-k, None) if k > 0 else slice(-k))

    if not op.return_indices:
        zv = np.partition(x, -k, axis=axis)[idx]
        return zv
    elif op.return_values:
        zi = np.argpartition(x, -k, axis=axis)[idx]
        idx2 = tuple(
            np.arange(s).reshape(
                (s,) + (1,) * (ndim - i - 1)
                ) if i != axis else zi for i, s in enumerate(x.shape))
        zv = x[idx2]
        return zv, zi.astype(idx_dtype)
    else:
        zi = np.argpartition(x, -k, axis=axis)[idx]
        return zi.astype(idx_dtype)


class TopKOp(theano.Op):
    """Operations related to finding k-largest elements.

    Parameters
    ----------
    axis: integer
        Defaults to ``-1``.
        The axis to perform the operation. Must be in range ``[-ndim, ndim)``, where
        ``ndim`` is the dimensionality of input tensor.

    idx_dtype: string
        Specify output dtype for indices, defaults to ``int64``, must be integer type.

    sorted: bool
        NOTE: NOT IMPLEMENTED YET
        Defaults to ``True``

        If True, the result array would be sorted in descending order.


    Notes
    -----
    - CPU and GPU ops don't produce same output order. This is expected.
    - The output order is not guaranteed. On the CPU, we use
      ``np.partition`` and ``np.argpartition`` that only make sure the
      k-th element is the correct one and that the other
      elements are on the correct side. On the GPU, they
      look sorted, but we do not test the correctness of this behavior.
    - By default, this Op gives two outputs: values and indices. However
      optimizers may remove a certain output if not needed.
    - Computing the gradient requests the computation of the indices in
      forward pass.
    - If the top-k-th value is not unique, we cannot guarantee the
      output indices being deterministically chosen.

    See Also
    --------
    topk
    argtopk
    argtopk_and_topk

    """

    # TODO more params
    '''
    only_top_kth: bool
        Defaults to ``False``

        If ``True``, will only find one exact top k-th element on given axis.

    '''

    # TODO c_code
    # TODO add opt, if k==1, use max/min reduce
    #      also if k is axis size, just copy input tensor
    # TODO add opt, to merge argtopk / topk
    __props__ = ('axis', 'sorted', 'return_values', 'return_indices', 'idx_dtype')

    def __init__(
            self,
            axis=-1,
            sorted=True,
            idx_dtype='int64',
            return_values=True,
            return_indices=True
            ):
        # numpy always uses int64 as output dtype for arg*() routines
        # however, we add "idx_dtype" param as memory is more precious on gpu
        if not isinstance(axis, int):
            raise TypeError(
                '"axis" parameter must be integer, got "%s"' % type(axis))
        if sorted:
            raise NotImplementedError(
                "The sorted parameter is not yet implemented. Use sorted=False for now.")
        if idx_dtype not in theano.tensor.integer_dtypes:
            raise TypeError(
                '"idx_dtype" parameter must be an integer dtype, got "%s"' % idx_dtype)

        if not (return_indices or return_values):
            raise ValueError(
                "Neither return_values nor return_indices is True, this isn't allowed")

        self.axis = axis
        self.sorted = sorted
        self.return_values = return_values
        self.return_indices = return_indices
        self.idx_dtype = idx_dtype

    def __str__(self):
        return '%(op)s{axis=%(axis)d, sorted=%(sorted)s}' % dict(
            op=self.__class__.__name__,
            axis=self.axis,
            sorted=self.sorted)

    def make_node(self, inp, kth):
        inp = theano.tensor.as_tensor_variable(inp)
        ndim = inp.ndim
        if ndim == 0:
            raise ValueError('Cannot take scalar as input')
        if not -ndim <= self.axis < ndim:
            raise IndexError(
                '"axis" parameter out of range,'
                ' expected integer within [%d, %d]' % (-ndim, ndim - 1))

        kth = theano.tensor.as_tensor_variable(kth)
        _check_tensor_is_scalar(kth)
        bcast = inp.type.broadcastable
        outs = []
        if self.return_values:
            outs.append(inp.type())
        if self.return_indices:
            outs.append(theano.tensor.TensorType(
                dtype=self.idx_dtype, broadcastable=bcast)())
        return theano.Apply(self, [inp, kth], outs)

    def perform(self, node, inputs, output_storage):
        x, k = inputs
        axis = self.axis
        if not self.return_indices:
            pzv = output_storage[0]
            pzv[0] = _topk_py_impl(self, x, k, axis, None)
        elif self.return_values:
            pzv = output_storage[0]
            pzi = output_storage[1]
            pzv[0], pzi[0] = _topk_py_impl(
                self, x, k, axis, node.outputs[1].dtype)
        else:
            pzi = output_storage[0]
            pzi[0] = _topk_py_impl(self, x, k, axis, node.outputs[0].dtype)

    def infer_shape(self, node, inp_shapes):
        shp = list(inp_shapes[0])
        shp[self.axis] = np.abs(node.inputs[1])
        shp = tuple(shp)
        return [shp for i in [self.return_values, self.return_indices] if i]

    def L_op(self, inputs, outputs, out_grads):
        x, k = inputs
        k_grad = grad_undefined(self, 1, k, 'topk: k is not differentiable')

        if not (self.return_indices or self.return_values):
            x_grad = grad_undefined(
                self, 0, x, 'topk: cannot get gradient'
                ' without both indices and values')
        else:
            x_shp = theano.tensor.shape(x)
            z_grad = out_grads[0]
            ndim = x.ndim
            axis = self.axis % ndim
            grad_indices = [
                arange(x_shp[i]).dimshuffle([0] + ['x'] * (ndim - i - 1))
                if i != axis else outputs[-1] for i in range(ndim)]
            x_grad = x.zeros_like(dtype=z_grad.dtype)
            x_grad = set_subtensor(x_grad[tuple(grad_indices)], z_grad)

        return [x_grad, k_grad]


def topk(x, kth, axis=-1, sorted=True, idx_dtype='int64'):
    """
    Returns the k-largest elements along an axis.

    Parameters
    ----------

    x: tensor instance

    kth: integer constant/variable
        Must not be 0. If negative, gives k-smallest elements instead.

    axis: integer or ``None``
        Upon which axis shall the operation be performed on.
        If ``None``, works on flattened array.

    sorted: bool
        NOTE: NOT IMPLEMENTED YET, USE ``False`` FOR NOW.
        Defaults to ``True``

        If True, the result array would be sorted in descending order.

    idx_dtype: string
        Specify output dtype used in indices, defaults to ``int64``, must be integer type.
        This option is here because indices are needed for gradient.

    Returns
    -------
    Tensor variable with same dtype as `x`.

    Notes
    -----
    - ``sorted=True`` is not supported yet.

    """
    if axis is None:
        x = theano.tensor.flatten(x)
        axis = 0
    return TopKOp(
        axis=axis,
        sorted=sorted,
        idx_dtype=idx_dtype)(x, kth)[0]


def argtopk(x, kth, axis=-1, sorted=True, idx_dtype='int64'):
    """
    Returns the indices of k-largest elements along an axis.

    Parameters
    ----------

    x: tensor instance

    kth: integer constant/variable
        Must not be 0. If negative, gives k-smallest elements instead.

    sorted: bool
        NOTE: NOT IMPLEMENTED YET, USE ``False`` FOR NOW.
        Defaults to ``True``

        If True, the result array of corresponding indices would be sorted in descending order.


    axis: integer, tuple/list of integers, or ``None``
        Upon which axis shall the operation be performed on.
        If ``None``, works on flattened array.

    idx_dtype: string
        Specify output dtype, defaults to ``int64``, must be integer type.

    Returns
    -------
    Tensor variable with dtype specified in `idx_dtype`.

    Notes
    -----
    - ``sorted=True`` is not supported yet.

    - If the top-k-th value is not unique, we cannot guarantee the output
      indices are deterministically chosen.

    """
    if axis is None:
        x = theano.tensor.flatten(x)
        axis = 0
    return TopKOp(
        axis=axis,
        sorted=sorted,
        idx_dtype=idx_dtype)(x, kth)[1]


def topk_and_argtopk(x, kth, axis=-1, sorted=True, idx_dtype='int64'):
    """
    Returns the results of both topk() and argtopk() in one Op.

    See the respective documentation for details.

    Returns
    -------
    tuple: (values, indices)

    """
    if axis is None:
        x = theano.tensor.flatten(x)
        axis = 0
    return TopKOp(
        axis=axis,
        sorted=sorted,
        idx_dtype=idx_dtype)(x, kth)
