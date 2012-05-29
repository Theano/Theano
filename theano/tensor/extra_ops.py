import theano
import numpy as np
import basic


class DiffOp(theano.Op):
    """Calculate the n-th order discrete difference along given axis.

    The first order difference is given by out[n] = a[n+1] - a[n]
    along the given axis, higher order differences are calculated by
    using diff recursively. Wraping of numpy.diff.

    Parameter:
    x -- Input vector.

    Keywords arguments:
    n -- The number of times values are differenced, default is 1.

    """

    def __init__(self, n=1, axis=-1):
        self.n = n
        self.axis = axis

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.n == other.n and
                self.axis == other.axis)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n) ^ hash(self.axis)

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

    def __str__(self):
        return self.__class__.__name__


def diff(x, n=1, axis=-1):
    """Calculate the n-th order discrete difference along given axis.

    The first order difference is given by out[n] = a[n+1] - a[n]
    along the given axis, higher order differences are calculated by
    using diff recursively. Wraping of numpy.diff.

    Parameter:
    x -- Input vector.

    Keywords arguments:
    n -- The number of times values are differenced, default is 1.

    """
    return DiffOp(n=n, axis=axis)(x)


class BinCountOp(theano.Op):
    """Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest
    value in x. If minlength is specified, there will be at least
    this number of bins in the output array (though it will be longer
    if necessary, depending on the contents of x). Each bin gives the
    number of occurrences of its index value in x. If weights is
    specified the input array is weighted by it, i.e. if a value n
    is found at position i, out[n] += weight[i] instead of out[n] += 1.
    Wraping of numpy.bincount

    Parameter:
    x -- 1 dimension, nonnegative ints

    Keywords arguments:
    weights -- Weights, array of the same shape as x.
    minlength -- A minimum number of bins for the output array.

    """

    compatible_type = ('int8', 'int16', 'int32', 'int64',
                       'uint8', 'uint16', 'uint32', 'uint64')
    """Tuple of all compatible dtype for the parameter of this op."""

    def __init__(self, minlength=None):
        self.minlength = minlength

    def __eq__(self, other):
        return (type(self) == type(other) and
               self.minlength == other.minlength)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.minlength)

    def make_node(self, x, weights):
        x = basic.as_tensor_variable(x)

        if x.dtype not in BinCountOp.compatible_type:
            raise TypeError("Inputs dtype must be an integer.")
        if x.ndim != 1:
            raise TypeError("Inputs must be of dimension 1.")

        if weights is None:
            weights = theano.gof.Constant(theano.gof.Generic(), None)
            out_type = x.type()
        else:
            weights = basic.as_tensor_variable(weights)
            out_type = weights.type()
            if weights.ndim != 1:
                raise TypeError("Weights cannot have a number of"
                                "dimension different of 1.")

        return theano.Apply(self, [x, weights], [out_type])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        weights = inputs[1]
        z = output_storage[0]

        if weights is not None and weights.shape != x.shape:
            raise TypeError("All inputs must have the same shape.")

        z[0] = np.bincount(x, weights=weights, minlength=self.minlength)

    def grad(self, inputs, outputs_gradients):
        return [None for i in inputs]

    def infer_shape(self, node, ins_shapes):
        x = node.inputs[0]
        m = basic.max(x) + 1
        if self.minlength != None:
            m = basic.maximum(m, self.minlength)
        return [[m]]

    def __str__(self):
        return self.__class__.__name__


def bincount(x, weights=None, minlength=None):
    """Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest
    value in x. If minlength is specified, there will be at least
    this number of bins in the output array (though it will be longer
    if necessary, depending on the contents of x). Each bin gives the
    number of occurrences of its index value in x. If weights is
    specified the input array is weighted by it, i.e. if a value n
    is found at position i, out[n] += weight[i] instead of out[n] += 1.
    Wraping of numpy.bincount

    Parameter:
    x -- 1 dimension, nonnegative ints

    Keywords arguments:
    weights -- Weights, array of the same shape as x.
    minlength -- A minimum number of bins for the output array.

    """
    return BinCountOp(minlength=minlength)(x, weights)


class SqueezeOp(theano.Op):
    """Remove single-dimensional entries from the shape of an array.

    It returns the input array, but with with all or a subset of the
    dimensions of length 1 removed. This is always x itself or a view
    into x. Wraping of numpy.squeeze.

    Parameter:
    x -- Input data, tensor variable.
    out_nd -- Output number of dimension for this op.

    """

    def __init__(self, out_nd):
        self.out_nd = out_nd

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.out_nd == other.out_nd)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.out_nd)

    def make_node(self, x):
        x = basic.as_tensor_variable(x)
        out_type = theano.tensor.TensorType(dtype=x.dtype,
                              broadcastable=[False] * self.out_nd)
        return theano.Apply(self, [x], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        z = output_storage[0]
        squeezed = np.squeeze(x)
        if squeezed.ndim != self.out_nd:
            raise TypeError("The number of dimension specified "
                            "is different from the one calculated.")
        z[0] = squeezed

    def grad(self, inputs, outputs_gradients):
        out = outputs_gradients[0]
        return [out.reshape(inputs[0].shape)]

    def __str__(self):
        return self.__class__.__name__


def squeeze(x, out_nd):
    """Remove single-dimensional entries from the shape of an array.

    It returns the input array, but with with all or a subset of the
    dimensions of length 1 removed. This is always x itself or a view
    into x. Wraping of numpy.squeeze.

    Parameter:
    x -- Input data, tensor variable.
    out_nd -- Output number of dimension for this op.

    """
    return SqueezeOp(out_nd=out_nd)(x)
