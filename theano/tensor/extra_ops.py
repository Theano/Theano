import numpy as np
import numpy

import theano
import basic
from theano import gof, tensor, function, scalar
from theano.sandbox.linalg.ops import diag


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
        # numpy return a view in that case.
        # TODO, make an optimization that remove this op in this case.
        if n == 0:
            self.view_map = {0: [0]}

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
        self.view_map = {0: [0]}
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


class RepeatOp(theano.Op):
    """Repeat elements of an array.

    It returns an array which has the same shape as x, except
    along the given axis. The axis is used to speficy along which
    axis to repeat values. By default, use the flattened input
    array, and return a flat output array.

    The number of repetitions for each element is repeat.
    repeats is broadcasted to fit the shape of the given axis.

    Parameter:
    x -- Input data, tensor variable.
    repeats -- int, tensor variable.

    Keywords arguments:
    axis -- int, optional.

    """

    def __init__(self, axis=None):
        self.axis = axis

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.axis == other.axis)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.axis)

    def make_node(self, x, repeats):
        x = basic.as_tensor_variable(x)
        repeats = basic.as_tensor_variable(repeats)
        if self.axis == None:
            out_type = theano.tensor.TensorType(dtype=x.dtype,
                                                broadcastable=[False])
        else:
            out_type = x.type
        return theano.Apply(self, [x, repeats], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        repeats = inputs[1]
        z = output_storage[0]
        z[0] = np.repeat(x, repeats=repeats, axis=self.axis)

    def grad(self, inputs, outputs_gradients):
        repeats = inputs[1]
        out = outputs_gradients[0]
        if inputs[0].ndim != 1:
            raise NotImplementedError()
        if repeats.ndim != 0:
            raise NotImplementedError()
        return [out.reshape([inputs[0].shape[0], repeats]).sum(axis=1), None]

    def infer_shape(self, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        repeats = node.inputs[1]
        out_shape = list(i0_shapes)

        if self.axis == None:
            res = 0
            for d in i0_shapes:
                res = res + d
            out_shape = (res * repeats, )
        else:
            if repeats.ndim == 0:
                out_shape[self.axis] = out_shape[self.axis] * repeats
            else:
                out_shape[self.axis] = theano.tensor.sum(repeats)
        return [out_shape]

    def __str__(self):
        return self.__class__.__name__


def repeat(x, repeats, axis=None):
    """Repeat elements of an array.

    It returns an array which has the same shape as x, except
    along the given axis. The axis is used to speficy along which
    axis to repeat values. By default, use the flattened input
    array, and return a flat output array.

    The number of repetitions for each element is repeat.
    repeats is broadcasted to fit the shape of the given axis.

    Parameter:
    x -- Input data, tensor variable.
    repeats -- int, tensor variable.

    Keywords arguments:
    axis -- int, optional.

    """
    return RepeatOp(axis=axis)(x, repeats)


class Bartlett(gof.Op):
    """
    An instance of this class returns the Bartlett spectral window in the
    time-domain. The Bartlett window is very similar to a triangular window,
    except that the end points are at zero. It is often used in signal
    processing for tapering a signal, without generating too much ripple in
    the frequency domain.

    input : (integer scalar) Number of points in the output window. If zero or
    less, an empty vector is returned.

    output : (vector of doubles) The triangular window, with the maximum value
    normalized to one (the value one appears only if the number of samples is
    odd), with the first and last samples equal to zero.
    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, M):
        M = tensor.as_tensor_variable(M)
        if M.ndim != 0:
            raise TypeError('%s only works on scalar input'
                            % self.__class__.__name__)
        elif (not M.dtype.startswith('int')) and \
              (not M.dtype.startswith('uint')):
        # dtype is a theano attribute here
            raise TypeError('%s only works on integer input'
                            % self.__class__.__name__)
        return gof.Apply(self, [M], [tensor.dvector()])

    def perform(self, node, inputs, out_):
        M = inputs[0]
        out, = out_
        out[0] = numpy.bartlett(M)

    def infer_shape(self, node, in_shapes):
        temp = node.inputs[0]
        M = tensor.switch(tensor.lt(temp, 0),
            tensor.cast(0, temp.dtype), temp)
        return [[M]]

    def grad(self, inputs, output_grads):
        return [None for i in inputs]


bartlett = Bartlett()


class FillDiagonal(gof.Op):
    """
    An instance of this class returns a copy of an array with all elements of
    the main diagonal set to a specified scalar value.

    inputs:

    a : Rectangular array of at least two dimensions.
    val : Scalar value to fill the diagonal whose type must be compatible with
    that of array 'a' (i.e. 'val' cannot be viewed as an upcast of 'a').

    output:

    An array identical to 'a' except that its main diagonal is filled with
    scalar 'val'. (For an array 'a' with a.ndim >= 2, the main diagonal is the
    list of locations a[i, i, ..., i] (i.e. with indices all identical).)

    Support rectangular matrix and tensor with more then 2 dimensions
    if the later have all dimensions are equals.

    """

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash_(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

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
            raise TypeError('%s: type of second parameter must be compatible'
                          ' with first\'s' % self.__class__.__name__)
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
            numpy.fill_diagonal(a, val)

        output_storage[0][0] = a

    def grad(self, inp, cost_grad):
        """
        Note: The gradient is currently implemented for matrices
        only.
        """
        a, val = inp
        grad = cost_grad[0]
        if (a.dtype.startswith('complex')):
            return [None, None]
        elif a.ndim > 2:
            raise NotImplementedError('%s: gradient is currently implemented'
                            ' for matrices only' % self.__class__.__name__)
        wr_a = fill_diagonal(grad, 0)  # valid for any number of dimensions
        wr_val = diag(grad).sum()  # diag is only valid for matrices
        return [wr_a, wr_val]


fill_diagonal = FillDiagonal()
