import numpy as np

import theano
from theano.tensor import tensor

from basic import mul


class SortOp(theano.Op):
    """
    This class is a wrapper for numpy sort function
    """
    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __eq__(self, other):
        return (type(self) == type(other) and self.order == other.order and
                self.kind == other.kind)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.order) ^ hash(self.kind)

    def __str__(self):
        return self.__class__.__name__ + "{%s, %s}" % (self.kind,
                                                       str(self.order))

    def make_node(self, input, axis=-1):
        input = theano.tensor.as_tensor_variable(input)
        if axis is None:
            axis = theano.Constant(theano.gof.generic, None)
            # axis=None flattens the array before sorting
            out_type = tensor(dtype=input.dtype, broadcastable=[False])
        else:
            axis = theano.tensor.as_tensor_variable(axis)
            out_type = input.type()
        return theano.Apply(self, [input, axis], [out_type])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = np.sort(a, axis, self.kind, self.order)

    def infer_shape(self, node, inputs_shapes):
        if (isinstance(node.inputs[1], theano.Constant) and
            node.inputs[1].data is None):
            # That means axis = None,
            # So the array is flattened before being sorted
            return [(mul(*inputs_shapes[0]),)]
        # axis should not be None
        # So there should be the same number of dimensions
        # in the input and output
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    #**** It need the argsort, so we can't do it now.
    #def grad(self, inputs, output_grads):
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
    Return a sorted copy of an array.
    a : Tensor
    Tensor to be sorted

    axis : Tensor
        Axis along which to sort. If None, the array is
        flattened before sorting.

    kind : {'quicksort', 'mergesort', 'heapsort'}, optional

        Sorting algorithm. Default is 'quicksort'.

    order : list, optional

        When `a` is a structured array, this argument specifies which
        fields to compare first, second, and so on. This list does not
        need to include all of the fields.

    """
    return SortOp(kind, order)(a, axis)


class ArgSortOp(theano.Op):
    """
    This class is a wrapper for numpy argsort function
    """
    def __init__(self, kind, order=None):
        self.kind = kind
        self.order = order

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.order == other.order and
                self.kind == other.kind)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.order) ^ hash(self.kind)

    def __str__(self):
        return (self.__class__.__name__
                + "{%s, %s}" % (self.kind, str(self.order)))

    def make_node(self, input, axis=-1):
        input = theano.tensor.as_tensor_variable(input)
        if axis is None:
            axis = theano.Constant(theano.gof.generic, None)
            bcast = [False]
        else:
            axis = theano.tensor.as_tensor_variable(axis)
            bcast = input.type.broadcastable
        return theano.Apply(self, [input, axis],
            [theano.tensor.TensorType(dtype="int64", broadcastable=bcast)()])

    def perform(self, node, inputs, output_storage):
        a = inputs[0]
        axis = inputs[1]
        z = output_storage[0]
        z[0] = theano._asarray(
                np.argsort(a, axis, self.kind, self.order),
                dtype=node.outputs[0].dtype)

    def infer_shape(self, node, inputs_shapes):
        if (isinstance(node.inputs[1], theano.Constant) and
                node.inputs[1].data is None):
            return [(mul(*inputs_shapes[0]),)]
        # axis should not be None, so there should be the same number of
        # dimensions in the input and output
        assert node.inputs[0].ndim == node.outputs[0].ndim
        assert inputs_shapes[1] == ()
        return [inputs_shapes[0]]

    def grad(self, inputs, output_grads):
        #No grad defined for intergers.
        return [None, None]
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
    return ArgSortOp(kind, order)(a, axis)
