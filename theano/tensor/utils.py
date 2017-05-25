from __future__ import absolute_import, print_function, division
from six import integer_types
import numpy as np

import theano
from theano import scalar
from theano.compat import izip
from theano.tensor import as_tensor_variable
from theano.tensor.var import TensorConstant
from theano.gof import Variable
from theano.gof.utils import hash_from_code
from theano.tensor.type_other import NoneConst

integer_dtypes = list(map(str, scalar.integer_types))


def hash_from_ndarray(data):
    """
    Return a hash from an ndarray.

    It takes care of the data, shapes, strides and dtype.

    """
    # We need to hash the shapes and strides as hash_from_code only hashes
    # the data buffer. Otherwise, this will cause problem with shapes like:
    # (1, 0) and (2, 0) and problem with inplace transpose.
    # We also need to add the dtype to make the distinction between
    # uint32 and int32 of zeros with the same shape and strides.

    # python hash are not strong, so I always use md5 in order not to have a
    # too long hash, I call it again on the concatenation of all parts.
    if not data.flags["C_CONTIGUOUS"]:
        # hash_from_code needs a C-contiguous array.
        data = np.ascontiguousarray(data)
    return hash_from_code(hash_from_code(data) +
                          hash_from_code(str(data.shape)) +
                          hash_from_code(str(data.strides)) +
                          hash_from_code(str(data.dtype)))


def shape_of_variables(fgraph, input_shapes):
    """
    Compute the numeric shape of all intermediate variables given input shapes.

    Parameters
    ----------
    fgraph
        The theano.FunctionGraph in question.
    input_shapes : dict
        A dict mapping input to shape.

    Returns
    -------
    shapes : dict
        A dict mapping variable to shape

    .. warning:: This modifies the fgraph. Not pure.

    Examples
    --------
    >>> import theano
    >>> x = theano.tensor.matrix('x')
    >>> y = x[512:]; y.name = 'y'
    >>> fgraph = theano.FunctionGraph([x], [y], clone=False)
    >>> d = shape_of_variables(fgraph, {x: (1024, 1024)})
    >>> d[y]
    (array(512), array(1024))
    >>> d[x]
    (array(1024), array(1024))
    """

    if not hasattr(fgraph, 'shape_feature'):
        fgraph.attach_feature(theano.tensor.opt.ShapeFeature())

    input_dims = [dimension for inp in fgraph.inputs
                  for dimension in fgraph.shape_feature.shape_of[inp]]

    output_dims = [dimension for shape in fgraph.shape_feature.shape_of.values()
                   for dimension in shape]

    compute_shapes = theano.function(input_dims, output_dims)

    if any([i not in fgraph.inputs for i in input_shapes.keys()]):
        raise ValueError(
            "input_shapes keys aren't in the fgraph.inputs. FunctionGraph()"
            " interface changed. Now by default, it clones the graph it receives."
            " To have the old behavior, give it this new parameter `clone=False`.")

    numeric_input_dims = [dim for inp in fgraph.inputs
                          for dim in input_shapes[inp]]
    numeric_output_dims = compute_shapes(*numeric_input_dims)

    sym_to_num_dict = dict(izip(output_dims, numeric_output_dims))

    l = {}
    for var in fgraph.shape_feature.shape_of:
        l[var] = tuple(sym_to_num_dict[sym]
                       for sym in fgraph.shape_feature.shape_of[var])
    return l


def check_and_normalize_axes(x, axis):
    """
    Check axes, normalize and convert them to a Python list of integers.
    Return an empty list if argument is None.

    Parameters
    ----------
    x: Tensor variable
    axis = Integer, tuple or list of integers

    Returns
    -------
    axis: list of integers
    """
    x = as_tensor_variable(x)
    if axis is None:
        axis = []
    elif (isinstance(axis, (integer_types, np.integer)) or
            (isinstance(axis, np.ndarray) and axis.ndim == 0)):
        axis = [int(axis)]
    elif isinstance(axis, (tuple, list, np.ndarray)):
        axis = [int(i) for i in axis]
    elif isinstance(axis, Variable):
        if NoneConst.equals(axis):
            axis = []
        elif not isinstance(axis, TensorConstant):
            raise TypeError("Computation needs a constant axis. Got %s" % axis)
        else:
            assert axis.dtype in integer_dtypes
            if (isinstance(axis.data, (integer_types, np.integer)) or
                    (isinstance(axis.data, np.ndarray) and axis.data.ndim == 0)):
                axis = [int(axis.data)]
            elif isinstance(axis.data, (list, np.ndarray)):
                axis = [int(i) for i in axis.data]
    if len(axis) > 0:
        for i in range(len(axis)):
            if axis[i] < 0:
                axis[i] += x.type.ndim
            if axis[i] < 0 or axis[i] >= x.type.ndim:
                raise ValueError("Computation needs a valid axis number for %d-D tensor. Got %d" % (x.type.ndim, axis[i]))
        axis = list(set(axis))
        axis.sort()
    return axis
