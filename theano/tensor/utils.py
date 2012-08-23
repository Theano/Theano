import numpy

import theano
from theano.gof.cc import hash_from_code


def hash_from_ndarray(data):
    """Return a hash from an ndarray

    It takes care of the data, shapes, strides and dtype.

    """
    # We need to hash the shapes and strides as hash_from_code only hashes
    # the data buffer. Otherwise, this will cause problem with shapes like:
    # (1, 0) and (2, 0) and problem with inplace transpose.
    # We also need to add the dtype to make the distinction between
    # uint32 and int32 of zeros with the same shape and strides.

    # python hash are not strong, so I always use md5 in order not to have a
    # too long hash, I call it again on the concatenation of all parts.
    if not data.flags["C_CONTIGUOUS"] and not data.flags["F_CONTIGUOUS"]:
        data = numpy.ascontiguousarray(data)
    return hash_from_code(hash_from_code(data) +
                          hash_from_code(str(data.shape)) +
                          hash_from_code(str(data.strides)) +
                          hash_from_code(str(data.dtype)))


def hash_from_dict(d):
    """Work around the fact that dict are not hashable in python

    This request that all object have a sorted order that depend only
    on the value of the object. This is true for integer/float/string

    We do not verify that the objects in the dict have this property.

    Also, we transform values that are list into tuple as list are not
    hashable.

    """
    items = d.items()
    items.sort()
    first_part = [k for k, v in items]
    second_part = []
    for k, v in items:
        if isinstance(v, (tuple, list)):
            second_part += [tuple(v)]
        else:
            second_part += [v]
    tuple_items = tuple(first_part + second_part)
    return hash(tuple_items)

def shape_of_variables(fgraph, input_shapes):
    """
    Compute the numeric shape of all intermediate variables given input shapes

    Inputs:
        fgraph - the theano.FunctionGraph in question
        input_shapes - a dict mapping input to shape

    Outputs:
        shapes - a dict mapping variable to shape

    WARNING : This modifies the fgraph. Not pure.

    >>> import theano
    >>> x = theano.tensor.matrix('x')
    >>> y = x[512:]; y.name = 'y'
    >>> fgraph = theano.FunctionGraph([x], [y])
    >>> shape_of_variables(fgraph, {x: (1024, 1024)})
    {y: (512, 1024), x: (1024, 1024)}
    """

    if not hasattr(fgraph, 'shape_feature'):
        fgraph.attach_feature(theano.tensor.opt.ShapeFeature())

    input_dims  = [dimension for inp in fgraph.inputs
                             for dimension in fgraph.shape_feature.shape_of[inp]]

    output_dims = [dimension for shape in fgraph.shape_feature.shape_of.values()
                             for dimension in shape]

    compute_shapes = theano.function(input_dims, output_dims)

    numeric_input_dims  = [dim for inp in fgraph.inputs
                               for dim in input_shapes[inp]]
    numeric_output_dims = compute_shapes(*numeric_input_dims)

    sym_to_num_dict = dict(zip(output_dims, numeric_output_dims))

    l = {}
    for var in fgraph.shape_feature.shape_of:
        l[var] = tuple(sym_to_num_dict[sym]
                       for sym in fgraph.shape_feature.shape_of[var])
    return l
