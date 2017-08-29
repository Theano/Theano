from __future__ import absolute_import, print_function, division
import numpy as np

import theano
from theano.compat import izip
from theano.gof.utils import hash_from_code


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

    # python hash are not strong, so use sha256 (md5 is not
    # FIPS compatible). To not have too long of hash, I call it again on
    # the concatenation of all parts.
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
