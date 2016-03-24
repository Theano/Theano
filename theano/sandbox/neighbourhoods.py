"""
.. warning:: This code is not recommanded. It is not finished, it is
slower than the version in sandbox/neighbours.py, and it does not work
on the GPU.

We only keep this version here as it is a little bit more generic, so
it cover more cases. But thoses cases aren't needed frequently, so you
probably don't want to use this version, go see neighbours.py!!!!!!!

"""
from __future__ import absolute_import, print_function, division
import numpy
from six.moves import xrange
import six.moves.builtins as builtins

import theano
from theano import gof, Op


class NeighbourhoodsFromImages(Op):
    """
    This extracts neighbourhoods from "images", but in a dimension-generic
    manner.

    In the 2D case, this is similar to downsampling, but instead of reducing
    a group of 2x2 pixels (for example) to a single new pixel in the output,
    you place those 4 pixels in a row.

    For example, say you have this 2x4 image::

            [ [ 0.5, 0.6, 0.7, 0.8 ],
              [ 0.1, 0.2, 0.3, 0.4 ] ]

    and you want to extract 2x2 neighbourhoods. This op would then produce::

            [ [ [ 0.5, 0.6, 0.1, 0.2 ] ], # the first 2x2 group of pixels
              [ [ 0.7, 0.8, 0.3, 0.4 ] ] ] # the second one

    So think of a 2D downsampling where each pixel of the resulting array
    is replaced by an array containing the (flattened) pixels of the
    corresponding neighbourhood.

    If you provide a stack of 2D images, or multiple stacks, each image
    will be treated independently, and the first dimensions of the array
    will be preserved as such.

    This also makes sense in the 1D or 3D case. Below I'll still be calling
    those "images", by analogy.

    In the 1D case, you're extracting subsequences from the original sequence.
    In the 3D case, you're extracting cuboids.
    If you ever find a 4D use, tell me! It should be possible, anyhow.

    Parameters
    ----------
    n_dims_before : int
        Number of dimensions preceding the "images".
    dims_neighbourhoods : tuple of ints
        Exact shape of windows to be extracted (e.g. (2,2) in the case above).
        n_dims_before + len(dims_neighbourhoods) should be equal to the
        number of dimensions in the input given to the op.
    strides : tuple of int
        Number of elements to skip when moving to the next neighbourhood,
        for each dimension of dims_neighbourhoods. There can be overlap
        between neighbourhoods, or gaps.
    ignore_border : bool
        If the dimensions of the neighbourhoods don't exactly divide the
        dimensions of the "images", you can either fill the last
        neighbourhood with zeros (False) or drop it entirely (True).
    inverse : bool
        You shouldn't have to use this. Only used by child class
        ImagesFromNeighbourhoods which simply reverses the assignment.

    """

    __props__ = ("n_dims_before", "dims_neighbourhoods", "strides",
                 "ignore_border", "inverse")

    def __init__(self, n_dims_before, dims_neighbourhoods,
                 strides=None, ignore_border=False, inverse=False):

        self.n_dims_before = n_dims_before
        self.dims_neighbourhoods = dims_neighbourhoods
        if strides is not None:
            self.strides = strides
        else:
            self.strides = dims_neighbourhoods
        self.ignore_border = ignore_border

        self.inverse = inverse

        self.code_string, self.code = self.make_py_code()

    def __str__(self):
        return '%s{%s,%s,%s,%s}' % (self.__class__.__name__,
                                    self.n_dims_before,
                                    self.dims_neighbourhoods,
                                    self.strides,
                                    self.ignore_border)

    def out_shape(self, input_shape):
        dims = list(input_shape[:self.n_dims_before])
        num_strides = [0 for i in xrange(len(self.strides))]
        neigh_flattened_dim = 1
        for i, ds in enumerate(self.dims_neighbourhoods):
            cur_stride = self.strides[i]
            input_dim = input_shape[i + self.n_dims_before]
            target_dim = input_dim // cur_stride
            if not self.ignore_border and (input_dim % cur_stride) != 0:
                target_dim += 1
            num_strides[i] = target_dim
            dims.append(target_dim)
            neigh_flattened_dim *= ds

        dims.append(neigh_flattened_dim)

        return dims, num_strides

    # for inverse mode
    # "output" here actually referes to the Op's input shape (but it's inverse
    # mode)
    def in_shape(self, output_shape):
        out_dims = list(output_shape[:self.n_dims_before])
        num_strides = []

        # in the inverse case we don't worry about borders:
        # they either have been filled with zeros, or have been cropped
        for i, ds in enumerate(self.dims_neighbourhoods):
            # the number of strides performed by NeighFromImg is
            # directly given by this shape
            num_strides.append(output_shape[self.n_dims_before + i])

            # our Op's output image must be at least this wide
            at_least_width = num_strides[i] * self.strides[i]

            # ... which gives us this number of neighbourhoods
            num_neigh = at_least_width // ds
            if at_least_width % ds != 0:
                num_neigh += 1

            # making the final Op's output dimension this wide
            out_dims.append(num_neigh * ds)

        return out_dims, num_strides

    def make_node(self, x):
        x = theano.tensor.as_tensor_variable(x)
        if self.inverse:
            # +1 in the inverse case
            if x.type.ndim != (self.n_dims_before +
                               len(self.dims_neighbourhoods) + 1):
                raise TypeError()
        else:
            if x.type.ndim != (self.n_dims_before +
                               len(self.dims_neighbourhoods)):
                raise TypeError()
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, inp, out):
        x, = inp
        z, = out
        if self.inverse:
            # +1 in the inverse case
            if len(x.shape) != (self.n_dims_before +
                                len(self.dims_neighbourhoods) + 1):
                raise ValueError("Images passed as input don't match the "
                                 "dimensions passed when this (inversed) "
                                 "Apply node was created")
            prod = 1
            for dim in self.dims_neighbourhoods:
                prod *= dim
            if x.shape[-1] != prod:
                raise ValueError(
                    "Last dimension of neighbourhoods (%s) is not"
                    " the product of the neighbourhoods dimensions"
                    " (%s)" % (str(x.shape[-1]), str(prod)))
        else:
            if len(x.shape) != (self.n_dims_before +
                                len(self.dims_neighbourhoods)):
                raise ValueError("Images passed as input don't match the "
                                 "dimensions passed when this Apply node "
                                 "was created")

        if self.inverse:
            input_shape, num_strides = self.in_shape(x.shape)
            out_shape, dummy = self.out_shape(input_shape)
        else:
            input_shape = x.shape
            out_shape, num_strides = self.out_shape(input_shape)

        if z[0] is None:
            if self.inverse:
                z[0] = numpy.zeros(input_shape)
            else:
                z[0] = numpy.zeros(out_shape)
            z[0] = theano._asarray(z[0], dtype=x.dtype)

        exec(self.code)

    def make_py_code(self):
        # TODO : need description for method and return
        code = self._py_outerloops()
        for i in xrange(len(self.strides)):
            code += self._py_innerloop(i)
        code += self._py_assignment()
        return code, builtins.compile(code, '<string>', 'exec')

    def _py_outerloops(self):
        # TODO : need description for method, parameter and return
        code_before = ""
        for dim_idx in xrange(self.n_dims_before):
            code_before += ('\t' * (dim_idx)) + \
                "for outer_idx_%d in xrange(input_shape[%d]):\n" % \
                (dim_idx, dim_idx)
        return code_before

    def _py_innerloop(self, inner_dim_no):
        # TODO : need description for method, parameter and return
        base_indent = ('\t' * (self.n_dims_before + inner_dim_no * 2))
        code_before = base_indent + \
            "for stride_idx_%d in xrange(num_strides[%d]):\n" % \
            (inner_dim_no, inner_dim_no)
        base_indent += '\t'
        code_before += base_indent + \
            "dim_%d_offset = stride_idx_%d * self.strides[%d]\n" %\
            (inner_dim_no, inner_dim_no, inner_dim_no)
        code_before += base_indent + \
            "max_neigh_idx_%d = input_shape[%d] - dim_%d_offset\n" % \
            (inner_dim_no, self.n_dims_before + inner_dim_no, inner_dim_no)
        code_before += base_indent + \
            ("for neigh_idx_%d in xrange(min(max_neigh_idx_%d,"
             " self.dims_neighbourhoods[%d])):\n") %\
            (inner_dim_no, inner_dim_no, inner_dim_no)

        return code_before

    def _py_flattened_idx(self):
        # TODO : need description for method and return
        return "+".join(["neigh_strides[%d]*neigh_idx_%d" % (i, i)
                        for i in xrange(len(self.strides))])

    def _py_assignment(self):
        # TODO : need description for method and return
        input_idx = "".join(["outer_idx_%d," % (i,)
                            for i in xrange(self.n_dims_before)])
        input_idx += "".join(["dim_%d_offset+neigh_idx_%d," %
                             (i, i) for i in xrange(len(self.strides))])
        out_idx = "".join(
            ["outer_idx_%d," % (i,) for i in xrange(self.n_dims_before)] +
            ["stride_idx_%d," % (i,) for i in xrange(len(self.strides))])
        out_idx += self._py_flattened_idx()

        # return_val = '\t' * (self.n_dims_before + len(self.strides)*2)
        # return_val += "print "+input_idx+"'\\n',"+out_idx+"\n"

        return_val = '\t' * (self.n_dims_before + len(self.strides) * 2)

        if self.inverse:
            # remember z and x are inversed:
            # z is the Op's output, but has input_shape
            # x is the Op's input, but has out_shape
            return_val += "z[0][%s] = x[%s]\n" % (input_idx, out_idx)
        else:
            return_val += "z[0][%s] = x[%s]\n" % (out_idx, input_idx)

        return return_val


class ImagesFromNeighbourhoods(NeighbourhoodsFromImages):
    # TODO : need description for class, parameters
    def __init__(self, n_dims_before, dims_neighbourhoods,
                 strides=None, ignore_border=False):
        NeighbourhoodsFromImages.__init__(self, n_dims_before,
                                          dims_neighbourhoods,
                                          strides=strides,
                                          ignore_border=ignore_border,
                                          inverse=True)
        # and that's all there is to it
