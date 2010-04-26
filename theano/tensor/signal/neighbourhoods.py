#!/usr/bin/python

import theano
from theano import gof, Op, tensor, Variable, Apply

import numpy
import __builtin__

class NeighbourhoodsFromImages(Op):
    def __init__(self, n_dims_before, dims_neighbourhoods, strides=None, ignore_border=False):
        """
        """
        self.n_dims_before = n_dims_before
        self.dims_neighbourhoods = dims_neighbourhoods
        self.strides = strides if not strides is None else dims_neighbourhoods
        self.ignore_border = ignore_border

        self.code = self.make_py_code()

    def _compute_neigh_strides(self):
        neigh_strides = [1 for i in range(len(self.strides))]
        cur_stride = 1
        for i in range(len(self.strides)-1, -1, -1):
            neigh_strides[i] = cur_stride
            cur_stride *= self.dims_neighbourhoods[i]
        return neigh_strides

    def __eq__(self, other):
        return type(self) == type(other) and \
                self.n_dims_before == other.n_dims_before and \
                self.dims_neighbourhoods == other.dims_neighbourhoods and \
                self.strides == other.strides and \
                self.ignore_border == other.ignore_border

    def __hash__(self):
        return hash(type(self)) ^ \
                hash(self.n_dims_before) ^ \
                hash(self.dims_neighbourhoods) ^ \
                hash(self.strides) ^ \
                hash(self.ignore_border)

    def __str__(self):
        return '%s{%s,%s,%s,%s}' % \
                (self.__class__.__name__, 
                 self.n_dims_before,
                 self.dims_neighbourhoods,
                 self.strides,
                 self.ignore_border)

    def out_shape(self, input_shape):
        dims = list(input_shape[:self.n_dims_before])
        num_strides = [0 for i in range(len(self.strides))]
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

    def make_node(self, x):
        if x.type.ndim != (self.n_dims_before + \
                len(self.dims_neighbourhoods)):
            raise TypeError()
        return gof.Apply(self, [x], [x.type()])

    def perform(self, node, (x,), (z,)):

        if len(x.shape) != (self.n_dims_before + len(self.dims_neighbourhoods)):
            raise ValueError("Images passed as input don't match the dimensions passed when this Apply node was created")

        out_shape, num_strides = self.out_shape(x.shape)
        neigh_strides = self._compute_neigh_strides()
        input_shape = x.shape

        if z[0] is None:
            z[0] = numpy.zeros(out_shape)
            z[0] = theano._asarray(z[0], dtype=x.dtype)

        exec(self.code)

    def make_py_code(self):
        code = self._py_outerloops()
        for i in xrange(len(self.strides)):
            code += self._py_innerloop(i)
        code += self._py_assignment()
        return __builtin__.compile(code, '<string>', 'exec')

    def _py_outerloops(self):
        code_before = ""
        for dim_idx in xrange(self.n_dims_before):
            code_before += ('\t' * (dim_idx)) + \
                "for outer_idx_%d in xrange(input_shape[%d]):\n" % \
                (dim_idx, dim_idx)
        return code_before

    def _py_innerloop(self, inner_dim_no):
        base_indent = ('\t' * (self.n_dims_before + inner_dim_no*2))
        code_before = base_indent + \
                "for stride_idx_%d in xrange(num_strides[%d]):\n" % \
                    (inner_dim_no,inner_dim_no)
        base_indent += '\t'
        code_before += base_indent + \
                "dim_%d_offset = stride_idx_%d * self.strides[%d]\n" %\
                         (inner_dim_no, inner_dim_no, inner_dim_no)
        code_before += base_indent + \
                "max_neigh_idx_%d = input_shape[%d] - dim_%d_offset\n" %\
                 (inner_dim_no,
                    self.n_dims_before+inner_dim_no, inner_dim_no)
        code_before += base_indent + \
                ("for neigh_idx_%d in xrange(min(max_neigh_idx_%d,"\
                +" self.dims_neighbourhoods[%d])):\n") % \
                    (inner_dim_no, inner_dim_no, inner_dim_no)
        
        return code_before

    def _py_flattened_idx(self):
        return "+".join(["neigh_strides[%d]*neigh_idx_%d" % (i,i) \
                    for i in range(len(self.strides))])

    def _py_assignment(self):
        input_idx = "".join(["outer_idx_%d," % (i,) \
                    for i in xrange(self.n_dims_before)])
        input_idx += "".join(["dim_%d_offset+neigh_idx_%d," % \
                    (i,i) for i in range(len(self.strides))])
        out_idx = "".join(\
                ["outer_idx_%d," % (i,) for i in \
                        range(self.n_dims_before)] + \
                ["stride_idx_%d," % (i,) for i in \
                        range(len(self.strides))])
        out_idx += self._py_flattened_idx()
        return '\t' * (self.n_dims_before + len(self.strides)*2) + \
                "z[0][%s] = x[%s]\n" % (out_idx, input_idx)



