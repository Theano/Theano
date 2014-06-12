import numpy
import theano


class GroupDot(theano.gof.Op):
    def __init__(self, n_groups):
        """
        Computes only the forward pass when doing the class like structure
        that Tomas proposed to speed up the output layer (which contains
        many softmax units)
        """
        self.n_groups = n_groups

    def __eq__(self, other):
        return type(self) == type(other) and self.n_groups == other.n_groups

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n_groups)

    def make_node(self, h, W, b, groups):
        h = theano.tensor.as_tensor_variable(h)
        W = theano.tensor.as_tensor_variable(W)
        b = theano.tensor.as_tensor_variable(b)
        groups = theano.tensor.as_tensor_variable(groups)
        assert h.ndim == 2
        assert W.ndim == 3
        assert b.ndim == 2
        assert groups.ndim == 1
        assert 'int' in groups.dtype
        return theano.gof.Apply(self,
                                [h, W, b, groups],
                                [h.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        shared = theano.tensor._shared

        self.h_g = shared(numpy.zeros((2, 2), dtype=node.inputs[0].dtype))
        self.W_g = shared(numpy.zeros((2, 2), dtype=node.inputs[1].dtype))
        self.b_g = shared(numpy.zeros((2,), dtype=node.inputs[2].dtype))
        self.out_g = shared(numpy.zeros((2, 2), dtype=node.outputs[0].dtype))

        out = theano.tensor.dot(self.h_g, self.W_g) + self.b_g
        updates = [(self.out_g, out)]
        self.step = theano.function([], [], name='GroupDotStep',
                                    updates=updates)

        return super(GroupDot, self).make_thunk(node, storage_map,
                                                compute_map, no_recycling)

    def perform(self, node, ins, outs):
        h_val, W_val, b_val, groups_val = ins
        out_val = outs[0]

        # This has been a problem in the past
        assert groups_val.max() < self.n_groups

        nw_shape = (h_val.shape[0], b_val.shape[1])
        if not (out_val[0] and out_val[0].shape == nw_shape):
            out_val[0] = numpy.empty(nw_shape, dtype=h_val.dtype)

        for pos in xrange(self.n_groups):
            mask = groups_val == pos
            if mask.sum() != 0:
                self.W_g.set_value(W_val[pos], borrow=True)
                self.b_g.set_value(b_val[pos], borrow=True)
                self.h_g.set_value(h_val[mask], borrow=True)
                self.step()
                out_val[0][mask] = self.out_g.get_value(borrow=True)

    def grad(self, inputs, grads):
        h, W, b, groups = inputs
        g, = grads
        rval = GroupDotGrad(n_groups=self.n_groups)(h, W, b, groups, g)
        return rval + [theano.gradient.grad_undefined(self, 3, groups)]


class GroupDotGrad(theano.gof.Op):
    def __init__(self, n_groups):
        """
        Computes only the forward pass when doing the class like structure
        that Tomas proposed to speed up the output layer (which contains
        many softmax units)
        """
        self.n_groups = n_groups

    def __eq__(self, other):
        return type(self) == type(other) and \
            self.n_groups == other.n_groups

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n_groups)

    def make_node(self, h, W, b, groups, g):
        h = theano.tensor.as_tensor_variable(h)
        W = theano.tensor.as_tensor_variable(W)
        b = theano.tensor.as_tensor_variable(b)
        g = theano.tensor.as_tensor_variable(g)
        groups = theano.tensor.as_tensor_variable(groups)

        assert h.ndim == 2
        assert W.ndim == 3
        assert b.ndim == 2
        assert groups.ndim == 1
        assert 'int' in groups.dtype
        return theano.gof.Apply(self,
                                [h, W, b, groups, g],
                                [h.type(), W.type(), b.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        shared = theano.tensor._shared

        self.W_g = shared(numpy.zeros((2, 2), dtype=node.inputs[1].dtype))
        self.h_g = shared(numpy.zeros((2, 2), dtype=node.inputs[0].dtype))
        self.g_g = shared(numpy.zeros((2, 2),
                                              dtype=node.inputs[4].dtype))
        self.gW_g = shared(numpy.zeros((2, 2), dtype=node.outputs[1].dtype))
        self.gh_g = shared(numpy.zeros((2, 2), dtype=node.outputs[1].dtype))
        self.gb_g = shared(numpy.zeros((2,), dtype=node.outputs[2].dtype))

        gW = theano.tensor.dot(self.h_g.T, self.g_g)
        gh = theano.tensor.dot(self.g_g, self.W_g.T)
        gb = self.g_g.sum(0)
        updates = [(self.gW_g, gW), (self.gb_g, gb), (self.gh_g, gh)]
        self.step = theano.function([], [], updates=updates,
                                    name='GroupDotGradStep')

        return super(GroupDotGrad, self).make_thunk(node, storage_map,
                                                    compute_map, no_recycling)

    def perform(self, node, ins, outs):
        h_val, W_val, b_val, groups_val, g_val = ins
        gh_val, gW_val, gb_val = outs

        if not (gh_val[0] and gh_val[0].shape == h_val.shape):
            gh_val[0] = numpy.empty_like(h_val)

        # These two can't be empty since the gradient computation
        # might not touch all the parts.
        if not (gW_val[0] and gW_val[0].shape == W_val.shape):
            gW_val[0] = numpy.zeros_like(W_val)

        if not (gb_val[0] and gb_val[0].shape == b_val.shape):
            gb_val[0] = numpy.zeros_like(b_val)

        # this has been a problem in the past
        assert groups_val.max() < self.n_groups

        for pos in xrange(self.n_groups):
            mask = groups_val == pos
            if mask.sum() != 0:
                self.W_g.set_value(W_val[pos], borrow=True)
                self.h_g.set_value(h_val[mask], borrow=True)
                self.g_g.set_value(g_val[mask], borrow=True)
                self.step()
                gh_val[0][mask] = self.gh_g.get_value(borrow=True)
                gW_val[0][pos] = self.gW_g.get_value(borrow=True)
                gb_val[0][pos] = self.gb_g.get_value(borrow=True)
