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
        return type(self) == type(other) and \
                self.n_groups == other.n_groups

    def __hash__(self):
        return hash(type(self)) ^ hash(self.n_groups)

    def make_node(self, vec, mat, bias, index):
        vec = theano.tensor.as_tensor_variable(vec)
        mat = theano.tensor.as_tensor_variable(mat)
        bias = theano.tensor.as_tensor_variable(bias)
        index = theano.tensor.as_tensor_variable(index)
        assert vec.ndim == 2
        assert mat.ndim == 3
        assert bias.ndim == 2
        assert index.ndim == 1
        assert 'int' in index.dtype
        return theano.gof.Apply(self,
                                [vec, mat, bias, index],
                                [vec.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        shared = theano.tensor._shared

        self.W = shared(numpy.zeros((2, 2), dtype='float32'))
        self.b = shared(numpy.zeros((2,), dtype='float32'))
        self.h = shared(numpy.zeros((2, 2), dtype='float32'))
        self.out = shared(numpy.zeros((2, 2), dtype='float32'))
        out = theano.tensor.dot(self.h, self.W) + self.b
        updates = [(self.out, out)]
        self.step = theano.function(
            [],
            [],
            name='step',
            updates=updates)
        self.tmp_h = None

        return super(GroupDot, self).make_thunk(node, storage_map,
                                                compute_map, no_recycling)

    def perform(self, node, ins, _outs):
        state_below, matrix, biases, groups = ins

        if not (_outs[0][0] and _outs[0][0].shape == state_below.shape):
            nw_shape = (state_below.shape[0], biases.shape[1])
            _outs[0][0] = numpy.zeros(nw_shape, dtype=state_below.dtype)
        for pos in xrange(self.n_groups):
            mask = groups == pos
            if mask.sum() != 0:
                self.W.set_value(matrix[pos], borrow=True)
                self.b.set_value(biases[pos], borrow=True)
                self.h.set_value(state_below[mask], borrow=True)
                self.step()
                values = self.out.get_value(borrow=True,
                                            return_internal_type=True)
                _outs[0][0][mask] = values

    def grad(self, inputs, grads):
        state_below, matrix, biases, groups = inputs
        gout, = grads
        rval = GroupDotGrad(n_groups=self.n_groups)(state_below,
                                                    matrix, biases,
                                                    groups, gout)
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

    def make_node(self, vec, mat, bias, index, grad_on_out):
        vec = theano.tensor.as_tensor_variable(vec)
        mat = theano.tensor.as_tensor_variable(mat)
        bias = theano.tensor.as_tensor_variable(bias)
        grad_on_out = theano.tensor.as_tensor_variable(grad_on_out)

        index = theano.tensor.as_tensor_variable(index)
        assert vec.ndim == 2
        assert mat.ndim == 3
        assert bias.ndim == 2
        assert index.ndim == 1
        assert 'int' in index.dtype
        return theano.gof.Apply(self,
                                [vec, mat, bias, index, grad_on_out],
                                [vec.type(), mat.type(), bias.type()])

    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        shared = theano.tensor._shared

        self.W = shared(numpy.zeros((2, 2), dtype='float32'))

        self.b = shared(numpy.zeros((2,), dtype='float32'))
        self.h = shared(numpy.zeros((2, 2), dtype='float32'))
        #self.out = shared(numpy.zeros((2,2), dtype='float32'))
        self.grad_on_out = shared(numpy.zeros((2, 2), dtype='float32'))
        self.gW = shared(numpy.zeros((2, 2), dtype='float32'))
        self.gh = shared(numpy.zeros((2, 2), dtype='float32'))
        self.gb = shared(numpy.zeros((2,), dtype='float32'))

        gW = theano.tensor.dot(self.h.T, self.grad_on_out)
        gh = theano.tensor.dot(self.grad_on_out, self.W.T)
        gb = self.grad_on_out.sum(0)

        updates = [(self.gW, gW), (self.gb, gb), (self.gh, gh)]
        self.step = theano.function([], [], updates=updates, name='grad_step')

        return super(GroupDotGrad, self).make_thunk(node, storage_map,
                                                    compute_map, no_recycling)

    def perform(self, node, ins, _outs):
        state_below, matrix, biases, groups, grad_on_out = ins
        if not (_outs[0][0] and _outs[0][0].shape == state_below.shape):
            _outs[0][0] = numpy.zeros_like(state_below)

        if not (_outs[1][0] and _outs[1][0].shape == matrix.shape):
            _outs[1][0] = numpy.zeros_like(matrix)

        if not (_outs[2][0] and _outs[2][0].shape == biases.shape):
            _outs[2][0] = numpy.zeros_like(biases)

        for pos in xrange(self.n_groups):
            mask = groups == pos
            if mask.sum() != 0:
                self.W.set_value(matrix[pos], borrow=True)
                self.b.set_value(biases[pos], borrow=True)

                self.h.set_value(state_below[mask],
                                 borrow=True)
                self.grad_on_out.set_value(grad_on_out[mask],
                                           borrow=True)
                self.step()
                gh = self.gh.get_value(borrow=True,
                                       return_internal_type=True)
                gW = self.gW.get_value(borrow=True,
                                       return_internal_type=True)
                gb = self.gb.get_value(borrow=True,
                                       return_internal_type=True)
                _outs[0][0][mask] = gh
                _outs[1][0][pos] += gW
                _outs[2][0][pos] += gb

    def grad(self, inputs, grads):
        raise NotImplemented
