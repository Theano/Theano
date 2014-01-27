import numpy as np
import numpy

import theano
from theano.tensor import basic
from theano import gof, scalar
from theano.tensor.sharedvar import tensor_constructor as _shared
from collections import OrderedDict
tensor = basic
from theano.gradient import DisconnectedType


class DiffOp(theano.Op):
    # See function diff for docstring
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

    The first order difference is given by out[i] = a[i + 1] - a[i]
    along the given axis, higher order differences are calculated by
    using diff recursively. Wraping of numpy.diff.

    :param x: Input tensor variable.

    :param n: The number of times values are differenced, default is 1.

    :param axis: The axis along which the difference is taken,
        default is the last axis.

    .. versionadded:: 0.6
    """
    return DiffOp(n=n, axis=axis)(x)


class BinCountOp(theano.Op):
    # See function bincount for docstring

    compatible_type = ('int8', 'int16', 'int32', 'int64',
                       'uint8', 'uint16', 'uint32', 'uint64')
    """Tuple of all compatible dtype for the parameter of this op."""

    def __init__(self, minlength=None):
        self.minlength = minlength
        if minlength is not None:
            numpy_ver = [int(n) for n in numpy.__version__.split('.')[:2]]
            if not bool(numpy_ver >= [1, 6]):
                raise NotImplementedError(
                    "BinCountOp with minlength attribute"
                    " requires NumPy 1.6 or higher.")

    def __eq__(self, other):
        return (type(self) == type(other) and
               self.minlength == other.minlength)

    def __hash__(self):
        return hash(type(self)) ^ hash(self.minlength)

    def make_node(self, x, weights):
        x = basic.as_tensor_variable(x)

        if x.dtype not in BinCountOp.compatible_type:
            raise TypeError("Inputs dtype must be an integer.")

        # Some dtypes are not supported by numpy's implementation of bincount.
        # Until another one is available, we should fail at graph construction
        # time, not wait for execution.
        int_bitwidth = theano.gof.python_int_bitwidth()
        if int_bitwidth == 64:
            numpy_unsupported_dtypes = ('uint64',)
        if int_bitwidth == 32:
            numpy_unsupported_dtypes = ('uint32', 'int64', 'uint64')
        intp_bitwidth = theano.gof.local_bitwidth()
        if intp_bitwidth == 32:
            out_type = basic.ivector()
        elif intp_bitwidth == 64:
            out_type = basic.lvector()

        if x.dtype in numpy_unsupported_dtypes:
            raise TypeError(
                    ("Input dtypes %s are not supported by numpy.bincount, "
                    % numpy_unsupported_dtypes), x.dtype)

        if x.ndim != 1:
            raise TypeError("Inputs must be of dimension 1.")

        if weights is None:
            weights = theano.gof.Constant(theano.gof.Generic(), None)
        else:
            weights = basic.as_tensor_variable(weights)
            out_type = basic.dvector()
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

        # Needed for numpy 1.4.1 compatibility
        if self.minlength:
            out = np.bincount(x, weights=weights, minlength=self.minlength)
        else:
            out = np.bincount(x, weights=weights)

        z[0] = theano._asarray(out, dtype=node.outputs[0].dtype)

    def grad(self, inputs, outputs_gradients):
        output = self(*inputs)

        if output.dtype.find('int') != -1:
            return [inp.zeros_like().astype(theano.config.floatX)
                    for inp in inputs]

        raise NotImplementedError()

    def infer_shape(self, node, ins_shapes):
        x = node.inputs[0]
        m = basic.max(x) + 1
        if self.minlength is not None:
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

    :param x: 1 dimension, nonnegative ints

    :param weights: array of the same shape as x with corresponding weights.
        Optional.
    :param minlength: A minimum number of bins for the output array.
        Optional.

    .. versionadded:: 0.6
    """
    return BinCountOp(minlength=minlength)(x, weights)


def squeeze(x):
    """Remove broadcastable dimensions from
    the shape of an array.

    It returns the input array, but with the
    broadcastable dimensions removed. This is
    always `x` itself or a view into `x`.

    :param x: Input data, tensor variable.

    :return: `x` without its broadcastable dimensions.

    .. versionadded:: 0.6
    """
    view = x.dimshuffle([i for i in range(x.ndim)
                         if not x.broadcastable[i]])
    return view


class RepeatOp(theano.Op):
    # See the repeat function for docstring

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

        if repeats.dtype not in tensor.discrete_dtypes:
            raise TypeError("repeats.dtype must be an integer.")

        # Some dtypes are not supported by numpy's implementation of repeat.
        # Until another one is available, we should fail at graph construction
        # time, not wait for execution.
        ptr_bitwidth = theano.gof.local_bitwidth()
        if ptr_bitwidth == 64:
            numpy_unsupported_dtypes = ('uint64',)
        if ptr_bitwidth == 32:
            numpy_unsupported_dtypes = ('uint32', 'int64', 'uint64')

        if repeats.dtype in numpy_unsupported_dtypes:
            raise TypeError(
                    ("dtypes %s are not supported by numpy.repeat "
                     "for the 'repeats' parameter, "
                     % str(numpy_unsupported_dtypes)), repeats.dtype)

        if self.axis is None:
            broadcastable = [False]
        else:
            try:
                const_reps = basic.get_scalar_constant_value(repeats)
            except basic.NotScalarConstantError:
                const_reps = None
            if const_reps == 1:
                broadcastable = x.broadcastable
            else:
                broadcastable = list(x.broadcastable)
                broadcastable[self.axis] = False

        out_type = theano.tensor.TensorType(x.dtype, broadcastable)

        return theano.Apply(self, [x, repeats], [out_type()])

    def perform(self, node, inputs, output_storage):
        x = inputs[0]
        repeats = inputs[1]
        z = output_storage[0]
        z[0] = np.repeat(x, repeats=repeats, axis=self.axis)

    def connection_pattern(self, node):

        return [[True], [False]]

    def grad(self, (x, repeats), (gz, )):
        if repeats.ndim == 0:
            if self.axis is None:
                axis = x.ndim
            else:
                if self.axis >= 0:
                    axis = self.axis + 1
                else:
                    axis = self.axis + x.ndim + 1

            shape = [x.shape[k] for k in range(x.ndim)]
            shape.insert(axis, repeats)

            return [gz.reshape(shape, x.ndim + 1).sum(axis=axis),
                    DisconnectedType()()]
        elif repeats.ndim == 1:
            # For this implementation, we would need to specify the length
            # of repeats in order to split gz in the right way to sum
            # the good part.
            raise NotImplementedError()
        else:
            raise ValueError()

    def infer_shape(self, node, ins_shapes):
        i0_shapes = ins_shapes[0]
        repeats = node.inputs[1]
        out_shape = list(i0_shapes)

        #uint64 shape are not supported.
        dtype = None
        if repeats.dtype in ['uint8', 'uint16', 'uint32']:
            dtype = 'int64'
        if self.axis is None:
            if repeats.ndim == 0:
                if len(i0_shapes) == 0:
                    out_shape = [repeats]
                else:
                    res = 1
                    for d in i0_shapes:
                        res = res * d
                    out_shape = (res * repeats, )
            else:
                out_shape = [theano.tensor.sum(repeats, dtype=dtype)]
        else:
            if repeats.ndim == 0:
                out_shape[self.axis] = out_shape[self.axis] * repeats
            else:
                out_shape[self.axis] = theano.tensor.sum(repeats, dtype=dtype)
        return [out_shape]

    def __str__(self):
        return self.__class__.__name__


def repeat(x, repeats, axis=None):
    """Repeat elements of an array.

    It returns an array which has the same shape as `x`, except
    along the given axis. The axis is used to speficy along which
    axis to repeat values. By default, use the flattened input
    array, and return a flat output array.

    The number of repetitions for each element is `repeat`.
    `repeats` is broadcasted to fit the length of the given `axis`.

    :param x: Input data, tensor variable.
    :param repeats: int, scalar or tensor variable.

    :param axis: int, optional.

    .. versionadded:: 0.6
    """
    return RepeatOp(axis=axis)(x, repeats)


class Bartlett(gof.Op):
    # See function bartlett for docstring
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
bartlett_ = Bartlett()


#I create a function only to have the doc show well.
def bartlett(M):
    """An instance of this class returns the Bartlett spectral window in the
    time-domain. The Bartlett window is very similar to a triangular window,
    except that the end points are at zero. It is often used in signal
    processing for tapering a signal, without generating too much ripple in
    the frequency domain.

    :param M: (integer scalar) Number of points in the output
        window. If zero or less, an empty vector is returned.

    :return: (vector of doubles) The triangular window, with the
        maximum value normalized to one (the value one appears only if
        the number of samples is odd), with the first and last samples
        equal to zero.

    .. versionadded:: 0.6

    """
    return bartlett_(M)


class FillDiagonal(gof.Op):
    # See function fill_diagonal for docstring
    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
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
        # diag is only valid for matrices
        import theano.sandbox.linalg
        wr_val = theano.sandbox.linalg.ops.diag(grad).sum()
        return [wr_a, wr_val]
fill_diagonal_ = FillDiagonal()


#I create a function only to have the doc show well.
def fill_diagonal(a, val):
    """ Returns a copy of an array with all
    elements of the main diagonal set to a specified scalar value.

    :param a: Rectangular array of at least two dimensions.
    :param val: Scalar value to fill the diagonal whose type must be
        compatible with that of array 'a' (i.e. 'val' cannot be viewed
        as an upcast of 'a').

    :return: An array identical to 'a' except that its main diagonal
        is filled with scalar 'val'. (For an array 'a' with a.ndim >=
        2, the main diagonal is the list of locations a[i, i, ..., i]
        (i.e. with indices all identical).)

    Support rectangular matrix and tensor with more than 2 dimensions
    if the later have all dimensions are equals.

    .. versionadded:: 0.6
    """
    return fill_diagonal_(a, val)


class BlockDot(gof.Op):
    def __init__(self, inplace=False):
        """
        Computes only the forward pass when doing the class like structure
        that Tomas proposed to speed up the output layer (which contains
        many softmax units)
        """
        self.inplace = inplace

    def __eq__(self, other):
        return type(self) == type(other) and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ hash(self.inplace)

    def make_node(self, *inputs):
        inputs = map(tensor.as_tensor_variable, inputs)
        # We restrict the op to matrix x matrix and vector x matrix !

        i_broadcastables = [input.type.broadcastable[1:] for input in
                            inputs[1:]]
        bx, by = i_broadcastables
        if len(by) == 2:  # y is a matrix
            bz = bx[:-1] + by[-1:]
        elif len(by) == 1:  # y is vector
            bz = bx[:-1]

        i_dtypes = [input.type.dtype for input in inputs]
        outputs = [tensor.tensor(scalar.upcast(*i_dtypes), (False,) + bz)]
        return gof.Apply(self, inputs, outputs)


    def make_thunk(self, node, storage_map, compute_map, no_recycling):
        node_input_storage = [storage_map[r] for r in node.inputs]
        node_output_storage = [storage_map[r] for r in node.outputs]
        node_input_compute = [compute_map[r] for r in node.inputs]
        node_output_compute = [compute_map[r] for r in node.outputs]
        def init_shape(var, exclude_leading_dim=True):
            shape = []
            if exclude_leading_dim:
                broad = var.broadcastable[1:]
            else:
                broad = var.broadcastable
            for b in broad:
                if b:
                    shape += [1]
                else:
                    shape += [2]
            return shape

        self.b = _shared(numpy.zeros(init_shape(node.inputs[0]),
                                     dtype=node.inputs[0].dtype))
        self.A = _shared(numpy.zeros(init_shape(node.inputs[1]),
                                    dtype=node.inputs[1].dtype))
        self.B = _shared(numpy.zeros(init_shape(node.inputs[2]),
                                    dtype=node.inputs[2].dtype))
        out = self.b + tensor.dot(self.A, self.B)
        self.out = _shared(numpy.zeros(init_shape(out, False), dtype=out.dtype))
        updates  = OrderedDict({self.out:out})
        self.step = theano.function(
            [],
            [],
            name='step',
            updates=updates)

        p = self.execute
        def rval(p=p, i=node_input_storage, o=node_output_storage, n=node):
            r = p(n, [x[0] for x in i], o)
            for o in node.outputs:
                compute_map[o][0] = True
            return r
        self.tmp_h = None
        rval.inputs = node_input_storage
        rval.outputs = node_output_storage
        rval.perform = p
        rval.lazy = False
        return rval


    def execute(self, node, inputs, _outs):

        b, A, B = inputs
        if A.ndim == 3 and B.ndim == 3:
            nw_shape = (A.shape[0], A.shape[1],
                    B.shape[2])
        elif A.ndim ==3 and B.ndim == 2:
            nw_shape = (A.shape[0], A.shape[1])
        elif A.ndim ==2 and B.ndim == 3:
            nw_shape = (A.shape[0], B.shape[2])
        elif A.ndim==2 and B.ndim == 2:
            nw_shape = (A.shape[0])


        if _outs[0][0] is not None and _outs[0][0].shape == nw_shape:
            pass
        else:
            _outs[0][0] = numpy.zeros(nw_shape, dtype=self.out.dtype)

        for dx in xrange(A.shape[0]):
            self.b.set_value(b[dx], borrow=True)
            self.A.set_value(A[dx], borrow=True)
            self.B.set_value(B[dx], borrow=True)
            self.step()
            values = self.out.get_value(borrow=True,
                                        return_internal_type=True)
            _outs[0][0][dx] = values

    def grad(self, inp, grads):

        b, x, y = inp
        gz, = grads
        xdim, ydim, gdim = x.type.ndim, y.type.ndim, gz.type.ndim

        #grad is scalar, so x is vector and y is vector
        if gdim == 1:
            bgrad = gz
            xgrad = gz.dimshuffle(0,'x') * y
            ygrad = gz.dimshuffle(0,'x') * x

        #x is vector, y is matrix, grad is vector
        elif xdim == 2 and ydim == 3:
            xgrad = block_dot(tensor.zeros_like(x), gz, y.dimshuffle(0,2,1))
            ygrad = block_dot(tensor.zeros_like(y), x.dimshuffle(0,1,'x'), gz.dimshuffle(0,'x',1))
            bgrad = gz

        #x is matrix, y is vector, grad is vector
        elif xdim == 3 and ydim == 2:
            xgrad = block_dot(tensor.zeros_like(x), gz.dimshuffle(0,1,'x'), y.dimshuffle(0,'x',1))
            ygrad = block_dot(tensor.zeros_like(y), x.dimshuffle(0,2,1), gz)
            bgrad = gz

        #x is matrix, y is matrix, grad is matrix
        elif xdim == ydim == 3:
            xgrad = block_dot(tensor.zeros_like(x), gz, y.dimshuffle(0,2,1))
            ygrad = block_dot(tensor.zeros_like(y), x.dimshuffle(0,2,1), gz)
            if b.ndim < gz.ndim:
                bgrad = gz.sum(1)
            else:
                bgrad = gz

        # If x or y contain broadcastable dimensions but only one of
        # them know that a matching dimensions is broadcastable, the
        # above code don't always return the right broadcast pattern.
        # This cause problem down the road. See gh-1461.
        if xgrad.broadcastable != x.broadcastable:
            xgrad = tensor.patternbroadcast(xgrad, x.broadcastable)
        if ygrad.broadcastable != y.broadcastable:
            ygrad = tensor.patternbroadcast(ygrad, y.broadcastable)

        rval = bgrad, xgrad, ygrad

        for elem in rval:
            assert elem.dtype.find('float') != -1

        return rval

    def R_op(self, inputs, eval_points):
        # R_op for a \dot b evaluted at c for a and d for b is
        # simply c \dot b + a \dot d


        assert len(inputs) == 3
        assert len(eval_points) == 3
        if eval_points[0] is None and eval_points[1] is None and\
            eval_points[2] is None:
            return [None]

        debugger_available = config.compute_test_value != 'off'

        if debugger_available:
            try:
                iv0 = gof.op.get_test_value(inputs[0])
            except AttributeError:
                gof.op.missing_test_message(
                    'first input passed to Dot.R_op has no test value')
                debugger_available = False

            try:
                iv1 = gof.op.get_test_value(inputs[1])
            except AttributeError:
                gof.op.missing_test_message(
                    'second input passed to Dot.R_op has no test value')
                debugger_available = False

            try:
                iv2 = gof.op.get_test_value(inputs[2])
            except AttributeError:
                gof.op.missing_test_message(
                    'second input passed to Dot.R_op has no test value')
                debugger_available = False

            if eval_points[0]:
                try:
                    ev0 = gof.op.get_test_value(eval_points[0])
                except AttributeError:
                    gof.op.missing_test_message(
                        'first eval point passed to Dot.R_op has no test value')
                    debugger_available = False
            if eval_points[1]:
                try:
                    ev1 = gof.op.get_test_value(eval_points[1])
                except AttributeError:
                    gof.op.missing_test_message(
                        'second eval point passed to Dot.R_op has no test value')
                    debugger_available = False
            if eval_points[2]:
                try:
                    ev2 = gof.op.get_test_value(eval_points[2])
                except AttributeError:
                    gof.op.missing_test_message(
                        'second eval point passed to Dot.R_op has no test value')
                    debugger_available = False


        if debugger_available:
            input_values = [iv0, iv1, iv2]
            eval_point_values = [ev0, ev1, ev2]

            for i in xrange(2):
                if eval_point_values[i] is not None and \
                   input_values[i].shape != eval_point_values[i].shape:
                    raise ValueError('input ' + str(i) + ' and eval_point ' +
                                     str(i) + ' to Dot.R_op '
                                     'should have the '
                                     'same shape, but their shapes are'
                                     ' %s and %s, respectively' % (
                            str(input_values[i].shape),
                            str(eval_point_values[i].shape)))
        if eval_points[0]:
            if eval_points[1]:
                ev0_0 = eval_points[0]
                ev0_1 = tensor.zeros_like(inputs[0])
            elif eval_points[2]:
                ev0_0 = tensor.zeros_like(inputs[0])
                ev0_1 = eval_points[0]
            else:
                ev0_0 = tensor.zeros_like(inputs[0])
                ev0_1 = tensor.zeros_like(inputs[0])
        else:
            ev0_0 = tensor.zeros_like(inputs[0])
            ev0_1 = tensor.zeros_like(inputs[0])

        if eval_points[0]:
            t1 = self(ev0_0, eval_points[0], inputs[1])
        if eval_points[1]:
            t2 = self(ev0_1, inputs[0], eval_points[1])

        if eval_points[0] and eval_points[1]:
            return [t1 + t2]
        elif eval_points[0]:
            return [t1]
        else:
            return [t2]

def block_dot(b, A,B):
    return BlockDot()(b, A,B)




