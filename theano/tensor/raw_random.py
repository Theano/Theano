"""Define random number Type (`RandomStateType`) and Op (`RandomFunction`)."""
from __future__ import absolute_import, print_function, division

import sys
from copy import copy

import numpy
from six import string_types
from six.moves import reduce, xrange

# local imports
import theano
from theano import tensor
from theano.tensor import opt
from theano import gof
from theano.compile import optdb

__docformat__ = "restructuredtext en"


class RandomStateType(gof.Type):
    """
    A Type wrapper for numpy.random.RandomState.

    The reason this exists (and `Generic` doesn't suffice) is that
    RandomState objects that would appear to be equal do not compare
    equal with the '==' operator.  This Type exists to provide an equals
    function that is used by DebugMode.

    """
    def __str__(self):
        return 'RandomStateType'

    def filter(self, data, strict=False, allow_downcast=None):
        if self.is_valid_value(data):
            return data
        else:
            raise TypeError()

    def is_valid_value(self, a):
        return type(a) == numpy.random.RandomState

    def values_eq(self, a, b):
        sa = a.get_state()
        sb = b.get_state()
        # Should always be the string 'MT19937'
        if sa[0] != sb[0]:
            return False
        # 1-D array of 624 unsigned integer keys
        if not numpy.all(sa[1] == sb[1]):
            return False
        # integer "pos" representing the position in the array
        if sa[2] != sb[2]:
            return False
        # integer "has_gauss"
        if sa[3] != sb[3]:
            return False
        # float "cached_gaussian".
        # /!\ It is not initialized if has_gauss == 0
        if sa[3] != 0:
            if sa[4] != sb[4]:
                return False
        return True

    def get_shape_info(self, obj):
        return None

    def get_size(self, shape_info):
        # The size is the data, that have constant size.
        state = numpy.random.RandomState().get_state()
        size = 0
        for elem in state:
            if isinstance(elem, str):
                size += len(elem)
            elif isinstance(elem, numpy.ndarray):
                size += elem.size * elem.itemsize
            elif isinstance(elem, int):
                size += numpy.dtype("int").itemsize
            elif isinstance(elem, float):
                size += numpy.dtype("float").itemsize
            else:
                raise NotImplementedError()
        return size

    @staticmethod
    def may_share_memory(a, b):
        return a is b

# Register RandomStateType's C code for ViewOp.
theano.compile.register_view_op_c_code(
    RandomStateType,
    """
    Py_XDECREF(%(oname)s);
    %(oname)s = %(iname)s;
    Py_XINCREF(%(oname)s);
    """,
    1)

random_state_type = RandomStateType()


class RandomFunction(gof.Op):
    """
    Op that draws random numbers from a numpy.random.RandomState object.

    Parameters
    ----------
    fn : string or function reference
        A member function of numpy.random.RandomState. A string will
        be interpreted as the name of a member function of
        numpy.random.RandomState.
        Technically, any function with a signature like the ones in
        numpy.random.RandomState will do. This function must accept
        the shape (sometimes called size) of the output as the last
        positional argument.
    outtype
        The theano Type of the output.
    args
        A list of default arguments for the function
        kwargs
        If the 'inplace' key is there, its value will be used to
        determine if the op operates inplace or not.
        If the 'ndim_added' key is there, its value indicates how
        many more dimensions this op will add to the output, in
        addition to the shape's dimensions (used in multinomial and
        permutation).

    """

    __props__ = ("fn", "outtype", "inplace", "ndim_added")

    def __init__(self, fn, outtype, inplace=False, ndim_added=0):
        self.__setstate__([fn, outtype, inplace, ndim_added])

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['exec_fn']
        if 'destroy_map' in d:
            del d['destroy_map']
        return d

    def __setstate__(self, dct):
        if isinstance(dct, dict):
            state = [dct['fn'],
                     dct['outtype'],
                     dct['inplace'],
                     dct['ndim_added']]
            self.__dict__.update(dct)
        else:
            state = dct
        fn, outtype, inplace, ndim_added = state
        self.fn = fn
        if isinstance(fn, string_types):
            self.exec_fn = getattr(numpy.random.RandomState, fn)
        else:
            self.exec_fn = fn
        self.outtype = outtype
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}
        self.ndim_added = ndim_added

    def __str__(self):
        return 'RandomFunction{%s}' % self.exec_fn.__name__

    def make_node(self, r, shape, *args):
        """
        Parameters
        ----------
        r
            A numpy.random.RandomState instance, or a Variable of Type
            RandomStateType that will contain a RandomState instance.
        shape
            An lvector with a shape defining how many samples
            to draw.  In the case of scalar distributions, it is the shape
            of the tensor output by this Op.  In that case, at runtime, the
            value associated with this lvector must have a length equal to
            the number of dimensions promised by `self.outtype`.
            In a more general case, the number of output dimensions,
            len(self.outtype), is equal to len(shape)+self.ndim_added.
            The special case where len(shape) == 0 means that the smallest
            shape compatible with the argument's shape will be used.
        args
            The values associated with these variables will be passed to the
            RandomState function during perform as extra "*args"-style
            arguments. These should be castable to variables of Type TensorType.

        Returns
        -------
        Apply
            Apply with two outputs. The first output is a gof.generic Variable
            from which to draw further random numbers.
            The second output is the outtype() instance holding the random
            draw.

        """
        shape_ = tensor.as_tensor_variable(shape, ndim=1)
        if shape == ():
            shape = shape_.astype('int64')
        else:
            shape = shape_
        assert shape.type.ndim == 1
        assert (shape.type.dtype == 'int64') or (shape.type.dtype == 'int32')
        if not isinstance(r.type, RandomStateType):
            print('WARNING: RandomState instances should be in RandomStateType', file=sys.stderr)
            if 0:
                raise TypeError('r must be RandomStateType instance', r)
        # the following doesn't work because we want to ignore the
        # broadcastable flags in shape.type
        # assert shape.type == tensor.lvector

        # convert args to TensorType instances
        # and append enough None's to match the length of self.args
        args = list(map(tensor.as_tensor_variable, args))

        return gof.Apply(self,
                         [r, shape] + args,
                         [r.type(), self.outtype()])

    def infer_shape(self, node, i_shapes):
        r, shp = node.inputs[0:2]

        # if shp is a constant array of len 0, then it means 'automatic shape'
        unknown_shape = len(getattr(shp, 'data', [0, 1, 2])) == 0

        # if ndim_added == 0 and shape != () then shape
        if self.ndim_added == 0 and not unknown_shape:
            sample_shp = shp
        else:
            # if shape == () then it will depend on args
            # if ndim_added != 0 and shape != () then it will depend on args
            # Use the default infer_shape implementation.
            raise tensor.ShapeError()

        return [None, [sample_shp[i] for i in xrange(node.outputs[1].ndim)]]

    def perform(self, node, inputs, out_):
        rout, out = out_
        # Use self.fn to draw shape worth of random numbers.
        # Numbers are drawn from r if self.inplace is True, and from a
        # copy of r if self.inplace is False
        r, shape, args = inputs[0], inputs[1], inputs[2:]
        assert type(r) == numpy.random.RandomState, (type(r), r)

        # If shape == [], that means no shape is enforced, and numpy is
        # trusted to draw the appropriate number of samples, numpy uses
        # shape "None" to represent that. Else, numpy expects a tuple.
        # TODO: compute the appropriate shape, and pass it to numpy.
        if len(shape) == 0:
            shape = None
        else:
            shape = tuple(shape)

        if (shape is not None and
                self.outtype.ndim != len(shape) + self.ndim_added):
            raise ValueError('Shape mismatch: self.outtype.ndim (%i) !='
                             ' len(shape) (%i) + self.ndim_added (%i)'
                             % (self.outtype.ndim, len(shape), self.ndim_added))
        if not self.inplace:
            r = copy(r)
        rout[0] = r
        rval = self.exec_fn(r, *(args + [shape]))
        if (not isinstance(rval, numpy.ndarray) or
                str(rval.dtype) != node.outputs[1].type.dtype):
            rval = theano._asarray(rval, dtype=node.outputs[1].type.dtype)

        # When shape is None, numpy has a tendency to unexpectedly
        # return a scalar instead of a higher-dimension array containing
        # only one element. This value should be reshaped
        if shape is None and rval.ndim == 0 and self.outtype.ndim > 0:
            rval = rval.reshape([1] * self.outtype.ndim)

        if len(rval.shape) != self.outtype.ndim:
            raise ValueError('Shape mismatch: "out" should have dimension %i,'
                             ' but the value produced by "perform" has'
                             ' dimension %i'
                             % (self.outtype.ndim, len(rval.shape)))

        # Check the output has the right shape
        if shape is not None:
            if self.ndim_added == 0 and shape != rval.shape:
                raise ValueError(
                    'Shape mismatch: "out" should have shape %s, but the'
                    ' value produced by "perform" has shape %s'
                    % (shape, rval.shape))
            elif (self.ndim_added > 0 and
                  shape != rval.shape[:-self.ndim_added]):
                raise ValueError(
                    'Shape mismatch: "out" should have shape starting with'
                    ' %s (plus %i extra dimensions), but the value produced'
                    ' by "perform" has shape %s'
                    % (shape, self.ndim_added, rval.shape))

        out[0] = rval

    def grad(self, inputs, outputs):
        return [theano.gradient.grad_undefined(self, k, inp,
                'No gradient defined through raw random numbers op')
                for k, inp in enumerate(inputs)]

    def R_op(self, inputs, eval_points):
        return [None for i in eval_points]


def _infer_ndim_bcast(ndim, shape, *args):
    """
    Infer the number of dimensions from the shape or the other arguments.

    Returns
    -------
    (int, variable, tuple) triple, where the variable is an integer vector,
    and the tuple contains Booleans
        The first element returned is the inferred number of dimensions.
        The second element is the shape inferred (combining symbolic and
        constant informations from shape and args).
        The third element is a broadcasting pattern corresponding to that shape.

    """

    # Find the minimum value of ndim required by the *args
    if args:
        args_ndim = max(arg.ndim for arg in args)
    else:
        args_ndim = 0

    if isinstance(shape, (tuple, list)):
        # there is a convention that -1 means the corresponding shape of a
        # potentially-broadcasted symbolic arg
        #
        # This case combines together symbolic and non-symbolic shape
        # information
        shape_ndim = len(shape)
        if ndim is None:
            ndim = shape_ndim
        else:
            if shape_ndim != ndim:
                raise ValueError('ndim should be equal to len(shape), but\n',
                                 'ndim = %s, len(shape) = %s, shape = %s'
                                 % (ndim, shape_ndim, shape))

        bcast = []
        pre_v_shape = []
        for i, s in enumerate(shape):
            if hasattr(s, 'type'):  # s is symbolic
                bcast.append(False)  # todo - introspect further
                pre_v_shape.append(s)
            else:
                if s >= 0:
                    pre_v_shape.append(tensor.as_tensor_variable(s))
                    bcast.append((s == 1))
                elif s == -1:
                    n_a_i = 0
                    for a in args:
                        # ndim: _   _   _   _   _   _
                        # ashp:         s0  s1  s2  s3
                        #           i
                        if i >= ndim - a.ndim:
                            n_a_i += 1
                            a_i = i + a.ndim - ndim
                            if not a.broadcastable[a_i]:
                                pre_v_shape.append(a.shape[a_i])
                                bcast.append(False)
                                break
                    else:
                        if n_a_i == 0:
                            raise ValueError((
                                'Auto-shape of -1 must overlap'
                                'with the shape of one of the broadcastable'
                                'inputs'))
                        else:
                            pre_v_shape.append(tensor.as_tensor_variable(1))
                            bcast.append(True)
                else:
                    ValueError('negative shape', s)
        # post-condition: shape may still contain both symbolic and
        # non-symbolic things
        if len(pre_v_shape) == 0:
            v_shape = tensor.constant([], dtype='int64')
        else:
            v_shape = tensor.stack(pre_v_shape)

    elif shape is None:
        # The number of drawn samples will be determined automatically,
        # but we need to know ndim
        if not args:
            raise TypeError(('_infer_ndim_bcast cannot infer shape without'
                             ' either shape or args'))
        template = reduce(lambda a, b: a + b, args)
        v_shape = template.shape
        bcast = template.broadcastable
        ndim = template.ndim
    else:
        v_shape = tensor.as_tensor_variable(shape)
        if v_shape.ndim != 1:
            raise TypeError(
                "shape must be a vector or list of scalar, got '%s'" % v_shape)

        if ndim is None:
            ndim = tensor.get_vector_length(v_shape)
        bcast = [False] * ndim

    if v_shape.ndim != 1:
        raise TypeError("shape must be a vector or list of scalar, got '%s'" %
                        v_shape)

    if (not (v_shape.dtype.startswith('int') or
             v_shape.dtype.startswith('uint'))):
        raise TypeError('shape must be an integer vector or list',
                        v_shape.dtype)

    if args_ndim > ndim:
        raise ValueError(
            'ndim should be at least as big as required by args value',
            (ndim, args_ndim), args)

    assert ndim == len(bcast)
    return ndim, tensor.cast(v_shape, 'int64'), tuple(bcast)


def _generate_broadcasting_indices(out_shape, *shapes):
    """
    Return indices over each shape that broadcast them to match out_shape.

    The first returned list is equivalent to numpy.ndindex(out_shape),
    the other returned lists are indices corresponding to the other shapes,
    such that looping over these indices produce tensors of shape out_shape.
    In particular, the indices over broadcasted dimensions should all be 0.

    The shapes should have the same length as out_shape. If they are longer,
    the right-most dimensions are ignored.

    """
    all_shapes = (out_shape,) + shapes
    # Will contain the return value: a list of indices for each argument
    ret_indices = [[()] for shape in all_shapes]

    for dim in xrange(len(out_shape)):
        # Temporary list to generate the indices
        _ret_indices = [[] for shape in all_shapes]

        out_range = list(range(out_shape[dim]))

        # Verify the shapes are compatible along that dimension
        # and generate the appropriate range: out_range, or [0, ..., 0]
        ranges = [out_range]
        for shape in shapes:
            if shape[dim] == out_shape[dim]:
                ranges.append(out_range)
            elif shape[dim] == 1:  # broadcast
                ranges.append([0] * out_shape[dim])
            else:
                raise ValueError(
                    'shape[%i] (%i) should be equal to out_shape[%i] (%i) or'
                    ' to 1'
                    % (dim, shape[dim], dim, out_shape[dim]), shape,
                    out_shape, shapes)

        for prev_index in zip(*ret_indices):
            for dim_index in zip(*ranges):
                for i in xrange(len(all_shapes)):
                    _ret_indices[i].append(prev_index[i] + (dim_index[i],))
        ret_indices = _ret_indices

    return ret_indices


def uniform(random_state, size=None, low=0.0, high=1.0, ndim=None, dtype=None):
    """
    Sample from a uniform distribution between low and high.

    If the size argument is ambiguous on the number of dimensions, ndim
    may be a plain integer to supplement the missing information.

    If size is None, the output shape will be determined by the shapes
    of low and high.

    If dtype is not specified, it will be inferred from the dtype of
    low and high, but will be at least as precise as floatX.

    """
    low = tensor.as_tensor_variable(low)
    high = tensor.as_tensor_variable(high)
    if dtype is None:
        dtype = tensor.scal.upcast(theano.config.floatX, low.dtype, high.dtype)
    ndim, size, bcast = _infer_ndim_bcast(ndim, size, low, high)
    op = RandomFunction('uniform',
                        tensor.TensorType(dtype=dtype, broadcastable=bcast))
    return op(random_state, size, low, high)


def normal(random_state, size=None, avg=0.0, std=1.0, ndim=None, dtype=None):
    """
    Sample from a normal distribution centered on avg with
    the specified standard deviation (std).

    If the size argument is ambiguous on the number of dimensions, ndim
    may be a plain integer to supplement the missing information.

    If size is None, the output shape will be determined by the shapes
    of avg and std.

    If dtype is not specified, it will be inferred from the dtype of
    avg and std, but will be at least as precise as floatX.

    """
    avg = tensor.as_tensor_variable(avg)
    std = tensor.as_tensor_variable(std)
    if dtype is None:
        dtype = tensor.scal.upcast(theano.config.floatX, avg.dtype, std.dtype)
    ndim, size, bcast = _infer_ndim_bcast(ndim, size, avg, std)
    op = RandomFunction('normal',
                        tensor.TensorType(dtype=dtype, broadcastable=bcast))
    return op(random_state, size, avg, std)


def binomial(random_state, size=None, n=1, p=0.5, ndim=None,
             dtype='int64', prob=None):
    """
    Sample n times with probability of success prob for each trial,
    return the number of successes.

    If the size argument is ambiguous on the number of dimensions, ndim
    may be a plain integer to supplement the missing information.

    If size is None, the output shape will be determined by the shapes
    of n and prob.

    """
    if prob is not None:
        p = prob
        print("DEPRECATION WARNING: the parameter prob to the binomal fct have been renamed to p to have the same name as numpy.", file=sys.stderr)
    n = tensor.as_tensor_variable(n)
    p = tensor.as_tensor_variable(p)
    ndim, size, bcast = _infer_ndim_bcast(ndim, size, n, p)
    if n.dtype == 'int64':
        try:
            numpy.random.binomial(n=numpy.asarray([2, 3, 4], dtype='int64'), p=numpy.asarray([.1, .2, .3], dtype='float64'))
        except TypeError:
            # THIS WORKS AROUND A NUMPY BUG on 32bit machine
            n = tensor.cast(n, 'int32')
    op = RandomFunction('binomial',
                        tensor.TensorType(dtype=dtype,
                                          broadcastable=(False,) * ndim))
    return op(random_state, size, n, p)


def random_integers_helper(random_state, low, high, size):
    """
    Helper function to draw random integers.

    This is a generalization of numpy.random.random_integers to the case where
    low and high are tensors.

    """
    # Figure out the output shape
    if size is not None:
        out_ndim = len(size)
    else:
        out_ndim = max(low.ndim, high.ndim)
    # broadcast low and high to out_ndim dimensions
    if low.ndim > out_ndim:
        raise ValueError(
            'low.ndim (%i) should not be larger than len(size) (%i)'
            % (low.ndim, out_ndim),
            low, size)
    if low.ndim < out_ndim:
        low = low.reshape((1,) * (out_ndim - low.ndim) + low.shape)

    if high.ndim > out_ndim:
        raise ValueError(
            'high.ndim (%i) should not be larger than len(size) (%i)'
            % (high.ndim, out_ndim), high, size)
    if high.ndim < out_ndim:
        high = high.reshape((1,) * (out_ndim - high.ndim) + high.shape)

    if size is not None:
        out_size = tuple(size)
    else:
        out_size = ()
        for dim in xrange(out_ndim):
            dim_len = max(low.shape[dim], high.shape[dim])
            out_size = out_size + (dim_len,)

    # Build the indices over which to loop
    out = numpy.ndarray(out_size)
    broadcast_ind = _generate_broadcasting_indices(out_size, low.shape,
                                                   high.shape)
    # Iterate over these indices, drawing one sample at a time from numpy
    for oi, li, hi in zip(*broadcast_ind):
        out[oi] = random_state.random_integers(low=low[li], high=high[hi])

    return out


def random_integers(random_state, size=None, low=0, high=1, ndim=None,
                    dtype='int64'):
    """
    Sample a random integer between low and high, both inclusive.

    If the size argument is ambiguous on the number of dimensions, ndim
    may be a plain integer to supplement the missing information.

    If size is None, the output shape will be determined by the shapes
    of low and high.

    """
    low = tensor.as_tensor_variable(low)
    high = tensor.as_tensor_variable(high)
    ndim, size, bcast = _infer_ndim_bcast(ndim, size, low, high)
    op = RandomFunction(random_integers_helper,
                        tensor.TensorType(dtype=dtype, broadcastable=bcast))
    return op(random_state, size, low, high)


def choice_helper(random_state, a, replace, p, size):
    """
    Helper function to draw random numbers using numpy's choice function.

    This is a generalization of numpy.random.choice that coerces
    `replace` to a bool and replaces `p` with None when p is a vector
    of 0 elements.

    """
    if a.ndim > 1:
        raise ValueError('a.ndim (%i) must be 0 or 1' % a.ndim)
    if p.ndim == 1:
        if p.size == 0:
            p = None
    else:
        raise ValueError('p.ndim (%i) must be 1' % p.ndim)
    replace = bool(replace)
    return random_state.choice(a, size, replace, p)


def choice(random_state, size=None, a=2, replace=True, p=None, ndim=None,
           dtype='int64'):
    """
    Choose values from `a` with or without replacement. `a` can be a 1-D array
    or a positive scalar. If `a` is a scalar, the samples are drawn from the
    range 0,...,a-1.

    If the size argument is ambiguous on the number of dimensions, ndim
    may be a plain integer to supplement the missing information.

    If size is None, a scalar will be returned.

    """
    # numpy.random.choice is only available for numpy versions >= 1.7
    major, minor, _ = numpy.version.short_version.split('.')
    if (int(major), int(minor)) < (1, 7):
        raise ImportError('choice requires at NumPy version >= 1.7 '
                          '(%s)' % numpy.__version__)
    a = tensor.as_tensor_variable(a)
    if isinstance(replace, bool):
        replace = tensor.constant(replace, dtype='int8')
    else:
        replace = tensor.as_tensor_variable(replace)
    # encode p=None as an empty vector
    p = tensor.as_tensor_variable(p or [])
    ndim, size, bcast = _infer_ndim_bcast(ndim, size)
    op = RandomFunction(choice_helper, tensor.TensorType(dtype=dtype,
                                                         broadcastable=bcast))
    return op(random_state, size, a, replace, p)


def poisson(random_state, size=None, lam=1.0, ndim=None, dtype='int64'):
    """
    Draw samples from a Poisson distribution.

    The Poisson distribution is the limit of the Binomial distribution for
    large N.

    Parameters
    ----------
    lam : float or ndarray-like of the same shape as size parameter
        Expectation of interval, should be >= 0.
    size: int or tuple of ints, optional
        Output shape. If the given shape is, e.g., (m, n, k), then m * n * k
        samples are drawn.
    dtype
        The dtype of the return value (which will represent counts).

    size or ndim must be given.

    """
    lam = tensor.as_tensor_variable(lam)

    ndim, size, bcast = _infer_ndim_bcast(ndim, size)

    op = RandomFunction("poisson", tensor.TensorType(dtype=dtype,
                                                     broadcastable=bcast))
    return op(random_state, size, lam)


def permutation_helper(random_state, n, shape):
    """
    Helper function to generate permutations from integers.

    permutation_helper(random_state, n, (1,)) will generate a permutation of
    integers 0..n-1.
    In general, it will generate as many such permutation as required by shape.
    For instance, if shape=(p,q), p*q permutations will be generated, and the
    output shape will be (p,q,n), because each permutation is of size n.

    If you wish to perform a permutation of the elements of an existing vector,
    see shuffle_row_elements.

    This is a generalization of numpy.random.permutation to tensors.
    Otherwise it behaves the same.

    """
    # n should be a 0-dimension array
    assert n.shape == ()
    # Note that it is important to convert `n` into an integer, because if it
    # is a long, the numpy permutation function will crash on Windows.
    n = int(n.item())

    if shape is None:
        # Draw only one permutation, equivalent to shape = ()
        shape = ()
    out_shape = list(shape)
    out_shape.append(n)
    out = numpy.empty(out_shape, int)
    for i in numpy.ndindex(*shape):
        out[i] = random_state.permutation(n)

    # print 'RETURNING', out.shape
    return out


def permutation(random_state, size=None, n=1, ndim=None, dtype='int64'):
    """
    Return permutations of the integers between 0 and n-1.

    Returns them as many times as required by size. For instance, if size=(p,q),
    p*q permutations will be generated, and the output shape will be (p,q,n),
    because each permutation is of size n.

    Theano tries to infer the number of dimensions from the length of
    the size argument and the shape of n, but you may always specify it
    with the `ndim` parameter.

    Notes
    -----
    Note that the output will then be of dimension ndim+1.

    """
    if size is None or size == ():
        if not(ndim is None or ndim == 1):
            raise TypeError(
                "You asked for just one permutation but asked for more then 1 dimensions.")
        ndim = 1
        size = ()
        bcast = ()
    else:
        ndim, size, bcast = _infer_ndim_bcast(ndim, size)
    # print "NDIM", ndim, size
    op = RandomFunction(permutation_helper,
                        tensor.TensorType(dtype=dtype,
                                          broadcastable=bcast + (False,)),
                        ndim_added=1)
    return op(random_state, size, n)


def multinomial_helper(random_state, n, pvals, size):
    """
    Helper function drawing from multinomial distributions.

    This is a generalization of numpy.random.multinomial to the case where
    n and pvals are tensors.

    """
    # Figure out the shape if it's None
    # Note: the output ndim will be ndim+1, because the multinomial
    # adds a dimension. The length of that dimension is pvals.shape[-1].
    if size is not None:
        ndim = len(size)
    else:
        ndim = max(n.ndim, pvals.ndim - 1)

    # broadcast n to ndim dimensions and pvals to ndim+1
    if n.ndim > ndim:
        raise ValueError('n.ndim (%i) should not be larger than len(size) (%i)'
                         % (n.ndim, ndim), n, size)
    if n.ndim < ndim:
        n = n.reshape((1,) * (ndim - n.ndim) + n.shape)

    if pvals.ndim - 1 > ndim:
        raise ValueError(
            'pvals.ndim-1 (%i) should not be larger than len(size) (%i)'
            % (pvals.ndim - 1, ndim),
            pvals, size)
    if pvals.ndim - 1 < ndim:
        pvals = pvals.reshape((1,) * (ndim - pvals.ndim + 1) + pvals.shape)

    if size is not None:
        size = tuple(size)
    else:
        size = ()
        for dim in xrange(ndim):
            dim_len = max(n.shape[dim], pvals.shape[dim])
            size = size + (dim_len,)
    out_size = size + (pvals.shape[-1],)

    # Build the indices over which to loop
    # Note that here, the rows (inner-most 1D subtensors) of pvals and out
    # are indexed, not their individual elements
    out = numpy.ndarray(out_size)
    broadcast_ind = _generate_broadcasting_indices(size, n.shape,
                                                   pvals.shape[:-1])
    # Iterate over these indices, drawing from one multinomial at a
    # time from numpy
    assert pvals.min() >= 0
    for mi, ni, pi in zip(*broadcast_ind):
        pvi = pvals[pi]

        # This might someday be fixed upstream
        # Currently numpy raises an exception in this method if the sum
        # of probabilities meets or exceeds 1.0.
        # In  perfect arithmetic this would be correct, but in float32 or
        # float64 it is too strict.
        pisum = numpy.sum(pvi)
        if 1.0 < pisum < 1.0 + 1e-5:  # correct if we went a little over
            # because mtrand.pyx has a ValueError that will trigger if
            # sum(pvals[:-1]) > 1.0
            pvi = pvi * (1.0 - 5e-5)
            # pvi = pvi * .9
            pisum = numpy.sum(pvi)
        elif pvi[-1] < 5e-5:  # will this even work?
            pvi = pvi * (1.0 - 5e-5)
            pisum = numpy.sum(pvi)
        assert pisum <= 1.0, pisum
        out[mi] = random_state.multinomial(n=n[ni],
                                           pvals=pvi.astype('float64'))
    return out


def multinomial(random_state, size=None, n=1, pvals=[0.5, 0.5],
                ndim=None, dtype='int64'):
    """
    Sample from one or more multinomial distributions defined by
    one-dimensional slices in pvals.

    Parameters
    ----------
    pvals
        A tensor of shape "nmulti+(L,)" describing each multinomial
        distribution.  This tensor must have the property that
        numpy.allclose(pvals.sum(axis=-1), 1) is true.
    size
        A vector of shape information for the output; this can also
        specify the "nmulti" part of pvals' shape.  A -1 in the k'th position
        from the right means to borrow the k'th position from the
        right in nmulti. (See examples below.)
        Default ``None`` means size=nmulti.
    n
        The number of experiments to simulate for each
        multinomial. This can be a scalar, or tensor, it will be
        broadcasted to have shape "nmulti".
    dtype
        The dtype of the return value (which will represent counts)

    Returns
    -------
    tensor
        Tensor of len(size)+1 dimensions, and shape[-1]==L, with
        the specified ``dtype``, with the experiment counts. See
        examples to understand the shape of the return value, which is
        derived from both size and pvals.shape. In return value rval,
        "numpy.allclose(rval.sum(axis=-1), n)" will be true.

    Extended Summary
    ----------------
    For example, to simulate n experiments from each multinomial in a batch of
    size B:

        size=None, pvals.shape=(B,L) --> rval.shape=[B,L]

        rval[i,j] is the count of possibility j in the i'th distribution (row)
        in pvals.

    Using size:

        size=(1,-1), pvals.shape=(A,B,L)
        --> rval.shape=[1,B,L], and requires that A==1.

        rval[k,i,j] is the count of possibility j in the distribution specified
        by pvals[k,i].

    Using size for broadcasting of pvals:

        size=(10, 1, -1), pvals.shape=(A, B, L)
        --> rval.shape=[10,1,B,L], and requires that A==1.

        rval[l,k,i,j] is the count of possibility j in the
        distribution specified by pvals[k,i], in the l'th of 10
        draws.

    """
    n = tensor.as_tensor_variable(n)
    pvals = tensor.as_tensor_variable(pvals)
    # until ellipsis is implemented (argh)
    tmp = pvals.T[0].T
    ndim, size, bcast = _infer_ndim_bcast(ndim, size, n, tmp)
    bcast = bcast + (pvals.type.broadcastable[-1],)
    op = RandomFunction(multinomial_helper,
                        tensor.TensorType(dtype=dtype,
                                          broadcastable=bcast),
                        ndim_added=1)
    return op(random_state, size, n, pvals)


@gof.local_optimizer([RandomFunction])
def random_make_inplace(node):
    op = node.op
    if isinstance(op, RandomFunction) and not op.inplace:
        # Read op_fn from op.state, not from op.fn, since op.fn
        # may not be picklable.
        op_fn, op_outtype, op_inplace, op_ndim_added = op._props()
        new_op = RandomFunction(op_fn, op_outtype, inplace=True,
                                ndim_added=op_ndim_added)
        return new_op.make_node(*node.inputs).outputs
    return False

optdb.register('random_make_inplace', opt.in2out(random_make_inplace,
                                                 ignore_newtrees=True),
               99, 'fast_run', 'inplace')


class RandomStreamsBase(object):

    def binomial(self, size=None, n=1, p=0.5, ndim=None, dtype='int64',
                 prob=None):
        """
        Sample n times with probability of success p for each trial and
        return the number of successes.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        """
        if prob is not None:
            p = prob
            print("DEPRECATION WARNING: the parameter prob to the binomal fct have been renamed to p to have the same name as numpy.", file=sys.stderr)
        return self.gen(binomial, size, n, p, ndim=ndim, dtype=dtype)

    def uniform(self, size=None, low=0.0, high=1.0, ndim=None, dtype=None):
        """
        Sample a tensor of given size whose element from a uniform
        distribution between low and high.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        """
        return self.gen(uniform, size, low, high, ndim=ndim, dtype=dtype)

    def normal(self, size=None, avg=0.0, std=1.0, ndim=None, dtype=None):
        """
        Sample from a normal distribution centered on avg with
        the specified standard deviation (std).

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        """
        return self.gen(normal, size, avg, std, ndim=ndim, dtype=dtype)

    def random_integers(self, size=None, low=0, high=1, ndim=None,
                        dtype='int64'):
        """
        Sample a random integer between low and high, both inclusive.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        """
        return self.gen(random_integers, size, low, high, ndim=ndim,
                        dtype=dtype)

    def choice(self, size=None, a=2, replace=True, p=None, ndim=None,
               dtype='int64'):
        """
        Choose values from `a` with or without replacement.

        `a` can be a 1-D array or a positive scalar.
        If `a` is a scalar, the samples are drawn from the range 0,...,a-1.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        """
        return self.gen(choice, size, a, replace, p, ndim=ndim, dtype=dtype)

    def poisson(self, size=None, lam=None, ndim=None, dtype='int64'):
        """
        Draw samples from a Poisson distribution.

        The Poisson distribution is the limit of the Binomial distribution for
        large N.

        If the size argument is ambiguous on the number of dimensions,
        ndim may be a plain integer to supplement the missing information.

        """
        return self.gen(poisson, size, lam, ndim=ndim, dtype=dtype)

    def permutation(self, size=None, n=1, ndim=None, dtype='int64'):
        """
        Return permutations of the integers between 0 and n-1.

        Returns them as many times as required by size. For instance,
        if size=(p,q), p*q permutations will be generated,
        and the output shape will be (p,q,n), because each
        permutation is of size n.

        Theano tries to infer the number of dimensions from the length
        of the size argument and the shape of n, but you may always
        specify it with the `ndim` parameter.

        Notes
        -----
        Note that the output will then be of dimension ndim+1.

        """
        return self.gen(permutation, size, n, ndim=ndim, dtype=dtype)

    def multinomial(self, size=None, n=1, pvals=[0.5, 0.5], ndim=None,
                    dtype='int64'):
        """
        Sample n times from a multinomial distribution defined by
        probabilities pvals, as many times as required by size. For
        instance, if size=(p,q), p*q samples will be drawn, and the
        output shape will be (p,q,len(pvals)).

        Theano tries to infer the number of dimensions from the length
        of the size argument and the shapes of n and pvals, but you may
        always specify it with the `ndim` parameter.

        Notes
        -----
        Note that the output will then be of dimension ndim+1.

        """
        return self.gen(multinomial, size, n, pvals, ndim=ndim, dtype=dtype)

    def shuffle_row_elements(self, input):
        """
        Return a variable with every row (rightmost index) shuffled.

        This uses permutation random variable internally, available via
        the ``.permutation`` attribute of the return value.

        """
        perm = self.permutation(size=input.shape[:-1], n=input.shape[-1],
                                ndim=input.ndim - 1)
        shuffled = tensor.permute_row_elements(input, perm)
        shuffled.permutation = perm
        return shuffled
