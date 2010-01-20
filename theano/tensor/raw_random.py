"""Define random number Type (`RandomStateType`) and Op (`RandomFunction`)."""
__docformat__ = "restructuredtext en"
import sys
from copy import copy
import numpy

#local imports
import basic as tensor
import opt
from theano import gof
from theano.compile import optdb

class RandomStateType(gof.Type):
    """A Type wrapper for numpy.RandomState

    The reason this exists (and `Generic` doesn't suffice) is that RandomState objects that
    would appear to be equal do not compare equal with the '==' operator.  This Type exists to
    provide an equals function that is used by DebugMode.
    
    """
    def __str__(self):
        return 'RandomStateType'

    def filter(self, data, strict=False):
        if self.is_valid_value(data):
            return data
        else:
            raise TypeError()

    def is_valid_value(self, a):
        return type(a) == numpy.random.RandomState

    def values_eq(self, a, b):
        sa = a.get_state()
        sb = b.get_state()
        for aa, bb in zip(sa, sb):
            if isinstance(aa, numpy.ndarray):
                if not numpy.all(aa == bb):
                    return False
            else:
                if not aa == bb:
                    return False
        return True

random_state_type = RandomStateType()


class RandomFunction(gof.Op):
    """Op that draws random numbers from a numpy.RandomState object

    """

    def __init__(self, fn, outtype, inplace=False, ndim_added=0 ):
        """
        :param fn: a member function of numpy.RandomState
        Technically, any function with a signature like the ones in numpy.random.RandomState
        will do.  This function must accept the shape (sometimes called size) of the output as
        the last positional argument.

        :type fn: string or function reference.  A string will be interpreted as the name of a
        member function of numpy.random.RandomState.

        :param outtype: the theano Type of the output

        :param args: a list of default arguments for the function

        :param kwargs:
            If the 'inplace' key is there, its value will be used to
            determine if the op operates inplace or not.
            If the 'ndim_added' key is there, its value indicates how
            many more dimensions this op will add to the output, in
            addition to the shape's dimensions (used in multinomial and
            permutation).
        """
        self.__setstate__([fn, outtype, inplace, ndim_added])

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.fn == other.fn\
            and self.outtype == other.outtype\
            and self.inplace == other.inplace\
            and self.ndim_added == other.ndim_added

    def __hash__(self):
        return hash(type(self)) ^ hash(self.fn) \
                ^ hash(self.outtype)  \
                ^ hash(self.inplace) ^ hash(self.ndim_added)

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state
        fn, outtype, inplace, ndim_added = state
        if isinstance(fn, str):
          self.fn = getattr(numpy.random.RandomState, fn)
        else:
          self.fn = fn
        #backport
        #self.fn = getattr(numpy.random.RandomState, fn) if isinstance(fn, str) else fn
        self.outtype = outtype
        self.inplace = inplace
        if self.inplace:
            self.destroy_map = {0: [0]}
        self.ndim_added = ndim_added

    def make_node(self, r, shape, *args):
        """
        :param r: a numpy.RandomState instance, or a Variable of Type RandomStateType that will
        contain a RandomState instance.

        :param shape: an lvector with a shape defining how many samples to draw.
        In the case of scalar distributions, it is the shape of the tensor output by this Op.
        In that case, at runtime, the value associated with this lvector must have a length
        equal to the number of dimensions promised by `self.outtype`.
        In general, the number of output dimenstions is equal to
        len(self.outtype)+self.ndim_added.

        :param args: the values associated with these variables will be passed to the RandomState
        function during perform as extra "*args"-style arguments.  These should be castable to
        variables of Type TensorType.

        :rtype: Apply

        :return: Apply with two outputs.  The first output is a gof.generic Variable from which
        to draw further random numbers.  The second output is the outtype() instance holding
        the random draw.

        """
        if shape == () or shape == []:
            shape = tensor.lvector()
        else:
            shape = tensor.as_tensor_variable(shape, ndim=1)
        #print 'SHAPE TYPE', shape.type, tensor.lvector
        assert shape.type.ndim == 1
        assert (shape.type.dtype == 'int64') or (shape.type.dtype == 'int32')
        if not isinstance(r.type, RandomStateType):
            print >> sys.stderr, 'WARNING: RandomState instances should be in RandomStateType'
            if 0:
                raise TypeError('r must be RandomStateType instance', r)
        # the following doesn't work because we want to ignore the broadcastable flags in
        # shape.type
        # assert shape.type == tensor.lvector 

        # convert args to TensorType instances
        # and append enough None's to match the length of self.args
        args = map(tensor.as_tensor_variable, args)

        return gof.Apply(self,
                         [r, shape] + args,
                         [r.type(), self.outtype()])

    def perform(self, node, inputs, (rout, out)):
        # Use self.fn to draw shape worth of random numbers.
        # Numbers are drawn from r if self.inplace is True, and from a copy of r if
        # self.inplace is False
        r, shape, args = inputs[0], inputs[1], inputs[2:]
        assert type(r) == numpy.random.RandomState
        r_orig = r
        if self.outtype.ndim != len(shape) + self.ndim_added:
            raise ValueError('Shape mismatch: self.outtype.ndim (%i) != len(shape) (%i) + self.ndim_added (%i)'\
                    %(self.outtype.ndim, len(shape), self.ndim_added))
        if not self.inplace:
            r = copy(r)
        rout[0] = r
        rval = self.fn(r, *(args + [tuple(shape)]))
        if not isinstance(rval, numpy.ndarray) \
               or str(rval.dtype) != node.outputs[1].type.dtype:
            out[0] = numpy.asarray(rval, dtype = node.outputs[1].type.dtype)
        else:
            out[0] = rval
        if len(rval.shape) != self.outtype.ndim:
            raise ValueError('Shape mismatch: "out" should have dimension %i, but the value produced by "perform" has dimension %i'\
                    % (self.outtype.ndim, len(rval.shape)))

    def grad(self, inputs, outputs):
        return [None for i in inputs]
    
def _infer_ndim(ndim, shape):
    """returns int, variable pair, such that the int is the length of the variable, and the
    variable is an integer or uint vector
    """

    if isinstance(shape, (tuple, list)):
        v_shape = tensor.TensorConstant(type=tensor.lvector, data=numpy.asarray(shape, dtype='int64'))
    else:
        v_shape = tensor.as_tensor_variable(shape)

    if not (v_shape.dtype.startswith('int') or v_shape.dtype.startswith('uint')):
        raise TypeError('shape must be an integer vector or list')

    if ndim is None:
        #infer ndim
        ndim = tensor.get_vector_length(v_shape)
    
    return ndim, v_shape

def uniform(random_state, size=(), low=0.0, high=1.0, ndim=None):
    """
    Sample from a uniform distribution between low and high.

    If the size argument is ambiguous on the number of
    dimensions, the first argument may be a plain integer
    to supplement the missing information.
    """
    ndim, size = _infer_ndim(ndim, size)
    op = RandomFunction('uniform', 
            tensor.TensorType(dtype = 'float64', broadcastable = (False,)*ndim) )
    return op(random_state, size, low, high)

def binomial(random_state, size=(), n=1, prob=0.5, ndim=None):
    """
    Sample n times with probability of success prob for each trial, return the number of
    successes.

    If the size argument is ambiguous on the number of dimensions, the first argument may be a
    plain integer to supplement the missing information.
    """
    ndim, size = _infer_ndim(ndim, size)
    op = RandomFunction('binomial', 
            tensor.TensorType(dtype = 'int64', broadcastable = (False,)*ndim) )
    return op(random_state, size, n, prob)

def normal(random_state, size=(),  avg=0.0, std=1.0, ndim=None):
    """
    Usage: normal(random_state, size,
    Sample from a normal distribution centered on avg with
    the specified standard deviation (std)

    If the size argument is ambiguous on the number of
    dimensions, the first argument may be a plain integer
    to supplement the missing information.
    """
    ndim, size = _infer_ndim(ndim, size)
    op = RandomFunction('normal', 
            tensor.TensorType(dtype = 'float64', broadcastable = (False,)*ndim) )
    return op(random_state, size, avg, std)


def random_integers(random_state, size=(), low=0, high=1, ndim=None):
    """
    Usage: random_integers(random_state, size, low=0, high=1)
    Sample a random integer between low and high, both inclusive.

    If the size argument is ambiguous on the number of
    dimensions, the first argument may be a plain integer
    to supplement the missing information.
    """
    ndim, size = _infer_ndim(ndim, size)
    op = RandomFunction('random_integers', 
            tensor.TensorType(dtype = 'int64', broadcastable = (False,)*ndim) )
    return op(random_state, size, low, high)

def permutation_helper(random_state, n, shape):
    """Helper function to generate permutations from integers.

    permutation_helper(random_state, n, (1,)) will generate a permutation of
    integers 0..n-1.
    In general, it will generate as many such permutation as required by shape.
    For instance, if shape=(p,q), p*q permutations will be generated, and the
    output shape will be (p,q,n), because each permutation is of size n.

    If you wish to perform a permutation of the elements of an existing vector,
    see shuffle (to be implemented).
    """
    # n should be a 0-dimension array
    assert n.shape == ()
    # Note that it is important to convert `n` into an integer, because if it
    # is a long, the numpy permutation function will crash on Windows.
    n = int(n.item())

    out_shape = list(shape)
    out_shape.append(n)
    out = numpy.zeros(out_shape, int)
    for i in numpy.ndindex(*shape):
        out[i] = random_state.permutation(n)

    print 'RETURNING', out.shape
    return out


def permutation(random_state, size=(), n=1, ndim=None):
    """
    Returns permutations of the integers between 0 and n-1, as many times
    as required by size. For instance, if size=(p,q), p*q permutations
    will be generated, and the output shape will be (p,q,n), because each
    permutation is of size n.

    Theano tries to infer the number of dimensions from the length of the size argument, but you
    may always specify it with the `ndim` parameter.

    .. note:: 
        Note that the output will then be of dimension ndim+1.
    """
    ndim, size = _infer_ndim(ndim, size)
    print "NDIM", ndim, size
    op = RandomFunction(permutation_helper, 
            tensor.TensorType(dtype='int64', broadcastable=(False,)*(ndim+1)),
            ndim_added=1)
    return op(random_state, size, n)

def multinomial(random_state, size=(), n=1, pvals=[0.5, 0.5], ndim=None):
    """
    Sample n times from a multinomial distribution defined by probabilities pvals,
    as many times as required by size. For instance, if size=(p,q), p*q
    samples will be drawn, and the output shape will be (p,q,len(pvals)).

    Theano tries to infer the number of dimensions from the length of the size argument, but you
    may always specify it with the `ndim` parameter.

    .. note:: 
        Note that the output will then be of dimension ndim+1.
    """
    ndim, size = _infer_ndim(ndim, size)
    op = RandomFunction('multinomial', 
            tensor.TensorType(dtype = 'int64', broadcastable = (False,)*(ndim+1)),
            ndim_added=1)
    return op(random_state, size, n, pvals)


@gof.local_optimizer([None])
def random_make_inplace(node):
    op = node.op
    if isinstance(op, RandomFunction) and not op.inplace:
        new_op = RandomFunction(op.fn, op.outtype, inplace=True, ndim_added=op.ndim_added)
        return new_op.make_node(*node.inputs).outputs
    return False

optdb.register('random_make_inplace', opt.in2out(random_make_inplace, ignore_newtrees=True), 99, 'fast_run', 'inplace')



class RandomStreamsBase(object):

    def binomial(self, size=(), n=1, prob=0.5, ndim=None):
        """
        Sample n times with probability of success prob for each trial, return the number of
        successes.

        If the size argument is ambiguous on the number of dimensions, the first argument may be a
        plain integer to supplement the missing information.
        """
        return self.gen(binomial, size, n, prob, ndim=ndim)

    def uniform(self,  size=(), low=0.0, high=1.0, ndim=None):
        """
        Sample a tensor of given size whose element from a uniform distribution between low and high.

        If the size argument is ambiguous on the number of
        dimensions, the first argument may be a plain integer
        to supplement the missing information.
        """
        return self.gen(uniform, size, low, high, ndim=ndim)

    def normal(self, size=(), avg=0.0, std=1.0, ndim=None):
        """
        Usage: normal(random_state, size,
        Sample from a normal distribution centered on avg with
        the specified standard deviation (std)

        If the size argument is ambiguous on the number of
        dimensions, the first argument may be a plain integer
        to supplement the missing information.
        """
        return self.gen(normal, size, avg, std, ndim=ndim)

    def random_integers(self, size=(), low=0, high=1, ndim=None):
        """
        Usage: random_integers(random_state, size, low=0, high=1)
        Sample a random integer between low and high, both inclusive.

        If the size argument is ambiguous on the number of
        dimensions, the first argument may be a plain integer
        to supplement the missing information.
        """
        return self.gen(random_integers, size, low, high, ndim=ndim)

    def permutation(self, size=(), n=1, ndim=None):
        """
        Returns permutations of the integers between 0 and n-1, as many times
        as required by size. For instance, if size=(p,q), p*q permutations
        will be generated, and the output shape will be (p,q,n), because each
        permutation is of size n.

        Theano tries to infer the number of dimensions from the length of the size argument, but you
        may always specify it with the `ndim` parameter.

        .. note:: 
            Note that the output will then be of dimension ndim+1.
        """
        return self.gen(permutation, size, n, ndim=ndim)

    def multinomial(self, size=(), n=1, pvals=[0.5, 0.5], ndim=None):
        """
        Sample n times from a multinomial distribution defined by probabilities pvals,
        as many times as required by size. For instance, if size=(p,q), p*q
        samples will be drawn, and the output shape will be (p,q,len(pvals)).

        Theano tries to infer the number of dimensions from the length of the size argument, but you
        may always specify it with the `ndim` parameter.

        .. note:: 
            Note that the output will then be of dimension ndim+1.
        """
        return self.gen(multinomial, size, n, pvals, ndim=ndim)

    def shuffle_row_elements(self, input):
        """Return a variable with every row (rightmost index) shuffled.
        
        This uses permutation random variable internally, available via the ``.permutation``
        attribute of the return value.
        """
        perm = self.permutation(size=input.shape[:-1], n=input.shape[-1], ndim=input.ndim-1)
        shuffled = tensor.permute_row_elements(input, perm)
        shuffled.permutation = perm
        return shuffled


