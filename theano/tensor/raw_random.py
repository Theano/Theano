"""Define random number Type (`RandomStateType`) and Op (`RandomFunction`)."""
__docformat__ = "restructuredtext en"
import sys
from copy import copy
import numpy

#local imports
import basic as tensor
import opt
from .. import gof
from ..compile import optdb

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

    def __init__(self, fn, outtype, *args, **kwargs):
        """
        :param fn: a member function of numpy.RandomState
        Technically, any function with a signature like the ones in numpy.random.RandomState
        will do.  This function must accept the shape (sometimes called size) of the output as
        the last positional argument.

        :type fn: string or function reference.  A string will be interpreted as the name of a
        member function of numpy.random.RandomState.

        :param outtype: the theano Type of the output

        :param args: a list of default arguments for the function

        :param kwargs: if the 'inplace' key is there, its value will be used to determine if the op operates inplace or not
        """
        self.__setstate__([fn, outtype, args, kwargs])

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.fn == other.fn\
            and self.outtype == other.outtype\
            and self.args == other.args\
            and self.inplace == other.inplace

    def __hash__(self):
        return hash(type(self)) ^ hash(self.fn) \
                ^ hash(self.outtype) ^ hash(self.args) ^ hash(self.inplace)

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state
        fn, outtype, args, kwargs = state
        self.fn = getattr(numpy.random.RandomState, fn) if isinstance(fn, str) else fn
        self.outtype = outtype
        self.args = tuple(tensor.as_ndarray_result(arg) for arg in args)
        self.inplace = kwargs.pop('inplace', False)
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, r, shape, *args):
        """
        :param r: a numpy.RandomState instance, or a Result of Type RandomStateType that will
        contain a RandomState instance.

        :param shape: an lvector with the shape of the tensor output by this Op.  At runtime,
        the value associated with this lvector must have a length that matches the number of
        dimensions promised by `self.outtype`.

        :param args: the values associated with these results will be passed to the RandomState
        function during perform as extra "*args"-style arguments.  These should be castable to
        results of Type NDArrayType.

        :rtype: Apply

        :return: Apply with two outputs.  The first output is a gof.generic Result from which
        to draw further random numbers.  The second output is the outtype() instance holding
        the random draw.

        """
        if shape == () or shape == []:
            shape = tensor.lvector()
        else:
            shape = tensor.as_ndarray_result(shape, ndim=1)
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

        # convert args to NDArrayType instances
        # and append enough None's to match the length of self.args
        args = map(tensor.as_ndarray_result, args)
        if len(args) > len(self.args):
            raise TypeError('Too many args for this kind of random generator')
        args += (None,) * (len(self.args) - len(args))
        assert len(args) == len(self.args)

        # build the inputs to this Apply by overlaying args on self.args
        inputs = []
        for arg, default in zip(args, self.args):
            assert arg is None or default.type.dtype == arg.type.dtype
            input = default if arg is None else arg
            inputs.append(input)

        return gof.Apply(self,
                         [r, shape] + inputs,
                         [r.type(), self.outtype()])

    def perform(self, node, inputs, (rout, out)):
        # Use self.fn to draw shape worth of random numbers.
        # Numbers are drawn from r if self.inplace is True, and from a copy of r if
        # self.inplace is False
        r, shape, args = inputs[0], inputs[1], inputs[2:]
        assert type(r) == numpy.random.RandomState
        r_orig = r
        assert self.outtype.ndim == len(shape)
        if not self.inplace:
            r = copy(r)
        rout[0] = r
        rval = self.fn(r, *(args + [shape]))
        if not isinstance(rval, numpy.ndarray) \
               or str(rval.dtype) != node.outputs[1].type.dtype:
            out[0] = numpy.asarray(rval, dtype = node.outputs[1].type.dtype)
        else:
            out[0] = rval

    def grad(self, inputs, outputs):
        return [None for i in inputs]
    


__oplist_constructor_list = []
"""List of functions to be listed as op constructors in the oplist (`gen_oplist`, doc/oplist.txt)."""
def constructor(f):
    """Add `f` to :doc:`oplist`.
    
    Make `f` appear as a constructor in the oplist (`gen_oplist`, doc/oplist.txt).
    """
    __oplist_constructor_list.append(f)
    return f
def __oplist_tag(thing, tag):
    tags = getattr(thing, '__oplist_tags', [])
    tags.append(tag)
    thing.__oplist_tags = tags

def random_function(fn, dtype, *rfargs, **rfkwargs):
    """
    Returns a wrapper around RandomFunction which automatically infers the number 
    of dimensions of the output from the given shape. If the shape cannot be inferred,
    the user can give an integer as first argument, which will be interpreted as the 
    number of dimensions.

    The number of dimensions for the following shape arguments can be inferred:
    - shape(x)
    - make_lvector(x, y, z, ...)
    - constants
    """
    @constructor
    def f(ndim, *args, **kwargs):
        if isinstance(ndim, int):
            r, shape, args = args[0], args[1], args[2:]
        else:
            r, shape, args = ndim, args[0], args[1:]
            if shape == () or shape == []:
                shape = tensor.NDArrayConstant(type = tensor.lvector, data = shape)
            else:
                shape = tensor.as_ndarray_result(shape)
            ndim = tensor.get_vector_length(shape)
            if ndim is None:
                raise ValueError('Cannot infer the number of dimensions from the shape argument.')
        # note: rf could be cached for future use
        rf = RandomFunction(fn, tensor.NDArrayType(dtype = dtype, broadcastable = (False,)*ndim), *rfargs, **rfkwargs)
        return rf(r, shape, *args, **kwargs)
    return f


# we need to provide defaults for all the functions in order to infer the argument types...

uniform = random_function('uniform', 'float64', 0.0, 1.0)
uniform.__doc__ = """
Usage: uniform(random_state, size, low=0.0, high=1.0)
Sample from a uniform distribution between low and high.

If the size argument is ambiguous on the number of
dimensions, the first argument may be a plain integer
to supplement the missing information.
"""

binomial = random_function('binomial', 'int64', 1, 0.5)
binomial.__doc__ = """
Usage: binomial(random_state, size, n=1, prob=0.5)
Sample n times with probability of success prob for each trial,
return the number of successes.

If the size argument is ambiguous on the number of
dimensions, the first argument may be a plain integer
to supplement the missing information.
"""

normal = random_function('normal', 'float64', 0.0, 1.0)
normal.__doc__ = """
Usage: normal(random_state, size, avg=0.0, std=1.0)
Sample from a normal distribution centered on avg with
the specified standard deviation (std)

If the size argument is ambiguous on the number of
dimensions, the first argument may be a plain integer
to supplement the missing information.
"""

random_integers = random_function('random_integers', 'int64', 0, 1)
random_integers.__doc__ = """
Usage: random_integers(random_state, size, low=0, high=1)
Sample a random integer between low and high, both inclusive.

If the size argument is ambiguous on the number of
dimensions, the first argument may be a plain integer
to supplement the missing information.
"""


@gof.local_optimizer([None])
def random_make_inplace(node):
    op = node.op
    if isinstance(op, RandomFunction) and not op.inplace:
        return RandomFunction(op.fn, op.outtype, *op.args, **dict(inplace=True)).make_node(*node.inputs).outputs
    return False

optdb.register('random_make_inplace', opt.in2out(random_make_inplace, ignore_newtrees=True), 99, 'fast_run', 'inplace')


