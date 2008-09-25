"""Random number generation for Theano graphs."""
import gof
import tensor
import numpy
import functools

from compile import SymbolicInputKit, SymbolicInput
from copy import copy

class RandomFunction(gof.Op):

    def __init__(self, fn, outtype, *args, **kwargs):
        """
        fn: a random function with the same signature as functions in numpy.random.RandomState
        outtype: the type of the output
        args: a list of default arguments for the function
        kwargs: if the 'inplace' key is there, its value will be used to determine if the op operates inplace or not
        """
        self.fn = fn
        self.outtype = outtype
        self.args = tuple(tensor.as_tensor(arg) for arg in args)
        self.inplace = kwargs.pop('inplace', False)
        if self.inplace:
            self.destroy_map = {0: [0]}

    def make_node(self, r, shape, *args):
        """
        in: r -> RandomState (gof.generic),
            shape -> lvector
            args -> the arguments expected by the numpy function
        out: r2 -> the new RandomState (gof.generic)
             out -> the random numbers we generated
        """
        args = map(tensor.as_tensor, args)
        shape = tensor.as_tensor(shape)
        assert shape.type == tensor.lvector
        assert len(args) <= len(self.args)
        args += (None,) * (len(self.args) - len(args))
        inputs = []
        for arg, default in zip(args, self.args):
            assert arg is None or default.type.dtype == arg.type.dtype
            input = default if arg is None else arg
            inputs.append(input)
        return gof.Apply(self,
                         [r, shape] + inputs,
                         [r.type(), self.outtype()])

    def perform(self, node, inputs, (rout, out)):
        r, shape, args = inputs[0], inputs[1], inputs[2:]
        assert self.outtype.ndim == len(shape)
        if not self.inplace:
            r = copy(r)
        rout[0] = r
        out[0] = self.fn(r, *(args + [shape]))

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.fn == other.fn\
            and self.outtype == other.outtype\
            and self.args == other.args\
            and self.inplace == other.inplace

    def __hash__(self):
        return hash(self.fn) ^ hash(self.outtype) ^ hash(self.args) ^ hash(self.inplace)


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
    def f(ndim, *args, **kwargs):
        if isinstance(ndim, int):
            r, shape, args = args[0], args[1], args[2:]
        else:
            r, shape, args = ndim, args[0], args[1:]
            shape = tensor.as_tensor(shape)
            ndim = tensor.get_vector_length(shape)
            if ndim is None:
                raise ValueError('Cannot infer the number of dimensions from the shape argument.')
        # note: rf should probably be cached for future use
        rf = RandomFunction(fn, tensor.Tensor(dtype = dtype, broadcastable = (False,)*ndim), *rfargs, **rfkwargs)
        return rf(r, shape, *args, **kwargs)
    return f


RS = numpy.random.RandomState

# we need to provide defaults for all the functions in order to infer the argument types...
uniform = random_function(RS.uniform, 'float64', 0.0, 1.0)
binomial = random_function(RS.binomial, 'int64', 1, 0.5)
normal = random_function(RS.normal, 'float64', 0.0, 1.0)
random_integers = random_function(RS.random_integers, 'int64', 0, 1)


@gof.local_optimizer
def random_make_inplace(node):
    op = node.op
    if isinstance(op, RandomFunction) and not op.inplace:
        return RandomFunction(op.fn, op.outtype, *op.args, **dict(inplace=True)).make_node(*node.inputs).outputs


import sys
from functools import partial
from collections import deque

class RandomKit(SymbolicInputKit):

    def __init__(self, name, value = None):
        super(RandomKit, self).__init__(name)
        self.value = value

    def gen(self, op, *args, **kwargs):
        r = gof.generic()
        new_r, out = op(r, *args, **kwargs)
        self.add_input(SymbolicInput(r, update = new_r))
        out.rng = r
        out.auto = self
        return out

    def distribute(self, value, indices, containers):
        rg = partial(numpy.random.RandomState(value).randint, sys.maxint)
        elems = deque(zip(indices, containers))
        i = 0
        while elems:
            index, container = elems.popleft()
            while i <= index:
                curr = rg()
                i += 1
            rs = numpy.random.RandomState(int(curr))
            container.data = rs

    def binomial(self, *args, **kwargs):
        return self.gen(binomial, *args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self.gen(uniform, *args, **kwargs)

    def normal(self, *args, **kwargs):
        return self.gen(normal, *args, **kwargs)

    def random_integers(self, *args, **kwargs):
        return self.gen(random_integers, *args, **kwargs)



rk = RandomKit('rk', 0xBAD5EED)


