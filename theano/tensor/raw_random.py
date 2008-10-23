"""Random number generation for Theano graphs."""
from .. import gof
import basic as tensor
import numpy
import functools

import opt
from .. import compile
from ..compile import SymbolicInputKit, SymbolicInput
from copy import copy


RS = numpy.random.RandomState

class RandomFunction(gof.Op):

    def __init__(self, fn, outtype, *args, **kwargs):
        """
        fn: a random function with the same signature as functions in numpy.random.RandomState
        outtype: the type of the output
        args: a list of default arguments for the function
        kwargs: if the 'inplace' key is there, its value will be used to determine if the op operates inplace or not
        """
        self.__setstate__([fn, outtype, args, kwargs])

    def make_node(self, r, shape, *args):
        """
        in: r -> RandomState (gof.generic),
            shape -> lvector
            args -> the arguments expected by the numpy function
        out: r2 -> the new RandomState (gof.generic)
             out -> the random numbers we generated
        """
        args = map(tensor.as_tensor, args)
        if shape == () or shape == []:
            shape = tensor.lvector()
        else:
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
        rval = self.fn(r, *(args + [shape]))
        if not isinstance(rval, numpy.ndarray):
            out[0] = numpy.asarray(rval)
        else:
            out[0] = rval

    def grad(self, inputs, outputs):
        return [None] * len(inputs)

    def __eq__(self, other):
        return type(self) == type(other) \
            and self.fn == other.fn\
            and self.outtype == other.outtype\
            and self.args == other.args\
            and self.inplace == other.inplace

    def __hash__(self):
        return hash(self.fn) ^ hash(self.outtype) ^ hash(self.args) ^ hash(self.inplace)

    def __getstate__(self):
        return self.state

    def __setstate__(self, state):
        self.state = state
        fn, outtype, args, kwargs = state
        self.fn = getattr(RS, fn) if isinstance(fn, str) else fn
        self.outtype = outtype
        self.args = tuple(tensor.as_tensor(arg) for arg in args)
        self.inplace = kwargs.pop('inplace', False)
        if self.inplace:
            self.destroy_map = {0: [0]}

    


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
                shape = tensor.TensorConstant(type = tensor.lvector, data = shape)
            else:
                shape = tensor.as_tensor(shape)
            ndim = tensor.get_vector_length(shape)
            if ndim is None:
                raise ValueError('Cannot infer the number of dimensions from the shape argument.')
        # note: rf should probably be cached for future use
        rf = RandomFunction(fn, tensor.Tensor(dtype = dtype, broadcastable = (False,)*ndim), *rfargs, **rfkwargs)
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

compile.optdb.register('random_make_inplace', opt.in2out(random_make_inplace), 99, 'fast_run', 'inplace')



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


class RModule(compile.FancyModule):

    def __init__(self, components = {}, **kwcomponents):
        super(RModule, self).__init__(components, **kwcomponents)
        self.random = RandomKit('rkit')
        self._components['_rkit'] = compile.KitComponent(self.random)

    def __wrapper__(self, x):
        x = compile.module.wrap(x)
        if isinstance(x, compile.Method):
            x.kits += [self.random]
        return x

    def _instance_seed(self, inst, seed, recursive = True):
        if recursive:
            for path, c in self.flat_components_map(True):
                if isinstance(c, RModule):
                    inst2 = inst
                    for name in path:
                        inst2 = inst2[name]
                    c._rkit.kit.distribute(seed, xrange(len(inst._rkit)), inst2._rkit)
        else:
            self._rkit.kit.distribute(seed, xrange(len(inst._rkit)), inst._rkit)
