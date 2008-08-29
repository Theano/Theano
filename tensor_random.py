"""Random number generation for Theano graphs."""
import gof
import tensor
import numpy
import functools

#from compile import State
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
        self.args = map(tensor.as_tensor, args)
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


uniform = random_function(numpy.random.RandomState.uniform, 'float64', 0.0, 1.0)



# T = tensor
# import compile

# x, y = T.matrices('xy')
# r = gof.generic()
# shp = T.make_lvector(2, 2, 2)
# r2, z = uniform(r, shp, x, y)
# f = compile.function([r, x, y], [z])

# print f(numpy.random.RandomState(1000), [[-1, -1], [-10, -10]], [[10, 1], [10, 1]])



@gof.local_optimizer
def random_make_inplace(node):
    op = node.op
    if isinstance(op, RandomFunction) and not op.inplace:
        return RandomFunction(op.fn, op.outtype, *op.args, **dict(inplace=True)).make_node(*node.inputs).outputs


# class RandomState(StateCollection):
    
#     def __init__(self, name = None):
#         self.states = []
#         self.name = name

#     def gen(self, op, *args, **kwargs):
#         r = gof.Generic()
#         new_r, out = op(*args, **kwargs)
#         state = State(r, new_r)
#         self.states.append(state)
#         return out

#     def make_states(self, init):
#         return [Container(numpy.random.RandomState(0)) for state in self.states]








































# class RandomState(object):
#     """The Theano version of numpy.RandomState

#     This class generates a sequence of L{Op} instances via the gen() and
#     gen_like() methods.

#     @ivar seed: an integer which determines the initial state of the L{Op}
#     instances returned by gen(), gen_like()
#     @type seed: int
#     """

#     def __init__(self, seed):
#         self.seed = seed

#     def gen(self, dist, shape=(), ndim=None):
#         """
#         @param dist: identifier of a sampling distribution. See L{_fn_from_dist}.
#         @param shape: tuple

#         @return: A tensor of random numbers, with given shape.
#         @rtype: L{Result} (output of L{Apply} of L{NumpyGenerator} instance)
#         """
#         self.seed += 1
#         fn = RandomState._fn_from_dist(dist)
#         if isinstance(shape, tuple):
#             return NumpyGenerator(self.seed-1, len(shape),fn) (shape)
#         return NumpyGenerator(self.seed - 1, ndim, fn)(shape)

#     def gen_like(self, dist, x):
#         """
#         @param dist: identifier of a sampling distribution. See L{_fn_from_dist}.
#         @param x: L{Result} of type L{Tensor}

#         @return: A tensor of random numbers, with the same shape as x.
#         @rtype: L{Result} (output of L{Apply} of L{NumpyGenerator} instance)
#         """
#         self.seed += 1
#         fn = RandomState._fn_from_dist(dist)
#         return NumpyGenerator(self.seed-1, x.type.ndim, fn)(tensor.shape(x))

#     def uniform_like(self, template, low=0.,high=1.):
#         """
#         Return a multivariate uniform(low,high)
#         random variable in a tensor of the same shape as template
#         (template can either be a tensor or a shape tuple). Each element of the
#         resulting tensor is sampled independently. low and high can
#         be scalars or have the same shape as the template (or broadcastable
#         to it).
#         """
#         return self.gen_like(('uniform',{'low':low,'high':high}),template)

#     def binomial_like(self, template, n=1, p=0.5):
#         """
#         Return a multivariate binomial(n,p) random variable in a tensor of the same shape as template
#         (template can either be a tensor or a shape tuple). Each element of the
#         resulting tensor is sampled independently. low and high can
#         be scalars or have the same shape as the template (or broadcastable
#         to it).
#         """
#         return self.gen_like(('binomial',{'n':n,'p':p}),template)

#     @staticmethod
#     def _fn_from_dist(dist, cache={}):
#         """Return a function from a distribution description

#         @param dist: identifier of a sampling distribution.
#         @type dist: callable or str or tuple(str, dict)

#         @param cache: The optional cache argument implements a closure, which ensures that
#         multiple requests for the same sampling function will get the same
#         sampling function. L{NumpyGenerator}.__hash__ depends on this.

#         @type cache: dict
#         """
#         if callable(dist):
#             return dist
#         if isinstance(dist, str):
#             return getattr(numpy.random.RandomState, dist)

#         name, kwargs = dist
#         key = (name, tuple(kwargs.items()))
#         if key not in cache:
#             fn = getattr(numpy.random.RandomState, name)
#             fn = functools.partial(fn, **kwargs)
#             cache[key] = fn
#         return cache[key]


# class NumpyGenerator(gof.op.Op):
#     """Supply a sequence of random tensors of a given shape, from a given
#     distribution.

#     @param seed: initial state for instances of this L{Op}.
#     @type seed: anything that numpy.random.RandomState accepts.
#     @param ndim: the rank of random tensors produced by this op.
#     @type ndim: non-negative integer
#     @param fn: a sampling function
#     @type fn: a callable that can reply to fn(numpy.RandomState(), size=<tuple>)
#     """
#     destroy_map = {0: [0]}

#     def __init__(self, seed, ndim, fn, **kwargs):
#         gof.op.Op.__init__(self, **kwargs)
#         self.seed = seed
#         self.ndim = ndim
#         self.fn = fn
#         assert numpy.random.RandomState(seed) #test the seed
#         assert 'int' in str(type(ndim))
#         assert callable(self.fn)

#     def __eq__(self, other):
#         return (type(self) is type(other))\
#                 and self.__class__ is NumpyGenerator \
#                 and self.seed == other.seed \
#                 and self.ndim == other.ndim \
#                 and self.fn == other.fn
#     def __hash__(self):
#         return self.seed ^ self.ndim ^ hash(self.fn)

#     def make_node(self, _shape):
#         #TODO: check for constant shape, and guess the broadcastable bits
#         shape = tensor.convert_to_int64(_shape)
#         if shape.type.ndim != 1:
#             raise TypeError('shape argument was not converted to 1-d tensor', _shape)

#         # we generate one random number with the distribution to determine what dtype to expect
#         output_dtype = str(self.fn(numpy.random.RandomState(18), size=(1,)).dtype)

#         inputs = [gof.Value(gof.type.generic, numpy.random.RandomState(self.seed)), shape]
#         outputs = [tensor.Tensor(dtype=output_dtype, broadcastable = [False]*self.ndim).make_result()]
#         return gof.Apply(op = self, inputs = inputs, outputs = outputs)

#     def grad(self, inputs, grad_outputs):
#         return [None, None]

#     def perform(self, node, input_storage, output_storage):
#         rng = input_storage[0]
#         shape = input_storage[1]
#         if self.ndim != len(shape):
#             raise ValueError('shape argument %s had the wrong length (!=%i)' %
#                     (shape, self.ndim) )
#         output_storage[0][0] = self.fn(rng, size=shape)

