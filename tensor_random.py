"""Random number generation for Theano graphs."""
import gof
import tensor
import numpy
import functools

class RandomState(object):
    """The Theano version of numpy.RandomState

    This class generates a sequence of L{Op} instances via the gen() and
    gen_like() methods.

    @ivar seed: an integer which determines the initial state of the L{Op}
    instances returned by gen(), gen_like()
    @type seed: int
    """
    @staticmethod
    def _fn_from_dist(dist, cache={}):
        """Return a function from a distribution description

        @param dist: identifier of a sampling distribution.
        @type dist: callable or str or tuple(str, dict)

        @param cache: The optional cache argument implements a closure, which ensures that
        multiple requests for the same sampling function will get the same
        sampling function. L{NumpyGenerator}.__hash__ depends on this.

        @type cache: dict
        """
        if callable(dist):
            return dist
        if isinstance(dist, str):
            return getattr(numpy.random.RandomState, dist)

        name, kwargs = dist
        key = (name, tuple(kwargs.items()))
        if key not in cache:
            fn = getattr(numpy.random.RandomState, name)
            fn = functools.partial(fn, **kwargs)
            cache[key] = fn
        return cache[key]

    def __init__(self, seed):
        self.seed = seed

    def gen(self, dist, shape=(), ndim=None):
        """
        @param dist: identifier of a sampling distribution. See L{_fn_from_dist}.
        @param shape: tuple

        @return: A tensor of random numbers, with given shape.
        @rtype: L{Result} (output of L{Apply} of L{NumpyGenerator} instance)
        """
        self.seed += 1
        fn = RandomState._fn_from_dist(dist)
        if isinstance(shape, tuple):
            return NumpyGenerator(self.seed-1, len(shape),fn) (shape)
        return NumpyGenerator(self.seed - 1, ndim, fn)(shape)

    def gen_like(self, dist, x):
        """
        @param dist: identifier of a sampling distribution. See L{_fn_from_dist}.
        @param x: L{Result} of type L{Tensor}

        @return: A tensor of random numbers, with the same shape as x.
        @rtype: L{Result} (output of L{Apply} of L{NumpyGenerator} instance)
        """
        self.seed += 1
        fn = RandomState._fn_from_dist(dist)
        return NumpyGenerator(self.seed-1, x.type.ndim, fn)(tensor.shape(x))

class NumpyGenerator(gof.op.Op):
    """Supply a sequence of random tensors of a given shape, from a given
    distribution.

    @param seed: initial state for instances of this L{Op}.
    @type seed: anything that numpy.random.RandomState accepts.
    @param ndim: the rank of random tensors produced by this op.
    @type ndim: non-negative integer
    @param fn: a sampling function
    @type fn: a callable that can reply to fn(numpy.RandomState(), size=<tuple>)
    """
    destroy_map = {0: [0]}

    def __init__(self, seed, ndim, fn, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        self.seed = seed
        self.ndim = ndim
        self.fn = fn
        assert numpy.random.RandomState(seed) #test the seed
        assert 'int' in str(type(ndim))
        assert callable(self.fn)

    def __eq__(self, other):
        return (type(self) is type(other))\
                and self.__class__ is NumpyGenerator \
                and self.seed == other.seed \
                and self.ndim == other.ndim \
                and self.fn == other.fn
    def __hash__(self):
        return self.seed ^ self.ndim ^ hash(self.fn)

    def make_node(self, _shape):
        #TODO: check for constant shape, and guess the broadcastable bits
        shape = tensor.convert_to_int64(_shape)
        if shape.type.ndim != 1:
            raise TypeError('shape argument was not converted to 1-d tensor', _shape)
        inputs = [gof.Value(gof.type.generic, numpy.random.RandomState(self.seed)), shape]
        outputs = [tensor.Tensor(dtype='float64', broadcastable = [False]*self.ndim).make_result()]
        return gof.Apply(op = self, inputs = inputs, outputs = outputs)

    def grad(self, inputs, grad_outputs):
        return [None]

    def perform(self, node, input_storage, output_storage):
        rng = input_storage[0]
        shape = input_storage[1]
        if self.ndim != len(shape):
            raise ValueError('shape argument %s had the wrong length (!=%i)' %
                    (shape, self.ndim) )
        output_storage[0][0] = self.fn(rng, size=shape)

def uniform(seed, template, low=0.,high=1.):
    """
    Return a multivariate uniform(low,high)
    random variable in a tensor of the same shape as template
    (template can either be a tensor or a shape tuple). Each element of the
    resulting tensor is sampled independently. low and high can
    be scalars or have the same shape as the template (or broadcastable
    to it).
    """
    return RandomState(seed).gen_like(('uniform',{'low':low,'high':high}),template)

def binomial(seed, template, n=1, p=0.5):
    """
    Return a multivariate binomial(n,p) random variable in a tensor of the same shape as template
    (template can either be a tensor or a shape tuple). Each element of the
    resulting tensor is sampled independently. low and high can
    be scalars or have the same shape as the template (or broadcastable
    to it).
    """
    return RandomState(seed).gen_like(('binomial',{'n':n,'p':p}),template)



