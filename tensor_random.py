import gof
import tensor
import numpy
import functools

def fn_from_dist(dist):
    if callable(dist):
        return dist
    if isinstance(dist, str):
        return getattr(numpy.random.RandomState, dist)

    name, kwargs = dist
    fn = getattr(numpy.random.RandomState, name)
    fn = functools.partial(fn, **kwargs)
    return fn

class RandomState(object):
    def __init__(self, seed):
        self.seed = seed

    def uniform(self, shape, ndim=None):
        return self.gen('uniform', shape, ndim)

    def uniform_like(self, x):
        return self.gen_like('uniform', x)
    
    def gen(self, dist, shape=(), ndim=None):
        self.seed += 1
        fn = fn_from_dist(dist)
        if isinstance(shape, tuple):
            return NumpyGenerator(self.seed-1, len(shape),fn) (shape)
        return NumpyGenerator(self.seed - 1, ndim, fn)(shape)

    def gen_like(self, dist, x):
        self.seed += 1
        fn = fn_from_dist(dist)
        return NumpyGenerator(self.seed-1, x.type.ndim, fn)(tensor.shape(x))

class NumpyGenerator(gof.op.Op):
    destroy_map = {0: [0]}

    def __init__(self, seed, ndim, fn, **kwargs):
        gof.op.Op.__init__(self, **kwargs)
        self.seed = seed
        self.ndim = ndim
        self.fn = fn

    def make_node(self, _shape):
        #TODO: check for constant shape, and guess the broadcastable bits
        shape = tensor.convert_to_int64(_shape)
        if shape.type.ndim != 1:
            raise TypeError('shape argument was not converted to 1-d tensor', _shape)
        inputs = [gof.Value(gof.type.generic, numpy.random.RandomState(self.seed)), shape]
        outputs = [gof.Result(tensor.Tensor(dtype='float64', broadcastable =
            [False]*self.ndim))]
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


