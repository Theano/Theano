from __future__ import absolute_import, print_function, division
from __future__ import print_function
import sys
import time
from six import iteritems
from theano.compile.pfunc import pfunc
from theano import tensor

import numpy
from six.moves import xrange

import theano.sandbox.cuda as tcn


def compare_fns(fns, input, reps=10):
    times = {}
    for implname, impl in iteritems(fns):
        try:
            print('TOPOSORT', implname)
            for i, n in enumerate(impl.maker.fgraph.toposort()):
                print(i, n)
        except Exception:
            pass
        t0 = time.time()
        for i in xrange(reps):
            impl(input)
        dt = time.time() - t0
        times[implname] = dt
    return times


def showtimes(times):
    for impl, dt in iteritems(times):
        print(impl, dt)


def cmp_sigmoids(shape):
    def numpy_sigmoid(input):
        1.0 / (1.0 + numpy.exp(-input))
    sinput = tensor.Tensor(
        dtype='float32', broadcastable=(0,) * len(shape))()
    shared_input = tcn.shared_constructor(
        numpy.random.rand(*shape),
        'shared_input')
    times = compare_fns(dict(
        numpy=numpy_sigmoid,
        theano_cpu=pfunc([sinput], 1.0 / (1.0 + tensor.exp(-sinput))),
        theano_gpu_onboard=pfunc(
            [sinput],
            [],
            updates=[(
                shared_input,
                1.0 / (1.0 + tensor.exp(-shared_input)))])),
        input=shared_input.value)
    showtimes(times)


def cmp_sigmoids_T(shape):
    def numpy_sigmoid(input):
        1.0 / (1.0 + numpy.exp(-input.T))
    sinput = tensor.Tensor(
        dtype='float32', broadcastable=(0,) * len(shape))()
    shared_input = tcn.shared_constructor(
        numpy.random.rand(*shape),
        'shared_input')
    times = compare_fns(dict(
        numpy=numpy_sigmoid,
        theano_cpu=pfunc([sinput], 1.0 / (1.0 + tensor.exp(-sinput.T))),
        theano_gpu_onboard=pfunc(
            [sinput],
            [],
            updates=[(
                shared_input,
                1.0 / (1.0 + tensor.exp(-shared_input.T)))])),
        input=shared_input.value)
    showtimes(times)

if __name__ == '__main__':
    eval(sys.argv[1])
    # cmp_sigmoids((640, 64*64)) # looks great in profiler
    # cmp_sigmoids((173, 74*49))
    # cmp_sigmoids_T((173, 74*49))
