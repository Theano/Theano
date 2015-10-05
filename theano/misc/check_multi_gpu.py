#! /usr/bin/env python
"""
This file compare the runtime of two independent dot products on one
and two GPU to measure the speedup.

This should be 2x if the GPUs are equivalent.
"""
import time

import numpy

import theano
from theano.sandbox.gpuarray import init_dev
from theano.sandbox.gpuarray.type import gpuarray_shared_constructor as shared
from theano.sandbox.gpuarray.blas import gpu_dot22


def main(dev1, dev2):
    init_dev(dev1, 'ctx1')
    init_dev(dev2, 'ctx2')

    val1a = shared(numpy.random.randn(1024, 1024).astype('float32'),
                   context_name='ctx1')
    val1b = shared(numpy.random.randn(1024, 1024).astype('float32'),
                   context_name='ctx1')
    val1c = shared(numpy.random.randn(1024, 1024).astype('float32'),
                   context_name='ctx1')
    val1d = shared(numpy.random.randn(1024, 1024).astype('float32'),
                   context_name='ctx1')

    val2a = shared(numpy.random.randn(1024, 1024).astype('float32'),
                   context_name='ctx2')
    val2b = shared(numpy.random.randn(1024, 1024).astype('float32'),
                   context_name='ctx2')

    f1 = theano.function([], [gpu_dot22(val1a, val1b),
                              gpu_dot22(val1c, val1d)])
    f2 = theano.function([], [gpu_dot22(val1a, val1b),
                              gpu_dot22(val2a, val2b)])

    r = f1()
    r[0].sync(), r[1].sync()
    r = None
    t = time.time()
    r = f1()
    r[0].sync(), r[1].sync()
    t2 = time.time()
    r = None

    print("one ctx %f" % (t2 - t,))

    r = f2()
    r[0].sync(), r[1].sync()
    r = None
    t = time.time()
    r = f2()
    r[0].sync(), r[1].sync()
    t2 = time.time()
    r = None

    print("two ctx %f" % (t2 - t,))

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        raise ValueError("This script require two device names.")
    main(sys.argv[1], sys.argv[2])
