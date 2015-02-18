#! /usr/bin/env python
"""
This file compare the runtime of two independent dot products on one
and two GPU to measure the speedup.

This should be 2x if the GPUs are equivalent.
"""
import time

import numpy

import pygpu
import theano
from theano.sandbox.gpuarray.type import (reg_context,
                                          gpuarray_shared_constructor as shared)
from theano.sandbox.gpuarray.blas import gpu_dot22

def main(dev1, dev2):
    ctx1 = pygpu.init(dev1)
    ctx2 = pygpu.init(dev2)
    print "ctx1", ctx1.devname
    print "ctx2", ctx2.devname

    reg_context('ctx1', ctx1)
    reg_context('ctx2', ctx2)

    val1a = shared(numpy.random.randn(1024, 1024).astype('float32'), context='ctx1')
    val1b = shared(numpy.random.randn(1024, 1024).astype('float32'), context='ctx1')
    val1c = shared(numpy.random.randn(1024, 1024).astype('float32'), context='ctx1')
    val1d = shared(numpy.random.randn(1024, 1024).astype('float32'), context='ctx1')

    val2a = shared(numpy.random.randn(1024, 1024).astype('float32'), context='ctx2')
    val2b = shared(numpy.random.randn(1024, 1024).astype('float32'), context='ctx2')

    f1 = theano.function([], [gpu_dot22(val1a, val1b), gpu_dot22(val1c, val1d)])
    f2 = theano.function([], [gpu_dot22(val1a, val1b), gpu_dot22(val2a, val2b)])

    f1()
    t = time.time()
    f1()
    t2 = time.time()

    print "one ctx", t2 - t

    f2()
    t = time.time()
    f2()
    t2 = time.time()

    print "two ctx", t2 - t

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        raise ValueError("This script require two device names.")
    main(sys.argv[1], sys.argv[2])
