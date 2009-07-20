from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor

import numpy

import gputensor as gpt


def test0():

    a = gpt.gpu_tensor_shared_constructor(numpy.random.rand(3,4), 'a')

    b = tensor.dmatrix()

    f = pfunc([b], [], updates=[(a, a+b)])

    a0 = a.value * 1.0
    f(numpy.ones((3,4)))
    print f.maker.env.toposort()

    assert numpy.all(a0 + 1.0 == a.value)

def test1():

    a = gpt.gpu_tensor_shared_constructor(numpy.random.rand(3,4), 'a')

    b = tensor.dmatrix()

    f = pfunc([b], [], updates=[(a, a+b)])
    for i, node in enumerate( f.maker.env.toposort()):
        print 'test1 toposort', i, node

    a0 = a.value * 1.0
    f(numpy.ones((3,4)))

    assert numpy.all(a0 + 1.0 == a.value)

