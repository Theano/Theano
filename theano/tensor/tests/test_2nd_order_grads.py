"""
Test for jacobian/hessian functions in Theano
"""
import unittest
from theano.tests  import unittest_tools as utt
from theano import function
import theano
from theano import tensor
import numpy


def test_jacobian():
    x = tensor.vector()
    y = x * 2
    Jx = tensor.jacobian(y, x)
    f = theano.function([x], Jx, allow_input_downcast=True)
    vx = numpy.arange(10).astype(theano.config.floatX)
    assert numpy.allclose(f(vx), numpy.eye(10) * 2)


def test_hessian():
    x = tensor.vector()
    y = tensor.sum(x ** 2)
    Hx = tensor.hessian(y, x)
    f = theano.function([x], Hx, allow_input_downcast=True)
    vx = numpy.arange(10).astype(theano.config.floatX)
    assert numpy.allclose(f(vx), numpy.eye(10) * 2)
