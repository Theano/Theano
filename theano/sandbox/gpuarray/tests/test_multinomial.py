import numpy

import theano
from theano import tensor
from theano.sandbox.gpuarray.multinomial import GPUAMultinomialFromUniform
from .config import mode_with_gpu

def test_multinomial0():
    # This tests the MultinomialFromUniform Op directly, not going through the
    # multinomial() call in GPU random generation.

    p = tensor.fmatrix()
    u = tensor.fvector()

    m = GPUAMultinomialFromUniform()(p, u)

    f = theano.function([p, u], m, mode=mode_with_gpu)

    assert f(numpy.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]), numpy.array([0.05, 0.05]))