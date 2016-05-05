from __future__ import absolute_import, print_function, division
import numpy
from nose.plugins.skip import SkipTest

import theano

mode_with_opt = theano.compile.mode.get_default_mode()
mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')


def test_viewop_gpu():
    from theano.sandbox import cuda
    if cuda.cuda_available is False:
        raise SkipTest('Optional package cuda disabled')
    _x = theano.tensor.fvector('x')
    x = cuda.gpu_from_host(_x)
    _out = theano.compile.ViewOp()(x)
    out = cuda.host_from_gpu(_out)
    f = theano.function([x],
                        out,
                        mode=mode_with_gpu)
    data = numpy.array([1, 2, 3], dtype='float32')
    assert numpy.allclose(f(data), data)
