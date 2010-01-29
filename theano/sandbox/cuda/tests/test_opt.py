import sys, time
from theano.compile.sharedvalue import shared
from theano.compile.pfunc import pfunc
from theano import tensor
import theano
import numpy

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
import theano.sandbox.cuda as cuda_ndarray
if cuda_ndarray.cuda_available == False:
    raise SkipTest('Optional package cuda disabled')

import theano.compile.mode

mode_with_gpu = theano.compile.mode.get_default_mode().including('gpu')

import theano.sandbox.cuda as cuda


def test_no_shared_var_graph():
    """Test that the InputToGpuOptimizer optimizer make graph that don't have shared variable compiled too.
    """
    a=tensor.fmatrix()
    b=tensor.fmatrix()
    f = theano.function([a,b],[a+b], mode=mode_with_gpu)
    l = f.maker.env.toposort()
    assert len(l)==4
    assert numpy.any(isinstance(x.op,cuda.GpuElemwise) for x in l)
    assert numpy.any(isinstance(x.op,cuda.GpuFromHost) for x in l)
    assert numpy.any(isinstance(x.op,cuda.HostFromGpu) for x in l)
