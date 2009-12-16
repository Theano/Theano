import sys, time
from theano.compile.sandbox.sharedvalue import shared
from theano.compile.sandbox.pfunc import pfunc
from theano import tensor
import theano
import numpy

# Skip test if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
try:
    import cuda_ndarray
except ImportError:
    raise SkipTest('Optional package cuda_ndarray not available')

import theano.sandbox.cuda as cuda


def test_no_shared_var_graph():
    """Test that the InputToGpuOptimizer optimizer make graph that don't have shared variable compiled too.
    """
    a=tensor.fmatrix()
    b=tensor.fmatrix()
    f = theano.function([a,b],[a+b])
    l = f.maker.env.toposort()
    assert len(l)==4
    assert any(isinstance(x.op,cuda.GpuElemwise) for x in l)
    assert any(isinstance(x.op,cuda.GpuFromHost) for x in l)
    assert any(isinstance(x.op,cuda.HostFromGpu) for x in l)
