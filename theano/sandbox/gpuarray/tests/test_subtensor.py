import numpy

import theano
# Avoid repeating the T_subtensor tests by not directly importing the class
from theano.tensor.tests import test_subtensor

from ..basic_ops import HostFromGpu, GpuFromHost
from ..subtensor import (GpuIncSubtensor, GpuSubtensor,
                         GpuAdvancedIncSubtensor1)

from ..type import gpuarray_shared_constructor

from .test_basic_ops import mode_with_gpu, fake_shared, fake_type

from theano.compile import DeepCopyOp

from theano import tensor


class G_subtensor(test_subtensor.T_subtensor):
    def shortDescription(self):
        return None

    def __init__(self, name):
        test_subtensor.T_subtensor.__init__(
            self, name,
            shared=fake_shared,
            type=fake_type,
            sub=GpuSubtensor,
            inc_sub=GpuIncSubtensor,
            adv_incsub1=GpuAdvancedIncSubtensor1,
            mode=mode_with_gpu,
            # avoid errors with limited devices
            dtype='float32',
            ignore_topo=(HostFromGpu, GpuFromHost,
                         DeepCopyOp))
        # GPU opt can't run in fast_compile only.
        self.fast_compile = False
        assert self.sub == GpuSubtensor


def test_advinc_subtensor1():
    """ Test the second case in the opt local_gpu_advanced_incsubtensor1 """
    for shp in [(3, 3), (3, 3, 3)]:
        shared = fake_shared
        xval = numpy.arange(numpy.prod(shp), dtype='float32').reshape(shp) + 1
        yval = numpy.empty((2,) + shp[1:], dtype='float32')
        yval[:] = 10
        x = shared(xval, name='x')
        y = tensor.tensor(dtype='float32',
                     broadcastable=(False,) * len(shp),
                     name='y')
        expr = tensor.advanced_inc_subtensor1(x, y, [0, 2])
        f = theano.function([y], expr, mode=mode_with_gpu)
        assert sum([isinstance(node.op, GpuAdvancedIncSubtensor1)
                    for node in f.maker.fgraph.toposort()]) == 1
        rval = f(yval)
        rep = xval.copy()
        rep[[0, 2]] += yval
        assert numpy.allclose(rval, rep)
