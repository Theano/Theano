from __future__ import absolute_import, print_function, division
import numpy

import theano
from theano import tensor
from theano.compile import DeepCopyOp
from theano.tensor.tests import test_subtensor

from ..basic_ops import HostFromGpu, GpuFromHost
from ..subtensor import (GpuIncSubtensor, GpuSubtensor,
                         GpuAdvancedSubtensor1,
                         GpuAdvancedIncSubtensor1)
from ..type import gpuarray_shared_constructor

from .config import mode_with_gpu


class G_subtensor(test_subtensor.T_subtensor):
    def shortDescription(self):
        return None

    def __init__(self, name):
        test_subtensor.T_subtensor.__init__(
            self, name,
            shared=gpuarray_shared_constructor,
            sub=GpuSubtensor,
            inc_sub=GpuIncSubtensor,
            adv_sub1=GpuAdvancedSubtensor1,
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
        shared = gpuarray_shared_constructor
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
