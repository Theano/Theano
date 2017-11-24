from __future__ import absolute_import, print_function, division

import theano
import theano.tensor.tests.test_sort
from .config import mode_with_gpu
from ..sort import GpuTopKOp
from nose.plugins.skip import SkipTest


class Test_GpuTopK(theano.tensor.tests.test_sort.Test_TopK):
    mode = mode_with_gpu
    dtype = 'float32'
    op_class = GpuTopKOp

    def setUp(self):
        raise SkipTest("GpuTopK is currently disabled")
