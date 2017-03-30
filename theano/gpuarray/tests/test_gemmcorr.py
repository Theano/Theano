from __future__ import absolute_import, print_function, division
import unittest
import numpy as np

import theano
from theano import config
from theano.tests import unittest_tools as utt

from theano.tensor.nnet.corr import CorrMM, CorrMM_gradWeights, CorrMM_gradInputs

from ..type import gpuarray_shared_constructor
from ..blas import GpuCorrMM, GpuCorrMM_gradWeights, GpuCorrMM_gradInputs
from .config import mode_with_gpu, mode_without_gpu, ref_cast


class TestCorrMM(unittest.TestCase):

    def run_conv_valid(self, inputs_shape, filters_shape,
                       border_mode='valid',
                       filter_dilation=(1, 1),
                       subsample=(1, 1),
                       verify_grad=False):
        inputs_shape = [inputs_shape[i] for i in (0, 3, 1, 2)]
        filters_shape = [filters_shape[i] for i in (0, 3, 1, 2)]

        inputs_val = np.random.random(inputs_shape).astype(config.floatX)
        filters_val = np.random.random(filters_shape).astype(config.floatX)

        inputs = gpuarray_shared_constructor(inputs_val)
        filters = gpuarray_shared_constructor(filters_val)

        conv_ref = CorrMM(border_mode=border_mode,
                          filter_dilation=filter_dilation,
                          subsample=subsample)(ref_cast(inputs),
                                               ref_cast(filters))
        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)

        conv = GpuCorrMM(border_mode=border_mode,
                         filter_dilation=filter_dilation,
                         subsample=subsample)(inputs, filters)
        f = theano.function([], conv, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res)

        if verify_grad:
            utt.verify_grad(GpuCorrMM(border_mode=border_mode,
                                      filter_dilation=filter_dilation,
                                      subsample=subsample),
                            [inputs_val, filters_val])

    def test_valid(self):
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            subsample=(2, 2))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            subsample=(2, 2))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            subsample=(3, 3))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            subsample=(3, 3))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            subsample=(3, 2))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            subsample=(1, 2))

    def test_border_mode(self):
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            border_mode='valid')
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            border_mode='half')
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            border_mode='full')
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            border_mode=(0, 0))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            border_mode=(1, 2))
        self.run_conv_valid(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            border_mode=(3, 2))

    def test_filter_dilation(self):
        inputs_shape = [16, 20, 12, 1]
        filters_shape = [10, 6, 5, 1]

        for filter_dilation in [(2, 1), (1, 2)]:
            for border_mode in ['valid', 'half', 'full']:
                self.run_conv_valid(inputs_shape=inputs_shape,
                                    filters_shape=filters_shape,
                                    filter_dilation=filter_dilation,
                                    border_mode=border_mode)

    def test_verify_gradients(self):
        # use a small example to check the gradients
        inputs_shape = [2, 7, 9, 1]
        filters_shape = [1, 3, 3, 1]

        for filter_dilation in [(2, 1), (1, 2)]:
            for border_mode in ['valid', 'half', 'full', (2, 1)]:
                self.run_conv_valid(inputs_shape=inputs_shape,
                                    filters_shape=filters_shape,
                                    filter_dilation=filter_dilation,
                                    border_mode=border_mode,
                                    verify_grad=True)

    def run_gradweight(self, inputs_shape, filters_shape, dCdH_shape,
                       subsample=(1, 1)):
        inputs_shape = [inputs_shape[i] for i in (0, 3, 1, 2)]
        filters_shape = [filters_shape[i] for i in (0, 3, 1, 2)]
        dCdH_shape = [dCdH_shape[i] for i in (0, 3, 1, 2)]

        inputs_val = np.random.random(inputs_shape).astype(config.floatX)
        dCdH_val = np.random.random(dCdH_shape).astype(config.floatX)
        inputs = gpuarray_shared_constructor(inputs_val)
        dCdH = gpuarray_shared_constructor(dCdH_val)
        shape = gpuarray_shared_constructor(np.array(filters_shape[2:]))

        if (subsample == (1, 1)):
            conv_ref = CorrMM_gradWeights(subsample=subsample)(
                ref_cast(inputs), ref_cast(dCdH))
            conv_gemm = GpuCorrMM_gradWeights(subsample=subsample)(
                inputs, dCdH)
        else:
            conv_ref = CorrMM_gradWeights(subsample=subsample)(
                ref_cast(inputs), ref_cast(dCdH), shape=shape)
            conv_gemm = GpuCorrMM_gradWeights(subsample=subsample)(
                inputs, dCdH, shape=shape)

        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
        f = theano.function([], conv_gemm, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res)

    def test_gradweight(self):
        self.run_gradweight(inputs_shape=(16, 10, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            dCdH_shape=(16, 5, 1, 10),
                            subsample=(1, 1))
        self.run_gradweight(inputs_shape=(16, 20, 10, 1),
                            filters_shape=(10, 6, 4, 1),
                            dCdH_shape=(16, 8, 4, 10),
                            subsample=(2, 2))
        self.run_gradweight(inputs_shape=(16, 20, 10, 1),
                            filters_shape=(10, 6, 3, 1),
                            dCdH_shape=(16, 5, 3, 10),
                            subsample=(3, 3))
        self.run_gradweight(inputs_shape=(16, 20, 12, 1),
                            filters_shape=(10, 6, 12, 1),
                            dCdH_shape=(16, 8, 1, 10),
                            subsample=(2, 1))

    def run_gradinput(self, inputs_shape, filters_shape,
                      subsample=(1, 1)):
        inputs_shape = [inputs_shape[i] for i in (0, 3, 1, 2)]
        filters_shape = [filters_shape[i] for i in (0, 3, 1, 2)]

        inputs_val = np.random.random(inputs_shape).astype(config.floatX)
        filters_val = np.random.random(filters_shape).astype(config.floatX)
        inputs = gpuarray_shared_constructor(inputs_val)
        filters = gpuarray_shared_constructor(filters_val)

        bottom_height = (inputs_shape[2] - 1) * subsample[0] + filters_shape[2]
        bottom_width = (inputs_shape[3] - 1) * subsample[1] + filters_shape[3]
        bottom_shape = gpuarray_shared_constructor(np.array([bottom_height, bottom_width]))

        if (subsample == (1, 1)):
            conv_ref = CorrMM_gradInputs(subsample=subsample)(
                kern=ref_cast(filters), topgrad=ref_cast(inputs))
            conv_gemm = GpuCorrMM_gradInputs(subsample=subsample)(
                kern=filters, topgrad=inputs)
        else:
            conv_ref = CorrMM_gradInputs(subsample=subsample)(
                kern=ref_cast(filters), topgrad=ref_cast(inputs),
                shape=bottom_shape)
            conv_gemm = GpuCorrMM_gradInputs(subsample=subsample)(
                kern=filters, topgrad=inputs, shape=bottom_shape)

        f_ref = theano.function([], conv_ref, mode=mode_without_gpu)
        f = theano.function([], conv_gemm, mode=mode_with_gpu)

        res_ref = f_ref()
        res = f()
        utt.assert_allclose(res_ref, res)

    def test_gradinput(self):
        self.run_gradinput(inputs_shape=(16, 15, 12, 10),
                           filters_shape=(10, 6, 12, 1))
        self.run_gradinput(inputs_shape=(16, 15, 12, 10),
                           filters_shape=(10, 6, 12, 1),
                           subsample=(2, 2))
        self.run_gradinput(inputs_shape=(16, 15, 12, 10),
                           filters_shape=(10, 6, 12, 1),
                           subsample=(3, 3))
        self.run_gradinput(inputs_shape=(16, 15, 12, 10),
                           filters_shape=(10, 6, 12, 1),
                           subsample=(3, 1))

    def test_large_input(self):
        # This tests the number-of-threads computation
        # by making (channels * height) > (max_threads_dim ** 2).
        # (See also issue #5165.)
        self.run_conv_valid(inputs_shape=(1, 1024, 3, 1024),
                            filters_shape=(1, 1, 1, 1024),
                            verify_grad=False)
        self.run_gradinput(inputs_shape=(1, 1024, 3, 1),
                           filters_shape=(1, 1, 1, 1024))
