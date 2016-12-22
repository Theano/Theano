from __future__ import absolute_import, division, print_function

import unittest
import numpy
import theano

from theano.tests import unittest_tools as utt
from .config import mode_with_gpu

# Skip tests if cuda_ndarray is not available.
from nose.plugins.skip import SkipTest
from theano.gpuarray.linalg import (cusolver_available, gpu_solve)

if not cusolver_available:
    raise SkipTest('Optional package scikits.cuda.cusolver not available')


class TestCusolver(unittest.TestCase):
    def run_gpu_solve(self, A_val, x_val):
        b_val = numpy.dot(A_val, x_val)
        b_val_trans = numpy.dot(A_val.T, x_val)

        A = theano.tensor.matrix("A", dtype="float32")
        b = theano.tensor.matrix("b", dtype="float32")
        b_trans = theano.tensor.matrix("b", dtype="float32")

        solver = gpu_solve(A, b)
        solver_trans = gpu_solve(A, b_trans, trans='T')
        fn = theano.function([A, b, b_trans], [solver, solver_trans], mode=mode_with_gpu)
        res = fn(A_val, b_val, b_val_trans)
        x_res = numpy.array(res[0])
        x_res_trans = numpy.array(res[1])
        utt.assert_allclose(x_res, x_val)
        utt.assert_allclose(x_res_trans, x_val)

    def test_diag_solve(self):
        numpy.random.seed(1)
        A_val = numpy.asarray([[2, 0, 0], [0, 1, 0], [0, 0, 1]],
                              dtype="float32")
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_val, x_val)

    def test_sym_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_sym = (A_val + A_val.T) / 2.0
        x_val = numpy.random.uniform(-0.4, 0.4, (A_val.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_sym, x_val)

    def test_orth_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        A_orth = numpy.linalg.svd(A_val)[0]
        x_val = numpy.random.uniform(-0.4, 0.4, (A_orth.shape[1],
                                     1)).astype("float32")
        self.run_gpu_solve(A_orth, x_val)

    def test_uni_rand_solve(self):
        numpy.random.seed(1)
        A_val = numpy.random.uniform(-0.4, 0.4, (5, 5)).astype("float32")
        x_val = numpy.random.uniform(-0.4, 0.4,
                                     (A_val.shape[1], 4)).astype("float32")
        self.run_gpu_solve(A_val, x_val)
