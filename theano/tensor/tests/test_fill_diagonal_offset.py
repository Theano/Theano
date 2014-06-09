import numpy as np
import numpy
import unittest
import pdb
import theano
from theano.tests import unittest_tools as utt

from theano.tensor.extra_ops import (CumsumOp, cumsum, CumprodOp, cumprod,
                                     BinCountOp, bincount, DiffOp, diff,
                                     squeeze, RepeatOp, repeat, Bartlett, bartlett,
                                     FillDiagonal, fill_diagonal, FillDiagonalOffset, 
                                     fill_diagonal_offset)
from theano import tensor as T
from theano import config, tensor, function


numpy_ver = [int(n) for n in numpy.__version__.split('.')[:2]]
numpy_16 = bool(numpy_ver >= [1, 6])


class TestFillDiagonalOffset(utt.InferShapeTester):

    rng = numpy.random.RandomState(43)

    def setUp(self):
        super(TestFillDiagonalOffset, self).setUp()
        self.op_class = FillDiagonalOffset
        self.op = fill_diagonal_offset

    def test_perform(self):
        x = tensor.matrix()
        y = tensor.scalar()
        z = tensor.scalar()

        test_offset = numpy.array(numpy.random.randint(0,5),
        	dtype = config.floatX)

        f = function([x, y, z], fill_diagonal_offset(x, y, z))
        for shp in [(8, 8), (5, 8), (8, 5)]:
            a = numpy.random.rand(*shp).astype(config.floatX)
            val = numpy.cast[config.floatX](numpy.random.rand())
            out = f(a, val, test_offset)
            # We can't use numpy.fill_diagonal as it is bugged.
            assert numpy.allclose(numpy.diag(out, test_offset), val)
            pdb.set_trace()
            assert (out == val).sum() == min(a.shape)

    def test_gradient(self):
    	test_offset = numpy.array(numpy.random.randint(0,5),
        				dtype = config.floatX)
        utt.verify_grad(fill_diagonal_offset, [numpy.random.rand(5, 8),
                                        numpy.random.rand(),
                                        test_offset],
                        n_tests=1, rng=TestFillDiagonalOffset.rng)
        utt.verify_grad(fill_diagonal_offset, [numpy.random.rand(8, 5),
                                        numpy.random.rand(),
                                        test_offset],
                        n_tests=1, rng=TestFillDiagonalOffset.rng)

    def test_infer_shape(self):
        x = tensor.dmatrix()
        y = tensor.dscalar()
        z = tensor.dscalar()
        test_offset = numpy.array(numpy.random.randint(0,5),
        				dtype = config.floatX)
        self._compile_and_check([x, y, z], [self.op(x, y, z)],
                                [numpy.random.rand(8, 5),
                                 numpy.random.rand(),
                                 test_offset],
                                self.op_class)

if __name__ == '__main__':
    unittest.main()