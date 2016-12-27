from __future__ import absolute_import, print_function, division

from nose.plugins.skip import SkipTest
import unittest
import numpy
import theano
import theano.tensor as tensor

import theano.sandbox.mkl as mkl
from theano.sandbox.mkl import mkl_lrn
from theano.tensor.nnet.lrn import lrn

if not mkl.mkl_available:
    raise SkipTest('Optional package MKL disabled')

if theano.config.mode == 'FAST_COMPILE':
    mode_with_mkl = theano.compile.mode.get_mode('FAST_RUN').including('mkl')
    mode_without_mkl = theano.compile.mode.get_mode('FAST_RUN').excluding('mkl')
else:
    mode_with_mkl = theano.compile.mode.get_default_mode().including('mkl')
    mode_without_mkl = theano.compile.mode.get_default_mode().excluding('mkl')


class test_mkl_lrn(unittest.TestCase):
    def ground_truth_normalizer(self, inp, k, n, alpha, beta):
        out = numpy.zeros(inp.shape)
        # for bc01
        for b in range(inp.shape[0]):
            for r in range(inp.shape[2]):
                for c in range(inp.shape[3]):
                    out[b, :, r, c] = self.ground_truth_normalize_row(row=inp[b, :, r, c],
                                                                      k=k,
                                                                      n=n,
                                                                      alpha=alpha,
                                                                      beta=beta)

        return out

    def ground_truth_normalize_row(self, row, k, n, alpha, beta):
        assert row.ndim == 1
        out = numpy.zeros(row.shape)
        alpha_n = float(alpha / n)  # not same with lasagne
        for i in range(row.shape[0]):
            s = k
            tot = 0
            for j in range(max(0, i - n // 2), min(row.shape[0], i + n // 2 + 1)):
                tot += 1
                sq = row[j] ** 2.
                assert sq > 0.
                assert s >= k
                assert alpha > 0.
                s += alpha_n * sq
                assert s >= k
            assert tot <= n
            assert s >= k
            s = s ** beta
            out[i] = row[i] / s

        return out

    def test_mkl_lrn_value(self):
        shape = [(2, 15, 3, 4), (256, 256, 27, 27)]  # NCHW
        n = 5
        k = 2
        alpha = 0.0001
        beta = 0.75

        x = tensor.dtensor4('x')
        z = lrn(x, alpha=alpha, beta=beta, k=k, n=n)
        fz = theano.function([x], z, mode=mode_with_mkl)
        # for shape[0]
        input_data = numpy.random.rand(*shape[0]).astype(theano.config.floatX)
        t = self.ground_truth_normalizer(input_data, k=k, n=n, alpha=alpha, beta=beta)
        assert (fz(input_data)).shape == t.shape
        assert numpy.allclose(fz(input_data), t)
        # for shape[1]. It is slow to compute t since the shape is large.
        # fz = theano.function([x], z, mode=mode_with_mkl)
        # input_data = numpy.random.rand(*shape[1]).astype(theano.config.floatX)
        # t = self.ground_truth_normalizer(input_data, k=k, n=n, alpha=alpha, beta=beta)
        # assert (fz(input_data)).shape == t.shape
        # assert numpy.allclose(fz(input_data), t)

    def test_lrn_wrong_dim(self):
        x = tensor.fmatrix('x')

        try:
            mkl_lrn.LRN()(x)
        except TypeError as e:
            pass
        except Exception as e:
            raise Exception('test_lrn.test_lrn_wrong_dim ' + str(e))

    def test_lrn_float32(self):
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float32'

        x = tensor.ftensor4('x')
        z = lrn(x)
        f = theano.function([x], z, mode=mode_with_mkl)
        imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)

        f(imval)
        assert f(imval).dtype == 'float32'

        theano.config.floatX = old_floatX

    def test_lrn_float64(self):
        old_floatX = theano.config.floatX
        theano.config.floatX == 'float64'

        x = tensor.dtensor4('x')
        z = lrn(x)
        f = theano.function([x], z, mode=mode_with_mkl)
        imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)

        f(imval)
        assert f(imval).dtype == 'float64'

        theano.config.floatX = old_floatX

    def test_lrn_eq(self):
        op1 = mkl_lrn.LRN(uniq_id=99)
        op2 = mkl_lrn.LRN(uniq_id=88)
        op3 = mkl_lrn.LRN(alpha=0.1, uniq_id=99)

        assert op1 == op2
        assert not op1 == op3

    def test_lrn_hash(self):
        op1 = mkl_lrn.LRN(uniq_id=99)
        op2 = mkl_lrn.LRN(uniq_id=88)
        op3 = mkl_lrn.LRN(alpha=0.1, uniq_id=99)

        assert hash(op1) == hash(op2)
        assert not hash(op1) == hash(op3)

    def test_lrn_topo(self):
        x = tensor.ftensor4('x')
        z = lrn(x)
        f = theano.function([x], z, mode=mode_with_mkl)
        inp = f.maker.fgraph.inputs
        out = f.maker.fgraph.outputs
        topo = f.maker.fgraph.toposort()

        assert len(inp) == 1
        assert len(out) == 1
        assert len(topo) == 3
        assert isinstance(topo[0].op, mkl.basic_ops.U2ILRN)
        assert isinstance(topo[1].op, mkl_lrn.LRN)
        assert isinstance(topo[2].op, mkl.basic_ops.I2U)


class test_lrn_grad(unittest.TestCase):
    def test_lrn_grad_wrong_dim(self):
        x = tensor.fmatrix('x')
        gz = tensor.fmatrix('gz')
        try:
            mkl_lrn.LRNGrad()(x, gz)
        except TypeError as e:
            pass
        except Exception as e:
            raise Exception('test_lrn_grad.test_lrn_grad_wrong_dim ' + str(e))

    def test_lrn_grad_float32(self):
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float32'

        x = tensor.ftensor4('x')
        z = lrn(x)
        zsum = tensor.sum(z)
        g = tensor.grad(zsum, [x])

        f = theano.function([x], g, mode=mode_with_mkl)
        imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)

        f(imval)
        assert f(imval)[0].dtype == 'float32'

        theano.config.floatX = old_floatX

    def test_lrn_grad_float64(self):
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float64'

        x = tensor.dtensor4('x')
        z = lrn(x)
        zsum = tensor.sum(z)
        g = tensor.grad(zsum, [x])

        f = theano.function([x], g, mode=mode_with_mkl)
        imval = numpy.random.rand(4, 2, 4, 4).astype(theano.config.floatX)

        f(imval)
        assert f(imval)[0].dtype == 'float64'

        theano.config.floatX = old_floatX

    def test_lrn_grad_eq(self):
        op1 = mkl_lrn.LRNGrad(uniq_id=99)
        op2 = mkl_lrn.LRNGrad(uniq_id=88)
        op3 = mkl_lrn.LRNGrad(alpha=0.1, uniq_id=99)

        assert op1 == op2
        assert not op1 == op3

    def test_lrn_grad_hash(self):
        op1 = mkl_lrn.LRNGrad(uniq_id=99)
        op2 = mkl_lrn.LRNGrad(uniq_id=88)
        op3 = mkl_lrn.LRNGrad(alpha=0.1, uniq_id=99)

        assert hash(op1) == hash(op2)
        assert not hash(op1) == hash(op3)


if __name__ == '__main__':
    unittest.main()
