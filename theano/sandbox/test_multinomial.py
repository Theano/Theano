import copy
import unittest
import numpy
import theano
from theano import config, function, tensor
import multinomial
from theano.compile.mode import get_default_mode, predefined_linkers
import theano.sandbox.cuda as cuda
from rng_mrg import MRG_RandomStreams


class TestMultinomial(unittest.TestCase):

    def setUp(self):
        self.mrng = MRG_RandomStreams(seed=1234)

    def test_pval_ok(self):
        # Tests that the remaning probability is assing to the last element if the
        # sum is below 1
        p_vals = numpy.asarray([[0.96, 0.01, 0.01, 0.01, 0.01]], dtype="float32")

        mult = self.mrng.multinomial(
            pvals=p_vals,
            dtype=theano.config.floatX
        ).eval()

        self.assertEqual(mult.sum(dtype="float64"), 1.)

    def test_pval_0(self):
        # Tests that the remaning probability is assing to the last element if the
        # sum is below 1
        p_vals = numpy.zeros((1, 10), dtype="float32")

        mult = self.mrng.multinomial(
            pvals=p_vals,
            dtype=theano.config.floatX
        ).eval()

        self.assertEqual(mult[0][-1], 1.)
        self.assertTrue(numpy.all(mult[0][:-1] == 0.))

    def test_pval_greater_1(self):
        # Tests that an error is trown if the sum of pvals is bigger than 1

        # This pvals is a special case where if the precise Kahan sum is use it sums
        # above 1 and if the normal sum is used it sums below 1
        p_vals = numpy.float32(1e-8) * numpy.ones(int(1e7) + 1, dtype=numpy.float32)
        p_vals[0] = numpy.float32(0.96)

        self.assertRaises(
            ValueError,
            self.mrng.multinomial(
                pvals=numpy.tile(p_vals, (1, 1)),
                dtype=theano.config.floatX
            ).eval
        )


def get_mode(gpu):
    mode = get_default_mode()
    mode = copy.copy(mode)
    if gpu:
        mode = mode.including('gpu', 'gpu_local_optimizations',
                              'local_cut_gpu_host_gpu',
                              'local_gpu_multinomial')
    if isinstance(mode.linker, theano.gof.PerformLinker):
        mode.linker = predefined_linkers['c|py']
    return mode


def run_with_c(f, gpu=False):
    mode = get_mode(gpu)
    f(mode, gpu)


def test_multinomial_0():
    # This tests the MultinomialFromUniform Op directly, not going through the
    # multinomial() call in GPU random generation.

    p = tensor.fmatrix()
    u = tensor.fvector()

    m = multinomial.MultinomialFromUniform('auto')(p, u)

    def body(mode, gpu):
        # the m*2 allows the multinomial to reuse output
        f = function([p, u], m * 2, allow_input_downcast=True, mode=mode)
        if gpu:
            assert any([type(node.op) is multinomial.GpuMultinomialFromUniform
                        for node in f.maker.fgraph.toposort()])

        # test that both first and second samples can be drawn
        assert numpy.allclose(f([[1, 0], [0, 1]], [.1, .1]),
                              [[2, 0], [0, 2]])

        # test that both second labels can be drawn
        r = f([[.2, .8], [.3, .7]], [.31, .31])
        assert numpy.allclose(r, [[0, 2], [0, 2]]), r

        # test that both first labels can be drawn
        r = f([[.2, .8], [.3, .7]], [.21, .21])
        assert numpy.allclose(r, [[0, 2], [2, 0]]), r

        # change the size to make sure output gets reallocated ok
        # and also make sure that the GPU version doesn't screw up the
        # transposed-ness
        r = f([[.2, .8]], [.25])
        assert numpy.allclose(r, [[0, 2]]), r

    run_with_c(body)
    if cuda.cuda_available:
        run_with_c(body, True)


# TODO: check a bigger example (make sure blocking on GPU is handled correctly)
def test_multinomial_large():
    # DEBUG_MODE will test this on GPU
    def body(mode, gpu):
        p = tensor.fmatrix()
        u = tensor.fvector()
        m = multinomial.MultinomialFromUniform('auto')(p, u)
        f = function([p, u], m * 2, allow_input_downcast=True, mode=mode)
        if gpu:
            assert any([type(node.op) is multinomial.GpuMultinomialFromUniform
                        for node in f.maker.fgraph.toposort()])

        pval = numpy.arange(10000 * 4, dtype='float32').reshape((10000, 4)) + 0.1
        pval = pval / pval.sum(axis=1)[:, None]
        uval = numpy.ones_like(pval[:, 0]) * 0.5
        mval = f(pval, uval)

        assert mval.shape == pval.shape
        if config.cast_policy == 'custom':
            assert mval.dtype == pval.dtype
        elif config.cast_policy == 'numpy+floatX':
            assert mval.dtype == config.floatX
        elif config.cast_policy == 'numpy':
            assert mval.dtype == 'float64'
        else:
            raise NotImplementedError(config.cast_policy)
        assert numpy.allclose(mval.sum(axis=1), 2)
        asdf = numpy.asarray([0, 0, 2, 0]) + 0 * pval
        assert numpy.allclose(mval, asdf)  # broadcast over all rows
    run_with_c(body)
    if cuda.cuda_available:
        run_with_c(body, True)


def test_multinomial_dtypes():
    p = tensor.dmatrix()
    u = tensor.dvector()
    m = multinomial.MultinomialFromUniform('auto')(p, u)
    assert m.dtype == 'float64', m.dtype

    p = tensor.fmatrix()
    u = tensor.fvector()
    m = multinomial.MultinomialFromUniform('auto')(p, u)
    assert m.dtype == 'float32', m.dtype

    p = tensor.fmatrix()
    u = tensor.fvector()
    m = multinomial.MultinomialFromUniform('float64')(p, u)
    assert m.dtype == 'float64', m.dtype


def test_gpu_opt():
    if not cuda.cuda_available:
        # Skip test if cuda_ndarray is not available.
        from nose.plugins.skip import SkipTest
        raise SkipTest('Optional package cuda not available')

    # We test the case where we put the op on the gpu when the output
    # is moved to the gpu.
    p = tensor.fmatrix()
    u = tensor.fvector()
    m = multinomial.MultinomialFromUniform('auto')(p, u)
    assert m.dtype == 'float32', m.dtype
    m_gpu = cuda.gpu_from_host(m)

    f = function([p, u], m_gpu, allow_input_downcast=True, mode=get_mode(True))
    assert any([type(node.op) is multinomial.GpuMultinomialFromUniform
                for node in f.maker.fgraph.toposort()])
    pval = numpy.arange(10000 * 4, dtype='float32').reshape((10000, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    mval = f(pval, uval)

    # Test with a row, it was failing in the past.
    r = tensor.frow()
    m = multinomial.MultinomialFromUniform('auto')(r, u)
    assert m.dtype == 'float32', m.dtype
    m_gpu = cuda.gpu_from_host(m)

    f = function([r, u], m_gpu, allow_input_downcast=True, mode=get_mode(True))
    assert any([type(node.op) is multinomial.GpuMultinomialFromUniform
                for node in f.maker.fgraph.toposort()])
    pval = numpy.arange(1 * 4, dtype='float32').reshape((1, 4)) + 0.1
    pval = pval / pval.sum(axis=1)[:, None]
    uval = numpy.ones_like(pval[:, 0]) * 0.5
    mval2 = f(pval, uval)
