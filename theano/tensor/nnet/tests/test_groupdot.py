from nose.plugins.skip import SkipTest
import numpy
import numpy as np
import __builtin__
import theano
from theano.gof.python25 import any
import theano.tensor as T
import theano.tests.unittest_tools as utt

def test_groupdot():
    x = T.fmatrix('x')
    w = T.tensor3('w',dtype='float32')
    b = T.fmatrix('b')
    c = T.vector('c',dtype='int32')
    z = T.nnet.GroupDot(51)(x, w, b, c)

    f = theano.function([x, w, b, c], z, name='cpu')

    def cmp(n_batch, n_hid, n_clust, n_classes):
        x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
        w = np.random.rand(n_clust, n_hid, n_classes).astype('float32')
        b = np.random.rand(n_clust, n_classes).astype('float32')
        c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')

        output = numpy.zeros(shape=(n_batch, n_classes))
        for i in range(n_batch):
            output[i] = np.dot(x[i, :], w[c[i], :, :]) + b[c[i]]
        out=f(x, w, b, c)
        assert numpy.allclose(out, output)

    cmp(50, 300, 20, 7000)
    cmp(100, 256, 51, 10000)


def verify_groupdotgrad(dims):
    n_clust, n_batch, n_hid, n_classes = dims
    c = np.random.randint(0, n_clust, size=(n_batch,)).astype('int32')

    def op_with_fixed_c(x, w, b):
        return T.nnet.GroupDot(n_clust)(x, w, b, c)

    x = numpy.arange(n_batch * n_hid, dtype='float32').reshape(n_batch, n_hid)
    w = np.random.rand(n_clust, n_hid, n_classes).astype('float32')
    b = np.random.rand(n_clust, n_classes).astype('float32')

    utt.verify_grad(op_with_fixed_c, [x, w, b], eps=1e-2)


def test_groupdotgrad():
    for dims in [(2, 5, 10, 15), (15, 11, 45, 25)]:
        yield verify_groupdotgrad, dims
