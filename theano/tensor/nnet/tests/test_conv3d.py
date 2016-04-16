from __future__ import absolute_import, print_function, division
import unittest
import theano
import theano.tensor as T
from theano import function, shared
from theano.tests import unittest_tools as utt
from theano.tensor.nnet.ConvTransp3D import convTransp3D, ConvTransp3D
from theano.tensor.nnet.ConvGrad3D import convGrad3D, ConvGrad3D
from theano.tensor.nnet.Conv3D import conv3D, Conv3D
from theano.tests.unittest_tools import attr
import numpy as N
from six.moves import xrange
import copy
import theano.sparse
if theano.sparse.enable_sparse:
    from scipy import sparse
from nose.plugins.skip import SkipTest

floatX = theano.config.floatX

# TODO: each individual test method should seed rng with utt.fetch_seed()
#      as it is right now, setUp does the seeding, so if you run just
#      a subset of the tests they will do different things than if you
#      run all of them


class DummyConv3D:

    """A dummy version of Conv3D passed to verify_grad
    Stores a fixed stride, since stride is not differentiable
    Exposes only one scalar argument, which is used as the position
    along a parametrically defined line, with 0 being at VwbVals
    Direction of the line is chosen randomly at construction
    The reason for locking the inputs to lie on this line is so that the
    verify_grad will not need to test hundreds of variables. Disadvantage
    is we can't be certain that all of them are correct, advantange is that
    this random projection lets us test lots of variables very quickly """

    def __init__(self, rng, VWbVals, d):
        """
        param: rng    Random number generator used to pick direction of the
            line
        param: VWbVals    tuple containing values to test V,W,b around
        param: d    shared variable for d, the stride
        """

        self.V, self.W, self.b = VWbVals
        self.dV = shared(rng.uniform(-1, 1,
                                     self.V.get_value(borrow=True).shape))
        self.dW = shared(rng.uniform(-1, 1,
                                     self.W.get_value(borrow=True).shape))
        self.db = shared(rng.uniform(-1, 1,
                                     self.b.get_value(borrow=True).shape))

        self.d = d

    def __call__(self, t):
        output = conv3D(self.V + t * self.dV, self.W + t * self.dW,
                        self.b + t * self.db, self.d)

        return output


class DummyConvGrad3D:

    def __init__(self, rng, VdHvals, d, WShape):
        """
        param: rng    Random number generator used to pick direction of the
            line
        param: VWbVals    tuple containing values to test V,W,b around
        param: d    shared variable for d, the stride
        """

        self.V, self.dCdH = VdHvals
        self.dV = shared(rng.uniform(-1, 1,
                                     self.V.get_value(borrow=True).shape))
        self.ddCdH = shared(rng.uniform(-1, 1,
                                    self.dCdH.get_value(borrow=True).shape))
        self.d = d
        self.WShape = WShape

    def __call__(self, t):

        output = convGrad3D(self.V + t * self.dV, self.d, self.WShape,
                            self.dCdH + t * self.ddCdH)
        return output


class DummyConvTransp3D:

    def __init__(self, rng, WbHvals, d, RShape):
        """
        param: rng    Random number generator used to pick direction of the
            line
        param: VWbVals    tuple containing values to test V,W,b around
        param: d    shared variable for d, the stride
        """

        self.W, self.b, self.H = WbHvals
        self.dW = rng.uniform(-1, 1, self.W.get_value(borrow=True).shape)
        self.db = rng.uniform(-1, 1, self.b.get_value(borrow=True).shape)
        self.dH = rng.uniform(-1, 1, self.H.get_value(borrow=True).shape)
        self.dW, self.db = shared(self.dW), shared(self.db),
        self.dH = shared(self.dH)

        self.d = d
        self.RShape = RShape

    def __call__(self, t):
        output = convTransp3D(self.W + t * self.dW, self.b + t * self.db,
                              self.d, self.H + t * self.dH, self.RShape)

        return output


class TestConv3D(utt.InferShapeTester):

    def setUp(self):
        super(TestConv3D, self).setUp()
        utt.seed_rng()
        self.rng = N.random.RandomState(utt.fetch_seed())

        mode = copy.copy(theano.compile.mode.get_default_mode())
        mode.check_py_code = False

        self.W = shared(N.ndarray(shape=(1, 1, 1, 1, 1), dtype=floatX))
        self.W.name = 'W'
        self.b = shared(N.zeros(1, dtype=floatX))
        self.b.name = 'b'
        self.rb = shared(N.zeros(1, dtype=floatX))
        self.rb.name = 'rb'
        self.V = shared(N.ndarray(shape=(1, 1, 1, 1, 1), dtype=floatX))
        self.V.name = 'V'
        self.d = shared(N.ndarray(shape=(3, ), dtype=int))
        self.d.name = 'd'

        self.H = conv3D(self.V, self.W, self.b, self.d)
        self.H.name = 'H'
        self.H_func = function([], self.H, mode=mode)
        self.H_shape_func = function([], self.H.shape, mode=mode)

        self.RShape = T.vector(dtype='int64')
        self.RShape.name = 'RShape'

        self.otherH = T.TensorType(floatX,
                        (False, False, False, False, False))(name='otherH')
        self.transp = convTransp3D(self.W, self.rb, self.d,
                                   self.otherH, self.RShape)
        self.transp.name = 'transp'
        self.transp_func = function([self.otherH, self.RShape],
                                    self.transp, mode=mode)

        self.R = convTransp3D(self.W, self.rb, self.d, self.H, self.RShape)
        self.R.name = 'R'
        self.R_func = function([self.RShape], self.R, mode=mode)
        self.R_shape_func = function([self.RShape], self.R.shape)

        diff = self.V - self.R
        diff.name = 'diff'
        sqr = T.sqr(diff)
        sqr.name = 'sqr'
        self.reconsObj = T.sum(sqr)
        self.reconsObj.name = 'reconsObj'
        self.reconsObjFunc = function([self.RShape], self.reconsObj, mode=mode)

        W_grad = T.grad(self.reconsObj, self.W)

        self.gradientsFunc = function([self.RShape],
                        [W_grad, T.grad(self.reconsObj,
                        self.H), T.grad(self.reconsObj, self.V),
                         T.grad(self.reconsObj, self.b)], mode=mode)

        self.check_c_against_python = function([self.RShape],
                        [T.grad(self.reconsObj, self.W), T.grad(self.reconsObj,
                        self.H), T.grad(self.reconsObj, self.V),
                         T.grad(self.reconsObj, self.b)], mode='DEBUG_MODE')

        self.dCdW_shape_func = function([self.RShape],
                        T.grad(self.reconsObj, self.W).shape, mode=mode)

    def random_tensor(self, *dims):
        return N.asarray(self.rng.uniform(-.05, .05, dims), dtype=floatX)

    def randomize(self):
        batchSize = self.rng.randint(1, 4)
        videoDur = self.rng.randint(8, 30)
        filterWidth = self.rng.randint(1, 8)
        filterHeight = self.rng.randint(1, 8)
        filterDur = self.rng.randint(1, 8)

        tsteps = self.rng.randint(1, 4)
        rsteps = self.rng.randint(1, 4)
        csteps = self.rng.randint(1, 4)

        videoDur = tsteps * filterDur + self.rng.randint(0, 3)
        videoWidth = csteps * filterWidth + self.rng.randint(0, 3)
        videoHeight = rsteps * filterHeight + self.rng.randint(0, 3)

        numFilters = self.rng.randint(1, 3)
        inputChannels = self.rng.randint(1, 3)
        self.d.get_value(borrow=True, return_internal_type=True)[0] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[1] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[2] = \
            self.rng.randint(1, 15)

        outputHeight = int((videoHeight - filterHeight) /
                           self.d.get_value(borrow=True)[0]) + 1
        outputWidth = int((videoWidth - filterWidth) /
                          self.d.get_value(borrow=True)[1]) + 1
        outputDur = int((videoDur - filterDur) /
                        self.d.get_value(borrow=True)[2]) + 1

        self.W.set_value(self.random_tensor(numFilters, filterHeight,
                    filterWidth, filterDur, inputChannels), borrow=True)
        self.b.set_value(self.random_tensor(numFilters), borrow=True)
        self.rb.set_value(self.random_tensor(inputChannels), borrow=True)

        self.V.set_value(self.random_tensor(batchSize, videoHeight,
                    videoWidth, videoDur, inputChannels), borrow=True)
        self.rb.set_value(self.random_tensor(inputChannels), borrow=True)

    def test_c_against_python(self):
        self.randomize()
        self.check_c_against_python(self.V.get_value(borrow=True).shape[1:4])

    @attr('slow')
    def test_c_against_mat_mul(self):
        # Use a filter of the same size as the image, so the convolution is
        # just a dense matrix multiply.
        # Check that dense matrix multiplication gives the same result as
        # convolution.

        batchSize = self.rng.randint(1, 10)
        videoDur = self.rng.randint(3, 10)
        videoWidth = self.rng.randint(1, 5)
        videoHeight = self.rng.randint(1, 5)

        filterWidth = videoWidth
        filterHeight = videoHeight
        filterDur = videoDur
        numFilters = self.rng.randint(1, 3)
        inputChannels = self.rng.randint(1, 4)

        self.d.get_value(borrow=True, return_internal_type=True)[0] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[1] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[2] = \
            self.rng.randint(1, 15)

        self.W.set_value(self.random_tensor(numFilters, filterHeight,
                filterWidth, filterDur, inputChannels), borrow=True)
        self.W.set_value(self.W.get_value(borrow=True) *
                (self.W.get_value(borrow=True) < 1e-5), borrow=True)

        self.b.set_value(self.random_tensor(numFilters), borrow=True)

        self.V.set_value(self.random_tensor(batchSize, videoHeight,
                videoWidth, videoDur, inputChannels), borrow=True)

        Hv = self.H_func()

        assert Hv.shape[1] == 1
        assert Hv.shape[2] == 1
        assert Hv.shape[3] == 1

        n = inputChannels * videoHeight * videoWidth * videoDur
        W_mat = N.zeros((n, numFilters))
        V_mat = N.zeros((batchSize, n))
        Hv_mat = N.zeros((batchSize, numFilters))
        for qi in xrange(0, numFilters):
            W_mat[:, qi] = \
                    self.W.get_value(borrow=True)[qi, :, :, :, :].reshape((n))
            Hv_mat[:, qi] = Hv[:, 0, 0, 0, qi]
        for qi in xrange(0, batchSize):
            V_mat[qi, :] = \
                    self.V.get_value(borrow=True)[qi, :, :, :, :].reshape((n))

        H_mat = N.dot(V_mat, W_mat) + self.b.get_value(borrow=True)

        tol = 1e-5
        if floatX == 'float32':
            tol = 1e-4

        if N.abs(H_mat - Hv_mat).max() > tol and not N.allclose(H_mat, Hv_mat):
            print(H_mat)
            print(Hv_mat)
            print('max error: ' + str(N.abs(H_mat - Hv_mat).max()))
            W.get_value(borrow=True)[W.get_value(borrow=True) != 0] += 1.0
            print('min non-zero kernel mag: ' + \
                str(N.abs(W.get_value(borrow=True)).min()))
            assert False

    def test_c_against_mat_transp_mul(self):
    # Use a filter of the same size as the image, so the convolution is just a
    # dense matrix multiply.
    # Check that dense matrix multiplication by the transpose of the matrix
    # gives the same result as ConvTransp.
        batchSize = self.rng.randint(1, 10)
        videoDur = self.rng.randint(3, 15)
        videoWidth = self.rng.randint(3, 15)
        videoHeight = self.rng.randint(3, 15)

        filterWidth = videoWidth
        filterHeight = videoHeight
        filterDur = videoDur
        numFilters = self.rng.randint(1, 15)
        inputChannels = self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[0] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[1] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[2] = \
            self.rng.randint(1, 15)

        self.W.set_value(self.random_tensor(numFilters, filterHeight,
                    filterWidth, filterDur, inputChannels), borrow=True)

        self.b.set_value(self.random_tensor(numFilters), borrow=True)

        self.V.set_value(self.random_tensor(batchSize, videoHeight,
                    videoWidth, videoDur, inputChannels), borrow=True)
        self.rb.set_value(self.random_tensor(inputChannels), borrow=True)

        H_shape = self.H_shape_func()

        assert H_shape[1] == 1
        assert H_shape[2] == 1
        assert H_shape[3] == 1

        Hv = self.random_tensor( * H_shape)

        Vv = self.transp_func(Hv, [videoHeight, videoWidth, videoDur])

        n = inputChannels * videoHeight * videoWidth * videoDur
        rbim = N.zeros((videoHeight, videoWidth, videoDur, inputChannels))
        for qi in xrange(0, inputChannels):
            rbim[:, :, :, qi] = self.rb.get_value(borrow=True)[qi]
        rbv = rbim.reshape((n))
        W_mat = N.zeros((numFilters, n))
        Vv_mat = N.zeros((n, batchSize))
        Hv_mat = N.zeros((numFilters, batchSize))
        for qi in xrange(0, numFilters):
            W_mat[qi, :] = \
                    self.W.get_value(borrow=True)[qi, :, :, :, :].reshape((n))
            Hv_mat[qi, :] = Hv[:, 0, 0, 0, qi]
        for qi in xrange(0, batchSize):
            Vv_mat[:, qi] = Vv[qi, :, :, :, :].reshape((n))

        V_mat = (N.dot(W_mat.transpose(), Hv_mat).transpose() + \
                 rbv).transpose()

        if N.abs(V_mat - Vv_mat).max() > 1e-5:
            print(V_mat)
            print(Vv_mat)

            for qq in xrange(V_mat.shape[0]):
                for qqq in xrange(Vv_mat.shape[1]):
                    if abs(V_mat[qq, qqq] - Vv_mat[qq, qqq]) > 1e-5:
                        print(('wrong at ' + str((qq, qqq)) + ': ' +
                        str(V_mat[qq, qqq], Vv_mat[qq, qqq])))
                        assert False

    def test_c_against_sparse_mat_transp_mul(self):
    # like test_c_against_mat_transp_mul but using a sparse matrix and a kernel
    # that is smaller than the image
        if not theano.sparse.enable_sparse:
            raise SkipTest('Optional package sparse disabled')

        batchSize = self.rng.randint(1, 3)
        filterWidth = self.rng.randint(1, 8)
        filterHeight = self.rng.randint(1, 8)
        filterDur = self.rng.randint(1, 8)

        self.d.get_value(borrow=True, return_internal_type=True)[0] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[1] = \
            self.rng.randint(1, 15)
        self.d.get_value(borrow=True, return_internal_type=True)[2] = \
            self.rng.randint(1, 15)

        dr = self.d.get_value(borrow=True)[0]
        dc = self.d.get_value(borrow=True)[1]
        dt = self.d.get_value(borrow=True)[2]

        numFilters = self.rng.randint(1, 3)
        row_steps = self.rng.randint(1, 4)
        col_steps = self.rng.randint(1, 4)
        time_steps = self.rng.randint(1, 4)

        #print (row_steps,col_steps,time_steps)

        videoDur = (time_steps - 1) * dt + filterDur + \
                      self.rng.randint(0, 3)
        videoWidth = (col_steps - 1) * dc + filterWidth + \
                      self.rng.randint(0, 3)
        videoHeight = (row_steps - 1) * dr + filterHeight + \
                      self.rng.randint(0, 3)

        inputChannels = self.rng.randint(1, 15)

        self.W.set_value(self.random_tensor(numFilters, filterHeight,
                filterWidth, filterDur, inputChannels), borrow=True)
        self.b.set_value(self.random_tensor(numFilters), borrow=True)
        # just needed so H_shape works
        self.V.set_value(self.random_tensor(batchSize, videoHeight, videoWidth,
                            videoDur, inputChannels), borrow=True)
        self.rb.set_value(self.random_tensor(inputChannels), borrow=True)

        H_shape = self.H_shape_func()

        # make index maps
        h = N.zeros(H_shape[1:], dtype='int32')
        r = N.zeros(H_shape[1:], dtype='int32')
        c = N.zeros(H_shape[1:], dtype='int32')
        t = N.zeros(H_shape[1:], dtype='int32')

        for qi in xrange(0, H_shape[4]):
            h[:, :, :, qi] = qi
        for qi in xrange(0, H_shape[1]):
            r[qi, :, :, :] = qi
        for qi in xrange(0, H_shape[2]):
            c[:, qi, :, :] = qi
        for qi in xrange(0, H_shape[3]):
            t[:, :, qi, :] = qi

        hn = H_shape[1] * H_shape[2] * H_shape[3] * H_shape[4]

        h = h.reshape((hn))
        r = r.reshape((hn))
        c = c.reshape((hn))
        t = t.reshape((hn))

        Hv = self.random_tensor(*H_shape)

        Vv = self.transp_func(Hv, [videoHeight, videoWidth, videoDur])

        n = inputChannels * videoHeight * videoWidth * videoDur
        rbim = N.zeros((videoHeight, videoWidth, videoDur, inputChannels))
        for qi in xrange(0, inputChannels):
            rbim[:, :, :, qi] = self.rb.get_value(borrow=True)[qi]
        rbv = rbim.reshape((n))

        W_mat = N.zeros((hn, n))
        Vv_mat = N.zeros((n, batchSize))
        Hv_mat = N.zeros((hn, batchSize))
        for qi in xrange(0, hn):
            hi = h[qi]
            ri = r[qi]
            ci = c[qi]
            ti = t[qi]

            placed_filter = N.zeros(self.V.get_value(borrow=True).shape[1:])

            placed_filter[
                    ri * dr:ri * dr + self.W.get_value(borrow=True).shape[1],
                    ci * dc:ci * dc + self.W.get_value(borrow=True).shape[2],
                    ti * dt:ti * dt + self.W.get_value(borrow=True).shape[3],
                    :] = self.W.get_value(borrow=True)[hi, :, :, :, :]

            W_mat[qi, :] = placed_filter.reshape((n))
            Hv_mat[qi, :] = Hv[:, ri, ci, ti, hi]
        for qi in xrange(0, batchSize):
            Vv_mat[:, qi] = Vv[qi, :, :, :, :].reshape((n))

        W_mat_T = sparse.csr_matrix(W_mat.transpose())

        temp = W_mat_T * Hv_mat
        V_mat = (temp.transpose() + rbv).transpose()

        if N.abs(V_mat - Vv_mat).max() > 1e-5:
            print('mul')
            print(V_mat)
            print('conv')
            print(Vv_mat)
            for i in xrange(0, n):
                for j in xrange(0, batchSize):
                    if abs(V_mat[i, j] - Vv_mat[i, j]) > 1e-5:
                        print(('wrong at %d,%d: %f mul versus %f conv'
                               % (i, j, V_mat[i, j], Vv_mat[i, j])))
            assert False

    def test_infer_shape(self):
        self.randomize()
        # Conv3D
        self._compile_and_check([], [self.H], [], Conv3D)

        # ConvTransp3D
        self._compile_and_check([self.RShape], [self.R],
                    [self.V.get_value(borrow=True).shape[1:4]], ConvTransp3D)

        # ConvGrad3D
        self._compile_and_check([self.RShape], [T.grad(self.reconsObj, self.W),
                                            T.grad(self.reconsObj, self.H),
                                            T.grad(self.reconsObj, self.V),
                                            T.grad(self.reconsObj, self.b)],
                    [self.V.get_value(borrow=True).shape[1:4]], ConvGrad3D)

    def test_gradient(self):
        self.randomize()
        rng, V, W, b, d, rb = self.rng, self.V, self.W, self.b, self.d, self.rb
        dCdH = shared(self.random_tensor(*self.H_shape_func()))
        testsPerDir = 2
        theano.tests.unittest_tools.verify_grad(DummyConv3D(rng, (V, W, b), d),
                                        [0.0], n_tests=testsPerDir)
        theano.tests.unittest_tools.verify_grad(DummyConvTransp3D(rng,
                        (W, rb, dCdH), d, V.get_value(borrow=True).shape[1:4]),
                                        [0.0], n_tests=testsPerDir)
        theano.tests.unittest_tools.verify_grad(DummyConvGrad3D(rng, (V, dCdH),
                        d, W.get_value(borrow=True).shape),
                                        [0.0], n_tests=testsPerDir)


if __name__ == '__main__':

    t = TestConv3D('setUp')
    t.setUp()
    t.test_infer_shape()
