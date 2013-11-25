import unittest

from nose.plugins.skip import SkipTest
import numpy

import theano
import theano.tensor as T
from theano.tests import unittest_tools as utt

from theano.tensor.signal import conv

from theano.tensor.basic import _allclose


class TestSignalConv2D(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()

    def validate(self, image_shape, filter_shape, verify_grad=True):

        image_dim = len(image_shape)
        filter_dim = len(filter_shape)
        input = T.TensorType('float64', [False] * image_dim)()
        filters = T.TensorType('float64', [False] * filter_dim)()

        bsize = image_shape[0]
        if image_dim != 3:
            bsize = 1
        nkern = filter_shape[0]
        if filter_dim != 3:
            nkern = 1

        ############# THEANO IMPLEMENTATION ############
        # we create a symbolic function so that verify_grad can work
        def sym_conv2d(input, filters):
            return conv.conv2d(input, filters)
        output = sym_conv2d(input, filters)
        theano_conv = theano.function([input, filters], output)

        # initialize input and compute result
        image_data = numpy.random.random(image_shape)
        filter_data = numpy.random.random(filter_shape)
        theano_output = theano_conv(image_data, filter_data)

        ############# REFERENCE IMPLEMENTATION ############
        out_shape2d = numpy.array(image_shape[-2:]) -\
                      numpy.array(filter_shape[-2:]) + 1
        ref_output = numpy.zeros(tuple(out_shape2d))

        # reshape as 3D input tensors to make life easier
        image_data3d = image_data.reshape((bsize,) + image_shape[-2:])
        filter_data3d = filter_data.reshape((nkern,) + filter_shape[-2:])
        # reshape theano output as 4D to make life easier
        theano_output4d = theano_output.reshape((bsize, nkern,) +
                                                theano_output.shape[-2:])

        # loop over mini-batches (if required)
        for b in range(bsize):

            # loop over filters (if required)
            for k in range(nkern):

                image2d = image_data3d[b, :, :]
                filter2d = filter_data3d[k, :, :]
                output2d = numpy.zeros(ref_output.shape)
                for row in range(ref_output.shape[0]):
                    for col in range(ref_output.shape[1]):
                        output2d[row, col] += (
                            image2d[row:row + filter2d.shape[0],
                                    col:col + filter2d.shape[1]] *
                            filter2d[::-1, ::-1]
                            ).sum()

                self.assertTrue(_allclose(theano_output4d[b, k, :, :],
                                          output2d))

        ############# TEST GRADIENT ############
        if verify_grad:
            utt.verify_grad(sym_conv2d, [image_data, filter_data])

    def test_basic(self):
        """
        Basic functionality of nnet.conv.ConvOp is already tested by
        its own test suite.  We just have to test whether or not
        signal.conv.conv2d can support inputs and filters of type
        matrix or tensor3.
        """
        if (not theano.tensor.nnet.conv.imported_scipy_signal and
            theano.config.cxx == ""):
            raise SkipTest("conv2d tests need SciPy or a c++ compiler")

        self.validate((1, 4, 5), (2, 2, 3), verify_grad=True)
        self.validate((7, 5), (5, 2, 3), verify_grad=False)
        self.validate((3, 7, 5), (2, 3), verify_grad=False)
        self.validate((7, 5), (2, 3), verify_grad=False)

    def test_fail(self):
        """
        Test that conv2d fails for dimensions other than 2 or 3.
        """
        self.assertRaises(Exception, conv.conv2d, T.dtensor4(), T.dtensor3())
        self.assertRaises(Exception, conv.conv2d, T.dtensor3(), T.dvector())

    def test_bug_josh_reported(self):
        """
        Test refers to a bug reported by Josh, when due to a bad merge these
        few lines of code failed. See
        http://groups.google.com/group/theano-dev/browse_thread/thread/8856e7ca5035eecb
        """
        m1 = theano.tensor.matrix()
        m2 = theano.tensor.matrix()
        conv.conv2d(m1, m2)
