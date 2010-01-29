import sys, time, unittest
import numpy
from scipy import signal

import theano
import theano.tensor as T
from theano import function, Mode
from theano.tests import unittest_tools as utt

from theano.tensor.signal import conv

from theano.tensor.basic import _allclose

class TestConv2D(unittest.TestCase):

    def setUp(self):
        utt.seed_rng()
        self.input = T.dtensor4('input')
        self.filters = T.dtensor4('filters')

    def validate(self, image_shape, filter_shape,
                 border_mode='valid', subsample=(1,1),
                 N_image_shape=None, N_filter_shape=None,
                 input=None, filters=None, 
                 unroll_batch=0, unroll_kern=0, unroll_patch=True,
                 verify_grad=True):

        if N_image_shape is None:
            N_image_shape = image_shape
        if N_filter_shape is None:
            N_filter_shape = filter_shape
    
        if not input:
            input = self.input
        if not filters:
            filters = self.filters
        
        ############# THEANO IMPLEMENTATION ############
        
        # we create a symbolic function so that verify_grad can work
        def sym_conv2d(input, filters):
            # define theano graph and function
            return conv.conv2d(input, filters, image_shape, filter_shape,
                          border_mode, subsample, unroll_batch=unroll_batch,
                          unroll_kern=unroll_kern, unroll_patch=unroll_patch)

        output = sym_conv2d(input, filters)
        theano_conv = theano.function([input, filters], output)
          
        # initialize input and compute result
        image_data  = numpy.random.random(N_image_shape)
        filter_data = numpy.random.random(N_filter_shape)
        theano_output = theano_conv(image_data, filter_data)

        ############# REFERENCE IMPLEMENTATION ############
        s = 1. if border_mode is 'full' else -1.
        out_shape2d = numpy.array(N_image_shape[-2:]) +\
                      s*numpy.array(N_filter_shape[-2:]) - s
        out_shape2d = numpy.ceil(out_shape2d / numpy.array(subsample))
        out_shape = (N_image_shape[0],N_filter_shape[0]) + tuple(out_shape2d)
        ref_output = numpy.zeros(out_shape)

        # loop over output feature maps
        for k in range(N_filter_shape[0]):
            # loop over input feature maps
            for l in range(N_filter_shape[1]):

                filter2d = filter_data[k,l,:,:]

                # loop over mini-batches
                for b in range(N_image_shape[0]):
                    image2d = image_data[b,l,:,:]
                    output2d = signal.convolve2d(image2d, filter2d, border_mode)

                    ref_output[b,k,:,:] +=\
                       output2d[::subsample[0],::subsample[1]]

        self.failUnless(_allclose(theano_output, ref_output))

        ############# TEST GRADIENT ############
        if verify_grad:
            utt.verify_grad(sym_conv2d, [image_data, filter_data])


    def test_basic(self):
        """
        Tests that basic convolutions work for odd and even dimensions of image and filter
        shapes, as well as rectangular images and filters.
        """
        self.validate((3,2,8,8), (4,2,5,5), 'valid')
        self.validate((3,2,7,5), (5,2,2,3), 'valid')
        self.validate((3,2,7,5), (5,2,3,2), 'valid')
        self.validate((3,2,8,8), (4,2,5,5), 'full')
        self.validate((3,2,7,5), (5,2,2,3), 'full')
        # test filter same size as input
        self.validate((3,2,3,3), (4,2,3,3), 'valid')

    def test_unroll_patch_false(self):
        """
        unroll_patch is True by default. Test basic convs with False.
        """
        self.validate((3,2,7,5), (5,2,2,3), 'valid', unroll_patch=False)
        self.validate((3,2,7,5), (5,2,2,3), 'full', unroll_patch=False)
        self.validate((3,2,3,3), (4,2,3,3), 'valid', unroll_patch=False)

    def test_unroll_special(self):
        """
        (unroll_kern, unroll_batch) in (0,1),(1,0) is special case.
        """
        self.validate((6,2,3,3), (3,2,2,2), 'valid', unroll_batch=1)

    def test_unroll_batch(self):
        """
        Test mini-batch unrolling for various legal values.
        """
        # mini-batch of size 6 is multiple of 2 and 3. Should work.
        self.validate((6,2,3,3), (3,2,2,2), 'valid', unroll_batch=2, verify_grad=False)
        self.validate((6,2,3,3), (3,2,2,2), 'valid', unroll_batch=3, verify_grad=False)

    def test_unroll_kern(self):
        """
        Test kernel unrolling for various legal values.
        """
        # 6 filters is a multiple of 2 and 3. Should work.
        self.validate((2,3,3,3), (6,3,2,2), 'valid', unroll_kern=2, verify_grad=False)
        self.validate((2,3,3,3), (6,3,2,2), 'valid', unroll_kern=3, verify_grad=False)

    def test_subsample(self):
        """
        Tests convolution where subsampling != (1,1)
        """
        self.validate((3,2,7,5), (5,2,2,3), 'valid', subsample=(2,2))
        self.validate((3,2,7,5), (5,2,2,3), 'full', subsample=(2,2))
        self.validate((3,2,7,5), (5,2,2,3), 'valid', subsample=(2,1))

    def test_invalid_filter_shape(self):
        """
        Tests scenario where filter_shape[1] != input_shape[1]
        """
        def f():
            self.validate((3,2,8,8), (4,3,5,5), 'valid')
        self.failUnlessRaises(AssertionError, f)

    def test_missing_info(self):
        """
        Test convolutions for various pieces of missing info.
        """
        self.validate(None, None, 
                      N_image_shape=(3,2,8,8), 
                      N_filter_shape=(4,2,5,5))
        self.validate((3,2,None,None), None,
                      N_image_shape=(3,2,8,8), 
                      N_filter_shape=(4,2,5,5))
        self.validate((None,2,None,None), (None,2,5,5),
                      N_image_shape=(3,2,8,8), 
                      N_filter_shape=(4,2,5,5))

    def test_full_mode(self):
        """
        Tests basic convolution in full mode and case where filter 
        is larger than the input image.
        """
        self.validate((3,2,5,5), (4,2,8,8), 'full')
        def f():
            self.validate((3,2,5,5), (4,2,8,8), 'valid')
        self.failUnlessRaises(Exception, f)

    def test_wrong_input(self):
        """
        Make sure errors are raised when image and kernel are not 4D tensors
        """
        try:
            self.validate((3,2,8,8), (4,2,5,5), 'valid', input = T.dmatrix())
            self.validate((3,2,8,8), (4,2,5,5), 'valid', filters = T.dvector())
            self.validate((3,2,8,8), (4,2,5,5), 'valid', input = T.dtensor3())
            # should never reach here
            self.fail()
        except: 
            pass
