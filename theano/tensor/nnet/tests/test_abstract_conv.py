import unittest
from theano.tensor.nnet.abstract_conv import get_conv_output_shape


class TestGetConvOutShape(unittest.TestCase):

    def test_basic(self):
        image_shape, kernel_shape = (3, 2, 8, 9), (4, 2, 5, 6)
        sub_sample = (1, 2)
        test1_params = get_conv_output_shape(
            image_shape, kernel_shape, 'valid', sub_sample)
        test2_params = get_conv_output_shape(
            image_shape, kernel_shape, 'half', sub_sample)
        test3_params = get_conv_output_shape(
            image_shape, kernel_shape, 'full', sub_sample)
        test4_params = get_conv_output_shape(
            image_shape, kernel_shape, (1, 2), sub_sample)

        self.assertTrue(test1_params == (3, 4, 4, 2))
        self.assertTrue(test2_params == (3, 4, 8, 5))
        self.assertTrue(test3_params == (3, 4, 12, 7))
        self.assertTrue(test4_params == (3, 4, 6, 4))
