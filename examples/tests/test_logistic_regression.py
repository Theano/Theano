
#
# UNIT TEST
#
import unittest
import numpy
from theano import gof

from theano.gradient import *
from theano import gradient

import sys
from theano import tensor as T
from theano.tensor import nnet
from theano.compile import module
from theano import printing, pprint
from theano import compile

import numpy as N

class test_logistic_regression_example(unittest.TestCase):

    def test_example(self): 
        """Test that the file execute without trouble"""
        from ..examples import logistic_regression
        logistic_regression.main()



if __name__ == '__main__':
    unittest.main()
