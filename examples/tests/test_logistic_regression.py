#!/usr/bin/env python
#
# UNIT TEST
#
import unittest
import numpy
from theano import gof

from theano.gradient import *
from theano import gradient
import theano

import sys
from theano import tensor as T
from theano.tensor import nnet
from theano.compile import module
from theano import printing, pprint
from theano import compile

import numpy as N

class test_logistic_regression_example(unittest.TestCase):

    def test_example_main(self): 
        """Test that the file execute without trouble"""
        import os
        sys.path.append(os.path.realpath(".."))
        import logistic_regression
        logistic_regression.main()

    def test_example_moduleN(self): 
        """Test that the LogisticRegressionN module execute the same with different mode"""
        import os
        sys.path.append(os.path.realpath(".."))
        import logistic_regression
        pprint.assign(nnet.crossentropy_softmax_1hot_with_bias_dx, printing.FunctionPrinter('xsoftmaxdx'))
        pprint.assign(nnet.crossentropy_softmax_argmax_1hot_with_bias, printing.FunctionPrinter('nll', 'softmax', 'argmax'))

        lrc = logistic_regression.LogisticRegressionN()
        lr0 = lrc.make(10, 2, seed=1827)
        lr1 = lrc.make(10, 2, mode=theano.Mode('c|py', 'fast_run'), seed=1827)
        lr2 = lrc.make(10, 2, mode=theano.Mode('py', 'fast_run'), seed=1827)
        lr3 = lrc.make(10, 2, mode=theano.Mode('py', 'merge'), seed=1827) #'FAST_RUN')
        lr4 = lrc.make(10, 2, mode=compile.FAST_RUN.excluding('fast_run'), seed=1827)
        #FAST_RUN, FAST_COMPILE, 
        data_x = N.random.randn(5, 10)
        data_y = (N.random.randn(5) > 0)

        def train(lr):
            for i in xrange(1000):
                lr.lr = 0.02
                xe = lr.update(data_x, data_y) 

        train(lr0)
        train(lr1)
        train(lr2)
        train(lr3)
        train(lr4)

        assert lr0==lr1
        assert lr0==lr2
        assert lr0==lr3
        assert lr0==lr4

    def test_example_module2(self): 
        """Test that the LogisticRegression2 module execute the same with different mode"""
        import os
        sys.path.append(os.path.realpath(".."))
        import logistic_regression
        lrc = logistic_regression.LogisticRegression2() #TODO: test 2==N
        lr0 = lrc.make(10,1827)
        lr1 = lrc.make(10, mode=theano.Mode('c|py', 'fast_run'), seed=1827)
        lr2 = lrc.make(10, mode=theano.Mode('py', 'fast_run'), seed=1827)
        lr3 = lrc.make(10, mode=theano.Mode('py', 'merge'), seed=1827) #'FAST_RU
        lr4 = lrc.make(10, mode=compile.FAST_RUN.excluding('fast_run'), seed=1827)
         #FAST_RUN, FAST_COMPILE, 
        data_x = N.random.randn(5, 10)
        data_y = (N.random.randn(5) > 0)
        data_y = data_y.reshape((data_y.shape[0],1))#need to be a column
        

        def train(lr):
            for i in xrange(1000):
                lr.lr = 0.02
                xe = lr.update(data_x, data_y) 

        train(lr0)
        train(lr1)
        train(lr2)
        train(lr3)
        train(lr4)

        assert lr0==lr1
        assert lr0==lr2
        assert lr0==lr3
        assert lr0==lr4

#        self.fail("NotImplementedError")

if __name__ == '__main__':
    from theano.tests import main
    main("test_logistic_regression")
