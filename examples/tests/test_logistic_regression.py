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
        from ..examples import logistic_regression
        logistic_regression.main()

    def test_example_moduleN(self): 
        """Test that the LogisticRegressionN module execute the same with different mode"""
        from ..examples import logistic_regression
        pprint.assign(nnet.crossentropy_softmax_1hot_with_bias_dx, printing.FunctionPrinter('xsoftmaxdx'))
        pprint.assign(nnet.crossentropy_softmax_argmax_1hot_with_bias, printing.FunctionPrinter('nll', 'softmax', 'argmax'))

        lrc = logistic_regression.LogisticRegressionN()
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
        train(lr1)
        train(lr2)
        train(lr3)
        train(lr4)

#        print 'lr1',lr1
        print 'lr2',lr2
        def assert_equal(l1,l2):
            if not (numpy.abs(l1.b-l2.b)<1e-8).all():
                print numpy.abs(l1.b-l2.b)<1e-10
                print numpy.abs(l1.b-l2.b)
                self.fail()
            if not (numpy.abs(l1.w-l2.w)<1e-8).all():
                print numpy.abs(l1.w-l2.w)<1e-10
                print numpy.abs(l1.w-l2.w)
                self.fail()
            assert l1.lr==l2.lr
        assert_equal(lr1,lr2)
        assert_equal(lr1,lr3)
        assert_equal(lr1,lr4)
        assert lr1==lr2
        assert lr1==lr3
        assert lr1==lr4

    def test_example_module2(self): 
        """Test that the LogisticRegression2 module execute the same with different mode"""
        from ..examples import logistic_regression
        lrc = logistic_regression.LogisticRegression2() #TODO: test 2==N
        lr0 = lrc.make(10,2)
#        lr0 = lrc.make(10,2,seed=1827)#error
#        lr1 = lrc.make(10, 2, mode=theano.Mode('c|py', 'fast_run'), seed=1827)
#         lr2 = lrc.make(10, 2, mode=theano.Mode('py', 'fast_run'), seed=1827)
#         lr3 = lrc.make(10, 2, mode=theano.Mode('py', 'merge'), seed=1827) #'FAST_RU

#         lr4 = lrc.make(10, 2, mode=compile.FAST_RUN.excluding('fast_run'), seed=1827)
#         #FAST_RUN, FAST_COMPILE, 
#         data_x = N.random.randn(5, 10)
#         data_y = (N.random.randn(5) > 0)

#         def train(lr):
#             for i in xrange(10000):
#                 lr.lr = 0.02
#                 xe = lr.update(data_x, data_y) 
#        train(lr1)
#        train(lr2)
#        train(lr3)
 #       train(lr4)

#        self.fail("NotImplementedError")

if __name__ == '__main__':
    from theano.tests import main
    main("test_wiki")
