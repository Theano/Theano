"""
 Tests for the Op decorator
"""

import unittest
from theano.tests  import unittest_tools as utt
from theano import function
import theano
from theano import tensor
from theano.tensor import dmatrix, dvector
import numpy as np 
from numpy import allclose
from theano.compile import as_op

class OpDecoratorTests(unittest.TestCase): 
    def test_1arg(self):
        x = dmatrix('x')
        
        @as_op(dmatrix, dvector)
        def diag(x): 
            return np.diag(x)

        fn = function([x], diag(x))
        r = fn([[1.5, 5],[2, 2]])
        r0 = np.array([1.5, 2])

        assert allclose(r, r0), (r, r0)

    def test_2arg(self):
        x = dmatrix('x')
        x.tag.test_value=np.zeros((2,2))
        y = dvector('y')
        y.tag.test_value=[0,0]
        
        @as_op([dmatrix, dvector], dvector)
        def diag_mult(x, y): 
            return np.diag(x) * y 

        fn = function([x, y], diag_mult(x, y))
        r = fn([[1.5, 5],[2, 2]], [1, 100])
        r0 = np.array([1.5, 200])
        print r

        assert allclose(r, r0), (r, r0)

